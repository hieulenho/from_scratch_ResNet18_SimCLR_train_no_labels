from __future__ import annotations

import copy
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """SimCLRv2-style MLP projection head.

    num_layers=3 gives: Linear-BN-ReLU -> Linear-BN-ReLU -> Linear.
    The hidden width defaults to the backbone feature dimension, matching the
    SimCLRv2 paper's practical setting for standard-width ResNets.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 128,
        hidden_dim: int | None = None,
        num_layers: int = 3,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        hidden_dim = hidden_dim or in_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.num_layers = num_layers

        layers: List[nn.Module] = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, proj_dim, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def make_prefix(self, include_hidden_layers: int) -> nn.Sequential:
        """Return a copy of the first N hidden projection blocks for fine-tuning.

        include_hidden_layers=1 corresponds to SimCLRv2 fine-tuning from the
        first layer of a 3-layer projection head.
        """
        if include_hidden_layers < 0:
            raise ValueError("include_hidden_layers must be >= 0")
        max_hidden = self.num_layers - 1
        if include_hidden_layers > max_hidden:
            raise ValueError(
                f"include_hidden_layers={include_hidden_layers} is invalid for "
                f"a {self.num_layers}-layer projection head; max is {max_hidden}."
            )
        modules = list(self.net.children())[: include_hidden_layers * 3]
        return nn.Sequential(*[copy.deepcopy(m) for m in modules])

    def prefix_out_dim(self, include_hidden_layers: int) -> int:
        return self.in_dim if include_hidden_layers == 0 else self.hidden_dim


class SimCLR(nn.Module):
    """Backbone encoder + SimCLRv2 projection head."""

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        proj_dim: int = 128,
        proj_hidden_dim: int | None = None,
        proj_layers: int = 3,
    ):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.projector = ProjectionHead(
            in_dim=feat_dim,
            proj_dim=proj_dim,
            hidden_dim=proj_hidden_dim or feat_dim,
            num_layers=proj_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x, return_features=True)
        z = self.projector(feats)
        return F.normalize(z, dim=1)


class SimCLRFineTuner(nn.Module):
    """Fine-tuning model initialized from a pretrained SimCLR model.

    The key SimCLRv2 change is include_proj_layers=1 by default: we keep the
    first hidden projection block as part of the supervised model instead of
    discarding the full projection head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: ProjectionHead,
        num_classes: int,
        include_proj_layers: int = 1,
    ):
        super().__init__()
        self.backbone = backbone
        self.include_proj_layers = include_proj_layers
        self.proj_prefix = projector.make_prefix(include_proj_layers)
        classifier_in_dim = projector.prefix_out_dim(include_proj_layers)
        self.classifier = nn.Linear(classifier_in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x, return_features=True)
        feats = self.proj_prefix(feats)
        return self.classifier(feats)


class _GatherWithGrad(torch.autograd.Function):
    """All-gather tensors while preserving gradients for the local rank."""

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        if not (dist.is_available() and dist.is_initialized()):
            return (x,)
        world_size = dist.get_world_size()
        outputs = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(outputs, x.contiguous())
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        if not (dist.is_available() and dist.is_initialized()):
            return grads[0]
        rank = dist.get_rank()
        return grads[rank]


def _concat_all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(_GatherWithGrad.apply(x), dim=0)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent loss that works for both single GPU and DDP multi-GPU training.

    In DDP, each rank uses its local samples as anchors and all gathered samples
    as negatives. DDP then averages the per-rank gradients, matching the global
    contrastive objective without double-scaling the loss.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    local_bsz = z1.shape[0]
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0

    z1_all = _concat_all_gather_with_grad(z1) if distributed else z1
    z2_all = _concat_all_gather_with_grad(z2) if distributed else z2
    global_bsz = z1_all.shape[0]
    z_all = torch.cat([z1_all, z2_all], dim=0)

    anchors = torch.cat([z1, z2], dim=0)
    logits = (anchors @ z_all.t()).float() / temperature

    local_index = rank * local_bsz + torch.arange(local_bsz, device=z1.device)
    self_index = torch.cat([local_index, global_bsz + local_index], dim=0)
    target = torch.cat([global_bsz + local_index, local_index], dim=0)

    logits.scatter_(1, self_index.view(-1, 1), float("-inf"))
    return F.cross_entropy(logits, target)
