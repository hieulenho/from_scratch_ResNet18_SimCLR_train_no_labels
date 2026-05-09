from __future__ import annotations

import argparse
import math
import os
import socket
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import get_dataloaders, get_simclr_dataloader
from resnet18 import build_resnet
from ssl_simclr import SimCLR, SimCLRFineTuner, nt_xent_loss


# -------------------------
# Distributed / HPC helpers
# -------------------------

def _first_host_from_pbs() -> str | None:
    nodefile = os.environ.get("PBS_NODEFILE", "")
    if not nodefile or not os.path.isfile(nodefile):
        return None
    with open(nodefile, "r", encoding="utf-8") as f:
        for line in f:
            host = line.strip()
            if host:
                return host
    return None


def _env_int(names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            try:
                return int(value)
            except ValueError:
                pass
    return default


def init_distributed_mode(args):
    """Initialize DDP from torchrun, OpenMPI, MPICH/PMI, MVAPICH, or PBS+mpirun env."""
    rank = _env_int(["RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK"], 0)
    world_size = _env_int(
        ["WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"],
        1,
    )
    local_rank = _env_int(
        ["LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "MV2_COMM_WORLD_LOCAL_RANK"],
        0,
    )

    args.distributed = bool(args.distributed or world_size > 1)
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank

    if not args.distributed:
        args.is_main = True
        return

    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", _first_host_from_pbs() or "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    args.is_main = args.rank == 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(args) -> bool:
    return getattr(args, "is_main", True)


def print_main(args, *values, **kwargs):
    if is_main_process(args):
        print(*values, **kwargs)


def setup_torch_for_speed(device: torch.device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def current_device(args) -> torch.device:
    if torch.cuda.is_available():
        idx = getattr(args, "local_rank", 0) % torch.cuda.device_count()
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def _make_scaler(device: torch.device, use_amp: bool):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.amp.GradScaler(enabled=False)


def _autocast(device: torch.device, use_amp: bool):
    return torch.amp.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=use_amp)


def _maybe_channels_last(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return x.contiguous(memory_format=torch.channels_last)
    return x


def set_sampler_epoch(loader, epoch: int):
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def reduce_stats(loss_sum: float, correct: int = 0, total: int = 0, device: torch.device | None = None):
    if not (dist.is_available() and dist.is_initialized()):
        return loss_sum, correct, total
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([loss_sum, float(correct), float(total)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), int(t[1].item()), int(t[2].item())


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


# -------------------------
# Models / checkpoints
# -------------------------

def build_backbone(arch: str, num_classes: int | None) -> nn.Module:
    return build_resnet(arch, num_classes=num_classes)


def build_simclr_model(args) -> SimCLR:
    backbone = build_backbone(args.arch, num_classes=args.num_classes)
    backbone.fc = nn.Identity()
    feat_dim = getattr(backbone, "feature_dim", 512)
    return SimCLR(
        backbone=backbone,
        feat_dim=feat_dim,
        proj_dim=args.proj_dim,
        proj_hidden_dim=args.proj_hidden_dim or feat_dim,
        proj_layers=args.proj_layers,
    )


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def save_ckpt(path: str, epoch: int, model: nn.Module, optimizer, scheduler, args, extra: Dict | None = None):
    payload = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_ckpt(path: str, model: nn.Module, optimizer=None, scheduler=None, map_location="cpu", strict: bool = True) -> Tuple[int, Dict]:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model", ckpt)
    state = strip_module_prefix(state)
    msg = unwrap_model(model).load_state_dict(state, strict=strict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    return start_epoch, {"checkpoint": ckpt, "load_msg": msg}


# -------------------------
# Optimizers / schedulers
# -------------------------

class LARS(optim.Optimizer):
    """Small LARS optimizer implementation for SimCLR-style large-batch training."""

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=1e-4, eta=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            eta = group["eta"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                update = grad
                # Do not apply weight decay or LARS trust ratio to bias/BN vectors.
                use_lars = p.ndim > 1
                if use_lars and weight_decay != 0:
                    update = update.add(p, alpha=weight_decay)
                if use_lars:
                    p_norm = torch.norm(p)
                    u_norm = torch.norm(update)
                    if p_norm > 0 and u_norm > 0:
                        update = update.mul(eta * p_norm / (u_norm + eps))
                state = self.state[p]
                if "mu" not in state:
                    state["mu"] = torch.zeros_like(p)
                mu = state["mu"]
                mu.mul_(momentum).add_(update)
                p.add_(mu, alpha=-lr)
        return loss


class WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epochs: int, warmup_epochs: int = 0, min_lr: float = 0.0, last_epoch: int = -1):
        self.max_epochs = max(1, max_epochs)
        self.warmup_epochs = max(0, warmup_epochs)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                lr = base_lr * epoch / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
                progress = min(1.0, max(0.0, progress))
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs


def make_optimizer(args, params):
    name = args.optimizer.lower()
    if name == "lars":
        return LARS(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError("optimizer must be one of: lars, sgd, adamw")


# -------------------------
# Train / eval loops
# -------------------------

def train_pretrain(args):
    device = current_device(args)
    setup_torch_for_speed(device)

    loader = get_simclr_dataloader(
        batch_size=args.batch_size,
        data_root=args.data_root,
        dataset=args.dataset,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download=args.download,
        use_cuda=(device.type == "cuda"),
        distributed=args.distributed,
    )

    model = build_simclr_model(args).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = make_optimizer(args, model.parameters())
    scheduler = WarmupCosineLR(
        optimizer,
        max_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = _make_scaler(device, use_amp)

    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch, info = load_ckpt(args.resume, model, optimizer=optimizer, scheduler=scheduler, strict=True)
            print_main(args, f"[Resume] loaded {args.resume}; start_epoch={start_epoch}; {info['load_msg']}")
        else:
            print_main(args, f"[Resume] not found: {args.resume}; ignored")

    if args.distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    os.makedirs(args.out_dir, exist_ok=True)
    print_main(
        args,
        f"[Pretrain] arch={args.arch} world_size={args.world_size} per_gpu_batch={args.batch_size} "
        f"global_batch={args.batch_size * args.world_size} host={socket.gethostname()}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        set_sampler_epoch(loader, epoch)
        model.train()
        loss_sum = 0.0
        n = 0
        t0 = time.time()

        for step, batch in enumerate(loader, start=1):
            if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (tuple, list)):
                (x1, x2), _ = batch
            else:
                x1, x2 = batch

            x1 = _maybe_channels_last(x1.to(device, non_blocking=True), device)
            x2 = _maybe_channels_last(x2.to(device, non_blocking=True), device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                z1 = model(x1)
                z2 = model(x2)
                loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bsz = x1.size(0)
            loss_sum += float(loss.item()) * bsz
            n += bsz
            if args.log_every > 0 and step % args.log_every == 0:
                print_main(args, f"  step {step:04d}/{len(loader)} | loss {loss.item():.4f}", flush=True)

        scheduler.step()
        loss_sum, _, n = reduce_stats(loss_sum, 0, n, device)
        epoch_loss = loss_sum / max(1, n)
        print_main(
            args,
            f"Epoch {epoch:03d}/{args.epochs} | pretrain_loss={epoch_loss:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6g} | time={time.time() - t0:.1f}s",
            flush=True,
        )

        if is_main_process(args) and ((epoch % args.save_every == 0) or (epoch == args.epochs)):
            ckpt_path = os.path.join(args.out_dir, f"simclrv2_{args.arch}_epoch_{epoch:03d}.pt")
            save_ckpt(ckpt_path, epoch, model, optimizer, scheduler, args, extra={"mode": "pretrain"})
            print_main(args, f"Saved: {ckpt_path}", flush=True)


def train_classification_epoch(model, loader, optimizer, scaler, device, args, epoch: int):
    set_sampler_epoch(loader, epoch)
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    use_amp = (device.type == "cuda") and (not args.no_amp)

    for x, y in loader:
        x = _maybe_channels_last(x.to(device, non_blocking=True), device)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += float(loss.item()) * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        total += y.numel()

    loss_sum, correct, total = reduce_stats(loss_sum, correct, total, device)
    return loss_sum / max(1, total), 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate_classification(model, loader, device, args):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    use_amp = (device.type == "cuda") and (not args.no_amp)
    for x, y in loader:
        x = _maybe_channels_last(x.to(device, non_blocking=True), device)
        y = y.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        total += y.numel()
    loss_sum, correct, total = reduce_stats(loss_sum, correct, total, device)
    return loss_sum / max(1, total), 100.0 * correct / max(1, total)


def train_finetune(args):
    if not args.ckpt:
        raise ValueError("--ckpt is required for mode=finetune")

    device = current_device(args)
    setup_torch_for_speed(device)

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_root=args.labeled_data_root or args.data_root,
        dataset=args.labeled_dataset,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download=args.download,
        use_cuda=(device.type == "cuda"),
        distributed=args.distributed,
    )

    simclr = build_simclr_model(args)
    _, info = load_ckpt(args.ckpt, simclr, strict=False)
    print_main(args, f"[FineTune] loaded pretrain ckpt: {args.ckpt}; {info['load_msg']}")

    model = SimCLRFineTuner(
        backbone=simclr.backbone,
        projector=simclr.projector,
        num_classes=args.num_classes,
        include_proj_layers=args.finetune_proj_layers,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = make_optimizer(args, model.parameters())
    scheduler = WarmupCosineLR(
        optimizer,
        max_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = _make_scaler(device, use_amp)

    if args.distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    os.makedirs(args.out_dir, exist_ok=True)
    print_main(
        args,
        f"[FineTune] arch={args.arch} finetune_proj_layers={args.finetune_proj_layers} "
        f"world_size={args.world_size} per_gpu_batch={args.batch_size}",
        flush=True,
    )

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_classification_epoch(model, train_loader, optimizer, scaler, device, args, epoch)
        val_loss, val_acc = evaluate_classification(model, test_loader, device, args)
        scheduler.step()

        print_main(
            args,
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | lr={scheduler.get_last_lr()[0]:.6g} | "
            f"time={time.time() - t0:.1f}s",
            flush=True,
        )

        if is_main_process(args):
            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = os.path.join(args.out_dir, f"finetune_{args.arch}_best.pt")
                save_ckpt(ckpt_path, epoch, model, optimizer, scheduler, args, extra={"best_acc": best_acc, "mode": "finetune"})
                print_main(args, f"Saved best: {ckpt_path} (acc={best_acc:.2f}%)", flush=True)
            if (epoch % args.save_every == 0) or (epoch == args.epochs):
                ckpt_path = os.path.join(args.out_dir, f"finetune_{args.arch}_epoch_{epoch:03d}.pt")
                save_ckpt(ckpt_path, epoch, model, optimizer, scheduler, args, extra={"best_acc": best_acc, "mode": "finetune"})
                print_main(args, f"Saved: {ckpt_path}", flush=True)


def train_supervised(args):
    device = current_device(args)
    setup_torch_for_speed(device)
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_root=args.labeled_data_root or args.data_root,
        dataset=args.labeled_dataset,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download=args.download,
        use_cuda=(device.type == "cuda"),
        distributed=args.distributed,
    )
    model = build_backbone(args.arch, num_classes=args.num_classes).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = make_optimizer(args, model.parameters())
    scheduler = WarmupCosineLR(optimizer, max_epochs=args.epochs, warmup_epochs=args.warmup_epochs, min_lr=args.min_lr)
    scaler = _make_scaler(device, (device.type == "cuda") and (not args.no_amp))
    if args.distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_classification_epoch(model, train_loader, optimizer, scaler, device, args, epoch)
        val_loss, val_acc = evaluate_classification(model, test_loader, device, args)
        scheduler.step()
        print_main(
            args,
            f"Epoch {epoch:03d}/{args.epochs} | train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
            flush=True,
        )
        if is_main_process(args) and val_acc > best_acc:
            best_acc = val_acc
            save_ckpt(os.path.join(args.out_dir, f"supervised_{args.arch}_best.pt"), epoch, model, optimizer, scheduler, args)


def linear_eval(args):
    """Kept for backward compatibility; finetune is recommended for SimCLRv2."""
    if not args.ckpt:
        raise ValueError("--ckpt is required for mode=linear-eval")
    # linear eval = finetune classifier only, no projection prefix, frozen backbone.
    device = current_device(args)
    setup_torch_for_speed(device)
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_root=args.labeled_data_root or args.data_root,
        dataset=args.labeled_dataset,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download=args.download,
        use_cuda=(device.type == "cuda"),
        distributed=args.distributed,
    )
    simclr = build_simclr_model(args)
    load_ckpt(args.ckpt, simclr, strict=False)
    model = SimCLRFineTuner(simclr.backbone, simclr.projector, args.num_classes, include_proj_layers=0).to(device)
    for name, p in model.named_parameters():
        if not name.startswith("classifier"):
            p.requires_grad = False
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = WarmupCosineLR(optimizer, max_epochs=args.epochs, warmup_epochs=args.warmup_epochs, min_lr=args.min_lr)
    scaler = _make_scaler(device, (device.type == "cuda") and (not args.no_amp))
    if args.distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_classification_epoch(model, train_loader, optimizer, scaler, device, args, epoch)
        val_loss, val_acc = evaluate_classification(model, test_loader, device, args)
        scheduler.step()
        print_main(args, f"[LinearEval] epoch={epoch:03d} train_acc={train_acc:.2f}% val_acc={val_acc:.2f}%")


# -------------------------
# CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ResNet34/50 + SimCLRv2 pretrain and fine-tune")

    parser.add_argument("--mode", choices=["pretrain", "simclr", "finetune", "fine-tune", "supervised", "linear-eval"], default="pretrain")
    parser.add_argument("--arch", choices=["resnet18", "resnet34", "resnet50"], default="resnet50")
    parser.add_argument("--num-classes", type=int, default=10)

    # data
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="stl10", help='Unlabeled dataset for pretrain: "stl10" or "folder"')
    parser.add_argument("--labeled-data-root", type=str, default="", help="Optional labeled data root for finetune/supervised")
    parser.add_argument("--labeled-dataset", type=str, default="stl10", help='Labeled dataset: "stl10" or ImageFolder "folder"')
    parser.add_argument("--download", action="store_true", help="Download STL10 if missing")
    parser.add_argument("--image-size", type=int, default=96)

    # training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128, help="Per-process/per-GPU batch size under DDP")
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["lars", "sgd", "adamw"], default="lars")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--log-every", type=int, default=50)

    # simclrv2
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--proj-hidden-dim", type=int, default=0, help="0 means use backbone feature dim")
    parser.add_argument("--proj-layers", type=int, default=3, help="SimCLRv2 uses a 3-layer MLP projection head")
    parser.add_argument("--finetune-proj-layers", type=int, default=1, help="Keep first N hidden projection blocks during fine-tuning")
    parser.add_argument("--temperature", type=float, default=0.2)

    # checkpoints
    parser.add_argument("--out-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="", help="Resume a pretraining run")
    parser.add_argument("--ckpt", type=str, default="", help="Pretrained SimCLR checkpoint for finetune/linear-eval")

    # distributed / PBS / MPI
    parser.add_argument("--distributed", action="store_true", help="Use DDP; also auto-enabled when mpirun/torchrun sets world_size>1")
    parser.add_argument("--master-port", type=int, default=29500)

    args = parser.parse_args()
    mode_alias = {"simclr": "pretrain", "fine-tune": "finetune"}
    args.mode = mode_alias.get(args.mode, args.mode)
    return args


def main():
    args = parse_args()
    init_distributed_mode(args)
    try:
        if args.mode == "pretrain":
            train_pretrain(args)
        elif args.mode == "finetune":
            train_finetune(args)
        elif args.mode == "supervised":
            train_supervised(args)
        elif args.mode == "linear-eval":
            linear_eval(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()
