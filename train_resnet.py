import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import get_dataloaders, get_simclr_dataloader
from resnet18 import resnet18
from Validate_loop import train_one_epoch, evaluate
from ssl_simclr import SimCLR, nt_xent_loss


def setup_torch_for_speed(device: torch.device):
    """Practical CUDA performance switches."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _make_scaler(device: torch.device, use_amp: bool):
    # On CPU, keep scaler disabled.
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.amp.GradScaler(enabled=False)


def _autocast(device: torch.device, use_amp: bool):
    # device_type must be 'cuda' or 'cpu'
    return torch.amp.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=use_amp)


def _maybe_channels_last(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return x.contiguous(memory_format=torch.channels_last)
    return x




def _save_simclr_ckpt(path: str, epoch: int, model: nn.Module, optimizer, scheduler, args):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        },
        path,
    )


def _load_simclr_ckpt(path: str, model: nn.Module, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    return start_epoch, ckpt


def train_simclr(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_torch_for_speed(device)

    loader = get_simclr_dataloader(
        batch_size=args.batch_size,
        data_root=args.data_root,
        dataset=args.dataset,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download=args.download,
    )

    backbone = resnet18(num_classes=10)
    backbone.fc = nn.Identity()

    model = SimCLR(backbone=backbone, feat_dim=512, proj_dim=args.proj_dim).to(device)

    
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = _make_scaler(device, use_amp)

    os.makedirs(args.out_dir, exist_ok=True)

    # Resume if requested
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch, _ = _load_simclr_ckpt(args.resume, model, optimizer=optimizer, scheduler=scheduler, device="cpu")
            print(f"[Resume] Loaded checkpoint: {args.resume} -> start at epoch {start_epoch}")
        else:
            print(f"[Resume] Not found: {args.resume} (ignored)")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        t0 = time.time()

        for step, batch in enumerate(loader, start=1):
           
            if (
                isinstance(batch, (tuple, list))
                and len(batch) == 2
                and isinstance(batch[0], (tuple, list))
                and len(batch[0]) == 2
            ):
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

            if args.log_every > 0 and (step % args.log_every == 0):
                print(f"  step {step:04d}/{len(loader)} | loss {loss.item():.4f}", flush=True)

        scheduler.step()

        epoch_loss = loss_sum / max(1, n)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"SimCLR Loss: {epoch_loss:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.6f} | "
            f"time: {dt:.1f}s"
        )

        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt_path = os.path.join(args.out_dir, f"simclr_epoch_{epoch:03d}.pt")
            _save_simclr_ckpt(ckpt_path, epoch, model, optimizer, scheduler, args)
            print(f"Saved: {ckpt_path}")


def train_supervised(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_torch_for_speed(device)

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        download=args.download,
    )

    model = resnet18(num_classes=10).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = _make_scaler(device, use_amp)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler, use_amp=use_amp
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"Train Acc {train_acc:.2f}% | "
            f"Val Acc {val_acc:.2f}%"
        )


def _extract_backbone_state(simclr_state: dict) -> dict:
    """
    Trích state_dict của backbone từ checkpoint SimCLR.
    Hỗ trợ các prefix hay gặp: backbone., module.backbone., encoder., module.encoder.
    """
    prefixes = ["backbone.", "module.backbone.", "encoder.", "module.encoder."]
    for p in prefixes:
        if any(k.startswith(p) for k in simclr_state.keys()):
            return {k[len(p):]: v for k, v in simclr_state.items() if k.startswith(p)}
    # Fallback: assume it's already backbone-compatible
    return simclr_state


def linear_eval(args):
    """
    Linear evaluation:
    - Load backbone from SimCLR checkpoint
    - Freeze backbone
    - Train a linear classifier on labeled STL10 train
    - Evaluate on labeled STL10 test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_torch_for_speed(device)

    # labeled loaders
    train_loader, test_loader = get_dataloaders(
        batch_size=args.linear_batch_size,
        data_root=args.data_root,
        num_workers=args.num_workers,
        download=args.download,
    )

    # build backbone
    backbone = resnet18(num_classes=10)
    backbone.fc = nn.Identity()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    simclr_state = ckpt["model"]
    backbone_state = _extract_backbone_state(simclr_state)

    missing = backbone.load_state_dict(backbone_state, strict=False)
    print("[LinearEval] Loaded backbone. Missing/Unexpected:", missing)

    backbone = backbone.to(device)
    if device.type == "cuda":
        backbone = backbone.to(memory_format=torch.channels_last)

    # freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    feat_dim = 512
    clf = nn.Linear(feat_dim, 10).to(device)

    opt = optim.SGD(clf.parameters(), lr=args.linear_lr, momentum=0.9, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.linear_epochs)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = _make_scaler(device, use_amp)

    for ep in range(1, args.linear_epochs + 1):
        # train linear head
        clf.train()
        correct = 0
        total = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x = _maybe_channels_last(x.to(device, non_blocking=True), device)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                f = backbone(x)

            opt.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                logits = clf(f)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / max(1, total)

        # eval on test
        clf.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = _maybe_channels_last(x.to(device, non_blocking=True), device)
                y = y.to(device, non_blocking=True)
                f = backbone(x)
                logits = clf(f)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = 100.0 * correct / max(1, total)
        print(f"[LinearEval] epoch {ep:02d}/{args.linear_epochs} | train_acc {train_acc:.2f}% | test_acc {test_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["simclr", "supervised", "linear-eval"], default="simclr")

    # data
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="stl10", help='"stl10" or "folder" (for unlabeled images)')
    parser.add_argument("--download", action="store_true", help="Download STL10 if missing")
    parser.add_argument("--image-size", type=int, default=96)

    # training (simclr/supervised)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile (PyTorch 2.x)")
    parser.add_argument("--log-every", type=int, default=50, help="Print every N steps inside an epoch (simclr)")

    # simclr
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="", help="Resume SimCLR from a checkpoint path")

    # linear eval
    parser.add_argument("--ckpt", type=str, default="", help="Path to simclr checkpoint for linear-eval")
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--linear-batch-size", type=int, default=256)
    parser.add_argument("--linear-lr", type=float, default=0.1)

    args = parser.parse_args()

    if args.mode == "simclr":
        train_simclr(args)
    elif args.mode == "supervised":
        train_supervised(args)
    else:  # linear-eval
        if not args.ckpt:
            raise ValueError("--ckpt is required for linear-eval")
        linear_eval(args)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()