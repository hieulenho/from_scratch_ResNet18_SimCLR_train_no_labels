# dataset.py
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

_IS_WIN = sys.platform.startswith("win")

STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD = [0.2241, 0.2215, 0.2239]


def _default_num_workers() -> int:
    if _IS_WIN:
        return 0
    cpu = os.cpu_count() or 4
    return min(8, max(2, cpu // 2))


def _resolve_stl10_root(data_root: str) -> str:
    p = Path(data_root)
    if p.name.lower() == "stl10_binary" and p.is_dir():
        p = p.parent

    candidates = [p, p / "STL-10", p / "STL10", p / "stl10", p / "stl-10"]
    for cand in candidates:
        if (cand / "stl10_binary").is_dir():
            return str(cand)

    raise FileNotFoundError(
        "Khong tim thay thu muc 'stl10_binary'. Hay dam bao ton tai 1 trong cac duong dan sau:\n"
        + "\n".join([f"- {str(c / 'stl10_binary')}" for c in candidates])
    )


def _loader_kwargs(num_workers: int, use_cuda: bool):
    kw = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0 and (not _IS_WIN):
        kw.update({"persistent_workers": True, "prefetch_factor": 2})
    return kw


def _make_loader(dataset, batch_size, shuffle, drop_last, num_workers, use_cuda, distributed):
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        drop_last=drop_last,
        **_loader_kwargs(num_workers=num_workers, use_cuda=use_cuda),
    )


def build_train_transform(image_size: int = 96):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
        ]
    )


def build_eval_transform(image_size: int = 96):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
        ]
    )


def get_dataloaders(
    batch_size: int = 128,
    data_root: str = "./data",
    dataset: str = "stl10",
    image_size: int = 96,
    num_workers: Optional[int] = None,
    download: bool = True,
    use_cuda: Optional[bool] = None,
    distributed: bool = False,
):
    """Return labeled train/test dataloaders for fine-tuning or supervised training.

    dataset="stl10": data_root contains STL10/stl10_binary or stl10_binary.
    dataset="folder": data_root has ImageFolder-style train and val/test folders.
    """
    if num_workers is None:
        num_workers = _default_num_workers()
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    dataset = dataset.lower().strip()
    if dataset == "stl10":
        stl_root = _resolve_stl10_root(data_root)
        train_set = datasets.STL10(
            root=stl_root,
            split="train",
            download=download,
            transform=build_train_transform(image_size),
        )
        test_set = datasets.STL10(
            root=stl_root,
            split="test",
            download=download,
            transform=build_eval_transform(image_size),
        )
    elif dataset == "folder":
        root = Path(data_root)
        train_dir = root / "train"
        val_dir = root / "val"
        if not val_dir.is_dir():
            val_dir = root / "test"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(
                "For dataset='folder', expected ImageFolder layout: data_root/train and data_root/val or data_root/test"
            )
        train_set = datasets.ImageFolder(str(train_dir), transform=build_train_transform(image_size))
        test_set = datasets.ImageFolder(str(val_dir), transform=build_eval_transform(image_size))
    else:
        raise ValueError('dataset must be "stl10" or "folder"')

    train_loader = _make_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        use_cuda=use_cuda,
        distributed=distributed,
    )
    test_loader = _make_loader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        use_cuda=use_cuda,
        distributed=distributed,
    )
    return train_loader, test_loader


class GaussianBlur(object):
    """Gaussian blur augmentation (PIL-based) used by SimCLR."""

    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image):
        import random

        if random.random() > self.p:
            return img
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius))


class TwoCropsTransform:
    """Create two differently augmented views of the same image."""

    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


class ViewsOnly(Dataset):
    """Wrapper that drops labels when an upstream dataset returns (views, label)."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return item[0]
        return item


class UnlabeledImageFolder(Dataset):
    """Loads images recursively from a directory and returns only images, no labels."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")
        self.samples = [
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in self.IMG_EXTS
        ]
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.root}. Supported extensions: {sorted(self.IMG_EXTS)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_simclr_transform(image_size: int = 96) -> transforms.Compose:
    """SimCLR/SimCLRv2-style augmentations for self-supervised pretraining."""
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
        ]
    )


def get_simclr_dataloader(
    batch_size: int = 256,
    data_root: str = "./data",
    dataset: str = "stl10",
    image_size: int = 96,
    num_workers: Optional[int] = None,
    download: bool = False,
    use_cuda: Optional[bool] = None,
    distributed: bool = False,
) -> DataLoader:
    """Unlabeled dataloader for SimCLR pretraining.

    dataset="stl10": uses STL10 split='unlabeled'.
    dataset="folder": recursively reads images from data_root.
    """
    if num_workers is None:
        num_workers = _default_num_workers()
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    transform = TwoCropsTransform(build_simclr_transform(image_size=image_size))
    dataset = dataset.lower().strip()
    if dataset == "stl10":
        stl_root = _resolve_stl10_root(data_root)
        base_ds = datasets.STL10(root=stl_root, split="unlabeled", download=download, transform=transform)
        ds = ViewsOnly(base_ds)
    elif dataset == "folder":
        ds = UnlabeledImageFolder(root=data_root, transform=transform)
    else:
        raise ValueError('dataset must be "stl10" or "folder"')

    return _make_loader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        use_cuda=use_cuda,
        distributed=distributed,
    )
