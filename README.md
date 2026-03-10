# ResNet18 + SimCLR on STL10

A compact PyTorch project for **image representation learning** with a custom **ResNet18** backbone.
This repository supports three training workflows in one codebase:

- **Supervised learning** on STL10
- **Self-supervised learning (SimCLR)** on unlabeled images
- **Linear evaluation** to measure representation quality after SimCLR pretraining

The core idea is simple: train a reusable ResNet18 backbone, learn stronger features with SimCLR, then evaluate those features with a frozen-backbone linear probe.

---

## Highlights

- ResNet18 implemented **from scratch** with `BasicBlock`
- Backbone can return either **classification logits** or **512-dim features**
- SimCLR projection head + **NT-Xent loss**
- Support for both:
  - **STL10 unlabeled split**
  - **custom unlabeled image folders**
- SimCLR **checkpointing** and **resume training**
- **Linear evaluation** from a SimCLR checkpoint
- Practical CUDA optimizations:
  - AMP
  - TF32
  - `channels_last`
  - `pin_memory`
  - `non_blocking=True`
  - optional `torch.compile`

---

## Project structure

```text
.
├── README.md
├── train_resnet.py
├── dataset.py
├── ssl_simclr.py
├── Validate_loop.py
├── resnet18.py
├── basicblock.py
├── __init__.py
└── hiunhooooo.ipynb
```

### What each file does

- **`train_resnet.py`**: main training entrypoint for `simclr`, `supervised`, and `linear-eval`
- **`dataset.py`**: STL10 loaders, SimCLR augmentations, and unlabeled folder loader
- **`ssl_simclr.py`**: SimCLR model wrapper and NT-Xent loss
- **`Validate_loop.py`**: supervised train / evaluation loop
- **`resnet18.py`**: custom ResNet18 backbone
- **`basicblock.py`**: residual block used by ResNet18
- **`hiunhooooo.ipynb`**: notebook workspace / experiments

---

## Training modes

### 1) Supervised
Train ResNet18 directly on labeled STL10 data.

### 2) SimCLR
Train the backbone on unlabeled images using contrastive learning.
The code supports:

- `dataset=stl10` → use **STL10 unlabeled** split
- `dataset=folder` → use a **custom unlabeled image directory**

### 3) Linear evaluation
Load a pretrained SimCLR backbone, **freeze it**, then train a linear classifier on top.
This is the standard way to evaluate learned representations.

---

## Model overview

### ResNet18 backbone
The backbone is written manually and can operate in two modes:

- `return_features=False` → return logits for classification
- `return_features=True` → return a pooled **512-dimensional feature vector**

This design allows the same encoder to be reused for both supervised classification and self-supervised pretraining.

### SimCLR head
The SimCLR module wraps the backbone and adds a projection head:

- Linear
- BatchNorm1d
- ReLU
- Linear
- L2 normalization

Training uses **NT-Xent loss** over two augmented views of the same image.

---

## Data pipeline

### Supervised pipeline
For labeled STL10 training / evaluation:

- random crop with padding
- random horizontal flip
- tensor conversion
- STL10 normalization

### SimCLR pipeline
For self-supervised learning:

- random resized crop
- random horizontal flip
- color jitter
- random grayscale
- Gaussian blur
- tensor conversion
- STL10 normalization

Two views are created from the same image using `TwoCropsTransform`.

---

## Installation

Recommended environment:

- Python 3.10+
- PyTorch
- torchvision
- Pillow

Install core dependencies:

```bash
pip install torch torchvision pillow
```

If you use NVIDIA GPU, install the PyTorch build that matches your CUDA version.

---

## Dataset setup

## Option A — STL10
Expected layout:

```text
./data/
└── stl10_binary/
```

The code uses:

- `train` split for supervised training
- `test` split for evaluation
- `unlabeled` split for SimCLR

## Option B — Custom unlabeled folder
You can also train SimCLR on your own image collection:

```text
./data/my_unlabeled_images/
├── img_001.jpg
├── img_002.png
└── nested_folder/
    └── img_003.webp
```

Supported extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

---

## Quick start

### Train SimCLR on STL10 unlabeled

```bash
python train_resnet.py \
  --mode simclr \
  --dataset stl10 \
  --data-root ./data \
  --download \
  --epochs 200 \
  --batch-size 256 \
  --lr 3e-4 \
  --proj-dim 128 \
  --temperature 0.2
```

### Train SimCLR on a custom unlabeled folder

```bash
python train_resnet.py \
  --mode simclr \
  --dataset folder \
  --data-root ./data/my_unlabeled_images \
  --epochs 200 \
  --batch-size 256
```

### Resume SimCLR from a checkpoint

```bash
python train_resnet.py \
  --mode simclr \
  --dataset stl10 \
  --data-root ./data \
  --resume ./checkpoints/simclr_epoch_050.pt \
  --epochs 200
```

### Train supervised on STL10

```bash
python train_resnet.py \
  --mode supervised \
  --data-root ./data \
  --download \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.1
```

### Run linear evaluation

```bash
python train_resnet.py \
  --mode linear-eval \
  --data-root ./data \
  --ckpt ./checkpoints/simclr_epoch_200.pt \
  --linear-epochs 20 \
  --linear-batch-size 256 \
  --linear-lr 0.1
```

---

## Important CLI arguments

### Shared arguments

- `--mode`: `simclr`, `supervised`, `linear-eval`
- `--data-root`: dataset path
- `--epochs`: epochs for `simclr` and `supervised`
- `--batch-size`: batch size for `simclr` and `supervised`
- `--lr`: learning rate for `simclr` and `supervised`
- `--weight-decay`: optimizer weight decay
- `--num-workers`: dataloader workers
- `--image-size`: input resolution
- `--no-amp`: disable mixed precision
- `--compile`: try `torch.compile`

### SimCLR-specific arguments

- `--dataset`: `stl10` or `folder`
- `--proj-dim`: projection head dimension
- `--temperature`: NT-Xent temperature
- `--out-dir`: checkpoint directory
- `--save-every`: save every N epochs
- `--resume`: resume checkpoint path
- `--log-every`: print every N steps

### Linear evaluation arguments

- `--ckpt`: SimCLR checkpoint path
- `--linear-epochs`: epochs for linear head training
- `--linear-batch-size`: batch size for linear eval
- `--linear-lr`: learning rate for linear head

---

## Recommended workflow

### Baseline route
1. Train `--mode supervised`
2. Record STL10 test accuracy

### Representation learning route
1. Pretrain with `--mode simclr`
2. Save SimCLR checkpoint
3. Run `--mode linear-eval`
4. Compare linear-eval accuracy against the supervised baseline

---

## Checkpoints

SimCLR checkpoints are saved to:

```text
./checkpoints/
```

File naming pattern:

```text
simclr_epoch_XXX.pt
```

Each checkpoint stores:

- current epoch
- model weights
- optimizer state
- scheduler state
- training arguments

---

## Performance notes

The current training script includes several practical speed optimizations for CUDA:

- `torch.backends.cudnn.benchmark = True`
- TF32 enabled for cuDNN and matrix multiplication
- mixed precision with `torch.amp`
- `channels_last` memory format on CUDA
- `pin_memory=True`
- `non_blocking=True` for tensor transfer
- `persistent_workers=True` on non-Windows when workers are used
- optional `torch.compile`

---

## Troubleshooting

### 1) STL10 download / path issue
The dataset helper currently resolves the STL10 folder **before** constructing the torchvision dataset object.
That means on a completely fresh machine, `--download` may still fail unless the expected `stl10_binary` directory already exists under your data root.

If that happens, either:

- prepare the STL10 folder manually first, or
- patch `dataset.py` so root resolution happens after download logic

### 2) Package layout note
The uploaded files are currently shown at the project root, but `resnet18.py` uses a relative import:

```python
from .basicblock import BasicBlock
```

If you run the repo as a flat script layout, you may need to adjust imports.
A cleaner layout is to place:

- `__init__.py`
- `basicblock.py`
- `resnet18.py`

inside a dedicated module folder.

### 3) README vs runtime tuning
The current dataloader helper uses `prefetch_factor=1` on non-Windows systems when workers are enabled.
If you benchmark performance, rely on the actual code settings rather than older README notes.

---

## Future improvements

- add a proper `requirements.txt`
- log training curves to TensorBoard or Weights & Biases
- add top-1 / top-5 reporting
- support checkpoint selection by best validation score
- add k-NN evaluation for representations
- package the backbone as a clean Python module

---

## Summary

This project is a strong compact foundation for:

- learning image features with SimCLR
- comparing supervised and self-supervised training
- reusing a custom ResNet18 backbone across multiple workflows

It is especially useful if you want a repo that is small enough to understand end-to-end, while still covering the full pipeline from pretraining to evaluation.
