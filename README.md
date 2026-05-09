# ResNet34/ResNet50 + SimCLRv2 on STL10 or Custom Image Folders

Project nay da duoc nang cap tu ResNet18 + SimCLR thanh workflow gan SimCLRv2 hon:

- Ho tro `resnet18`, `resnet34`, `resnet50`.
- ResNet50 dung `Bottleneck`; ResNet34 dung `BasicBlock`.
- SimCLRv2 projection head mac dinh 3 tang MLP.
- Workflow chinh la `pretrain` -> `finetune`, thay vi chi dung `linear-eval`.
- Fine-tune mac dinh giu lai 1 hidden block dau cua projection head (`--finetune-proj-layers 1`).
- Ho tro DistributedDataParallel qua `mpirun`/PBS `para_gpu` de chay 2-3 node song song.

## Files chinh

```text
train_resnet.py
ssl_simclr.py
dataset.py
resnet18/
  basicblock.py
  resnet18.py
  __init__.py
resnet.sh
hpc/
  resnet34_pretrain_para_gpu.sh
  resnet50_pretrain_para_gpu.sh
  resnet34_finetune_para_gpu.sh
  resnet50_finetune_para_gpu.sh
  smoke_test_cpu.sh
HPC_USAGE.md
```

## Cai dat

```bash
pip install -r requirements.txt
```

## Pretrain SimCLRv2

```bash
python train_resnet.py \
  --mode pretrain \
  --arch resnet50 \
  --dataset stl10 \
  --data-root ./data \
  --image-size 96 \
  --epochs 800 \
  --batch-size 128 \
  --optimizer lars \
  --lr 0.3 \
  --warmup-epochs 40 \
  --weight-decay 1e-4 \
  --proj-layers 3 \
  --proj-dim 128 \
  --temperature 0.2
```

## Fine-tune tu checkpoint pretrain

```bash
python train_resnet.py \
  --mode finetune \
  --arch resnet50 \
  --ckpt ./checkpoints/simclrv2_resnet50_epoch_800.pt \
  --labeled-dataset stl10 \
  --data-root ./data \
  --image-size 96 \
  --num-classes 10 \
  --epochs 100 \
  --batch-size 128 \
  --optimizer lars \
  --lr 0.05 \
  --weight-decay 0.0 \
  --proj-layers 3 \
  --finetune-proj-layers 1
```

## Chay tren HPC01 bang PBS para_gpu

Doc chi tiet trong `HPC_USAGE.md`. Cach nhanh:

```bash
nano resnet.sh
qsub resnet.sh
qstat
```

`resnet.sh` mac dinh chay ResNet50 pretrain tren 3 node, moi node 1 GPU. Doi `select=2` neu muon chay 2 node.

Fine-tune:

```bash
qsub hpc/resnet50_finetune_para_gpu.sh
```

## Kiem tra nhanh

```bash
bash hpc/smoke_test_cpu.sh
```

## Luu y

- `--batch-size` la batch moi GPU; global batch = `batch-size x so GPU`.
- ResNet50 tren Tesla P100 16GB co the can `--batch-size 64` neu het VRAM.
- `linear-eval` van con de tuong thich nguoc, nhung workflow khuyen nghi la `pretrain` va `finetune`.
