#!/bin/bash
#PBS -N simclrv2_resnet50_pretrain
#PBS -j oe
#PBS -m abe
#PBS -M hieulenho8@gmail.com
#PBS -q long_gpu
#PBS -l select=1:ncpus=1:ngpus=1:mem=24G

cd "$PBS_O_WORKDIR" || exit 1

mkdir -p checkpoints logs

LOG_FILE="logs/${PBS_JOBNAME}_${PBS_JOBID}.log"
exec > "$LOG_FILE" 2>&1

set -ex

PYTHON=/opt/apps/python/3.8.10/bin/python3
export PATH=/opt/apps/python/3.8.10/bin:$PATH
export OMP_NUM_THREADS=1

echo "===== JOB INFO ====="
echo "Job ID: $PBS_JOBID"
echo "Job name: $PBS_JOBNAME"
echo "Workdir: $PBS_O_WORKDIR"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Current dir: $(pwd)"

echo "===== FILE CHECK ====="
ls -lah
test -f train_resnet.py
test -f dataset.py
test -f ssl_simclr.py
test -d resnet18
echo "===== PYTHON ====="
echo "PYTHON=$PYTHON"
$PYTHON --version
$PYTHON -c "import sys; print(sys.executable)"

echo "===== GPU ====="
nvidia-smi || true

echo "===== TORCH/CUDA ====="
$PYTHON -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.c$

echo "===== START TRAIN ====="

$PYTHON train_resnet.py \
  --mode pretrain \
  --arch resnet50 \
  --dataset stl10 \
  --data-root ./data \
  --download \
  --image-size 96 \
  --epochs 300 \
  --batch-size 128 \
  --optimizer lars \
  --lr 0.3 \
  --warmup-epochs 15 \
  --weight-decay 1e-4 \
  --proj-layers 3 \
  --proj-dim 128 \
  --temperature 0.2 \
  --num-workers 1 \
  --out-dir ./checkpoints \
  --save-every 20 \
  --log-every 20

echo "===== END TRAIN ====="
echo "End time: $(date)"
