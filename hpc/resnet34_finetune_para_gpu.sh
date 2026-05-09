#!/bin/bash
#PBS -N simclrv2_resnet34_finetune
#PBS -j oe
#PBS -m abe
#PBS -M hieulenho8@gmail.com
#PBS -l select=2:ncpus=4:ngpus=1:mpiprocs=1:mem=32G
#PBS -q para_gpu

cd "$PBS_O_WORKDIR"
# module load cuda/11.8
# source ~/venvs/torch/bin/activate

export OMP_NUM_THREADS=4
UNIQUE_NODEFILE=${PBS_O_WORKDIR}/pbs_unique_nodes_${PBS_JOBID}.txt
sort -u "$PBS_NODEFILE" > "$UNIQUE_NODEFILE"
export MASTER_ADDR=$(head -n 1 "$UNIQUE_NODEFILE")
export MASTER_PORT=${MASTER_PORT:-29503}
export WORLD_SIZE=$(wc -l < "$UNIQUE_NODEFILE")
mkdir -p checkpoints logs

PRETRAIN_CKPT=${PRETRAIN_CKPT:-./checkpoints/simclrv2_resnet34_epoch_800.pt}

mpirun -np "$WORLD_SIZE" -ppn 1 -f "$UNIQUE_NODEFILE" \
  python train_resnet.py \
    --mode finetune \
    --distributed \
    --arch resnet34 \
    --ckpt "$PRETRAIN_CKPT" \
    --labeled-dataset stl10 \
    --data-root ./data \
    --image-size 96 \
    --num-classes 10 \
    --epochs 100 \
    --batch-size 192 \
    --optimizer lars \
    --lr 0.05 \
    --warmup-epochs 0 \
    --weight-decay 0.0 \
    --proj-layers 3 \
    --finetune-proj-layers 1 \
    --proj-dim 128 \
    --num-workers 4 \
    --out-dir ./checkpoints \
    --save-every 10 \
    --log-every 20
