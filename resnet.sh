#!/bin/bash
#PBS -N simclrv2_resnet50_pretrain
#PBS -j oe
#PBS -m abe
#PBS -M hieulenho8@gmail.com
#PBS -l select=3:ncpus=4:ngpus=1:mpiprocs=1:mem=32G
#PBS -q para_gpu

cd "$PBS_O_WORKDIR"

# Edit these paths for the server environment.
# module load cuda/11.8
# source ~/venvs/torch/bin/activate

export OMP_NUM_THREADS=4
UNIQUE_NODEFILE=${PBS_O_WORKDIR}/pbs_unique_nodes_${PBS_JOBID}.txt
sort -u "$PBS_NODEFILE" > "$UNIQUE_NODEFILE"
export MASTER_ADDR=$(head -n 1 "$UNIQUE_NODEFILE")
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$(wc -l < "$UNIQUE_NODEFILE")

mkdir -p checkpoints logs

echo "PBS_NODEFILE=$PBS_NODEFILE"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"
cat "$UNIQUE_NODEFILE"

mpirun -np "$WORLD_SIZE" -ppn 1 -f "$UNIQUE_NODEFILE" \
  python train_resnet.py \
    --mode pretrain \
    --distributed \
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
    --temperature 0.2 \
    --num-workers 4 \
    --out-dir ./checkpoints \
    --save-every 20 \
    --log-every 20
