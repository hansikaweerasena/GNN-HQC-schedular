#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/mosaic_%j.log

module purge
module load conda
conda activate /blue/prabhat/$USER/conda/envs/hqc_gnn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /blue/prabhat/hansikam.lokukat/NoC/traceGNN/GNN-HQC-schedular

# Raise open-file limit: 7 DataLoader workers × prefetch × 80 layers × 3 tensors
# easily exceeds the default 1024, causing "Too many open files" in the feeder thread.

python scripts/train_hipergator_n.py \
    --sched_cfg configs.scheduler_config_n \
    --cost_cfg  cost_config_v3_n.json \
    --run_tag   v1_batched_expA