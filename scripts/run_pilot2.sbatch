#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=20:00:00
#SBATCH --output=logs/softmax_%j.log

module purge
module load conda
conda activate /blue/prabhat/$USER/conda/envs/hqc_gnn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /blue/prabhat/hansikam.lokukat/NoC/traceGNN/GNN-HQC-schedular

# echo "=== DRY RUN: SINKHORN, seed=0 ==="

# python scripts/train_hipergator.py \
#     --sched_cfg configs.scheduler_config \
#     --cost_cfg cost_config_v3.json \
#     --capacity_mode sinkhorn \
#     --seed 0 \
#     --no_early_stop \
#     --run_tag pilot30_sinkhorn


echo "=== DRY RUN: SOFTMAX, seed=0 ==="

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config_temp \
    --cost_cfg cost_config_v3.json \
    --capacity_mode softmax \
    --seed 0 \
    --no_early_stop \
    --run_tag pilot30_softmax