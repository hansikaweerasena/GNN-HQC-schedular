#!/bin/bash
#SBATCH --job-name=qnn_train
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --chdir=/blue/prabhat/hansikam.lokukat/NoC/traceGNN/GNN-HQC-schedular
#SBATCH --output=/blue/prabhat/hansikam.lokukat/NoC/traceGNN/GNN-HQC-schedular/logs/mosaic_%j.log

module purge
module load conda
conda activate /blue/prabhat/hansikam.lokukat/conda/envs/hqc_gnn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1

cd /blue/prabhat/hansikam.lokukat/NoC/traceGNN/GNN-HQC-schedular

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp1n_99.json \
    --run_tag   v1_batched_tp1n_99

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp1n_98.json \
    --run_tag   v1_batched_tp1n_98    