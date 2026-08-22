#!/bin/bash
#SBATCH --job-name=tee
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --chdir=/blue/prabhat/hansikam.lokukat/NoC/traceGNN/GNN-HQC-schedular
#SBATCH --output=/blue/prabhat/hansikam.lokukat/NoC/traceGNN/GNN-HQC-schedular/logs/global_%j.log

module purge
module load conda
conda activate /blue/prabhat/hansikam.lokukat/conda/envs/hqc_gnn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1

cd /blue/prabhat/hansikam.lokukat/NoC/traceGNN/GNN-HQC-schedular

python scripts/train_hipergator_glo.py \
    --sched_cfg configs.scheduler_config_glo \
    --cost_cfg cost_config_v3_6.json \
    --run_tag global30_tp6 \
    --global_features \
    --capacity_mode sinkhorn