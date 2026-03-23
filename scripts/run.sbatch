#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/mosaic_%j.log

module purge
module load conda
conda activate /blue/prabhat/$USER/conda/envs/hqc_gnn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /blue/prabhat/hansikam.lokukat/NoC/traceGNN/GNN-HQC-schedular

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_v3.json \
    --run_tag   baseline_v1