#!/bin/bash
#SBATCH --job-name=gpucheck
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:10:00
#SBATCH --output=gpucheck_%j.out

module purge
module load conda
conda activate /blue/prabhat/$USER/conda/envs/hqc_gnn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  configs/cost_config_v3.json \
    --run_tag   dry \
    --dry_run