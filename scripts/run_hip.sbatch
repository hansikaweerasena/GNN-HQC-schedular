#!/bin/bash
#SBATCH --job-name=qnn_train
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=120:00:00
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
    --cost_cfg  cost_config_tp2n_99.json \
    --run_tag   v1_batched_tp2n_99

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp3n_99.json \
    --run_tag   v1_batched_tp3n_99

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp4n_99.json \
    --run_tag   v1_batched_tp4n_99

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp5n_99.json \
    --run_tag   v1_batched_tp5n_99

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp6n_99.json \
    --run_tag   v1_batched_tp6n_99

python scripts/train_hipergator_n.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp7n_99.json \
    --run_tag   v1_batched_tp7n_99

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp3n_98.json \
    --run_tag   v1_batched_tp3n_98    

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp2n_98.json \
    --run_tag   v1_batched_tp2n_98    

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp4n_98.json \
    --run_tag   v1_batched_tp4n_98            

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp5n_98.json \
    --run_tag   v1_batched_tp5n_98           

python scripts/train_hipergator.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp6n_98.json \
    --run_tag   v1_batched_tp6n_98         

python scripts/train_hipergator_n.py \
    --sched_cfg configs.scheduler_config \
    --cost_cfg  cost_config_tp7n_98.json \
    --run_tag   v1_batched_tp7n_98             