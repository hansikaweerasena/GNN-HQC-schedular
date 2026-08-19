#!/bin/bash
# run_pilot.sbatch — fixed-N=30 S-vs-R pilot, all arms/seeds in parallel.
#
#   sbatch run_pilot.sbatch
#
# Launches 6 independent jobs as a SLURM array:
#   task 0,1,2 -> arm S (sinkhorn),  seeds 0,1,2
#   task 3,4,5 -> arm R (softmax + CapacityPenalty), seeds 0,1,2
#
# Why three seeds per arm and not one. Gate criterion 6 is "arm S is not
# systematically worse than arm R beyond normal run-to-run variance". With one
# run per arm that variance is unknown, so the criterion is unevaluable and a
# single unlucky init could fail a working implementation (or pass a broken
# one). Three seeds is the minimum that gives a spread to compare against.
#
# Arms sharing a seed share their initialisation, data order, and dropout
# stream, so the S-vs-R difference at a given seed is attributable to the
# capacity mechanism alone. This is the paired comparison; seeds give the
# spread.
#
# If GPU allocation is tight, run %2 or %3 concurrency (see --array below) or
# drop to seeds 0,1. Do not drop to a single seed.

#SBATCH --job-name=mosaic_pilot
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/pilot_%A_%a.log

set -euo pipefail

module purge
module load conda
conda activate /blue/prabhat/$USER/conda/envs/hqc_gnn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /blue/prabhat/hansikam.lokukat/NoC/traceGNN/GNN-HQC-schedular

MODES=(sinkhorn sinkhorn sinkhorn softmax softmax softmax)
SEEDS=(0        1        2        0       1       2)

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== pilot task ${SLURM_ARRAY_TASK_ID}: capacity_mode=${MODE} seed=${SEED} ==="

python scripts/train_hipergator.py \
    --sched_cfg     configs.scheduler_config \
    --cost_cfg      cost_config_v3.json \
    --capacity_mode "${MODE}" \
    --seed          "${SEED}" \
    --no_early_stop \
    --run_tag       pilot30

# --no_early_stop is required, not optional. With early stopping the two arms
# can halt at different epochs and would then be compared at different points
# on the temperature schedule (T anneals to its floor around epoch 115 of 140),
# so an EFCL gap could reflect schedule position rather than mechanism.
