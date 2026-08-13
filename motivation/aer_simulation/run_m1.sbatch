#!/bin/bash
#SBATCH --job-name=mosaic_m1
#SBATCH --array=0-29
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/m1_%a.out

# M1: one seed per array task. Each task is ~35 s per circuit at 8q/depth 20,
# so a single seed finishes in a couple of minutes; the array exists so the
# whole sweep returns in wall-clock minutes rather than ~20 min serial.
#
#   mkdir -p logs results
#   sbatch run_m1.sbatch
#   python -m mosaic_aer.exp_m1 --report 'results/m1_*.json' --csv m1_all.csv

module load conda 2>/dev/null || true
# conda activate mosaic

mkdir -p logs results

python -m mosaic_aer.exp_m1 \
    --seed0 "${SLURM_ARRAY_TASK_ID}" \
    --seeds 1 \
    --depth 20 \
    --families hotcore \
    --out "results/m1_${SLURM_ARRAY_TASK_ID}.json"
