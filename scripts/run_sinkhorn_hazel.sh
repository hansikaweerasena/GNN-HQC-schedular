#!/bin/bash

#SBATCH --job-name=mosaic_global
#SBATCH --account=liuquantumproj_gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/liuquantumproj/hlokuka/GNN-HQC-runs/logs/mosaic_global_%j.out
#SBATCH --error=/share/liuquantumproj/hlokuka/GNN-HQC-runs/logs/mosaic_global_%j.err

set -euo pipefail

CODE_ROOT="/home/hlokuka/GNN-HQC-schedular"
RUN_ROOT="/share/liuquantumproj/hlokuka/GNN-HQC-runs"
ENV_ROOT="/share/liuquantumproj/hlokuka/conda/envs/hqc_gnn"
PYTHON="$ENV_ROOT/bin/python"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "============================================================"
echo "MOSAIC HAZEL DRY RUN"
echo "============================================================"
echo "Job ID:              $SLURM_JOB_ID"
echo "Node:                $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python:              $PYTHON"
echo "Code root:           $CODE_ROOT"
echo "Results root:        $RUN_ROOT/results"
echo "Date:                $(date)"
echo

echo "---- GPU ----"
nvidia-smi
echo

echo "---- Environment ----"
"$PYTHON" - <<'PY'
import sys
import torch
import torch_geometric
import qiskit

print("Python:", sys.version)
print("Python executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("PyG:", torch_geometric.__version__)
print("Qiskit:", qiskit.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the allocated GPU.")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory GB:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
)
PY

echo
echo "---- Repository ----"
cd "$CODE_ROOT"

echo "Git commit:"
git rev-parse HEAD

echo "Git status:"
git status --short

echo
echo "---- Starting MOSAIC dry run ----"

"$PYTHON" scripts/train_hipergator_glo.py \
    --sched_cfg configs.scheduler_config_glo \
    --cost_cfg cost_config_v3.json \
    --run_tag gf30 \
    --results_root "$RUN_ROOT/results" \
    --global_features

echo
echo "============================================================"
echo "MOSAIC DRY RUN COMPLETED SUCCESSFULLY"
echo "Finished: $(date)"
echo "============================================================"