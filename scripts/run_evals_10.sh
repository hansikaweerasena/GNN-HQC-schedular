#!/usr/bin/env bash
# run_evals.sh — Run eval_scheduler_v1 and eval_scheduler_v2 for each run directory.
# Usage: bash run_evals.sh

# set -e  # exit on first error

DIRS=(
    # tp2n_99_10
    # tp2n_98_10
    tp2n_99_10_cap8
    # tp2n_98_10_cap8
    # add more directories here
)

RESULTS_BASE="../results"

for DIR in "${DIRS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Evaluating: $DIR"
    echo "========================================"

    echo "[v1] Synthetic circuits..."
    python eval_scheduler_v1.py \
        --run_dir "${RESULTS_BASE}/${DIR}" \
        --n_circuits 300 \
        --is_range

    echo "[v2] MQT Bench zero-shot..."
    python eval_scheduler_v2.py \
        --run_dir "${RESULTS_BASE}/${DIR}" \
        --qubit_min 10 \
        --qubit_max 16

    echo "  Done: $DIR"
done

echo ""
echo "All evaluations complete."
