#!/usr/bin/env bash
# run_evals.sh — Run eval_scheduler_v1 and eval_scheduler_v2 for each run directory.
# Usage: bash run_evals.sh

# set -e  # exit on first error

DIRS=(
    tp2n_98
    tp1n_98
    tp3n_98
    tp4n_98
    tp5n_98
    tp6n_98
    tp7n_98
    # tp5n_98
    # tp6n_98
    # tp7n_98
    # tp1n_98
    # tp7n_99
    # tp2n_98
    # tp6n_99_50
    # tp3n_98
    # tp4n_98
    # tp5n_98
    # tp6n_98
    # tp7n_98
    # add more directories here
)

RESULTS_BASE="../results"

for DIR in "${DIRS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Evaluating: $DIR"
    echo "========================================"

    # echo "[v1] Synthetic circuits..."
    # python eval_scheduler_v1.py \
    #     --run_dir "${RESULTS_BASE}/${DIR}" \
    #     --n_circuits 300 \
    #     --is_range

    echo "[v2] MQT Bench zero-shot..."
    python eval_scheduler_v2.py \
        --run_dir "${RESULTS_BASE}/${DIR}" \
        --qubit_min 28 \
        --qubit_max 32

    echo "  Done: $DIR"
done

echo ""
echo "All evaluations complete."
