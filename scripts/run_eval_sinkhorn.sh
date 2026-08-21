#!/usr/bin/env bash
# run_eval_sinkhorn.sh — synthetic baseline comparison for the Sinkhorn pilot.
#
#   bash run_eval_sinkhorn.sh
#
# Fixed N=30 only. No --is_range: the model was trained at N=30 with
# C_total=40, and range mode would sweep sizes the pilot never covered
# (and past N=40 there is no feasible assignment at all).

set -euo pipefail

RUN_DIR="../results/global30_sinkhorn"

# echo "=== v1: synthetic circuits, N=30, MOSAIC vs B1/B3/B4/B5 ==="
# python eval_scheduler_v1.py \
#     --run_dir    "${RUN_DIR}" \
#     --checkpoint best \
#     --n_circuits 300 \
#     --seed       99999 \
#     --num_qubits 30 \
#     --efcl-hardener

echo "=== v2: MQT benchmark circuits, N=30, MOSAIC vs B1/B3/B4/B5 ==="
python eval_scheduler_v2.py \
    --run_dir "${RUN_DIR}" \
    --checkpoint best \
    --qubit_min 30 --qubit_max 30 \
    --efcl-hardener

echo ""
echo "Results -> ${RUN_DIR}/eval_syn_best/"
