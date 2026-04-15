#!/bin/bash
# Batch size sweep: 3 datasets × 3 batch sizes × 4 methods × 3 seeds = 108 runs.
#
# Usage:
#   bash experiments/pretrain/sweep_batch_size.sh              # run all
#   bash experiments/pretrain/sweep_batch_size.sh cifar100      # one dataset
#   bash experiments/pretrain/sweep_batch_size.sh cifar100 64   # one dataset + BS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$SCRIPT_DIR"

FILTER_DATASET="${1:-all}"
FILTER_BS="${2:-all}"

# Dataset-specific batch sizes and epochs.
# All values lie inside the locked-in 5-point sweep {8, 16, 32, 48, 64};
# BS=128 was dropped because it falls past the convnextv2_nano saturation
# point and adds wall-clock without adding new physics.
declare -A BATCH_SIZES=(
    [cifar100]="32 48 64"
    [cifar10]="32 48 64"
    [flowers102]="16 32 64"
)
declare -A EPOCHS=(
    [cifar100]=400
    [cifar10]=400
    [flowers102]=2000
)

SEEDS="42 43 44"
METHODS="sigreg w1"
ACCUM_FLAGS=("" "--accumulate")

count=0
total=0

# Count total
for DATASET in cifar100 cifar10 flowers102; do
    [[ "$FILTER_DATASET" != "all" && "$FILTER_DATASET" != "$DATASET" ]] && continue
    for BS in ${BATCH_SIZES[$DATASET]}; do
        [[ "$FILTER_BS" != "all" && "$FILTER_BS" != "$BS" ]] && continue
        for METHOD in $METHODS; do
            for ACCUM in "${ACCUM_FLAGS[@]}"; do
                for SEED in $SEEDS; do
                    total=$((total + 1))
                done
            done
        done
    done
done

echo "=== Batch Size Sweep ==="
echo "  Total runs: $total"
echo ""

for DATASET in cifar100 cifar10 flowers102; do
    [[ "$FILTER_DATASET" != "all" && "$FILTER_DATASET" != "$DATASET" ]] && continue
    for BS in ${BATCH_SIZES[$DATASET]}; do
        [[ "$FILTER_BS" != "all" && "$FILTER_BS" != "$BS" ]] && continue
        for METHOD in $METHODS; do
            for ACCUM in "${ACCUM_FLAGS[@]}"; do
                for SEED in $SEEDS; do
                    count=$((count + 1))
                    echo "[$count/$total] dataset=$DATASET bs=$BS method=$METHOD ${ACCUM:+pooled} seed=$SEED"
                    $PYTHON train.py \
                        --dataset "$DATASET" \
                        --batch-size "$BS" \
                        --regularizer "$METHOD" \
                        $ACCUM \
                        --seed "$SEED" \
                        --epochs "${EPOCHS[$DATASET]}" \
                        --save-dir "$REPO_ROOT/runs/sweep" \
                        --lambd 0.05 \
                        --proj-dim 64 \
                        --proj-hidden 512 \
                        --num-proj 1024 \
                        --save-interval 25
                done
            done
        done
    done
done

echo ""
echo "=== All $total runs complete ==="
