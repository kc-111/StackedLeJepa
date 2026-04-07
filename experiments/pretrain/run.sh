#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$PYTHONPATH"
cd "$SCRIPT_DIR"

case "${1:-help}" in
  sigreg)
    python trainer.py \
      --regularizer sigreg \
      --save-dir "$REPO_ROOT/runs" \
      "${@:2}"
    ;;
  w1)
    python trainer.py \
      --regularizer w1 \
      --accumulate \
      --save-dir "$REPO_ROOT/runs" \
      "${@:2}"
    ;;
  w2)
    python trainer.py \
      --regularizer w2 \
      --accumulate \
      --save-dir "$REPO_ROOT/runs" \
      "${@:2}"
    ;;
  accum-sigreg)
    python trainer.py \
      --regularizer sigreg \
      --accumulate \
      --save-dir "$REPO_ROOT/runs" \
      "${@:2}"
    ;;
  sweep)
    bash sweep_batch_size.sh "${@:2}"
    ;;
  eval)
    python evaluate.py \
      --checkpoint "${2:?Usage: run.sh eval <checkpoint_path>}" \
      "${@:3}"
    ;;
  *)
    echo "Usage: $0 {sigreg|w1|w2|accum-sigreg|sweep|eval} [extra args...]"
    echo ""
    echo "  sigreg        SIGReg (non-accumulated)"
    echo "  w1            W1 pooled (1-large-pass)"
    echo "  w2            W2 pooled (1-large-pass)"
    echo "  accum-sigreg  SIGReg pooled (1-large-pass)"
    echo "  sweep [args]  Batch size sweep (see sweep_batch_size.sh)"
    echo "  eval <ckpt>   Evaluate checkpoint"
    echo ""
    echo "All commands accept extra args, e.g.:"
    echo "  $0 w1 --dataset cifar10 --batch-size 64 --epochs 400"
    exit 1
    ;;
esac
