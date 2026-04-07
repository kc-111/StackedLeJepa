#!/bin/bash
# Run synthetic experiments with parallel workers.
# Usage:
#   bash experiments/synthetic/run_parallel.sh exp1_1     # run exp1_1 with 4 workers
#   bash experiments/synthetic/run_parallel.sh exp1_3 8   # run exp1_3 with 8 workers
#   bash experiments/synthetic/run_parallel.sh all        # run all experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
NUM_WORKERS="${2:-4}"

run_experiment() {
    local script="$1"
    shift
    echo "=== Running $script with $NUM_WORKERS workers ==="

    # Launch workers in parallel
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        $PYTHON "$SCRIPT_DIR/$script" --num-workers "$NUM_WORKERS" --worker-id "$i" "$@" &
    done
    wait
    echo "  Workers done. Generating plots..."

    # Final pass: load all results, generate plots + summary
    $PYTHON "$SCRIPT_DIR/$script" "$@"
    echo "  Done: $script"
    echo
}

case "${1:-help}" in
    exp1_1)
        run_experiment exp1_1_full_batch.py "${@:3}"
        ;;
    exp1_2)
        run_experiment exp1_2_accumulation.py "${@:3}"
        ;;
    exp1_3)
        run_experiment exp1_3_ddp_sim.py "${@:3}"
        ;;
    exp1_4)
        # Timing has no worker sharding, just run directly
        $PYTHON "$SCRIPT_DIR/exp1_4_timing.py" "${@:3}"
        ;;
    exp1_5)
        run_experiment exp1_5_fifo.py "${@:3}"
        ;;
    all)
        run_experiment exp1_1_full_batch.py "${@:3}"
        run_experiment exp1_2_accumulation.py "${@:3}"
        run_experiment exp1_3_ddp_sim.py "${@:3}"
        $PYTHON "$SCRIPT_DIR/exp1_4_timing.py" "${@:3}"
        run_experiment exp1_5_fifo.py "${@:3}"
        ;;
    *)
        echo "Usage: $0 {exp1_1|exp1_2|exp1_3|exp1_4|exp1_5|all} [num_workers] [extra args...]"
        echo ""
        echo "  exp1_1  Full batch baseline"
        echo "  exp1_2  Standard vs pooled accumulation"
        echo "  exp1_3  DDP simulation"
        echo "  exp1_4  Timing comparison (no parallelism)"
        echo "  exp1_5  FIFO buffer"
        echo "  all     Run all experiments"
        echo ""
        echo "Default: 4 workers. Extra args are passed to the script."
        echo "Example: $0 exp1_3 8 --distributions blobs ring"
        exit 1
        ;;
esac
