#!/bin/bash
# Benchmark script for GEMV implementations
# Usage: ./run_benchmarks.sh [single|multi|both] [preset]

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/build"

# Make BLIS available if installed locally
if [ -d "$SCRIPT_DIR/../blis_install/lib" ]; then
    export LD_LIBRARY_PATH="$SCRIPT_DIR/../blis_install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

MODE="${1:-both}"
PRESET="${2:-all}"

# Detect physical cores (not hyperthreads)
PCORES=$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l)
if [ "$PCORES" -eq 0 ]; then
    PCORES=$(nproc 2>/dev/null || echo 6)
fi
LAST_PCORE=$((PCORES - 1))

echo "==============================================="
echo "GEMV Benchmark Runner"
echo "Physical cores: $PCORES (pinning with taskset)"
echo "==============================================="
echo ""

if [ "$MODE" = "single" ] || [ "$MODE" = "both" ]; then
    echo "=== SINGLE-THREADED COMPARISON (pinned to core 0) ==="
    echo "Command: taskset -c 0 OMP_NUM_THREADS=1 ./benchmark $PRESET"
    echo ""
    OMP_NUM_THREADS=1 taskset -c 0 ./benchmark "$PRESET"
    echo ""
fi

if [ "$MODE" = "multi" ] || [ "$MODE" = "both" ]; then
    echo "=== MULTI-THREADED COMPARISON ($PCORES physical cores, pinned 0-$LAST_PCORE) ==="
    echo "Command: taskset -c 0-$LAST_PCORE OMP_NUM_THREADS=$PCORES ./benchmark $PRESET"
    echo ""
    OMP_NUM_THREADS=$PCORES taskset -c 0-"$LAST_PCORE" ./benchmark "$PRESET"
    echo ""
fi

echo "==============================================="
echo "Benchmark complete!"
echo "==============================================="
