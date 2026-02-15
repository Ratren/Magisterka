#!/bin/bash
# Benchmark script for GEMV implementations
# Usage: ./run_benchmarks.sh [single|multi|both] [preset]

cd "$(dirname "$0")/build"

MODE="${1:-both}"
PRESET="${2:-all}"

echo "==============================================="
echo "GEMV Benchmark Runner"
echo "==============================================="
echo ""

if [ "$MODE" = "single" ] || [ "$MODE" = "both" ]; then
    echo "=== SINGLE-THREADED COMPARISON ==="
    echo "Command: OMP_NUM_THREADS=1 ./benchmark $PRESET"
    echo ""
    OMP_NUM_THREADS=1 ./benchmark "$PRESET"
    echo ""
fi

if [ "$MODE" = "multi" ] || [ "$MODE" = "both" ]; then
    NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 6)

    echo "=== MULTI-THREADED COMPARISON ($NCORES threads) ==="
    echo "Command: OMP_NUM_THREADS=$NCORES ./benchmark $PRESET"
    echo ""
    OMP_NUM_THREADS=$NCORES ./benchmark "$PRESET"
    echo ""
fi

echo "==============================================="
echo "Benchmark complete!"
echo "==============================================="
