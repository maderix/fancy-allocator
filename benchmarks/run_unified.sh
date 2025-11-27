#!/bin/bash
# Unified benchmark runner - ensures identical workload for all allocators

set -e

MAX_THREADS="${1:-128}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  Unified Memory Allocator Benchmark"
echo "  Max threads: $MAX_THREADS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Compile
echo "Compiling..."
g++ -O3 -pthread -march=native -o bench_unified bench_unified.cpp
g++ -O3 -pthread -march=native -o bench_unified_fancy bench_unified_fancy.cpp
echo "Done."
echo ""

# Results file
RESULTS="results/unified_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p results

echo "allocator,threads,elapsed_us,allocs,frees,ops_per_sec,mops_per_sec" > "$RESULTS"

# Run Fancy
echo "Running Fancy..."
./bench_unified_fancy "$MAX_THREADS" | tail -n +2 | tee -a "$RESULTS"
echo ""

# Run glibc
echo "Running glibc..."
./bench_unified glibc "$MAX_THREADS" | tail -n +2 | tee -a "$RESULTS"
echo ""

# Run jemalloc
echo "Running jemalloc..."
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 ./bench_unified jemalloc "$MAX_THREADS" | tail -n +2 | tee -a "$RESULTS"
echo ""

# Run tcmalloc
echo "Running tcmalloc..."
LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4 ./bench_unified tcmalloc "$MAX_THREADS" | tail -n +2 | tee -a "$RESULTS"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  Results Summary (Mops/sec)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Threads | Fancy    | glibc    | jemalloc | tcmalloc |"
echo "--------|----------|----------|----------|----------|"

for t in 1 2 4 8 16 32 64 128; do
    if [ "$t" -le "$MAX_THREADS" ]; then
        FANCY=$(grep "^fancy,$t," "$RESULTS" | cut -d, -f7 | head -1)
        GLIBC=$(grep "^glibc,$t," "$RESULTS" | cut -d, -f7 | head -1)
        JEMALLOC=$(grep "^jemalloc,$t," "$RESULTS" | cut -d, -f7 | head -1)
        TCMALLOC=$(grep "^tcmalloc,$t," "$RESULTS" | cut -d, -f7 | head -1)
        printf "%7d | %8.2f | %8.2f | %8.2f | %8.2f |\n" \
            "$t" "${FANCY:-0}" "${GLIBC:-0}" "${JEMALLOC:-0}" "${TCMALLOC:-0}"
    fi
done

echo ""
echo "Results saved to: $RESULTS"
