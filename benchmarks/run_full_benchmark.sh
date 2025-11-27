#!/bin/bash
# Comprehensive benchmark with latency percentiles, RSS tracking, fragmentation
# Generates CSV data and plots

set -e

MAX_THREADS="${1:-128}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/full_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Comprehensive Memory Allocator Benchmark"
echo "  Max threads: $MAX_THREADS"
echo "  Results dir: $RESULTS_DIR"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Compile
echo "Compiling benchmarks..."
g++ -O3 -pthread -march=native -o bench_full bench_full.cpp
g++ -O3 -pthread -march=native -o bench_full_fancy bench_full_fancy.cpp
echo "Done."
echo ""

# Combined results file
RESULTS_CSV="$RESULTS_DIR/results.csv"
echo "allocator,threads,mops_sec,alloc_p50_us,alloc_p90_us,alloc_p99_us,alloc_p999_us,free_p50_us,free_p90_us,free_p99_us,free_p999_us,peak_rss_mb,frag_pct" > "$RESULTS_CSV"

# Run Fancy
echo "═══════════════════════════════════════════════════════════════════════"
echo "Running Fancy allocator..."
echo "═══════════════════════════════════════════════════════════════════════"
./bench_full_fancy "$MAX_THREADS" "$RESULTS_DIR" | tee "$RESULTS_DIR/fancy_raw.csv"
tail -n +2 "$RESULTS_DIR/fancy_raw.csv" >> "$RESULTS_CSV"
echo ""

# Run glibc
echo "═══════════════════════════════════════════════════════════════════════"
echo "Running glibc malloc..."
echo "═══════════════════════════════════════════════════════════════════════"
./bench_full glibc "$MAX_THREADS" "$RESULTS_DIR" | tee "$RESULTS_DIR/glibc_raw.csv"
tail -n +2 "$RESULTS_DIR/glibc_raw.csv" >> "$RESULTS_CSV"
echo ""

# Run jemalloc
echo "═══════════════════════════════════════════════════════════════════════"
echo "Running jemalloc..."
echo "═══════════════════════════════════════════════════════════════════════"
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 ./bench_full jemalloc "$MAX_THREADS" "$RESULTS_DIR" | tee "$RESULTS_DIR/jemalloc_raw.csv"
tail -n +2 "$RESULTS_DIR/jemalloc_raw.csv" >> "$RESULTS_CSV"
echo ""

# Run tcmalloc
echo "═══════════════════════════════════════════════════════════════════════"
echo "Running tcmalloc..."
echo "═══════════════════════════════════════════════════════════════════════"
LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4 ./bench_full tcmalloc "$MAX_THREADS" "$RESULTS_DIR" | tee "$RESULTS_DIR/tcmalloc_raw.csv"
tail -n +2 "$RESULTS_DIR/tcmalloc_raw.csv" >> "$RESULTS_CSV"
echo ""

# Generate plots
echo "═══════════════════════════════════════════════════════════════════════"
echo "Generating plots..."
echo "═══════════════════════════════════════════════════════════════════════"
python3 plot_benchmark.py "$RESULTS_DIR"
echo ""

# Print summary
echo "═══════════════════════════════════════════════════════════════════════"
echo "  RESULTS SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Throughput (Mops/sec):"
echo "─────────────────────────────────────────────────────────────────────"
printf "%-10s" "Threads"
for alloc in fancy glibc jemalloc tcmalloc; do
    printf "│ %-10s" "$alloc"
done
echo ""
echo "─────────────────────────────────────────────────────────────────────"

for t in 1 2 4 8 16 32 64 128; do
    if [ "$t" -le "$MAX_THREADS" ]; then
        printf "%-10s" "$t"
        for alloc in fancy glibc jemalloc tcmalloc; do
            val=$(grep "^$alloc,$t," "$RESULTS_CSV" | cut -d, -f3)
            printf "│ %-10.2f" "${val:-0}"
        done
        echo ""
    fi
done

echo ""
echo "Allocation Latency P99 (μs):"
echo "─────────────────────────────────────────────────────────────────────"
printf "%-10s" "Threads"
for alloc in fancy glibc jemalloc tcmalloc; do
    printf "│ %-10s" "$alloc"
done
echo ""
echo "─────────────────────────────────────────────────────────────────────"

for t in 1 2 4 8 16 32 64 128; do
    if [ "$t" -le "$MAX_THREADS" ]; then
        printf "%-10s" "$t"
        for alloc in fancy glibc jemalloc tcmalloc; do
            val=$(grep "^$alloc,$t," "$RESULTS_CSV" | cut -d, -f6)
            printf "│ %-10.2f" "${val:-0}"
        done
        echo ""
    fi
done

echo ""
echo "Peak RSS (MB) @ max threads:"
echo "─────────────────────────────────────────────────────────────────────"
for alloc in fancy glibc jemalloc tcmalloc; do
    val=$(grep "^$alloc,$MAX_THREADS," "$RESULTS_CSV" | cut -d, -f12)
    printf "  %-10s: %.1f MB\n" "$alloc" "${val:-0}"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "Results saved to: $RESULTS_DIR"
echo "  - results.csv: All metrics"
echo "  - rss_*.csv: RSS timelines"
echo "  - *.png: Plots"
echo "═══════════════════════════════════════════════════════════════════════"
