#!/bin/bash
#
# Memory Allocator Benchmark Suite
# One-shot script to install dependencies, compile, and run all benchmarks
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
MAX_THREADS="${1:-128}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Check and Install Dependencies
# ─────────────────────────────────────────────────────────────────────────────
install_dependencies() {
    print_header "Phase 1: Checking Dependencies"

    local MISSING_DEPS=()

    # Check for g++
    if ! command -v g++ &> /dev/null; then
        MISSING_DEPS+=("g++")
    else
        print_step "g++ found: $(g++ --version | head -1)"
    fi

    # Check for jemalloc
    if ! ldconfig -p 2>/dev/null | grep -q libjemalloc || ! [ -f /usr/include/jemalloc/jemalloc.h ]; then
        MISSING_DEPS+=("libjemalloc-dev")
    else
        print_step "jemalloc found"
    fi

    # Check for tcmalloc
    if ! ldconfig -p 2>/dev/null | grep -q libtcmalloc; then
        MISSING_DEPS+=("libgoogle-perftools-dev")
    else
        print_step "tcmalloc found"
    fi

    # Install missing dependencies
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        print_warn "Missing dependencies: ${MISSING_DEPS[*]}"
        print_step "Installing dependencies (requires sudo)..."

        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y "${MISSING_DEPS[@]}"
        elif command -v dnf &> /dev/null; then
            # Map package names for Fedora/RHEL
            local FEDORA_DEPS=()
            for dep in "${MISSING_DEPS[@]}"; do
                case "$dep" in
                    "libjemalloc-dev") FEDORA_DEPS+=("jemalloc-devel") ;;
                    "libgoogle-perftools-dev") FEDORA_DEPS+=("gperftools-devel") ;;
                    *) FEDORA_DEPS+=("$dep") ;;
                esac
            done
            sudo dnf install -y "${FEDORA_DEPS[@]}"
        elif command -v pacman &> /dev/null; then
            # Map package names for Arch
            local ARCH_DEPS=()
            for dep in "${MISSING_DEPS[@]}"; do
                case "$dep" in
                    "libjemalloc-dev") ARCH_DEPS+=("jemalloc") ;;
                    "libgoogle-perftools-dev") ARCH_DEPS+=("gperftools") ;;
                    *) ARCH_DEPS+=("$dep") ;;
                esac
            done
            sudo pacman -S --noconfirm "${ARCH_DEPS[@]}"
        else
            print_error "Unknown package manager. Please install manually: ${MISSING_DEPS[*]}"
            exit 1
        fi
        print_step "Dependencies installed successfully"
    else
        print_step "All dependencies satisfied"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Compile Benchmarks
# ─────────────────────────────────────────────────────────────────────────────
compile_benchmarks() {
    print_header "Phase 2: Compiling Benchmarks"

    cd "$SCRIPT_DIR"

    # Compiler flags
    CXXFLAGS="-O3 -pthread -march=native"

    print_step "Compiling Fancy allocator benchmarks..."
    g++ $CXXFLAGS -o bench_fancy_threads bench_fancy_threads.cpp

    print_step "Compiling jemalloc benchmarks..."
    g++ $CXXFLAGS -o bench_jemalloc_threads bench_jemalloc_threads.cpp -ljemalloc

    print_step "Compiling glibc malloc benchmarks..."
    g++ $CXXFLAGS -o bench_glibc_threads bench_glibc_threads.cpp

    print_step "Compiling tcmalloc benchmarks..."
    # Check if tcmalloc dev symlink exists, if not use full path
    TCMALLOC_LIB="-ltcmalloc"
    if ! g++ -ltcmalloc -x c++ - -o /dev/null 2>/dev/null <<< "int main(){}"; then
        if [ -f /lib/x86_64-linux-gnu/libtcmalloc.so.4 ]; then
            TCMALLOC_LIB="/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        elif [ -f /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 ]; then
            TCMALLOC_LIB="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        else
            print_warn "tcmalloc not found, skipping tcmalloc benchmarks"
            TCMALLOC_LIB=""
        fi
    fi
    if [ -n "$TCMALLOC_LIB" ]; then
        g++ $CXXFLAGS -o bench_tcmalloc_threads bench_tcmalloc_threads.cpp $TCMALLOC_LIB
    else
        echo "#!/bin/bash" > bench_tcmalloc_threads
        echo "echo 'tcmalloc,1,0,0,0,0'" >> bench_tcmalloc_threads
        chmod +x bench_tcmalloc_threads
    fi

    # Compile small-only benchmarks
    print_step "Compiling small-allocation benchmarks..."

    cat > /tmp/bench_fancy_small.cpp << 'EOF'
#include "HEADER_PATH"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>

static FancyPerThreadAllocator* g_alloc = nullptr;
static std::atomic<long long> g_totalOps{0};

void worker(int iters) {
    void* ptrs[128];
    for (int i = 0; i < iters; i++) {
        for (int j = 0; j < 128; j++) ptrs[j] = g_alloc->allocate(64);
        for (int j = 0; j < 128; j++) g_alloc->deallocate(ptrs[j]);
    }
    g_totalOps.fetch_add(iters * 256LL, std::memory_order_relaxed);
}

int main(int argc, char* argv[]) {
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int max_threads = (argc > 1) ? std::atoi(argv[1]) : 128;
    const int ITERS = 50000;

    std::cout << "allocator,threads,mops_per_sec" << std::endl;

    for (int t : thread_counts) {
        if (t > max_threads) break;
        FancyPerThreadAllocator alloc(512ULL * 1024ULL * 1024ULL, true);
        g_alloc = &alloc;
        g_totalOps = 0;
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (int i = 0; i < t; i++) threads.emplace_back(worker, ITERS);
        for (auto& th : threads) th.join();
        double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "fancy," << t << "," << g_totalOps.load() / secs / 1e6 << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
}
EOF
    sed "s|HEADER_PATH|$PROJECT_DIR/memory_allocator.h|g" /tmp/bench_fancy_small.cpp > /tmp/bench_fancy_small_final.cpp
    g++ $CXXFLAGS -o bench_fancy_small /tmp/bench_fancy_small_final.cpp

    # Small benchmarks for other allocators
    for alloc in glibc jemalloc tcmalloc; do
        local LIB_FLAG=""
        [[ "$alloc" == "jemalloc" ]] && LIB_FLAG="-ljemalloc"
        [[ "$alloc" == "tcmalloc" ]] && LIB_FLAG="$TCMALLOC_LIB"

        # Skip tcmalloc if not available
        if [[ "$alloc" == "tcmalloc" ]] && [[ -z "$TCMALLOC_LIB" ]]; then
            echo "#!/bin/bash" > bench_tcmalloc_small
            echo "echo 'tcmalloc,1,0'" >> bench_tcmalloc_small
            chmod +x bench_tcmalloc_small
            continue
        fi

        cat > /tmp/bench_${alloc}_small.cpp << EOF
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>

static std::atomic<long long> g_totalOps{0};

void worker(int iters) {
    void* ptrs[128];
    for (int i = 0; i < iters; i++) {
        for (int j = 0; j < 128; j++) ptrs[j] = malloc(64);
        for (int j = 0; j < 128; j++) free(ptrs[j]);
    }
    g_totalOps.fetch_add(iters * 256LL, std::memory_order_relaxed);
}

int main(int argc, char* argv[]) {
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int max_threads = (argc > 1) ? std::atoi(argv[1]) : 128;
    const int ITERS = 50000;

    std::cout << "allocator,threads,mops_per_sec" << std::endl;

    for (int t : thread_counts) {
        if (t > max_threads) break;
        g_totalOps = 0;
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (int i = 0; i < t; i++) threads.emplace_back(worker, ITERS);
        for (auto& th : threads) th.join();
        double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "${alloc}," << t << "," << g_totalOps.load() / secs / 1e6 << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
}
EOF
        g++ $CXXFLAGS -o bench_${alloc}_small /tmp/bench_${alloc}_small.cpp $LIB_FLAG
    done

    print_step "All benchmarks compiled successfully"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Run Benchmarks
# ─────────────────────────────────────────────────────────────────────────────
run_benchmarks() {
    print_header "Phase 3: Running Benchmarks (max $MAX_THREADS threads)"

    mkdir -p "$RESULTS_DIR"

    cd "$SCRIPT_DIR"

    # System info
    echo "Benchmark run: $TIMESTAMP" > "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
    echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
    echo "Cores: $(nproc) logical, $(lscpu | grep 'Core(s) per socket' | cut -d: -f2 | xargs) physical per socket" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
    echo "Memory: $(free -h | awk '/^Mem:/{print $2}')" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
    echo "Kernel: $(uname -r)" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
    echo "Compiler: $(g++ --version | head -1)" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"

    # ─── Mixed Workload Benchmarks ───
    print_step "Running mixed workload benchmarks (60% 64B, 30% 1KB, 10% 8KB)..."

    MIXED_RESULTS="$RESULTS_DIR/mixed_workload_$TIMESTAMP.csv"
    echo "allocator,threads,elapsed_us,total_ops,ops_per_sec,mops_per_sec" > "$MIXED_RESULTS"

    for bench in fancy glibc jemalloc tcmalloc; do
        print_step "  Running $bench..."
        ./bench_${bench}_threads $MAX_THREADS | tail -n +2 >> "$MIXED_RESULTS"
    done

    # ─── Small Allocation Benchmarks ───
    print_step "Running small allocation benchmarks (64B only)..."

    SMALL_RESULTS="$RESULTS_DIR/small_alloc_$TIMESTAMP.csv"
    echo "allocator,threads,mops_per_sec" > "$SMALL_RESULTS"

    for bench in fancy glibc jemalloc tcmalloc; do
        print_step "  Running $bench..."
        ./bench_${bench}_small $MAX_THREADS | tail -n +2 >> "$SMALL_RESULTS"
    done

    print_step "Benchmarks complete!"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Generate Report
# ─────────────────────────────────────────────────────────────────────────────
generate_report() {
    print_header "Phase 4: Generating Report"

    REPORT="$RESULTS_DIR/report_$TIMESTAMP.txt"

    cat > "$REPORT" << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    MEMORY ALLOCATOR BENCHMARK REPORT                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

EOF

    cat "$RESULTS_DIR/system_info_$TIMESTAMP.txt" >> "$REPORT"

    cat >> "$REPORT" << 'EOF'

═══════════════════════════════════════════════════════════════════════════════
                      MIXED WORKLOAD (60% 64B, 30% 1KB, 10% 8KB)
═══════════════════════════════════════════════════════════════════════════════

EOF

    # Parse and format mixed results
    echo "Threads | Fancy     | glibc     | jemalloc  | tcmalloc  |" >> "$REPORT"
    echo "--------|-----------|-----------|-----------|-----------|" >> "$REPORT"

    for threads in 1 2 4 8 16 32 64 128; do
        if [ $threads -le $MAX_THREADS ]; then
            FANCY=$(grep "^fancy,$threads," "$RESULTS_DIR/mixed_workload_$TIMESTAMP.csv" | cut -d, -f6 | head -1)
            GLIBC=$(grep "^glibc,$threads," "$RESULTS_DIR/mixed_workload_$TIMESTAMP.csv" | cut -d, -f6 | head -1)
            JEMALLOC=$(grep "^jemalloc,$threads," "$RESULTS_DIR/mixed_workload_$TIMESTAMP.csv" | cut -d, -f6 | head -1)
            TCMALLOC=$(grep "^tcmalloc,$threads," "$RESULTS_DIR/mixed_workload_$TIMESTAMP.csv" | cut -d, -f6 | head -1)

            printf "%7d | %9.1f | %9.1f | %9.1f | %9.1f |\n" \
                $threads "${FANCY:-0}" "${GLIBC:-0}" "${JEMALLOC:-0}" "${TCMALLOC:-0}" >> "$REPORT"
        fi
    done

    cat >> "$REPORT" << 'EOF'

                                                        (values in Mops/s)

═══════════════════════════════════════════════════════════════════════════════
                         SMALL ALLOCATIONS (64 bytes only)
═══════════════════════════════════════════════════════════════════════════════

EOF

    echo "Threads | Fancy     | glibc     | jemalloc  | tcmalloc  |" >> "$REPORT"
    echo "--------|-----------|-----------|-----------|-----------|" >> "$REPORT"

    for threads in 1 2 4 8 16 32 64 128; do
        if [ $threads -le $MAX_THREADS ]; then
            FANCY=$(grep "^fancy,$threads," "$RESULTS_DIR/small_alloc_$TIMESTAMP.csv" | cut -d, -f3 | head -1)
            GLIBC=$(grep "^glibc,$threads," "$RESULTS_DIR/small_alloc_$TIMESTAMP.csv" | cut -d, -f3 | head -1)
            JEMALLOC=$(grep "^jemalloc,$threads," "$RESULTS_DIR/small_alloc_$TIMESTAMP.csv" | cut -d, -f3 | head -1)
            TCMALLOC=$(grep "^tcmalloc,$threads," "$RESULTS_DIR/small_alloc_$TIMESTAMP.csv" | cut -d, -f3 | head -1)

            printf "%7d | %9.1f | %9.1f | %9.1f | %9.1f |\n" \
                $threads "${FANCY:-0}" "${GLIBC:-0}" "${JEMALLOC:-0}" "${TCMALLOC:-0}" >> "$REPORT"
        fi
    done

    cat >> "$REPORT" << 'EOF'

                                                        (values in Mops/s)

═══════════════════════════════════════════════════════════════════════════════
                                   SUMMARY
═══════════════════════════════════════════════════════════════════════════════

EOF

    # Calculate peak performance
    FANCY_PEAK_MIXED=$(grep "^fancy," "$RESULTS_DIR/mixed_workload_$TIMESTAMP.csv" | cut -d, -f6 | sort -rn | head -1)
    FANCY_PEAK_SMALL=$(grep "^fancy," "$RESULTS_DIR/small_alloc_$TIMESTAMP.csv" | cut -d, -f3 | sort -rn | head -1)
    JEMALLOC_PEAK_MIXED=$(grep "^jemalloc," "$RESULTS_DIR/mixed_workload_$TIMESTAMP.csv" | cut -d, -f6 | sort -rn | head -1)
    JEMALLOC_PEAK_SMALL=$(grep "^jemalloc," "$RESULTS_DIR/small_alloc_$TIMESTAMP.csv" | cut -d, -f3 | sort -rn | head -1)

    echo "Peak Performance:" >> "$REPORT"
    echo "  Fancy (mixed):     ${FANCY_PEAK_MIXED:-N/A} Mops/s" >> "$REPORT"
    echo "  Fancy (small):     ${FANCY_PEAK_SMALL:-N/A} Mops/s" >> "$REPORT"
    echo "  jemalloc (mixed):  ${JEMALLOC_PEAK_MIXED:-N/A} Mops/s" >> "$REPORT"
    echo "  jemalloc (small):  ${JEMALLOC_PEAK_SMALL:-N/A} Mops/s" >> "$REPORT"
    echo "" >> "$REPORT"

    if [ -n "$FANCY_PEAK_SMALL" ] && [ -n "$JEMALLOC_PEAK_SMALL" ]; then
        RATIO=$(echo "scale=2; $FANCY_PEAK_SMALL / $JEMALLOC_PEAK_SMALL" | bc)
        echo "Fancy vs jemalloc (small allocs): ${RATIO}x faster" >> "$REPORT"
    fi

    cat >> "$REPORT" << 'EOF'

═══════════════════════════════════════════════════════════════════════════════
Raw data files:
EOF
    echo "  - $RESULTS_DIR/mixed_workload_$TIMESTAMP.csv" >> "$REPORT"
    echo "  - $RESULTS_DIR/small_alloc_$TIMESTAMP.csv" >> "$REPORT"
    echo "  - $RESULTS_DIR/system_info_$TIMESTAMP.txt" >> "$REPORT"

    print_step "Report saved to: $REPORT"

    # Display report
    echo ""
    cat "$REPORT"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
    print_header "Memory Allocator Benchmark Suite"

    echo "Project directory: $PROJECT_DIR"
    echo "Benchmark directory: $SCRIPT_DIR"
    echo "Max threads: $MAX_THREADS"
    echo ""

    install_dependencies
    compile_benchmarks
    run_benchmarks
    generate_report

    print_header "Benchmark Complete!"
    echo "Results saved in: $RESULTS_DIR"
}

main "$@"
