// Fancy allocator - thread scaling benchmark
// Usage: ./bench_fancy_threads [max_threads]
#include "../memory_allocator.h"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <cstdlib>

static FancyPerThreadAllocator* g_alloc = nullptr;
static std::atomic<long long> g_totalOps{0};

// Clean benchmark: batch alloc/free with deterministic sizes
void worker(int iters) {
    void* ptrs[128];

    // Deterministic mixed sizes: 60% small (64B), 30% medium (1KB), 10% large (8KB)
    // Pattern repeats every 10 allocations
    size_t sizes[10] = {64, 64, 64, 64, 64, 64, 1024, 1024, 1024, 8192};

    for (int i = 0; i < iters; i++) {
        // Batch allocate
        for (int j = 0; j < 128; j++) {
            ptrs[j] = g_alloc->allocate(sizes[j % 10]);
        }
        // Batch free
        for (int j = 0; j < 128; j++) {
            g_alloc->deallocate(ptrs[j]);
        }
    }
    g_totalOps.fetch_add(iters * 256LL, std::memory_order_relaxed);
}

int main(int argc, char* argv[]) {
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int max_threads = (argc > 1) ? std::atoi(argv[1]) : 128;
    const int ITERS = 50000;  // Per-thread iterations

    std::cout << "allocator,threads,elapsed_us,total_ops,ops_per_sec,mops_per_sec" << std::endl;

    for (int t : thread_counts) {
        if (t > max_threads) break;

        FancyPerThreadAllocator alloc(256ULL * 1024ULL * 1024ULL, true);
        g_alloc = &alloc;
        g_totalOps = 0;

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> workers;
        for (int i = 0; i < t; i++)
            workers.emplace_back(worker, ITERS);
        for (auto& w : workers) w.join();

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count();

        long long ops = g_totalOps.load();
        double secs = elapsed / 1e6;
        double mops = ops / secs / 1e6;

        std::cout << "fancy," << t << "," << elapsed << "," << ops
                  << "," << (ops * 1000000LL / elapsed) << "," << mops << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
}
