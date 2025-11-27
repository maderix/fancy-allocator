// jemalloc - thread scaling benchmark
// Compile: g++ -O3 -pthread -o bench_jemalloc_threads bench_jemalloc_threads.cpp -ljemalloc
// Usage: ./bench_jemalloc_threads [max_threads]
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <cstdlib>

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
            ptrs[j] = malloc(sizes[j % 10]);
        }
        // Batch free
        for (int j = 0; j < 128; j++) {
            free(ptrs[j]);
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

        std::cout << "jemalloc," << t << "," << elapsed << "," << ops
                  << "," << (ops * 1000000LL / elapsed) << "," << mops << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
}
