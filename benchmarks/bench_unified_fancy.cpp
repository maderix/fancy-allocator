// Fancy allocator benchmark - uses same workload as bench_unified
// Compile: g++ -O3 -pthread -march=native -o bench_unified_fancy bench_unified_fancy.cpp

#include "bench_common.h"
#include "../memory_allocator.h"

static FancyPerThreadAllocator* g_alloc = nullptr;

int main(int argc, char* argv[]) {
    int maxThreads = (argc > 1) ? std::atoi(argv[1]) : 128;

    int threadCounts[] = {1, 2, 4, 8, 16, 32, 64, 128};

    printHeader();

    for (int t : threadCounts) {
        if (t > maxThreads) break;

        // Create fresh allocator for each thread count
        // Disable reclamation for better benchmark performance
        FancyPerThreadAllocator alloc(64ULL * 1024ULL * 1024ULL, false);
        g_alloc = &alloc;

        runBenchmark("fancy", t,
            [](size_t sz) { return g_alloc->allocate(sz); },
            [](void* p) { g_alloc->deallocate(p); }
        );

        g_alloc = nullptr;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    return 0;
}
