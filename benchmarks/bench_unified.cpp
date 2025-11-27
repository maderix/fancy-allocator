// Unified benchmark - uses malloc/free which can be overridden via LD_PRELOAD
// Compile: g++ -O3 -pthread -march=native -o bench_unified bench_unified.cpp
// Run:
//   ./bench_unified glibc 16
//   LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 ./bench_unified jemalloc 16
//   LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4 ./bench_unified tcmalloc 16

#include "bench_common.h"
#include <cstdlib>

int main(int argc, char* argv[]) {
    const char* name = (argc > 1) ? argv[1] : "malloc";
    int maxThreads = (argc > 2) ? std::atoi(argv[2]) : 128;

    int threadCounts[] = {1, 2, 4, 8, 16, 32, 64, 128};

    printHeader();

    for (int t : threadCounts) {
        if (t > maxThreads) break;

        runBenchmark(name, t,
            [](size_t sz) { return malloc(sz); },
            [](void* p) { free(p); }
        );

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    return 0;
}
