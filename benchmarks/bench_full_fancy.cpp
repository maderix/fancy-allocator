// Full comprehensive benchmark for Fancy allocator
#include "bench_comprehensive.h"
#include "../memory_allocator.h"

static FancyPerThreadAllocator* g_alloc = nullptr;

int main(int argc, char* argv[]) {
    int maxThreads = (argc > 1) ? std::atoi(argv[1]) : 128;
    const char* rssDir = (argc > 2) ? argv[2] : ".";

    int threadCounts[] = {1, 2, 4, 8, 16, 32, 64, 128};

    printResultHeader();

    for (int t : threadCounts) {
        if (t > maxThreads) break;

        FancyPerThreadAllocator alloc(64ULL * 1024ULL * 1024ULL, true);
        g_alloc = &alloc;

        auto result = runComprehensiveBenchmark("fancy", t,
            [](size_t sz) { return g_alloc->allocate(sz); },
            [](void* p) { g_alloc->deallocate(p); }
        );

        printResult(result);

        // Save RSS timeline
        std::string rssFile = std::string(rssDir) + "/rss_fancy_" + std::to_string(t) + "t.csv";
        saveRSSTimeline(result, rssFile);

        g_alloc = nullptr;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    return 0;
}
