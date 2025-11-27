// Full comprehensive benchmark - uses malloc/free (override via LD_PRELOAD)
#include "bench_comprehensive.h"
#include <cstdlib>

int main(int argc, char* argv[]) {
    const char* name = (argc > 1) ? argv[1] : "malloc";
    int maxThreads = (argc > 2) ? std::atoi(argv[2]) : 128;
    const char* rssDir = (argc > 3) ? argv[3] : ".";

    int threadCounts[] = {1, 2, 4, 8, 16, 32, 64, 128};

    printResultHeader();

    for (int t : threadCounts) {
        if (t > maxThreads) break;

        auto result = runComprehensiveBenchmark(name, t,
            [](size_t sz) { return malloc(sz); },
            [](void* p) { free(p); }
        );

        printResult(result);

        // Save RSS timeline
        std::string rssFile = std::string(rssDir) + "/rss_" + name + "_" + std::to_string(t) + "t.csv";
        saveRSSTimeline(result, rssFile);

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    return 0;
}
