// Comprehensive benchmark with latency percentiles, RSS tracking, fragmentation
#pragma once

#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>
#include <sys/resource.h>
#include <unistd.h>

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

constexpr int OPS_PER_THREAD = 250000;
constexpr int RING_SIZE = 100000;
constexpr int LATENCY_SAMPLES = 10000;  // Sample every N ops for latency
constexpr int RSS_SAMPLE_INTERVAL_MS = 100;

// ─────────────────────────────────────────────────────────────────────────────
// Latency tracking
// ─────────────────────────────────────────────────────────────────────────────

struct LatencyStats {
    std::vector<double> allocLatenciesUs;
    std::vector<double> freeLatenciesUs;

    void reserve(size_t n) {
        allocLatenciesUs.reserve(n);
        freeLatenciesUs.reserve(n);
    }
};

struct PercentileStats {
    double allocP50, allocP90, allocP99, allocP999;
    double freeP50, freeP90, freeP99, freeP999;
    double allocMean, freeMean;
};

inline double percentile(std::vector<double>& data, double p) {
    if (data.empty()) return 0;
    std::sort(data.begin(), data.end());
    size_t idx = static_cast<size_t>(p * data.size());
    if (idx >= data.size()) idx = data.size() - 1;
    return data[idx];
}

inline PercentileStats computePercentiles(std::vector<LatencyStats>& allStats) {
    std::vector<double> allAlloc, allFree;
    for (auto& s : allStats) {
        allAlloc.insert(allAlloc.end(), s.allocLatenciesUs.begin(), s.allocLatenciesUs.end());
        allFree.insert(allFree.end(), s.freeLatenciesUs.begin(), s.freeLatenciesUs.end());
    }

    PercentileStats ps{};
    if (!allAlloc.empty()) {
        ps.allocMean = std::accumulate(allAlloc.begin(), allAlloc.end(), 0.0) / allAlloc.size();
        ps.allocP50 = percentile(allAlloc, 0.50);
        ps.allocP90 = percentile(allAlloc, 0.90);
        ps.allocP99 = percentile(allAlloc, 0.99);
        ps.allocP999 = percentile(allAlloc, 0.999);
    }
    if (!allFree.empty()) {
        ps.freeMean = std::accumulate(allFree.begin(), allFree.end(), 0.0) / allFree.size();
        ps.freeP50 = percentile(allFree, 0.50);
        ps.freeP90 = percentile(allFree, 0.90);
        ps.freeP99 = percentile(allFree, 0.99);
        ps.freeP999 = percentile(allFree, 0.999);
    }
    return ps;
}

// ─────────────────────────────────────────────────────────────────────────────
// RSS tracking
// ─────────────────────────────────────────────────────────────────────────────

inline size_t getRSSBytes() {
    std::ifstream statm("/proc/self/statm");
    size_t size, resident;
    statm >> size >> resident;
    return resident * sysconf(_SC_PAGESIZE);
}

struct RSSTimeline {
    std::vector<double> timestamps;  // seconds from start
    std::vector<size_t> rssBytes;
    size_t peakRSS = 0;
    size_t initialRSS = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark results
// ─────────────────────────────────────────────────────────────────────────────

struct BenchmarkResult {
    std::string allocatorName;
    int numThreads;

    // Throughput
    double elapsedUs;
    size_t totalAllocs;
    size_t totalFrees;
    double opsPerSec;
    double mopsPerSec;

    // Latency percentiles (microseconds)
    PercentileStats latency;

    // Memory
    RSSTimeline rss;
    size_t requestedBytes;    // Sum of requested allocation sizes
    size_t peakRSSBytes;      // Peak RSS during benchmark
    double fragmentationPct;  // (peakRSS - requested) / requested * 100
};

// ─────────────────────────────────────────────────────────────────────────────
// Worker with latency tracking
// ─────────────────────────────────────────────────────────────────────────────

template<typename AllocFn, typename FreeFn>
void comprehensiveWorker(
    AllocFn alloc_fn,
    FreeFn free_fn,
    std::atomic<size_t>& allocCount,
    std::atomic<size_t>& freeCount,
    std::atomic<size_t>& bytesRequested,
    std::atomic<int>& readyCount,
    std::atomic<bool>& startFlag,
    std::atomic<int>& doneCount,
    LatencyStats& latencyStats
) {
    struct Slot { void* ptr; int ttl; size_t size; };
    std::vector<Slot> ring(RING_SIZE, {nullptr, 0, 0});
    int pos = 0;

    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> catDist(1, 100);
    std::uniform_int_distribution<int> smallDist(16, 256);
    std::uniform_int_distribution<int> medDist(512, 2048);
    std::uniform_int_distribution<int> largeDist(4096, 32768);
    std::uniform_int_distribution<int> ttlDist(50, 2000);

    size_t localAllocs = 0, localFrees = 0, localBytes = 0;
    int sampleCounter = 0;

    latencyStats.reserve(OPS_PER_THREAD / LATENCY_SAMPLES + 100);

    readyCount.fetch_add(1, std::memory_order_release);
    while (!startFlag.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    for (int i = 0; i < OPS_PER_THREAD; i++) {
        auto& slot = ring[pos];
        bool sampleLatency = (++sampleCounter % LATENCY_SAMPLES == 0);

        // Free
        if (slot.ptr && slot.ttl <= 0) {
            if (sampleLatency) {
                auto t1 = std::chrono::high_resolution_clock::now();
                free_fn(slot.ptr);
                auto t2 = std::chrono::high_resolution_clock::now();
                latencyStats.freeLatenciesUs.push_back(
                    std::chrono::duration<double, std::micro>(t2 - t1).count());
            } else {
                free_fn(slot.ptr);
            }
            slot.ptr = nullptr;
            localFrees++;
        }

        if (slot.ptr && slot.ttl > 0) slot.ttl--;

        // Allocate
        if (!slot.ptr) {
            int c = catDist(rng);
            size_t sz;
            if (c <= 60) sz = smallDist(rng);
            else if (c <= 90) sz = medDist(rng);
            else sz = largeDist(rng);

            void* p;
            if (sampleLatency) {
                auto t1 = std::chrono::high_resolution_clock::now();
                p = alloc_fn(sz);
                auto t2 = std::chrono::high_resolution_clock::now();
                latencyStats.allocLatenciesUs.push_back(
                    std::chrono::duration<double, std::micro>(t2 - t1).count());
            } else {
                p = alloc_fn(sz);
            }

            if (p) {
                memset(p, 0x42, sz > 64 ? 64 : sz);
                slot.ptr = p;
                slot.ttl = ttlDist(rng);
                slot.size = sz;
                localAllocs++;
                localBytes += sz;
            }
        }

        pos = (pos + 1) % RING_SIZE;
    }

    doneCount.fetch_add(1, std::memory_order_release);

    // Cleanup
    for (auto& s : ring) {
        if (s.ptr) {
            free_fn(s.ptr);
            s.ptr = nullptr;
            localFrees++;
        }
    }

    allocCount.fetch_add(localAllocs, std::memory_order_relaxed);
    freeCount.fetch_add(localFrees, std::memory_order_relaxed);
    bytesRequested.fetch_add(localBytes, std::memory_order_relaxed);
}

// ─────────────────────────────────────────────────────────────────────────────
// RSS sampler thread
// ─────────────────────────────────────────────────────────────────────────────

inline void rssSampler(
    std::atomic<bool>& stopFlag,
    RSSTimeline& timeline,
    std::chrono::high_resolution_clock::time_point startTime
) {
    timeline.initialRSS = getRSSBytes();
    while (!stopFlag.load(std::memory_order_acquire)) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - startTime).count();
        size_t rss = getRSSBytes();

        timeline.timestamps.push_back(elapsed);
        timeline.rssBytes.push_back(rss);
        if (rss > timeline.peakRSS) timeline.peakRSS = rss;

        std::this_thread::sleep_for(std::chrono::milliseconds(RSS_SAMPLE_INTERVAL_MS));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Run comprehensive benchmark
// ─────────────────────────────────────────────────────────────────────────────

template<typename AllocFn, typename FreeFn>
BenchmarkResult runComprehensiveBenchmark(
    const char* name,
    int numThreads,
    AllocFn alloc_fn,
    FreeFn free_fn
) {
    BenchmarkResult result;
    result.allocatorName = name;
    result.numThreads = numThreads;

    std::atomic<size_t> allocCount(0), freeCount(0), bytesRequested(0);
    std::atomic<int> readyCount(0), doneCount(0);
    std::atomic<bool> startFlag(false), stopRSS(false);

    std::vector<LatencyStats> latencyStats(numThreads);

    // Start RSS sampler
    auto startTime = std::chrono::high_resolution_clock::now();
    std::thread rssSamplerThread(rssSampler, std::ref(stopRSS), std::ref(result.rss), startTime);

    // Create worker threads
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back([&, i]() {
            comprehensiveWorker(alloc_fn, free_fn, allocCount, freeCount, bytesRequested,
                               readyCount, startFlag, doneCount, latencyStats[i]);
        });
    }

    // Wait for setup
    while (readyCount.load(std::memory_order_acquire) < numThreads) {
        std::this_thread::yield();
    }

    // Start timing
    auto benchStart = std::chrono::high_resolution_clock::now();
    startFlag.store(true, std::memory_order_release);

    // Wait for completion
    while (doneCount.load(std::memory_order_acquire) < numThreads) {
        std::this_thread::yield();
    }

    auto benchEnd = std::chrono::high_resolution_clock::now();
    result.elapsedUs = std::chrono::duration<double, std::micro>(benchEnd - benchStart).count();

    // Stop RSS sampler
    stopRSS.store(true, std::memory_order_release);
    rssSamplerThread.join();

    // Join workers
    for (auto& t : threads) t.join();

    // Compute results
    result.totalAllocs = allocCount.load();
    result.totalFrees = freeCount.load();
    result.requestedBytes = bytesRequested.load();
    result.peakRSSBytes = result.rss.peakRSS;

    size_t totalOps = result.totalAllocs + result.totalFrees;
    result.opsPerSec = totalOps * 1e6 / result.elapsedUs;
    result.mopsPerSec = result.opsPerSec / 1e6;

    // Fragmentation: how much more memory than requested
    if (result.requestedBytes > 0) {
        result.fragmentationPct = 100.0 * (double)(result.peakRSSBytes - result.rss.initialRSS) / result.requestedBytes - 100.0;
        if (result.fragmentationPct < 0) result.fragmentationPct = 0;
    }

    // Latency percentiles
    result.latency = computePercentiles(latencyStats);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Output helpers
// ─────────────────────────────────────────────────────────────────────────────

inline void printResultHeader() {
    std::cout << "allocator,threads,mops_sec,"
              << "alloc_p50_us,alloc_p90_us,alloc_p99_us,alloc_p999_us,"
              << "free_p50_us,free_p90_us,free_p99_us,free_p999_us,"
              << "peak_rss_mb,frag_pct" << std::endl;
}

inline void printResult(const BenchmarkResult& r) {
    std::cout << r.allocatorName << "," << r.numThreads << ","
              << r.mopsPerSec << ","
              << r.latency.allocP50 << "," << r.latency.allocP90 << ","
              << r.latency.allocP99 << "," << r.latency.allocP999 << ","
              << r.latency.freeP50 << "," << r.latency.freeP90 << ","
              << r.latency.freeP99 << "," << r.latency.freeP999 << ","
              << (r.peakRSSBytes / 1024.0 / 1024.0) << ","
              << r.fragmentationPct << std::endl;
}

inline void saveRSSTimeline(const BenchmarkResult& r, const std::string& filename) {
    std::ofstream f(filename);
    f << "time_sec,rss_mb" << std::endl;
    for (size_t i = 0; i < r.rss.timestamps.size(); i++) {
        f << r.rss.timestamps[i] << "," << (r.rss.rssBytes[i] / 1024.0 / 1024.0) << std::endl;
    }
}
