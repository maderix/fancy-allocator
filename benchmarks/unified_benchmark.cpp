// Unified Memory Allocator Benchmark
// Tests all allocators with IDENTICAL workload matching README methodology
//
// Compile:
//   g++ -O3 -pthread -march=native -o unified_benchmark unified_benchmark.cpp -ljemalloc -ldl
//
// Usage:
//   ./unified_benchmark [max_threads] [allocator]
//   allocator: all, fancy, glibc, jemalloc, tcmalloc

#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>

// Include Fancy allocator
#include "../memory_allocator.h"

// ─────────────────────────────────────────────────────────────────────────────
// Allocator abstraction
// ─────────────────────────────────────────────────────────────────────────────

struct AllocatorInterface {
    std::function<void*(size_t)> alloc;
    std::function<void(void*)> dealloc;
    std::string name;
};

// Global Fancy allocator instance
static FancyPerThreadAllocator* g_fancy = nullptr;

// tcmalloc function pointers (loaded dynamically to avoid link issues)
typedef void* (*tc_malloc_t)(size_t);
typedef void (*tc_free_t)(void*);
static tc_malloc_t tc_malloc_fn = nullptr;
static tc_free_t tc_free_fn = nullptr;

// jemalloc function pointers (loaded dynamically)
typedef void* (*je_malloc_t)(size_t);
typedef void (*je_free_t)(void*);
static je_malloc_t je_malloc_fn = nullptr;
static je_free_t je_free_fn = nullptr;

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark parameters (matching README exactly)
// ─────────────────────────────────────────────────────────────────────────────

constexpr int OPS_PER_THREAD = 250000;
constexpr int RING_SIZE = 100000;

// ─────────────────────────────────────────────────────────────────────────────
// Worker function - IDENTICAL for all allocators
// ─────────────────────────────────────────────────────────────────────────────

void worker(
    std::function<void*(size_t)> alloc_fn,
    std::function<void(void*)> free_fn,
    std::atomic<size_t>& allocCount,
    std::atomic<size_t>& freeCount
) {
    struct Slot { void* ptr; int ttl; };
    std::vector<Slot> ring(RING_SIZE, {nullptr, 0});
    int pos = 0;

    // Random number generation - same seed methodology for all
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> catDist(1, 100);
    std::uniform_int_distribution<int> smallDist(16, 256);
    std::uniform_int_distribution<int> medDist(512, 2048);
    std::uniform_int_distribution<int> largeDist(4096, 32768);
    std::uniform_int_distribution<int> ttlDist(50, 2000);

    size_t localAllocs = 0;
    size_t localFrees = 0;

    for (int i = 0; i < OPS_PER_THREAD; i++) {
        auto& slot = ring[pos];

        // Free old allocation if TTL expired
        if (slot.ptr && slot.ttl <= 0) {
            free_fn(slot.ptr);
            slot.ptr = nullptr;
            localFrees++;
        }

        // Decrement TTL
        if (slot.ptr && slot.ttl > 0) {
            slot.ttl--;
        }

        // Allocate if slot is empty
        if (!slot.ptr) {
            int c = catDist(rng);
            size_t sz;
            if (c <= 60) {
                sz = smallDist(rng);      // 60% small (16-256)
            } else if (c <= 90) {
                sz = medDist(rng);        // 30% medium (512-2048)
            } else {
                sz = largeDist(rng);      // 10% large (4096-32768)
            }

            void* p = alloc_fn(sz);
            if (p) {
                // Touch memory to ensure it's actually allocated
                memset(p, 0x42, sz > 64 ? 64 : sz);
                slot.ptr = p;
                slot.ttl = ttlDist(rng);
                localAllocs++;
            }
        }

        pos = (pos + 1) % RING_SIZE;
    }

    // Cleanup remaining allocations
    for (auto& s : ring) {
        if (s.ptr) {
            free_fn(s.ptr);
            s.ptr = nullptr;
            localFrees++;
        }
    }

    allocCount.fetch_add(localAllocs, std::memory_order_relaxed);
    freeCount.fetch_add(localFrees, std::memory_order_relaxed);
}

// ─────────────────────────────────────────────────────────────────────────────
// Run benchmark for a specific allocator
// ─────────────────────────────────────────────────────────────────────────────

void runBenchmark(const std::string& name, int numThreads,
                  std::function<void*(size_t)> alloc_fn,
                  std::function<void(void*)> free_fn) {

    std::atomic<size_t> allocCount(0), freeCount(0);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back(worker, alloc_fn, free_fn,
                            std::ref(allocCount), std::ref(freeCount));
    }
    for (auto& t : threads) {
        t.join();
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    size_t totalOps = allocCount.load() + freeCount.load();
    double opsPerSec = totalOps * 1000000.0 / elapsed;
    double mOpsPerSec = opsPerSec / 1e6;

    std::cout << name << "," << numThreads << ","
              << elapsed << "," << allocCount.load() << "," << freeCount.load() << ","
              << static_cast<long long>(opsPerSec) << ","
              << mOpsPerSec << std::endl;
}

// ─────────────────────────────────────────────────────────────────────────────
// Load tcmalloc dynamically
// ─────────────────────────────────────────────────────────────────────────────

bool loadTcmalloc() {
    void* handle = dlopen("libtcmalloc.so.4", RTLD_NOW);
    if (!handle) {
        handle = dlopen("/lib/x86_64-linux-gnu/libtcmalloc.so.4", RTLD_NOW);
    }
    if (!handle) {
        std::cerr << "Warning: Could not load tcmalloc: " << dlerror() << std::endl;
        return false;
    }

    tc_malloc_fn = (tc_malloc_t)dlsym(handle, "tc_malloc");
    tc_free_fn = (tc_free_t)dlsym(handle, "tc_free");

    if (!tc_malloc_fn || !tc_free_fn) {
        // Fall back to standard names (tcmalloc interposes malloc/free)
        tc_malloc_fn = (tc_malloc_t)dlsym(handle, "malloc");
        tc_free_fn = (tc_free_t)dlsym(handle, "free");
    }

    return tc_malloc_fn && tc_free_fn;
}


// ─────────────────────────────────────────────────────────────────────────────
// Load jemalloc dynamically
// ─────────────────────────────────────────────────────────────────────────────

bool loadJemalloc() {
    // Use RTLD_GLOBAL | RTLD_NOW to fix TLS allocation issues
    void* handle = dlopen("libjemalloc.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        handle = dlopen("/lib/x86_64-linux-gnu/libjemalloc.so.2", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!handle) {
        std::cerr << "Warning: Could not load jemalloc: " << dlerror() << std::endl;
        return false;
    }

    // jemalloc exports standard malloc/free names
    je_malloc_fn = (je_malloc_t)dlsym(handle, "malloc");
    je_free_fn = (je_free_t)dlsym(handle, "free");

    if (!je_malloc_fn || !je_free_fn) {
        std::cerr << "Warning: Could not find malloc/free in jemalloc" << std::endl;
        return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    int maxThreads = (argc > 1) ? std::atoi(argv[1]) : 128;
    std::string targetAllocator = (argc > 2) ? argv[2] : "all";

    int threadCounts[] = {1, 2, 4, 8, 16, 32, 64, 128};

    std::cout << "# Unified Memory Allocator Benchmark" << std::endl;
    std::cout << "# Methodology: " << OPS_PER_THREAD << " ops/thread, "
              << RING_SIZE << " ring slots, mixed sizes (60% small, 30% med, 10% large)" << std::endl;
    std::cout << "# Memory is touched (memset) to ensure actual allocation" << std::endl;
    std::cout << "#" << std::endl;
    std::cout << "allocator,threads,elapsed_us,allocs,frees,ops_per_sec,mops_per_sec" << std::endl;

    // Load dynamic libraries BEFORE any allocations to avoid TLS issues
    bool hasJemalloc = loadJemalloc();
    bool hasTcmalloc = loadTcmalloc();

    for (int t : threadCounts) {
        if (t > maxThreads) break;

        // ─── Fancy ───
        if (targetAllocator == "all" || targetAllocator == "fancy") {
            FancyPerThreadAllocator fancy(64ULL * 1024ULL * 1024ULL, true);
            g_fancy = &fancy;

            runBenchmark("fancy", t,
                [](size_t sz) { return g_fancy->allocate(sz); },
                [](void* p) { g_fancy->deallocate(p); }
            );

            g_fancy = nullptr;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }

        // ─── glibc ───
        if (targetAllocator == "all" || targetAllocator == "glibc") {
            runBenchmark("glibc", t,
                [](size_t sz) { return malloc(sz); },
                [](void* p) { free(p); }
            );
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }

        // ─── jemalloc ───
        if ((targetAllocator == "all" || targetAllocator == "jemalloc") && hasJemalloc) {
            runBenchmark("jemalloc", t,
                [](size_t sz) { return je_malloc_fn(sz); },
                [](void* p) { je_free_fn(p); }
            );
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }

        // ─── tcmalloc ───
        if ((targetAllocator == "all" || targetAllocator == "tcmalloc") && hasTcmalloc) {
            runBenchmark("tcmalloc", t,
                [](size_t sz) { return tc_malloc_fn(sz); },
                [](void* p) { tc_free_fn(p); }
            );
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    return 0;
}
