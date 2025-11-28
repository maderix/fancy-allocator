// Common benchmark code - identical workload for all allocators
#pragma once

#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <atomic>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// WORKLOAD DESCRIPTION
// ─────────────────────────────────────────────────────────────────────────────
//
// This benchmark simulates a realistic memory allocation workload:
//
// - Ring buffer model: 100,000 slots per thread
// - Each slot holds a pointer with a random TTL (time-to-live)
// - On each iteration:
//   1. If slot's TTL expired → free the memory
//   2. If slot is empty → allocate new memory with random size
//   3. Advance to next slot (ring buffer wraps around)
//
// Size distribution (matches real-world patterns):
//   - 60% small:  16-256 bytes   (hot path, frequent allocations)
//   - 30% medium: 512-2048 bytes (buffers, strings)
//   - 10% large:  4096-32768 bytes (data structures)
//
// Memory is touched (memset) to ensure actual page allocation.
//
// TIMING: Only the allocation/free loop is timed.
//         RNG init, ring buffer setup, and thread creation are EXCLUDED.
//
// ─────────────────────────────────────────────────────────────────────────────

constexpr int OPS_PER_THREAD = 250000;
constexpr int RING_SIZE = 100000;

// ─────────────────────────────────────────────────────────────────────────────
// Worker function template - IDENTICAL workload for all allocators
// ─────────────────────────────────────────────────────────────────────────────

template<typename AllocFn, typename FreeFn>
void worker(
    AllocFn alloc_fn,
    FreeFn free_fn,
    std::atomic<size_t>& allocCount,
    std::atomic<size_t>& freeCount,
    std::atomic<int>& readyCount,
    std::atomic<bool>& startFlag,
    std::atomic<int>& doneCount
) {
    // ─── SETUP PHASE (not timed) ───
    struct Slot { void* ptr; int ttl; };
    std::vector<Slot> ring(RING_SIZE, {nullptr, 0});
    int pos = 0;

    // Pre-initialize RNG (not timed)
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> catDist(1, 100);
    std::uniform_int_distribution<int> smallDist(16, 256);
    std::uniform_int_distribution<int> medDist(512, 2048);
    std::uniform_int_distribution<int> largeDist(4096, 32768);
    std::uniform_int_distribution<int> ttlDist(50, 2000);

    // Pre-compute ALL random values to exclude RNG from timing
    std::vector<size_t> preSizes(OPS_PER_THREAD);
    std::vector<int> preTTLs(OPS_PER_THREAD);
    for (int i = 0; i < OPS_PER_THREAD; i++) {
        int c = catDist(rng);
        if (c <= 60) {
            preSizes[i] = smallDist(rng);
        } else if (c <= 90) {
            preSizes[i] = medDist(rng);
        } else {
            preSizes[i] = largeDist(rng);
        }
        preTTLs[i] = ttlDist(rng);
    }

    size_t localAllocs = 0;
    size_t localFrees = 0;

    // Warm up: do a dummy allocation to initialize thread-local structures
    // This ensures arena creation happens BEFORE timing starts
    void* warmup = alloc_fn(64);
    if (warmup) free_fn(warmup);

    // Signal ready and wait for start
    readyCount.fetch_add(1, std::memory_order_release);
    while (!startFlag.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // ─── TIMED SECTION STARTS HERE ───
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

        // Allocate if slot is empty (using pre-computed random values)
        if (!slot.ptr) {
            size_t sz = preSizes[i];

            void* p = alloc_fn(sz);
            if (p) {
                // Touch memory to ensure actual allocation
                memset(p, 0x42, sz > 64 ? 64 : sz);
                slot.ptr = p;
                slot.ttl = preTTLs[i];
                localAllocs++;
            }
        }

        pos = (pos + 1) % RING_SIZE;
    }

    // Signal work complete
    doneCount.fetch_add(1, std::memory_order_release);
    // ─── TIMED SECTION ENDS HERE ───

    // Cleanup (not timed)
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
// Run benchmark with proper timing (excludes setup/teardown)
// ─────────────────────────────────────────────────────────────────────────────

template<typename AllocFn, typename FreeFn>
void runBenchmark(const char* name, int numThreads, AllocFn alloc_fn, FreeFn free_fn) {
    std::atomic<size_t> allocCount(0), freeCount(0);
    std::atomic<int> readyCount(0);
    std::atomic<bool> startFlag(false);
    std::atomic<int> doneCount(0);

    // Create threads (setup happens in parallel, not timed)
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back([&]() {
            worker(alloc_fn, free_fn, allocCount, freeCount,
                   readyCount, startFlag, doneCount);
        });
    }

    // Wait for all threads to finish setup
    while (readyCount.load(std::memory_order_acquire) < numThreads) {
        std::this_thread::yield();
    }

    // ─── START TIMING ───
    auto start = std::chrono::high_resolution_clock::now();
    startFlag.store(true, std::memory_order_release);

    // Wait for all threads to finish work
    while (doneCount.load(std::memory_order_acquire) < numThreads) {
        std::this_thread::yield();
    }

    // ─── END TIMING ───
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    // Join threads (cleanup happens here, not timed)
    for (auto& t : threads) {
        t.join();
    }

    size_t totalOps = allocCount.load() + freeCount.load();
    double opsPerSec = totalOps * 1000000.0 / elapsed;
    double mOpsPerSec = opsPerSec / 1e6;

    std::cout << name << "," << numThreads << ","
              << elapsed << "," << allocCount.load() << "," << freeCount.load() << ","
              << static_cast<long long>(opsPerSec) << ","
              << mOpsPerSec << std::endl;
}

inline void printHeader() {
    std::cout << "# WORKLOAD: Ring buffer (100K slots), 250K ops/thread" << std::endl;
    std::cout << "# SIZES: 60% small (16-256B), 30% medium (512-2KB), 10% large (4-32KB)" << std::endl;
    std::cout << "# TIMING: Excludes RNG init, ring buffer setup, thread creation" << std::endl;
    std::cout << "#" << std::endl;
    std::cout << "allocator,threads,elapsed_us,allocs,frees,ops_per_sec,mops_per_sec" << std::endl;
}
