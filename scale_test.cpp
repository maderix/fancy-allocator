#include "memory_allocator.h"
#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <atomic>
#include <iomanip>

// Simple interface for allocators
class AllocInterface {
public:
    virtual ~AllocInterface() {}
    virtual void* allocate(size_t sz) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual const char* getName() const = 0;
};

// System allocator
class SystemAllocator : public AllocInterface {
public:
    void* allocate(size_t sz) override { return std::malloc(sz); }
    void deallocate(void* ptr) override { std::free(ptr); }
    const char* getName() const override { return "System malloc/free"; }
};

// Fancy allocator with reclamation OFF
class FancyAllocatorNoReclaim : public AllocInterface {
public:
    FancyAllocatorNoReclaim() : fancy_(64ULL * 1024ULL * 1024ULL, false) {}
    
    void* allocate(size_t sz) override { return fancy_.allocate(sz); }
    void deallocate(void* ptr) override { fancy_.deallocate(ptr); }
    const char* getName() const override { return "Fancy (Reclamation OFF)"; }
    
private:
    FancyPerThreadAllocator fancy_;
};

// Fancy allocator with reclamation ON
class FancyAllocatorWithReclaim : public AllocInterface {
public:
    FancyAllocatorWithReclaim() : fancy_(64ULL * 1024ULL * 1024ULL, true) {}
    
    void* allocate(size_t sz) override { return fancy_.allocate(sz); }
    void deallocate(void* ptr) override { fancy_.deallocate(ptr); }
    const char* getName() const override { return "Fancy (Reclamation ON)"; }
    
private:
    FancyPerThreadAllocator fancy_;
};

// Optional TCMalloc support
#ifdef USE_TCMALLOC
#include <gperftools/tcmalloc.h>

class TCMallocAllocator : public AllocInterface {
public:
    void* allocate(size_t sz) override { return tc_malloc(sz); }
    void deallocate(void* ptr) override { tc_free(ptr); }
    const char* getName() const override { return "TCMalloc"; }
};
#endif

// Optional JEMalloc support
#ifdef USE_JEMALLOC
#include <jemalloc/jemalloc.h>

class JEMallocAllocator : public AllocInterface {
public:
    void* allocate(size_t sz) override { return malloc(sz); }
    void deallocate(void* ptr) override { free(ptr); }
    const char* getName() const override { return "JEMalloc"; }
};
#endif

// Worker function for the ephemeral HPC scenario
void ephemeralWorker(AllocInterface* alloc, int ops, int ringSize, std::atomic<size_t>& allocCount, std::atomic<size_t>& freeCount) {
    struct Slot { void* ptr; int ttl; };
    std::vector<Slot> ring(ringSize, {nullptr, 0});
    int pos = 0;

    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> catDist(1, 100);
    std::uniform_int_distribution<int> smallDist(16, 256);
    std::uniform_int_distribution<int> medDist(512, 2048);
    std::uniform_int_distribution<int> largeDist(4096, 32768);
    std::uniform_int_distribution<int> ttlDist(50, 2000);

    for (int i = 0; i < ops; i++) {
        auto& slot = ring[pos];
        // free if expired
        if (slot.ptr && slot.ttl <= 0) {
            alloc->deallocate(slot.ptr);
            slot.ptr = nullptr;
            freeCount++;
        }
        // decrement TTL
        if (slot.ptr && slot.ttl > 0) {
            slot.ttl--;
        }
        // if empty => allocate new
        if (!slot.ptr) {
            int c = catDist(rng);
            size_t sz = 0;
            if (c <= 60) sz = smallDist(rng);
            else if (c <= 90) sz = medDist(rng);
            else sz = largeDist(rng);

            void* p = alloc->allocate(sz);
            if (p) {
                slot.ptr = p;
                slot.ttl = ttlDist(rng);
                allocCount++;
            }
        }
        pos = (pos + 1) % ringSize;
    }
    // final free
    for (auto& s : ring) {
        if (s.ptr) {
            alloc->deallocate(s.ptr);
            s.ptr = nullptr;
            freeCount++;
        }
    }
}

// Run the test with a specific allocator
void runTest(AllocInterface* alloc, int threads, int opsPerThread, int ringSize) {
    std::cout << "Testing " << alloc->getName() << "..." << std::endl;
    
    std::atomic<size_t> allocCount(0);
    std::atomic<size_t> freeCount(0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> workers;
    workers.reserve(threads);
    
    for (int i = 0; i < threads; i++) {
        workers.emplace_back(ephemeralWorker, alloc, opsPerThread, ringSize, std::ref(allocCount), std::ref(freeCount));
    }
    
    for (auto& worker : workers) {
        worker.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "-- " << alloc->getName() << " --" << std::endl;
    std::cout << "Elapsed (us): " << elapsed << std::endl;
    std::cout << "Alloc calls: " << allocCount << ", Free calls: " << freeCount << std::endl;
    std::cout << std::endl;
}

int main() {
    // Reduced test parameters to avoid memory exhaustion
    const int threads = 128;           // Reduced from 512
    const int opsPerThread = 250000;   // Reduced from 1000000
    const int ringSize = 100000;       // Reduced from 500000
    
    std::cout << "=== High-Scale HPC Ephemeral Test (Reduced Memory) ===" << std::endl;
    std::cout << "Threads: " << threads << ", Ops/Thread: " << opsPerThread << ", Ring Size: " << ringSize << std::endl;
    std::cout << std::endl;
    
    // Run tests with different allocators
    {
        SystemAllocator sysAlloc;
        runTest(&sysAlloc, threads, opsPerThread, ringSize);
    }
    
    {
        FancyAllocatorNoReclaim fancyNoReclaim;
        runTest(&fancyNoReclaim, threads, opsPerThread, ringSize);
    }
    
    {
        FancyAllocatorWithReclaim fancyWithReclaim;
        runTest(&fancyWithReclaim, threads, opsPerThread, ringSize);
    }
    
#ifdef USE_TCMALLOC
    {
        TCMallocAllocator tcmAlloc;
        runTest(&tcmAlloc, threads, opsPerThread, ringSize);
    }
#endif
    
#ifdef USE_JEMALLOC
    {
        JEMallocAllocator jemAlloc;
        runTest(&jemAlloc, threads, opsPerThread, ringSize);
    }
#endif
    
    std::cout << "All tests completed." << std::endl;
    return 0;
} 