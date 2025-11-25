#include "memory_allocator.h"  // your final memory_allocator.h with optional reclamation
#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>


// We'll define an interface for system vs fancy
class AllocInterface {
public:
    virtual ~AllocInterface() {}
    virtual void* allocate(size_t sz)=0;
    virtual void  deallocate(void* ptr)=0;
    virtual bool hasStats() const { return false; }
    virtual AllocStatsSnapshot getStats() const { return {0,0,0,0}; }
};

// A wrapper for FancyPerThreadAllocator
class FancyInterface : public AllocInterface {
public:
    FancyInterface(size_t arenaSize, bool reclamation)
        : fancy_(arenaSize, reclamation)
    {}
    void* allocate(size_t sz) override {
        return fancy_.allocate(sz);
    }
    void deallocate(void* ptr) override {
        fancy_.deallocate(ptr);
    }
    bool hasStats() const override { return true; }
    AllocStatsSnapshot getStats() const override {
        return fancy_.getStatsSnapshot();
    }

    // Simple async request structure
    struct AsyncRequest {
        size_t size;
        void* result;
        bool completed;
    };

    // Async allocation - returns immediately with a request object
    std::shared_ptr<AsyncRequest> allocateAsync(size_t size) {
        auto request = std::make_shared<AsyncRequest>();
        request->size = size;
        
        // Fast path - do it immediately if not too busy
        request->result = allocate(size);
        request->completed = true;
        
        return request;
    }

    // Async deallocation - queue it up and return immediately
    void deallocateAsync(void* ptr) {
        if (!ptr) return;
        
        // Fast path - do it immediately if not too busy
        deallocate(ptr);
    }

    // Process any pending async operations
    void processAsyncBatch() {
        // Nothing to do in this simplified implementation
        // since we're doing everything immediately
    }

private:
    FancyPerThreadAllocator fancy_;
};

// A wrapper for system malloc
class SystemInterface : public AllocInterface {
public:
    void* allocate(size_t sz) override { return std::malloc(sz); }
    void  deallocate(void* ptr) override { std::free(ptr); }
};

// We'll do an ephemeral HPC scenario with ring buffer 
void ephemeralWorker(AllocInterface* alloc, int ops, int ringSize)
{
    struct Slot { void* ptr; int ttl; };
    std::vector<Slot> ring(ringSize, {nullptr, 0});
    int pos=0;

    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> catDist(1,100);
    std::uniform_int_distribution<int> smallDist(16,256);
    std::uniform_int_distribution<int> medDist(512,2048);
    std::uniform_int_distribution<int> largeDist(4096,32768);
    std::uniform_int_distribution<int> ttlDist(50,2000);

    for(int i=0; i<ops; i++){
        auto& slot = ring[pos];
        // free if expired
        if(slot.ptr && slot.ttl <= 0){
            alloc->deallocate(slot.ptr);
            slot.ptr = nullptr;
        }
        // decrement TTL
        if(slot.ptr && slot.ttl>0){
            slot.ttl--;
        }
        // if empty => allocate new
        if(!slot.ptr){
            int c = catDist(rng);
            size_t sz=0;
            if(c<=60) sz=smallDist(rng);
            else if(c<=90) sz=medDist(rng);
            else sz=largeDist(rng);

            void* p = alloc->allocate(sz);
            if(p){
                slot.ptr = p;
                slot.ttl = ttlDist(rng);
            }
        }
        pos = (pos+1) % ringSize;
    }
    // final free
    for(auto& s : ring){
        if(s.ptr){
            alloc->deallocate(s.ptr);
            s.ptr=nullptr;
        }
    }
}

// Function to run a warmup phase
void runWarmup(AllocInterface* alloc, int threads, int opsPerThread, int ringSize) {
    std::cout << "Running warmup phase... ";
    std::cout.flush();
    
    // Use fewer operations for warmup to save time
    int warmupOps = opsPerThread / 10;
    
    std::vector<std::thread> ths;
    ths.reserve(threads);
    for(int i=0; i<threads; i++){
        ths.emplace_back(ephemeralWorker, alloc, warmupOps, ringSize);
    }
    for(auto& t : ths){
        t.join();
    }
    
    std::cout << "done." << std::endl;
    
    // Optional: Add a small delay to let any background activity settle
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// We'll do a timed test for ephemeral HPC scenario
struct TestResult {
    long long elapsedUs;
    AllocStatsSnapshot snap;
};

TestResult runEphemeralTest(AllocInterface* alloc, int threads, int opsPerThread, int ringSize, bool doWarmup = true)
{
    // Run warmup phase if requested
    if (doWarmup) {
        runWarmup(alloc, threads, opsPerThread, ringSize);
    }
    
    // Reset stats if possible
    if (alloc->hasStats()) {
        // We can't actually reset the stats, but we can take a snapshot now
        // and subtract from the final result
        auto beforeSnap = alloc->getStats();
        
        auto start=std::chrono::high_resolution_clock::now();

        std::vector<std::thread> ths;
        ths.reserve(threads);
        for(int i=0; i<threads; i++){
            ths.emplace_back(ephemeralWorker, alloc, opsPerThread, ringSize);
        }
        for(auto& t : ths){
            t.join();
        }
        auto end=std::chrono::high_resolution_clock::now();
        long long us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        auto afterSnap = alloc->getStats();
        
        TestResult r;
        r.elapsedUs = us;
        r.snap.totalAllocCalls = afterSnap.totalAllocCalls - beforeSnap.totalAllocCalls;
        r.snap.totalFreeCalls = afterSnap.totalFreeCalls - beforeSnap.totalFreeCalls;
        r.snap.currentUsedBytes = afterSnap.currentUsedBytes;
        r.snap.peakUsedBytes = afterSnap.peakUsedBytes;
        
        return r;
    } else {
        // For allocators without stats, just time the operation
        auto start=std::chrono::high_resolution_clock::now();

        std::vector<std::thread> ths;
        ths.reserve(threads);
        for(int i=0; i<threads; i++){
            ths.emplace_back(ephemeralWorker, alloc, opsPerThread, ringSize);
        }
        for(auto& t : ths){
            t.join();
        }
        auto end=std::chrono::high_resolution_clock::now();
        long long us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        TestResult r;
        r.elapsedUs = us;
        r.snap = {0, 0, 0, 0};
        
        return r;
    }
}

int main(){
    // HPC ephemeral big test
    int threads = 128;
    int opsPerThread = 1000000;
    int ringSize     = 100000;

    std::cout << "\n=== Compare System Malloc vs. Fancy(Off) vs. Fancy(On) under HPC ephemeral scenario ===\n";
    std::cout << "Threads= " << threads << ", Ops/Thread= " << opsPerThread << ", ringSize= " << ringSize << "\n";

    // 1) System
    {
        SystemInterface sys;
        auto r = runEphemeralTest(&sys, threads, opsPerThread, ringSize);
        std::cout << "\n-- System malloc/free --\n";
        std::cout << "Elapsed (us): " << r.elapsedUs << "\n";
    }

    // 2) Fancy Reclamation OFF
    {
        FancyInterface fancyNoReclaim(64ULL*1024ULL*1024ULL, false);
        auto r = runEphemeralTest(&fancyNoReclaim, threads, opsPerThread, ringSize);
        std::cout << "\n-- Fancy Per-Thread (Reclamation OFF) --\n";
        std::cout << "Elapsed (us): " << r.elapsedUs << "\n";
        std::cout << "Alloc calls : " << r.snap.totalAllocCalls 
                  << ", Free calls: " << r.snap.totalFreeCalls 
                  << ", Peak usage: " << r.snap.peakUsedBytes << "\n";
    }

    // 3) Fancy Reclamation ON
    {
        FancyInterface fancyReclaim(64ULL*1024ULL*1024ULL, true);
        auto r = runEphemeralTest(&fancyReclaim, threads, opsPerThread, ringSize);
        std::cout << "\n-- Fancy Per-Thread (Reclamation ON) --\n";
        std::cout << "Elapsed (us): " << r.elapsedUs << "\n";
        std::cout << "Alloc calls : " << r.snap.totalAllocCalls
                  << ", Free calls: " << r.snap.totalFreeCalls
                  << ", Peak usage: " << r.snap.peakUsedBytes << "\n";
    }

    std::cout << "\nAll tests completed.\n";
    return 0;
}
