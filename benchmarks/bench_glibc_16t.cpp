// Benchmark for jemalloc only
// Link with -ljemalloc to intercept malloc/free
#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <atomic>
#include <cstdlib>

void worker(int ops, int ringSize, std::atomic<size_t>& allocCount, std::atomic<size_t>& freeCount) {
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
        if (slot.ptr && slot.ttl <= 0) {
            free(slot.ptr);
            slot.ptr = nullptr;
            freeCount++;
        }
        if (slot.ptr && slot.ttl > 0) {
            slot.ttl--;
        }
        if (!slot.ptr) {
            int c = catDist(rng);
            size_t sz = 0;
            if (c <= 60) sz = smallDist(rng);
            else if (c <= 90) sz = medDist(rng);
            else sz = largeDist(rng);

            void* p = malloc(sz);
            if (p) {
                slot.ptr = p;
                slot.ttl = ttlDist(rng);
                allocCount++;
            }
        }
        pos = (pos + 1) % ringSize;
    }
    for (auto& s : ring) {
        if (s.ptr) {
            free(s.ptr);
            s.ptr = nullptr;
            freeCount++;
        }
    }
}

int main() {
    const int threads = 16;
    const int opsPerThread = 250000;
    const int ringSize = 100000;

    std::cout << "=== Glibc Benchmark ===" << std::endl;
    std::cout << "Threads: " << threads << ", Ops/Thread: " << opsPerThread << std::endl;

    std::atomic<size_t> allocCount(0);
    std::atomic<size_t> freeCount(0);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> workers;
    for (int i = 0; i < threads; i++) {
        workers.emplace_back(worker, opsPerThread, ringSize, std::ref(allocCount), std::ref(freeCount));
    }
    for (auto& w : workers) {
        w.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Elapsed (us): " << elapsed << std::endl;
    std::cout << "Allocs: " << allocCount << ", Frees: " << freeCount << std::endl;
    std::cout << "Ops/sec: " << (allocCount + freeCount) * 1000000ULL / elapsed << std::endl;

    return 0;
}
