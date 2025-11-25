// tcmalloc - thread scaling benchmark
// Compile: clang++ -O3 -pthread -o bench_tcmalloc_threads bench_tcmalloc_threads.cpp /lib/x86_64-linux-gnu/libtcmalloc.so.4
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
        if (slot.ptr && slot.ttl > 0) slot.ttl--;
        if (!slot.ptr) {
            int c = catDist(rng);
            size_t sz = (c <= 60) ? smallDist(rng) : (c <= 90) ? medDist(rng) : largeDist(rng);
            void* p = malloc(sz);
            if (p) { slot.ptr = p; slot.ttl = ttlDist(rng); allocCount++; }
        }
        pos = (pos + 1) % ringSize;
    }
    for (auto& s : ring) {
        if (s.ptr) { free(s.ptr); s.ptr = nullptr; freeCount++; }
    }
}

int main(int argc, char* argv[]) {
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int max_threads = (argc > 1) ? std::atoi(argv[1]) : 128;

    std::cout << "allocator,threads,elapsed_us,allocs,ops_per_sec" << std::endl;

    for (int t : thread_counts) {
        if (t > max_threads) break;

        std::atomic<size_t> allocCount(0), freeCount(0);
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> workers;
        for (int i = 0; i < t; i++)
            workers.emplace_back(worker, 250000, 100000, std::ref(allocCount), std::ref(freeCount));
        for (auto& w : workers) w.join();

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "tcmalloc," << t << "," << elapsed << "," << allocCount
                  << "," << (allocCount + freeCount) * 1000000ULL / elapsed << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    return 0;
}
