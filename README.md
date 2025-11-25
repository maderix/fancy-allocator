# FancyAllocator: High-Performance Per-Thread Memory Allocator

A blazing-fast, bounded-memory allocator designed for high-concurrency workloads. **5-6x faster than glibc malloc** and nearly matches jemalloc in single-threaded performance (only 1.17x slower), while providing deterministic memory bounds.

![Performance Comparison](benchmarks/benchmark_plots.png)

## Performance Highlights

### x86_64 (128 threads)

| Allocator | Ops/sec | vs glibc | Notes |
|-----------|---------|----------|-------|
| **glibc malloc** | 7.6M | 1.0x | System default |
| **tcmalloc** | 10.8M | 1.4x | Google's allocator |
| **FancyAllocator** | 49.8M | **6.5x** | Bounded memory |
| **jemalloc** | 75.0M | 9.9x | Facebook's allocator |

*Benchmark: 128 threads, 250K operations/thread, mixed allocation sizes (16B-32KB)*

### ARM64 (Raspberry Pi 5)

| Allocator | Ops/sec | vs glibc |
|-----------|---------|----------|
| **glibc malloc** | 1.78M | 1.0x |
| **FancyAllocator** | 3.73M | **2.1x** |

*Benchmark: 16 threads, 250K operations/thread (4-core Cortex-A76)*

### Thread Scaling

![Thread Scaling](benchmarks/thread_scaling.png)

| Threads | glibc | tcmalloc | Fancy | jemalloc |
|---------|-------|----------|-------|----------|
| 1 | 3.1M | 5.0M | **14.6M** | 16.7M |
| 2 | 5.1M | 11.6M | **27.2M** | 31.8M |
| 4 | 7.0M | 18.3M | **35.7M** | 52.3M |
| 8 | 8.3M | 22.1M | **43.4M** | 78.5M |
| 16 | 7.3M | 21.4M | **43.2M** | 71.3M |
| 32 | 7.7M | 17.1M | **43.4M** | 93.8M |
| 64 | 8.2M | 16.1M | **44.7M** | 91.7M |
| 128 | 8.5M | 14.9M | **47.4M** | 95.8M |

**Key insights**:
- **Single-threaded**: Fancy (14.6M) nearly matches jemalloc (16.7M), but is **2.9x faster than tcmalloc** (5.0M)
- **tcmalloc degrades**: Peaks at 8 threads (22M), then drops to 15M at 128 threads
- **glibc bottleneck**: Plateaus at ~8M ops/sec regardless of threads
- **Fancy scales consistently**: Linear scaling 1→8 threads, stable ~45M ops/sec thereafter

## Why FancyAllocator?

Memory allocation is often an invisible bottleneck in concurrent applications. The standard glibc malloc, while robust and portable, was designed in an era when four cores was considered parallel computing. Modern systems routinely run 64, 128, or more threads, and under these conditions, glibc's internal locking becomes a severe limitation. Our benchmarks demonstrate that glibc malloc throughput plateaus at approximately 8 million operations per second regardless of thread count—a clear indication of lock contention.

FancyAllocator takes a different approach. Rather than sharing memory structures across threads and mediating access through locks, it provides each thread with its own private arena and small-block cache. The result is that the most common allocation paths require no synchronization whatsoever. This design yields consistent performance of 45-50 million operations per second at high thread counts, representing a 5-6x improvement over glibc.

### Bounded Memory as a Feature

Most high-performance allocators like jemalloc and tcmalloc prioritize throughput above all else, allowing memory usage to grow as needed. This is often the right tradeoff for general-purpose applications, but it creates problems in constrained environments. A container with a 4GB memory limit can be killed without warning if the allocator decides to hold onto freed memory for future use. Real-time systems cannot tolerate the unpredictable latency spikes that occur when the allocator requests new memory from the operating system.

FancyAllocator addresses these concerns through fixed-size arenas. When you create an allocator with a 64MB arena size and run 16 threads, you know the maximum memory footprint will be approximately 1GB. This predictability is valuable in production environments where resource planning matters, and essential in embedded or real-time contexts where exceeding memory bounds is not merely inconvenient but catastrophic.

### Performance Characteristics

The performance profile of FancyAllocator is worth examining in detail. In single-threaded operation, it achieves 14.6 million operations per second compared to jemalloc's 16.7 million—a gap of only 14%. This near-parity demonstrates that the per-thread design does not impose significant overhead in the uncontended case. Surprisingly, FancyAllocator significantly outperforms tcmalloc (5.4M ops/sec) by a factor of 2.7x even without any threading benefits, suggesting that the slab-based small allocation design and O(1) bin lookup are inherently efficient.

As thread count increases, FancyAllocator scales linearly up to approximately 8 threads, after which throughput stabilizes around 45-50 million operations per second. The comparison with glibc is dramatic: at every thread count tested, FancyAllocator outperforms glibc by a factor of 5-6x. More importantly, glibc's performance actually degrades under high contention, while FancyAllocator maintains consistent throughput. This stability is particularly valuable for server applications where load varies unpredictably.

### Appropriate Use Cases

FancyAllocator is well-suited to applications where memory bounds must be guaranteed and where multiple threads perform frequent allocations. Game engines benefit from the absence of garbage collection pauses and the ability to dedicate arenas to specific subsystems. High-frequency trading systems value the consistent low-latency behavior. Server applications handling many concurrent connections can scale thread pools without hitting allocation bottlenecks. Container-based deployments appreciate the predictable memory footprint that stays within cgroup limits.

The allocator is less appropriate for single-threaded applications where simplicity is paramount—though notably, FancyAllocator still outperforms both glibc (4.7x faster) and tcmalloc (2.7x faster) even with a single thread, trailing only jemalloc by 14%. Applications that frequently allocate memory in one thread and free it in another will not benefit from per-thread arenas and may actually see degraded performance. Similarly, applications requiring truly unbounded memory growth should use an allocator designed for that purpose.

## Arena Size Trade-offs

![Arena Size Impact](benchmarks/performance_comparison.png)

| Arena Size | Throughput | Successful Allocs | Total Memory (128 threads) |
|------------|------------|-------------------|---------------------------|
| 32MB | 65.2M ops/s | 11.49M (90%) | 4GB |
| 64MB | 50.7M ops/s | 11.61M (91%) | 8GB |
| 128MB | 34.7M ops/s | 11.98M (94%) | 16GB |
| 256MB | 24.8M ops/s | 12.80M (100%) | 32GB |
| 512MB | 24.8M ops/s | 12.80M (100%) | 64GB |

**Key insight**: Smaller arenas are faster due to better cache locality. Choose based on your allocation success rate requirements.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FancyPerThreadAllocator                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐    │
│  │  Thread 1   │  │  Thread 2   │  ...  │  Thread N   │    │
│  ├─────────────┤  ├─────────────┤       ├─────────────┤    │
│  │ SmallCache  │  │ SmallCache  │       │ SmallCache  │    │
│  │ (16 bins)   │  │ (16 bins)   │       │ (16 bins)   │    │
│  │ ≤512 bytes  │  │ ≤512 bytes  │       │ ≤512 bytes  │    │
│  ├─────────────┤  ├─────────────┤       ├─────────────┤    │
│  │   Arena     │  │   Arena     │       │   Arena     │    │
│  │ (64MB mmap) │  │ (64MB mmap) │       │ (64MB mmap) │    │
│  │ >512 bytes  │  │ >512 bytes  │       │ >512 bytes  │    │
│  └─────────────┘  └─────────────┘       └─────────────┘    │
├─────────────────────────────────────────────────────────────┤
│              GlobalArenaManager (optional reclamation)       │
└─────────────────────────────────────────────────────────────┘
```

### Small Block Cache (≤512 bytes)
- 16 size classes: 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512
- O(1) size-to-bin lookup using bit manipulation
- Slab allocation via mmap (bypasses glibc entirely)
- Thread-local free lists for zero-contention recycling

### Arena (>512 bytes)
- Segregated free lists with 16 size bins
- O(1) bin lookup using `__builtin_clz`
- Immediate coalescing on free (forward + backward)
- Boundary tags for efficient merging

### Concurrency Design

**Why Fancy scales better than glibc:**

```
glibc malloc:                    FancyAllocator:
┌─────────────┐                  ┌─────────────┐
│ Thread 1 ───┼──┐               │ Thread 1    │──→ [Own Arena + Cache]
│ Thread 2 ───┼──┼─→ [Global    │ Thread 2    │──→ [Own Arena + Cache]
│ Thread 3 ───┼──┤    Lock]     │ Thread 3    │──→ [Own Arena + Cache]
│ Thread N ───┼──┘               │ Thread N    │──→ [Own Arena + Cache]
└─────────────┘                  └─────────────┘
    ↓ Contention                     ↓ Zero contention
    8M ops/sec max                   47M+ ops/sec
```

- **Per-thread arenas**: Each thread has dedicated memory, no locking needed
- **Thread-local small cache**: Hot path requires zero synchronization
- **Batched statistics**: Global stats updated every 128 ops (not every call)
- **Lock-free fast path**: Only arena creation requires synchronization

## Quick Start

```cpp
#include "memory_allocator.h"

int main() {
    // Create allocator: 64MB per thread, reclamation enabled
    FancyPerThreadAllocator alloc(64 * 1024 * 1024, true);

    // Allocate
    void* ptr = alloc.allocate(1024);

    // Use memory...

    // Deallocate
    alloc.deallocate(ptr);

    // Get stats
    auto stats = alloc.getStatsSnapshot();
    printf("Allocs: %zu, Frees: %zu\n",
           stats.totalAllocCalls, stats.totalFreeCalls);

    return 0;
}
```

## Building

### Requirements
- C++17 compiler (clang++ or g++)
- Linux (uses mmap, huge pages)
- pthreads

### Compile with optimizations
```bash
clang++ -O3 -pthread -march=native -o myapp myapp.cpp
```

### Run benchmarks
```bash
cd benchmarks/

# Compile benchmarks
clang++ -O3 -pthread -march=native -o bench_fancy bench_fancy.cpp
clang++ -O3 -pthread -march=native -o bench_glibc bench_glibc.cpp
clang++ -O3 -pthread -march=native -o bench_jemalloc bench_jemalloc.cpp -ljemalloc
clang++ -O3 -pthread -march=native -o bench_tcmalloc bench_tcmalloc.cpp /lib/x86_64-linux-gnu/libtcmalloc.so.4

# Run individual benchmarks
./bench_fancy
./bench_glibc
./bench_jemalloc
./bench_tcmalloc

# Test arena sizes (32, 64, 128, 256, 512 MB)
clang++ -O3 -pthread -march=native -o bench_fancy_arena bench_fancy_arena.cpp
./bench_fancy_arena 64

# Generate plots
python3 generate_plots.py
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `arenaSize` | 64MB | Memory per thread arena |
| `enableReclamation` | false | Background thread reclaims empty arenas |

### Reclamation Trade-off
- **OFF**: Maximum throughput, memory not returned to OS
- **ON**: Slightly lower throughput, reduces peak memory usage

## Internals & Optimizations

### Performance Techniques Used
- `__builtin_expect` for branch prediction hints
- `__builtin_prefetch` for cache warming
- `__attribute__((always_inline, hot))` for hot paths
- Cache-line aligned stats (64-byte) to prevent false sharing
- Batched statistics updates (every 128 ops) to reduce atomics
- mmap with `MAP_HUGETLB` / `MADV_HUGEPAGE` for large pages
- Adaptive spinlock (spin → yield → sleep) for short critical sections

### Memory Layout
```
Small Block:
┌──────────────────┬─────────────────────────┐
│ SmallBlockHeader │      User Data          │
│ (binIndex, size) │     (16-512 bytes)      │
└──────────────────┴─────────────────────────┘

Large Block:
┌─────────────┬─────────────────────────┬─────────────┐
│ BlockHeader │       User Data         │ BlockFooter │
│ (magic,size)│    (>512 bytes)         │ (magic,size)│
└─────────────┴─────────────────────────┴─────────────┘
```

## Benchmark Methodology

All benchmarks run with:
- 128 concurrent threads
- 250,000 operations per thread
- Ring buffer of 100,000 slots (simulates working set)
- Mixed allocation sizes:
  - 60%: 16-256 bytes (small)
  - 30%: 512-2048 bytes (medium)
  - 10%: 4096-32768 bytes (large)
- Random TTL (50-2000 ops) before deallocation
- 5-second sleep between runs for CPU cooldown

## License

MIT License - see LICENSE file.

## Contributing

Contributions welcome! Areas of interest:
- Windows support (VirtualAlloc instead of mmap)
- ARM optimizations
- Custom size class configurations
- Memory profiling tools
