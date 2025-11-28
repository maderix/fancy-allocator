#ifndef MEMORY_ALLOCATOR_H
#define MEMORY_ALLOCATOR_H

#include <cstddef>     // size_t
#include <cstdint>     // uint32_t
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <sched.h>     // sched_yield
#include <sys/mman.h>  // mmap, munmap, madvise
#include <unordered_set>
#include <unordered_map>
#include <cstdio>      // fprintf for debug output
#include <new>         // std::align
#include <cerrno>      // EINVAL, ENOMEM for posix_memalign

//-------------------------------------------------------
// Debug/Safety Configuration
// Compile with -DFANCY_DEBUG to enable all safety checks
// Compile with -DFANCY_DEBUG_VERBOSE for detailed logging
//-------------------------------------------------------
#ifdef FANCY_DEBUG
    #define FANCY_CANARY_CHECK 1      // Buffer overflow detection
    #define FANCY_POISON_CHECK 1      // Use-after-free detection
    #define FANCY_DOUBLE_FREE_CHECK 1 // Double-free detection
    #define FANCY_STATS_DETAILED 1    // Per-bin statistics
    #define FANCY_STATS_ENABLED 1     // Basic stats in hot path
#else
    #define FANCY_CANARY_CHECK 0
    #define FANCY_POISON_CHECK 0
    #define FANCY_DOUBLE_FREE_CHECK 0
    #define FANCY_STATS_DETAILED 0
    #define FANCY_STATS_ENABLED 0     // No stats overhead in release
#endif

// Debug constants
static constexpr uint32_t CANARY_VALUE = 0xDEADCAFE;     // Buffer overflow canary
static constexpr uint32_t POISON_FREED = 0xFEEDFACE;    // Freed memory poison
static constexpr uint32_t POISON_UNINIT = 0xBAADF00D;   // Uninitialized memory
static constexpr size_t   FREED_TRACKER_SIZE = 16384;   // Track last N freed pointers

//-------------------------------------------------------
// Memory allocation helpers using mmap for better performance
//-------------------------------------------------------
inline void* allocatePages(size_t size) {
    // Round up to page size (4KB)
    size = (size + 4095) & ~4095ULL;

    // Try huge pages first for large allocations (2MB+)
    #ifdef MAP_HUGETLB
    if (size >= 2 * 1024 * 1024) {
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED) return p;
    }
    #endif

    // Fall back to regular pages with transparent huge page hint
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED) {
        #ifdef MADV_HUGEPAGE
        madvise(p, size, MADV_HUGEPAGE);  // THP hint
        #endif
    }
    return (p == MAP_FAILED) ? nullptr : p;
}

inline void freePages(void* ptr, size_t size) {
    if (ptr) {
        size = (size + 4095) & ~4095ULL;
        munmap(ptr, size);
    }
}

//-------------------------------------------------------
// Alignment helper functions
//-------------------------------------------------------
inline bool isPowerOfTwo(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

inline size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline void* alignPointer(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
}

// Maximum supported alignment (must be power of 2)
static constexpr size_t MAX_ALIGNMENT = 4096;
static constexpr size_t DEFAULT_ALIGNMENT = 16;  // Default for malloc compatibility

//-------------------------------------------------------
// Adaptive Spinlock - faster than std::mutex for short critical sections
//-------------------------------------------------------
class AdaptiveSpinlock {
    std::atomic<int> state_{0};  // 0=free, 1=locked, 2=contended
public:
    void lock() {
        int expected = 0;
        if (state_.compare_exchange_weak(expected, 1, std::memory_order_acquire))
            return;  // Fast path: got lock immediately

        // Spin briefly with pause instruction
        for (int i = 0; i < 100; i++) {
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");
            #endif
            expected = 0;
            if (state_.compare_exchange_weak(expected, 1, std::memory_order_acquire))
                return;
        }

        // Fall back to sched_yield for longer waits
        while (state_.exchange(2, std::memory_order_acquire) != 0) {
            sched_yield();
        }
    }

    void unlock() {
        state_.store(0, std::memory_order_release);
    }
};

//-------------------------------------------------------
// 1) Stats - Basic and Detailed
//-------------------------------------------------------
struct AllocStatsSnapshot {
    size_t totalAllocCalls;
    size_t totalFreeCalls;
    size_t currentUsedBytes;
    size_t peakUsedBytes;
};

// Detailed per-bin statistics (enabled with FANCY_DEBUG)
struct DetailedStatsSnapshot {
    AllocStatsSnapshot basic;

    // Per-bin stats (small allocations)
    size_t smallBinAllocCounts[16];
    size_t smallBinFreeCounts[16];
    size_t smallBinCurrentCount[16];

    // Per-bin stats (large allocations)
    size_t largeBinAllocCounts[16];
    size_t largeBinFreeCounts[16];
    size_t largeBinCurrentCount[16];

    // Fragmentation metrics
    size_t totalArenaBytes;        // Total arena memory
    size_t usedArenaBytes;         // Actually used memory
    size_t freeBlockCount;         // Number of free blocks
    size_t largestFreeBlock;       // Largest contiguous free block
    double fragmentationRatio;     // 1.0 - (largest / total_free)

    // Safety check stats
    size_t doubleFreeAttempts;
    size_t bufferOverflowsDetected;
    size_t useAfterFreeDetected;
    size_t corruptedBlocksDetected;

    void print() const {
        fprintf(stderr, "\n=== FancyAllocator Detailed Stats ===\n");
        fprintf(stderr, "Total alloc calls:    %zu\n", basic.totalAllocCalls);
        fprintf(stderr, "Total free calls:     %zu\n", basic.totalFreeCalls);
        fprintf(stderr, "Current used bytes:   %zu\n", basic.currentUsedBytes);
        fprintf(stderr, "Peak used bytes:      %zu\n", basic.peakUsedBytes);

        fprintf(stderr, "\n--- Small Bin Stats (<=512B) ---\n");
        const size_t smallSizes[] = {16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512};
        for (int i = 0; i < 16; i++) {
            if (smallBinAllocCounts[i] > 0) {
                fprintf(stderr, "  Bin %2d (%3zuB): allocs=%zu, frees=%zu, live=%zu\n",
                       i, smallSizes[i], smallBinAllocCounts[i], smallBinFreeCounts[i], smallBinCurrentCount[i]);
            }
        }

        fprintf(stderr, "\n--- Large Bin Stats (>512B) ---\n");
        const char* largeSizeNames[] = {"1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K",
                                        "256K", "512K", "1M", "2M", "4M", "8M", "16M", "huge"};
        for (int i = 0; i < 16; i++) {
            if (largeBinAllocCounts[i] > 0) {
                fprintf(stderr, "  Bin %2d (%s): allocs=%zu, frees=%zu, live=%zu\n",
                       i, largeSizeNames[i], largeBinAllocCounts[i], largeBinFreeCounts[i], largeBinCurrentCount[i]);
            }
        }

        fprintf(stderr, "\n--- Fragmentation ---\n");
        fprintf(stderr, "Total arena bytes:    %zu\n", totalArenaBytes);
        fprintf(stderr, "Used arena bytes:     %zu\n", usedArenaBytes);
        fprintf(stderr, "Free block count:     %zu\n", freeBlockCount);
        fprintf(stderr, "Largest free block:   %zu\n", largestFreeBlock);
        fprintf(stderr, "Fragmentation ratio:  %.2f%%\n", fragmentationRatio * 100.0);

        if (doubleFreeAttempts > 0 || bufferOverflowsDetected > 0 ||
            useAfterFreeDetected > 0 || corruptedBlocksDetected > 0) {
            fprintf(stderr, "\n--- SAFETY VIOLATIONS ---\n");
            if (doubleFreeAttempts > 0)
                fprintf(stderr, "  Double-free attempts:    %zu\n", doubleFreeAttempts);
            if (bufferOverflowsDetected > 0)
                fprintf(stderr, "  Buffer overflows:        %zu\n", bufferOverflowsDetected);
            if (useAfterFreeDetected > 0)
                fprintf(stderr, "  Use-after-free:          %zu\n", useAfterFreeDetected);
            if (corruptedBlocksDetected > 0)
                fprintf(stderr, "  Corrupted blocks:        %zu\n", corruptedBlocksDetected);
        }
        fprintf(stderr, "=====================================\n\n");
    }
};

// Cache-line aligned to prevent false sharing between cores
struct alignas(64) AllocStats {
    std::atomic<size_t> totalAllocCalls{0};
    char pad1[64 - sizeof(std::atomic<size_t>)];
    std::atomic<size_t> totalFreeCalls{0};
    char pad2[64 - sizeof(std::atomic<size_t>)];
    std::atomic<size_t> currentUsedBytes{0};
    char pad3[64 - sizeof(std::atomic<size_t>)];
    std::atomic<size_t> peakUsedBytes{0};
    char pad4[64 - sizeof(std::atomic<size_t>)];

#if FANCY_STATS_DETAILED
    // Per-bin statistics (only in debug mode)
    std::atomic<size_t> smallBinAllocs[16] = {};
    std::atomic<size_t> smallBinFrees[16] = {};
    std::atomic<size_t> largeBinAllocs[16] = {};
    std::atomic<size_t> largeBinFrees[16] = {};

    // Safety violation counters
    std::atomic<size_t> doubleFreeAttempts{0};
    std::atomic<size_t> bufferOverflows{0};
    std::atomic<size_t> useAfterFree{0};
    std::atomic<size_t> corruptedBlocks{0};
#endif

    AllocStatsSnapshot snapshot() const {
        AllocStatsSnapshot snap;
        snap.totalAllocCalls  = totalAllocCalls.load(std::memory_order_relaxed);
        snap.totalFreeCalls   = totalFreeCalls.load(std::memory_order_relaxed);
        snap.currentUsedBytes = currentUsedBytes.load(std::memory_order_relaxed);
        snap.peakUsedBytes    = peakUsedBytes.load(std::memory_order_relaxed);
        return snap;
    }
};

//-------------------------------------------------------
// 2) Thread-Local small-block cache
//    Expanded to 16 size classes for reduced fragmentation
//-------------------------------------------------------
static constexpr int SMALL_BIN_COUNT = 16;
static constexpr size_t SMALL_BIN_SIZE[SMALL_BIN_COUNT] = {
    16, 32, 48, 64, 80, 96, 112, 128,      // 8 tiny (16B quantum)
    160, 192, 224, 256,                     // 4 small (32B quantum)
    320, 384, 448, 512                      // 4 medium (64B quantum)
};
static constexpr size_t MAX_SMALL_SIZE = 512;

// Magic for small block headers (different from Arena::MAGIC)
// Reduced to 16-bit for smaller header
static constexpr uint16_t SMALL_MAGIC = 0xFACE;

// Header must be 16 bytes for 16-byte aligned user data (required for malloc/SIMD compatibility)
// Layout: [header 16B][user data 16B-aligned]
struct alignas(16) SmallBlockHeader {
    uint16_t magic;       // 2 bytes - Always SMALL_MAGIC
    uint8_t binIndex;     // 1 byte  - 0-15 bins (max 255)
    uint8_t flags;        // 1 byte  - Reserved
    uint32_t userSize;    // 4 bytes - For realloc support
#if FANCY_CANARY_CHECK
    uint32_t canaryHead;  // 4 bytes - Debug only
    uint32_t pad_;        // 4 bytes - Pad to 16 in debug
#else
    uint64_t pad_;        // 8 bytes - Pad to 16 for user data alignment
#endif
};
static_assert(sizeof(SmallBlockHeader) == 16, "SmallBlockHeader must be 16 bytes for alignment");
struct SmallFreeBlock {
    SmallBlockHeader hdr;
    SmallFreeBlock* next;
};

// Canary trailer for buffer overflow detection (placed after user data)
struct SmallBlockTrailer {
    uint32_t canaryTail;
};

// O(1) size-to-bin lookup - use bit manipulation directly
__attribute__((always_inline))
inline int fastFindSmallBin(size_t size) {
    if (__builtin_expect(size > MAX_SMALL_SIZE, 0)) return -1;
    if (size <= 16) return 0;

    // Tiny: 16B quantum (bins 0-7) -> sizes 1-128
    if (size <= 128) {
        return ((size - 1) >> 4);  // 0-7
    }
    // Small: 32B quantum (bins 8-11) -> sizes 129-256
    if (size <= 256) {
        return 8 + ((size - 129) >> 5);  // 8-11
    }
    // Medium: 64B quantum (bins 12-15) -> sizes 257-512
    return 12 + ((size - 257) >> 6);  // 12-15
}

class ThreadLocalSmallCache {
    static constexpr size_t SLAB_SIZE = 64 * 1024;  // 64KB slabs
    static constexpr size_t MAX_SLABS_PER_BIN = 64; // Track up to 64 slabs per bin

public:
    ThreadLocalSmallCache() {
        for (int i=0; i<SMALL_BIN_COUNT; i++){
            freeList_[i] = nullptr;
            slabCurrent_[i] = nullptr;
            slabEnd_[i] = nullptr;
            slabCount_[i] = 0;
        }
        ownerSlot_ = 0;
        localAllocCalls_ = 0;
        localFreeCalls_ = 0;
        localBytesAllocated_ = 0;
        localBytesFreed_ = 0;
#if FANCY_DOUBLE_FREE_CHECK
        freedTrackerIdx_ = 0;
        std::memset(freedTracker_, 0, sizeof(freedTracker_));
#endif
    }
    ~ThreadLocalSmallCache() {
        // Clean up all slabs
        for (int bin = 0; bin < SMALL_BIN_COUNT; bin++) {
            for (size_t i = 0; i < slabCount_[bin]; i++) {
                if (slabs_[bin][i]) {
                    freePages(slabs_[bin][i], SLAB_SIZE);
                }
            }
        }
    }

    void setOwnerSlot(uint16_t slot) { ownerSlot_ = slot; }

    // Flush local stats to global (called periodically or on thread exit)
    void flushStats(AllocStats& stats) {
        if (localAllocCalls_ > 0) {
            stats.totalAllocCalls.fetch_add(localAllocCalls_, std::memory_order_relaxed);
            localAllocCalls_ = 0;
        }
        if (localFreeCalls_ > 0) {
            stats.totalFreeCalls.fetch_add(localFreeCalls_, std::memory_order_relaxed);
            localFreeCalls_ = 0;
        }
        if (localBytesAllocated_ > localBytesFreed_) {
            stats.currentUsedBytes.fetch_add(localBytesAllocated_ - localBytesFreed_, std::memory_order_relaxed);
        } else if (localBytesFreed_ > localBytesAllocated_) {
            stats.currentUsedBytes.fetch_sub(localBytesFreed_ - localBytesAllocated_, std::memory_order_relaxed);
        }
        localBytesAllocated_ = 0;
        localBytesFreed_ = 0;
    }

    // find bin index for a requested size - O(1) lookup
    int findBin(size_t size) {
        return fastFindSmallBin(size);
    }

#if FANCY_DOUBLE_FREE_CHECK
    bool isRecentlyFreed(void* ptr) {
        for (size_t i = 0; i < FREED_TRACKER_SIZE; i++) {
            if (freedTracker_[i] == ptr) return true;
        }
        return false;
    }

    void trackFreed(void* ptr) {
        freedTracker_[freedTrackerIdx_] = ptr;
        freedTrackerIdx_ = (freedTrackerIdx_ + 1) % FREED_TRACKER_SIZE;
    }

    void untrackFreed(void* ptr) {
        for (size_t i = 0; i < FREED_TRACKER_SIZE; i++) {
            if (freedTracker_[i] == ptr) {
                freedTracker_[i] = nullptr;
                return;
            }
        }
    }
#endif

    __attribute__((always_inline, hot))
    void* allocateSmall(size_t reqSize, AllocStats& stats) {
        int bin = findBin(reqSize);
        if (__builtin_expect(bin < 0, 0)) return nullptr;

        void* userPtr = nullptr;
        char* block = freeList_[bin];

        // Fast path: pop from free list
        if (__builtin_expect(block != nullptr, 1)) {
            // Get next pointer (stored in user area after header)
            char* next = *reinterpret_cast<char**>(block + sizeof(SmallBlockHeader));
            freeList_[bin] = next;

            auto* hdr = reinterpret_cast<SmallBlockHeader*>(block);
            hdr->magic = SMALL_MAGIC;
            hdr->userSize = reqSize;
#if FANCY_CANARY_CHECK
            hdr->canaryHead = CANARY_VALUE;
#endif

            userPtr = block + sizeof(SmallBlockHeader);

#if FANCY_POISON_CHECK
            uint32_t* poisonCheck = reinterpret_cast<uint32_t*>(userPtr);
            if (*poisonCheck == POISON_FREED) {
                size_t binSize = SMALL_BIN_SIZE[bin];
                bool corrupted = false;
                for (size_t i = sizeof(void*); i < binSize; i += sizeof(uint32_t)) {
                    if (i + sizeof(uint32_t) <= binSize) {
                        uint32_t* check = reinterpret_cast<uint32_t*>((char*)userPtr + i);
                        if (*check != POISON_FREED) {
                            corrupted = true;
                            break;
                        }
                    }
                }
                if (corrupted) {
#if FANCY_STATS_DETAILED
                    stats.useAfterFree.fetch_add(1, std::memory_order_relaxed);
#endif
#ifdef FANCY_DEBUG_VERBOSE
                    fprintf(stderr, "[FANCY] USE-AFTER-FREE detected at %p (bin %d)\n", userPtr, bin);
#endif
                }
            }
            std::memset(userPtr, (POISON_UNINIT & 0xFF), SMALL_BIN_SIZE[bin]);
#endif

#if FANCY_CANARY_CHECK
            char* canaryPos = (char*)userPtr + reqSize;
            *reinterpret_cast<uint32_t*>(canaryPos) = CANARY_VALUE;
#endif

#if FANCY_DOUBLE_FREE_CHECK
            untrackFreed(userPtr);
#endif

#if FANCY_STATS_DETAILED
            stats.smallBinAllocs[bin].fetch_add(1, std::memory_order_relaxed);
#endif
#if FANCY_STATS_ENABLED
            if (__builtin_expect((++localAllocCalls_ & 0x3FF) == 0, 0)) {
                localBytesAllocated_ += localAllocCalls_ * 32;
                flushStats(stats);
            }
#endif
            return userPtr;
        }

        // Slow path: allocate from slab (bump pointer)
        size_t totalSz = sizeof(SmallBlockHeader) + SMALL_BIN_SIZE[bin];
#if FANCY_CANARY_CHECK
        totalSz += sizeof(uint32_t);
#endif
        // Always round up to 16 bytes for alignment (required for SIMD compatibility)
        totalSz = alignUp(totalSz, 16);
        if (__builtin_expect(slabCurrent_[bin] + totalSz > slabEnd_[bin], 0)) {
            char* slab = (char*)allocatePages(SLAB_SIZE);
            if (!slab) return nullptr;
            slabCurrent_[bin] = slab;
            slabEnd_[bin] = slab + SLAB_SIZE;
            // Track slab for cleanup
            if (slabCount_[bin] < MAX_SLABS_PER_BIN) {
                slabs_[bin][slabCount_[bin]++] = slab;
            }
        }

        char* newBlock = slabCurrent_[bin];
        slabCurrent_[bin] += totalSz;

        auto* hdr = reinterpret_cast<SmallBlockHeader*>(newBlock);
        hdr->magic = SMALL_MAGIC;
        hdr->binIndex = bin;
        hdr->flags = 0;
        hdr->userSize = reqSize;
#if FANCY_CANARY_CHECK
        hdr->canaryHead = CANARY_VALUE;
#endif

        userPtr = newBlock + sizeof(SmallBlockHeader);

#if FANCY_CANARY_CHECK
        char* canaryPos = (char*)userPtr + reqSize;
        *reinterpret_cast<uint32_t*>(canaryPos) = CANARY_VALUE;
#endif

#if FANCY_STATS_DETAILED
        stats.smallBinAllocs[bin].fetch_add(1, std::memory_order_relaxed);
#endif

#if FANCY_STATS_ENABLED
        localAllocCalls_++;
        localBytesAllocated_ += totalSz;
#endif

        return userPtr;
    }

    __attribute__((always_inline, hot))
    void freeSmall(void* userPtr, AllocStats& stats) {
        if (__builtin_expect(!userPtr, 0)) return;

        char* blockStart = (char*)userPtr - sizeof(SmallBlockHeader);
        auto* hdr = reinterpret_cast<SmallBlockHeader*>(blockStart);

        // First verify this is actually a small block
        if (__builtin_expect(hdr->magic != SMALL_MAGIC, 0)) {
#if FANCY_STATS_DETAILED
            stats.corruptedBlocks.fetch_add(1, std::memory_order_relaxed);
#endif
#ifdef FANCY_DEBUG_VERBOSE
            fprintf(stderr, "[FANCY] INVALID SMALL BLOCK at %p (bad magic 0x%X)\n", userPtr, hdr->magic);
#endif
            return;
        }

        int bin = (int)hdr->binIndex;

        // Safety check - corrupted block
        if (__builtin_expect(bin < 0 || bin >= SMALL_BIN_COUNT, 0)) {
#if FANCY_STATS_DETAILED
            stats.corruptedBlocks.fetch_add(1, std::memory_order_relaxed);
#endif
#ifdef FANCY_DEBUG_VERBOSE
            fprintf(stderr, "[FANCY] CORRUPTED BLOCK at %p (invalid bin %d)\n", userPtr, bin);
#endif
            return;
        }

#if FANCY_CANARY_CHECK
        // Check head canary
        if (hdr->canaryHead != CANARY_VALUE) {
#if FANCY_STATS_DETAILED
            stats.corruptedBlocks.fetch_add(1, std::memory_order_relaxed);
#endif
#ifdef FANCY_DEBUG_VERBOSE
            fprintf(stderr, "[FANCY] HEAD CANARY CORRUPTED at %p (expected 0x%X, got 0x%X)\n",
                   userPtr, CANARY_VALUE, hdr->canaryHead);
#endif
        }
        // Check tail canary
        char* canaryPos = (char*)userPtr + hdr->userSize;
        uint32_t tailCanary = *reinterpret_cast<uint32_t*>(canaryPos);
        if (tailCanary != CANARY_VALUE) {
#if FANCY_STATS_DETAILED
            stats.bufferOverflows.fetch_add(1, std::memory_order_relaxed);
#endif
#ifdef FANCY_DEBUG_VERBOSE
            fprintf(stderr, "[FANCY] BUFFER OVERFLOW at %p (tail canary expected 0x%X, got 0x%X)\n",
                   userPtr, CANARY_VALUE, tailCanary);
#endif
        }
#endif

#if FANCY_DOUBLE_FREE_CHECK
        // Check for double-free
        if (isRecentlyFreed(userPtr)) {
#if FANCY_STATS_DETAILED
            stats.doubleFreeAttempts.fetch_add(1, std::memory_order_relaxed);
#endif
#ifdef FANCY_DEBUG_VERBOSE
            fprintf(stderr, "[FANCY] DOUBLE-FREE detected at %p (bin %d)\n", userPtr, bin);
#endif
            return;  // Don't actually free - prevent corruption
        }
        trackFreed(userPtr);
#endif

#if FANCY_POISON_CHECK
        // Poison freed memory
        size_t binSize = SMALL_BIN_SIZE[bin];
        uint32_t* poisonStart = reinterpret_cast<uint32_t*>(userPtr);
        for (size_t i = 0; i < binSize; i += sizeof(uint32_t)) {
            poisonStart[i / sizeof(uint32_t)] = POISON_FREED;
        }
#endif

#if FANCY_STATS_DETAILED
        stats.smallBinFrees[bin].fetch_add(1, std::memory_order_relaxed);
#endif

        // Push to free list - store next ptr in user area
        *reinterpret_cast<char**>(blockStart + sizeof(SmallBlockHeader)) = freeList_[bin];
        freeList_[bin] = blockStart;

#if FANCY_STATS_ENABLED
        // Lightweight stats (batch flush every 1024 ops)
        if (__builtin_expect((++localFreeCalls_ & 0x3FF) == 0, 0)) {
            localBytesFreed_ += localFreeCalls_ * 32;
            flushStats(stats);
        }
#endif
    }

    // Get user-requested size for a small allocation
    size_t getSmallAllocSize(void* userPtr) {
        if (!userPtr) return 0;
        char* blockStart = (char*)userPtr - sizeof(SmallBlockHeader);
        auto* hdr = reinterpret_cast<SmallBlockHeader*>(blockStart);
        if (hdr->magic != SMALL_MAGIC) return 0;
        return hdr->userSize;
    }

private:
    // Per-bin free list (linked list through user area)
    char* freeList_[SMALL_BIN_COUNT];
    // Slab allocator for each bin (avoids calling glibc)
    char* slabCurrent_[SMALL_BIN_COUNT];
    char* slabEnd_[SMALL_BIN_COUNT];
    // Track slabs for cleanup on thread exit
    char* slabs_[SMALL_BIN_COUNT][MAX_SLABS_PER_BIN];
    size_t slabCount_[SMALL_BIN_COUNT];
    // Owner allocator slot for cross-thread deallocation
    uint16_t ownerSlot_;
    // Thread-local stats to avoid atomic contention
    size_t localAllocCalls_;
    size_t localFreeCalls_;
    size_t localBytesAllocated_;
    size_t localBytesFreed_;
#if FANCY_DOUBLE_FREE_CHECK
    // Circular buffer tracking recently freed pointers
    void* freedTracker_[FREED_TRACKER_SIZE];
    size_t freedTrackerIdx_;
#endif
};

//-------------------------------------------------------
// 3) Arena for large blocks
//    With segregated free lists for O(1) bin lookup
//-------------------------------------------------------
class Arena {
public:
    static constexpr uint32_t MAGIC = 0xCAFEBABE;

    // Segregated bin sizes for large allocations
    static constexpr int LARGE_BIN_COUNT = 16;
    static constexpr size_t LARGE_BIN_SIZE[LARGE_BIN_COUNT] = {
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
        262144, 524288, 1048576, 2097152, 4194304, 8388608,
        16777216, 0xFFFFFFFFFFFFFFFF  // Last bin for huge allocations
    };

    // O(1) bin lookup using CLZ
    static int findLargeBin(size_t size) {
        if (size <= 1024) return 0;
        if (size > 16777216) return LARGE_BIN_COUNT - 1;
        // Use leading zeros to find log2 and map to bin
        int log2 = 63 - __builtin_clzll(size - 1);
        return std::min(std::max(0, log2 - 9), LARGE_BIN_COUNT - 1);
    }

    // All block structures must be 16-byte aligned for better SIMD compatibility
    struct alignas(16) BlockHeader {
        uint32_t magic;
        uint16_t ownerSlot;   // Owner allocator slot (for cross-thread free)
        uint16_t alignOffset; // Offset from header end to user ptr
        size_t   totalSize;
        size_t   userSize;
        bool     isFree;
        char     pad_[7];     // Pad to 16-byte boundary
    };
    static_assert(sizeof(BlockHeader) == 32, "BlockHeader must be 32 bytes");
    static_assert(sizeof(BlockHeader) % 16 == 0, "BlockHeader must be 16-byte aligned");

    struct alignas(16) BlockFooter {
        uint32_t magic;
        uint32_t padding_;   // Explicit padding
        size_t   totalSize;
        bool     isFree;
        char     pad_[7];    // Pad to 16-byte boundary
    };
    static_assert(sizeof(BlockFooter) == 32, "BlockFooter must be 32 bytes");
    static_assert(sizeof(BlockFooter) % 16 == 0, "BlockFooter must be 16-byte aligned");

    struct alignas(16) FreeBlock {
        BlockHeader hdr;
        FreeBlock* next;
        char pad_[8];  // Pad to 48 bytes (multiple of 16)
    };
    static_assert(sizeof(FreeBlock) % 16 == 0, "FreeBlock must be 16-byte aligned");

    // Helper to round up to alignment (16-byte default for SIMD)
    static constexpr size_t ARENA_ALIGNMENT = 16;
    static constexpr size_t arenaAlignUp(size_t size) {
        return (size + ARENA_ALIGNMENT - 1) & ~(ARENA_ALIGNMENT - 1);
    }

    Arena(size_t arenaSize)
        : arenaSize_(arenaSize), usedBytes_(0)
    {
        // Use mmap for better performance and huge page support
        memory_ = (char*)allocatePages(arenaSize_);
        if (!memory_) {
            throw std::bad_alloc();
        }

        // Initialize all bins to nullptr
        for (int i = 0; i < LARGE_BIN_COUNT; i++) {
            freeBins_[i] = nullptr;
        }

        auto* fb = reinterpret_cast<FreeBlock*>(memory_);
        fb->hdr.magic=MAGIC;
        fb->hdr.totalSize=arenaSize_;
        fb->hdr.userSize=0;
        fb->hdr.isFree=true;
        fb->next=nullptr;

        auto* foot = getFooter(&fb->hdr);
        foot->magic=MAGIC;
        foot->totalSize=arenaSize_;
        foot->isFree=true;

        // Insert into appropriate bin
        insertIntoBin(fb);
    }
    ~Arena(){
        if(memory_){
            freePages(memory_, arenaSize_);
            memory_ = nullptr;
        }
    }

    size_t usedBytes() const { return usedBytes_; }
    size_t arenaSize() const { return arenaSize_; }
    bool fullyFree() const { return usedBytes_ == 0; }

    // Get the user-requested size of an allocation (for realloc)
    // O(1) lookup using back-offset
    size_t getAllocSize(void* userPtr) const {
        if (!userPtr) return 0;
        char* ptr = (char*)userPtr;

        // Fast path: default alignment
        auto* hdr = (BlockHeader*)(ptr - sizeof(BlockHeader));
        if (hdr->magic == MAGIC && !hdr->isFree && hdr->alignOffset == 0) {
            return hdr->userSize;
        }

        // Non-default alignment: use back-offset
        uint16_t backOffset = *reinterpret_cast<const uint16_t*>(ptr - sizeof(uint16_t));
        if (backOffset > 0 && backOffset <= MAX_ALIGNMENT) {
            hdr = (BlockHeader*)(ptr - sizeof(BlockHeader) - backOffset);
            if (hdr->magic == MAGIC && !hdr->isFree && hdr->alignOffset == backOffset) {
                return hdr->userSize;
            }
        }
        return 0;
    }

    // Check if this arena owns the given pointer
    bool ownsPointer(void* ptr) const {
        return ptr >= memory_ && ptr < memory_ + arenaSize_;
    }

    // Heap validation - walks all blocks and checks consistency
    struct HeapValidationResult {
        bool valid;
        size_t totalBlocks;
        size_t freeBlocks;
        size_t usedBlocks;
        size_t corruptedBlocks;
        size_t totalFreeBytes;
        size_t largestFreeBlock;
        double fragmentationRatio;
        std::vector<const char*> errors;
    };

    HeapValidationResult validateHeap() const {
        HeapValidationResult result = {};
        result.valid = true;

        char* pos = memory_;
        char* end = memory_ + arenaSize_;
        size_t totalFree = 0;
        size_t largestFree = 0;

        while (pos < end) {
            auto* hdr = reinterpret_cast<BlockHeader*>(pos);

            // Check magic number
            if (hdr->magic != MAGIC) {
                result.valid = false;
                result.corruptedBlocks++;
                result.errors.push_back("Invalid magic number in block header");
                break;  // Can't continue - don't know block size
            }

            // Check block size sanity
            if (hdr->totalSize == 0 || hdr->totalSize > arenaSize_) {
                result.valid = false;
                result.corruptedBlocks++;
                result.errors.push_back("Invalid block size");
                break;
            }

            // Check footer
            auto* foot = reinterpret_cast<BlockFooter*>(pos + hdr->totalSize - sizeof(BlockFooter));
            if (foot->magic != MAGIC) {
                result.valid = false;
                result.corruptedBlocks++;
                result.errors.push_back("Invalid magic number in block footer");
            }
            if (foot->totalSize != hdr->totalSize) {
                result.valid = false;
                result.corruptedBlocks++;
                result.errors.push_back("Header/footer size mismatch");
            }
            if (foot->isFree != hdr->isFree) {
                result.valid = false;
                result.corruptedBlocks++;
                result.errors.push_back("Header/footer free flag mismatch");
            }

            result.totalBlocks++;
            if (hdr->isFree) {
                result.freeBlocks++;
                totalFree += hdr->totalSize;
                if (hdr->totalSize > largestFree) {
                    largestFree = hdr->totalSize;
                }
            } else {
                result.usedBlocks++;
            }

            pos += hdr->totalSize;
        }

        result.totalFreeBytes = totalFree;
        result.largestFreeBlock = largestFree;
        if (totalFree > 0) {
            result.fragmentationRatio = 1.0 - (double)largestFree / (double)totalFree;
        } else {
            result.fragmentationRatio = 0.0;
        }

        return result;
    }

    // Get fragmentation metrics
    struct FragmentationMetrics {
        size_t totalArenaBytes;
        size_t usedBytes;
        size_t freeBytes;
        size_t freeBlockCount;
        size_t largestFreeBlock;
        size_t smallestFreeBlock;
        double fragmentationRatio;   // 0.0 = no fragmentation, 1.0 = fully fragmented
        double utilizationRatio;     // usedBytes / totalArenaBytes
    };

    FragmentationMetrics getFragmentation() const {
        FragmentationMetrics m = {};
        m.totalArenaBytes = arenaSize_;
        m.usedBytes = usedBytes_;
        m.freeBytes = arenaSize_ - usedBytes_;
        m.smallestFreeBlock = SIZE_MAX;

        // Walk the free lists
        for (int bin = 0; bin < LARGE_BIN_COUNT; bin++) {
            FreeBlock* cur = freeBins_[bin];
            while (cur) {
                m.freeBlockCount++;
                if (cur->hdr.totalSize > m.largestFreeBlock) {
                    m.largestFreeBlock = cur->hdr.totalSize;
                }
                if (cur->hdr.totalSize < m.smallestFreeBlock) {
                    m.smallestFreeBlock = cur->hdr.totalSize;
                }
                cur = cur->next;
            }
        }

        if (m.freeBlockCount == 0) {
            m.smallestFreeBlock = 0;
        }

        if (m.freeBytes > 0 && m.largestFreeBlock > 0) {
            m.fragmentationRatio = 1.0 - (double)m.largestFreeBlock / (double)m.freeBytes;
        }
        m.utilizationRatio = (double)m.usedBytes / (double)m.totalArenaBytes;

        return m;
    }

    void destroy() {
        // unmap
        if(memory_) {
            freePages(memory_, arenaSize_);
            memory_=nullptr;
        }
    }

    __attribute__((hot))
    void* allocate(size_t reqSize, size_t alignment, AllocStats& stats, uint16_t ownerSlot = 0) {
        // No lock needed - each thread has its own Arena

        // Ensure alignment is at least ARENA_ALIGNMENT and power of 2
        if (alignment < ARENA_ALIGNMENT) alignment = ARENA_ALIGNMENT;
        if (!isPowerOfTwo(alignment) || alignment > MAX_ALIGNMENT) {
            return nullptr;  // Invalid alignment
        }

        constexpr size_t overhead = sizeof(BlockHeader) + sizeof(BlockFooter);
        // Add extra space for alignment padding
        const size_t alignPadding = (alignment > ARENA_ALIGNMENT) ? (alignment - 1) : 0;
        const size_t totalNeeded = arenaAlignUp(reqSize + overhead + alignPadding);
        const int startBin = findLargeBin(totalNeeded);

        // Search bins from startBin upward for a fit
        for (int bin = startBin; bin < LARGE_BIN_COUNT; bin++) {
            FreeBlock* cur = freeBins_[bin];
            if (__builtin_expect(cur == nullptr, 0)) continue;

            FreeBlock* prev = nullptr;
            do {
                if (__builtin_expect(cur->hdr.totalSize >= totalNeeded, 1)) {
                    char* start = (char*)cur;
                    char* baseUserArea = start + sizeof(BlockHeader);

                    // Calculate aligned user pointer
                    char* alignedUserArea = (char*)alignPointer(baseUserArea, alignment);
                    uint16_t alignOffset = (uint16_t)(alignedUserArea - baseUserArea);

                    // Adjust needed size to account for actual alignment
                    size_t actualNeeded = arenaAlignUp(reqSize + overhead + alignOffset);

                    // Check if block is still large enough after alignment adjustment
                    if (actualNeeded > cur->hdr.totalSize) {
                        // This block doesn't fit after alignment, try next
                        prev = cur;
                        cur = cur->next;
                        continue;
                    }

                    // Remove from current bin (common: head of list)
                    if (__builtin_expect(prev == nullptr, 1))
                        freeBins_[bin] = cur->next;
                    else
                        prev->next = cur->next;

                    // Split if worthwhile - ensure leftover is also aligned
                    size_t leftover = cur->hdr.totalSize - actualNeeded;
                    const size_t minSplitSize = arenaAlignUp(sizeof(FreeBlock) + overhead);
                    if (leftover >= minSplitSize) {
                        char* leftoverAddr = start + actualNeeded;
                        auto* leftoverFB = (FreeBlock*)leftoverAddr;
                        leftoverFB->hdr.magic = MAGIC;
                        leftoverFB->hdr.totalSize = leftover;
                        leftoverFB->hdr.userSize = 0;
                        leftoverFB->hdr.isFree = true;
                        leftoverFB->hdr.ownerSlot = 0;
                        leftoverFB->hdr.alignOffset = 0;
                        leftoverFB->next = nullptr;

                        auto* leftoverFoot = getFooter(&leftoverFB->hdr);
                        leftoverFoot->magic = MAGIC;
                        leftoverFoot->totalSize = leftover;
                        leftoverFoot->isFree = true;

                        insertIntoBin(leftoverFB);
                    } else {
                        actualNeeded = cur->hdr.totalSize;
                    }

                    // Mark allocated with alignment info
                    cur->hdr.isFree = false;
                    cur->hdr.userSize = reqSize;
                    cur->hdr.totalSize = actualNeeded;
                    cur->hdr.ownerSlot = ownerSlot;
                    cur->hdr.alignOffset = alignOffset;

                    // Store back-offset right before user pointer for O(1) header lookup
                    // Only needed when alignOffset > 0 (non-default alignment)
                    if (alignOffset > 0) {
                        uint16_t* backOffset = reinterpret_cast<uint16_t*>(alignedUserArea) - 1;
                        *backOffset = alignOffset;
                    }

                    auto* foot = getFooter(&cur->hdr);
                    foot->magic = MAGIC;
                    foot->totalSize = actualNeeded;
                    foot->isFree = false;

                    usedBytes_ += actualNeeded;

#if FANCY_STATS_DETAILED
                    int allocBin = findLargeBin(actualNeeded);
                    stats.largeBinAllocs[allocBin].fetch_add(1, std::memory_order_relaxed);
#endif

#if FANCY_STATS_ENABLED
                    // Batch stats every 512 ops (increased for better scaling)
                    if (__builtin_expect((++localAllocCount_ & 0x1FF) == 0, 0)) {
                        stats.totalAllocCalls.fetch_add(localAllocCount_, std::memory_order_relaxed);
                        stats.currentUsedBytes.fetch_add(localAllocBytes_ + actualNeeded, std::memory_order_relaxed);
                        localAllocCount_ = 0;
                        localAllocBytes_ = 0;
                    } else {
                        localAllocBytes_ += actualNeeded;
                    }
#endif

                    return alignedUserArea;
                }
                prev = cur;
                cur = cur->next;
            } while (cur);
        }
        return nullptr;
    }

    // Get block header from user pointer, accounting for alignment offset
    // O(1) lookup using back-offset stored before user pointer
    __attribute__((always_inline))
    BlockHeader* getHeaderFromUserPtr(void* userPtr) {
        char* ptr = (char*)userPtr;

        // Fast path: check default alignment (alignOffset == 0)
        // Header is directly before user pointer
        auto* hdr = (BlockHeader*)(ptr - sizeof(BlockHeader));
        if (__builtin_expect(hdr->magic == MAGIC && !hdr->isFree && hdr->alignOffset == 0, 1)) {
            return hdr;
        }

        // Non-default alignment: read back-offset stored before user pointer
        // The offset tells us how far the header is
        uint16_t backOffset = *reinterpret_cast<uint16_t*>(ptr - sizeof(uint16_t));
        if (backOffset > 0 && backOffset <= MAX_ALIGNMENT) {
            hdr = (BlockHeader*)(ptr - sizeof(BlockHeader) - backOffset);
            if (hdr->magic == MAGIC && !hdr->isFree && hdr->alignOffset == backOffset) {
                return hdr;
            }
        }

        return nullptr;
    }

    __attribute__((hot))
    void deallocate(void* userPtr, AllocStats& stats) {
        if (__builtin_expect(!userPtr, 0)) return;

        auto* hdr = getHeaderFromUserPtr(userPtr);
        if (__builtin_expect(!hdr, 0)) {
#if FANCY_STATS_DETAILED
            stats.corruptedBlocks.fetch_add(1, std::memory_order_relaxed);
#endif
#ifdef FANCY_DEBUG_VERBOSE
            fprintf(stderr, "[FANCY] Could not find block header for %p\n", userPtr);
#endif
            return;
        }
        char* start = (char*)hdr;
        // Magic already validated by getHeaderFromUserPtr

#if FANCY_DOUBLE_FREE_CHECK
        // Check if already free (double-free)
        if (hdr->isFree) {
#if FANCY_STATS_DETAILED
            stats.doubleFreeAttempts.fetch_add(1, std::memory_order_relaxed);
#endif
#ifdef FANCY_DEBUG_VERBOSE
            fprintf(stderr, "[FANCY] DOUBLE-FREE detected at %p (large block)\n", userPtr);
#endif
            return;
        }
#endif

        // Validate footer
        auto* foot = getFooter(hdr);
        if (__builtin_expect(foot->magic != MAGIC, 0)) {
#if FANCY_STATS_DETAILED
            stats.bufferOverflows.fetch_add(1, std::memory_order_relaxed);
#endif
#ifdef FANCY_DEBUG_VERBOSE
            fprintf(stderr, "[FANCY] BUFFER OVERFLOW at %p (footer magic corrupted: 0x%X)\n",
                   userPtr, foot->magic);
#endif
        }

#if FANCY_STATS_DETAILED
        int bin = findLargeBin(hdr->totalSize);
        stats.largeBinFrees[bin].fetch_add(1, std::memory_order_relaxed);
#endif

        hdr->isFree = true;
        size_t sz = hdr->totalSize;
        usedBytes_ -= sz;

        // Update footer free flag to match header
        auto* footerToUpdate = getFooter(hdr);
        footerToUpdate->isFree = true;

#if FANCY_POISON_CHECK
        // Poison freed memory (after header+alignOffset, before footer)
        // Account for alignment offset so we don't poison past the footer
        char* poisonStart = (char*)userPtr;
        size_t usableSize = sz - sizeof(BlockHeader) - sizeof(BlockFooter) - hdr->alignOffset;
        if (usableSize <= sz && usableSize >= sizeof(uint32_t)) {  // Sanity check
            // Poison in 4-byte chunks
            uint32_t* p = reinterpret_cast<uint32_t*>(poisonStart);
            size_t count = usableSize / sizeof(uint32_t);
            for (size_t i = 0; i < count; i++) {
                p[i] = POISON_FREED;
            }
        }
#endif

#if FANCY_STATS_ENABLED
        // Batch stats every 512 ops (increased for better scaling)
        if (__builtin_expect((++localFreeCount_ & 0x1FF) == 0, 0)) {
            stats.totalFreeCalls.fetch_add(localFreeCount_, std::memory_order_relaxed);
            stats.currentUsedBytes.fetch_sub(localFreeBytes_ + sz, std::memory_order_relaxed);
            localFreeCount_ = 0;
            localFreeBytes_ = 0;
        } else {
            localFreeBytes_ += sz;
        }
#endif

        auto* fb = (FreeBlock*)hdr;

        // Coalesce and insert
        coalesceForward(fb);
        FreeBlock* result = coalesceBackwardAndGet(fb);
        insertIntoBin(result);
    }

    void coalesceAll(){
        // No lock needed - each thread has its own Arena
        // we do merges on free anyway
    }

private:
    BlockFooter* getFooter(BlockHeader* hdr){
        char* footAddr=(char*)hdr + hdr->totalSize - sizeof(BlockFooter);
        return (BlockFooter*)footAddr;
    }

    // Insert a free block into the appropriate size bin
    void insertIntoBin(FreeBlock* fb) {
        int bin = findLargeBin(fb->hdr.totalSize);
        fb->next = freeBins_[bin];
        freeBins_[bin] = fb;
    }

    // Remove a free block from its bin
    void removeFromBin(BlockHeader* h){
        int bin = findLargeBin(h->totalSize);
        FreeBlock* prev = nullptr;
        FreeBlock* cur = freeBins_[bin];
        while(cur){
            if(&cur->hdr == h){
                if(!prev) freeBins_[bin] = cur->next;
                else prev->next = cur->next;
                cur->next = nullptr;
                return;
            }
            prev = cur;
            cur = cur->next;
        }
        // Block might be in a different bin if size changed, search all bins
        for (int b = 0; b < LARGE_BIN_COUNT; b++) {
            if (b == bin) continue;
            prev = nullptr;
            cur = freeBins_[b];
            while(cur){
                if(&cur->hdr == h){
                    if(!prev) freeBins_[b] = cur->next;
                    else prev->next = cur->next;
                    cur->next = nullptr;
                    return;
                }
                prev = cur;
                cur = cur->next;
            }
        }
    }

    void coalesceForward(FreeBlock* blk){
        char* nxtAddr=(char*)blk + blk->hdr.totalSize;
        if(nxtAddr>=memory_+arenaSize_) return;
        auto* nxtHdr=(BlockHeader*)nxtAddr;
        if(nxtHdr->magic==MAGIC && nxtHdr->isFree){
            removeFromBin(nxtHdr);
            blk->hdr.totalSize += nxtHdr->totalSize;
            auto* foot=getFooter(&blk->hdr);
            foot->magic= MAGIC;
            foot->totalSize= blk->hdr.totalSize;
            foot->isFree= true;
        }
    }

    // Returns the resulting block after potential merge with previous
    FreeBlock* coalesceBackwardAndGet(FreeBlock* blk){
        if((char*)blk == memory_) return blk;
        char* footAddr=(char*)blk - sizeof(BlockFooter);
        if(footAddr<memory_) return blk;
        auto* foot=(BlockFooter*)footAddr;
        if(foot->magic==MAGIC && foot->isFree){
            size_t prevSz= foot->totalSize;
            char* prevAddr=(char*)blk - prevSz;
            auto* prevHdr=(BlockHeader*)prevAddr;
            if(prevHdr->magic==MAGIC && prevHdr->isFree){
                // Remove prev block from its bin, merge into prev
                removeFromBin(prevHdr);
                prevHdr->totalSize += blk->hdr.totalSize;
                auto* newFoot=getFooter(prevHdr);
                newFoot->magic= MAGIC;
                newFoot->totalSize= prevHdr->totalSize;
                newFoot->isFree= true;

                // Return the previous block (now contains merged data)
                return (FreeBlock*)prevHdr;
            }
        }
        return blk;
    }

    char* memory_;
    size_t arenaSize_;
    size_t usedBytes_;  // Per-thread arena, no atomic needed
    FreeBlock* freeBins_[LARGE_BIN_COUNT];  // Segregated free lists
    AdaptiveSpinlock spinlock_;

    // Local counters for batched stats (reduces atomic contention)
    size_t localAllocCount_ = 0;
    size_t localAllocBytes_ = 0;
    size_t localFreeCount_ = 0;
    size_t localFreeBytes_ = 0;
};

//-------------------------------------------------------
// 4) ThreadLocalData
//-------------------------------------------------------
struct ThreadLocalData {
    Arena* arena;
    ThreadLocalSmallCache smallCache;
};

//-------------------------------------------------------
// 5) GlobalArenaManager with optional reclamation
//-------------------------------------------------------
class GlobalArenaManager {
public:
    GlobalArenaManager(bool enableReclamation)
        : stopThread_(false)
        , enableReclamation_(enableReclamation)
    {
        if(enableReclamation_){
            bgThread_ = std::thread([this]{ this->bgLoop(); });
        }
    }
    ~GlobalArenaManager(){
        {
            std::lock_guard<std::mutex> lk(mgrMutex_);
            stopThread_ = true;
        }
        cv_.notify_all();
        if(bgThread_.joinable()){
            bgThread_.join();
        }
        // clean up all arenas
        for(auto* a: arenas_){
            a->destroy();
            delete a;
        }
    }

    Arena* createArena(size_t arenaSize){
        std::lock_guard<std::mutex> lk(mgrMutex_);
        auto* a=new Arena(arenaSize);
        arenas_.push_back(a);
        return a;
    }

    // Get all arenas for validation/stats
    std::vector<Arena*> getArenas() const {
        std::lock_guard<std::mutex> lk(mgrMutex_);
        return arenas_;
    }

    // Get aggregate stats across all arenas
    void getArenaStats(size_t& totalBytes, size_t& usedBytes, size_t& freeBlocks,
                       size_t& largestFree, double& fragRatio) const {
        std::lock_guard<std::mutex> lk(mgrMutex_);
        totalBytes = 0;
        usedBytes = 0;
        freeBlocks = 0;
        largestFree = 0;
        double totalFree = 0;

        for (auto* arena : arenas_) {
            auto frag = arena->getFragmentation();
            totalBytes += frag.totalArenaBytes;
            usedBytes += frag.usedBytes;
            freeBlocks += frag.freeBlockCount;
            if (frag.largestFreeBlock > largestFree) {
                largestFree = frag.largestFreeBlock;
            }
            totalFree += frag.freeBytes;
        }

        if (totalFree > 0) {
            fragRatio = 1.0 - (double)largestFree / totalFree;
        } else {
            fragRatio = 0.0;
        }
    }

    // Validate all arenas
    bool validateAllArenas(std::vector<std::string>& errors) const {
        std::lock_guard<std::mutex> lk(mgrMutex_);
        bool allValid = true;

        for (size_t i = 0; i < arenas_.size(); i++) {
            auto result = arenas_[i]->validateHeap();
            if (!result.valid) {
                allValid = false;
                for (auto* err : result.errors) {
                    errors.push_back(std::string("Arena ") + std::to_string(i) + ": " + err);
                }
            }
        }
        return allValid;
    }

private:
    void bgLoop(){
        // runs every 1 second
        while(true){
            std::unique_lock<std::mutex> lk(mgrMutex_);
            cv_.wait_for(lk, std::chrono::seconds(1), [this]{return stopThread_;});
            if(stopThread_) break;
            if(!enableReclamation_) continue; 

            // pass
            for(size_t i=0; i<arenas_.size(); ){
                auto* ar= arenas_[i];
                ar->coalesceAll();
                if(ar->fullyFree()){
                    // reclaim
                    ar->destroy();
                    delete ar;
                    arenas_.erase(arenas_.begin()+i);
                } else {
                    i++;
                }
            }
        }
    }

    std::vector<Arena*> arenas_;
    bool stopThread_;
    bool enableReclamation_;
    std::thread bgThread_;
    mutable std::mutex mgrMutex_;  // mutable for const methods
    std::condition_variable cv_;
};

//-------------------------------------------------------
// 6) The main facade
//-------------------------------------------------------
class FancyPerThreadAllocator {
public:
    explicit FancyPerThreadAllocator(size_t defaultArenaSize, bool enableReclamation=false)
        : defaultArenaSize_(defaultArenaSize)
        , instanceSlot_(nextSlot_.fetch_add(1, std::memory_order_relaxed))
    {
        manager_ = std::make_shared<GlobalArenaManager>(enableReclamation);
    }

    ~FancyPerThreadAllocator() {
        // Clean up our entry in the thread-local cache
        // Note: This only cleans up the current thread's entry
        if (instanceSlot_ < TLD_FAST_SLOTS) {
            if (tldCache_.keys[instanceSlot_] == this) {
                delete tldCache_.values[instanceSlot_];
                tldCache_.keys[instanceSlot_] = nullptr;
                tldCache_.values[instanceSlot_] = nullptr;
            }
        } else {
            auto it = tldCache_.overflow.find(this);
            if (it != tldCache_.overflow.end()) {
                delete it->second;
                tldCache_.overflow.erase(it);
            }
        }
    }

    // Disable copy (allocator manages unique resources)
    FancyPerThreadAllocator(const FancyPerThreadAllocator&) = delete;
    FancyPerThreadAllocator& operator=(const FancyPerThreadAllocator&) = delete;

    AllocStatsSnapshot getStatsSnapshot() const {
        return stats_.snapshot();
    }

    // Get detailed stats (only meaningful with FANCY_DEBUG)
    DetailedStatsSnapshot getDetailedStats() const {
        DetailedStatsSnapshot ds = {};
        ds.basic = stats_.snapshot();

#if FANCY_STATS_DETAILED
        // Copy per-bin stats
        for (int i = 0; i < 16; i++) {
            ds.smallBinAllocCounts[i] = stats_.smallBinAllocs[i].load(std::memory_order_relaxed);
            ds.smallBinFreeCounts[i] = stats_.smallBinFrees[i].load(std::memory_order_relaxed);
            ds.smallBinCurrentCount[i] = ds.smallBinAllocCounts[i] - ds.smallBinFreeCounts[i];

            ds.largeBinAllocCounts[i] = stats_.largeBinAllocs[i].load(std::memory_order_relaxed);
            ds.largeBinFreeCounts[i] = stats_.largeBinFrees[i].load(std::memory_order_relaxed);
            ds.largeBinCurrentCount[i] = ds.largeBinAllocCounts[i] - ds.largeBinFreeCounts[i];
        }

        ds.doubleFreeAttempts = stats_.doubleFreeAttempts.load(std::memory_order_relaxed);
        ds.bufferOverflowsDetected = stats_.bufferOverflows.load(std::memory_order_relaxed);
        ds.useAfterFreeDetected = stats_.useAfterFree.load(std::memory_order_relaxed);
        ds.corruptedBlocksDetected = stats_.corruptedBlocks.load(std::memory_order_relaxed);
#endif

        // Get arena-level stats
        manager_->getArenaStats(ds.totalArenaBytes, ds.usedArenaBytes,
                                ds.freeBlockCount, ds.largestFreeBlock,
                                ds.fragmentationRatio);

        return ds;
    }

    // Validate heap integrity
    bool validateHeap(std::vector<std::string>& errors) const {
        return manager_->validateAllArenas(errors);
    }

    // Quick heap check (returns false if any corruption detected)
    bool isHeapValid() const {
        std::vector<std::string> errors;
        return validateHeap(errors);
    }

    // Print detailed stats to stderr
    void printDetailedStats() const {
        auto ds = getDetailedStats();
        ds.print();
    }

    // Check for potential memory leaks (live allocations at shutdown)
    struct LeakReport {
        size_t liveAllocations;
        size_t liveBytes;
        bool hasLeaks() const { return liveAllocations > 0; }

        void print() const {
            if (hasLeaks()) {
                fprintf(stderr, "\n=== MEMORY LEAK REPORT ===\n");
                fprintf(stderr, "Live allocations: %zu\n", liveAllocations);
                fprintf(stderr, "Live bytes:       %zu\n", liveBytes);
                fprintf(stderr, "==========================\n\n");
            } else {
                fprintf(stderr, "\n=== No memory leaks detected ===\n\n");
            }
        }
    };

    LeakReport checkLeaks() const {
        LeakReport report = {};
        auto snap = stats_.snapshot();
        report.liveAllocations = snap.totalAllocCalls - snap.totalFreeCalls;
        report.liveBytes = snap.currentUsedBytes;
        return report;
    }

    __attribute__((always_inline, hot))
    void* allocate(size_t size) {
        if (__builtin_expect(size == 0, 0)) size = 1;
        auto* tld = getThreadData();
        // small path - now covers up to 512 bytes (60%+ of allocations)
        if (__builtin_expect(size <= MAX_SMALL_SIZE, 1)) {
            return tld->smallCache.allocateSmall(size, stats_);
        }
        // large path
        return tld->arena->allocate(size, alignof(std::max_align_t), stats_);
    }

    __attribute__((always_inline, hot))
    void deallocate(void* ptr) {
        if (__builtin_expect(!ptr, 0)) return;
        auto* tld = getThreadData();

        // Fast path: Check for small block first (most common case ~60%)
        // Small blocks have SMALL_MAGIC at ptr - sizeof(SmallBlockHeader)
        char* headerStart = (char*)ptr - sizeof(SmallBlockHeader);
        auto* smallHdr = reinterpret_cast<SmallBlockHeader*>(headerStart);
        if (__builtin_expect(smallHdr->magic == SMALL_MAGIC, 1)) {
            // Validate binIndex to avoid false positives from alignment padding
            if (__builtin_expect(smallHdr->binIndex < SMALL_BIN_COUNT, 1)) {
                tld->smallCache.freeSmall(ptr, stats_);
                return;
            }
        }

        // Large/aligned allocation: check arena ownership
        if (tld->arena->ownsPointer(ptr)) {
            tld->arena->deallocate(ptr, stats_);
            return;
        }

        // Cross-thread deallocation: find the owning arena
        deallocateCrossThread(ptr);
    }

    // Aligned allocation (for SIMD: 16/32/64-byte alignment)
    __attribute__((always_inline, hot))
    void* allocateAligned(size_t size, size_t alignment) {
        if (__builtin_expect(size == 0, 0)) size = 1;
        if (!isPowerOfTwo(alignment) || alignment > MAX_ALIGNMENT) {
            return nullptr;  // Invalid alignment
        }

        // For alignments <= 16, use regular allocate (already 16-byte aligned)
        if (alignment <= 16 && size <= MAX_SMALL_SIZE) {
            return allocate(size);
        }

        // For larger alignments or sizes, use arena with explicit alignment
        auto* tld = getThreadData();
        return tld->arena->allocate(size, alignment, stats_,
                                    static_cast<uint16_t>(instanceSlot_));
    }

    // Reallocate: grow/shrink allocation, preserving data
    void* reallocate(void* ptr, size_t newSize) {
        if (!ptr) return allocate(newSize);
        if (newSize == 0) {
            deallocate(ptr);
            return nullptr;
        }

        // Get current allocation size
        size_t oldSize = getAllocSize(ptr);
        if (oldSize == 0) {
            // Unknown allocation, can't reallocate
            return nullptr;
        }

        // If shrinking significantly or growing, allocate new block
        if (newSize <= oldSize && newSize >= oldSize / 2) {
            // No need to reallocate for minor shrinkage
            return ptr;
        }

        // Allocate new block
        void* newPtr = allocate(newSize);
        if (!newPtr) return nullptr;

        // Copy data
        size_t copySize = (newSize < oldSize) ? newSize : oldSize;
        std::memcpy(newPtr, ptr, copySize);

        // Free old block
        deallocate(ptr);
        return newPtr;
    }

    // Calloc: allocate and zero-initialize
    void* callocate(size_t nmemb, size_t size) {
        // Check for overflow
        size_t totalSize = nmemb * size;
        if (size != 0 && totalSize / size != nmemb) {
            return nullptr;  // Overflow
        }

        void* ptr = allocate(totalSize);
        if (ptr) {
            std::memset(ptr, 0, totalSize);
        }
        return ptr;
    }

    // Get allocation size (for realloc)
    size_t getAllocSize(void* ptr) const {
        if (!ptr) return 0;

        // Check if small allocation (with binIndex validation)
        char* headerStart = (char*)ptr - sizeof(SmallBlockHeader);
        auto* smallHdr = reinterpret_cast<SmallBlockHeader*>(headerStart);
        if (smallHdr->magic == SMALL_MAGIC && smallHdr->binIndex < SMALL_BIN_COUNT) {
            return smallHdr->userSize;
        }

        // Check all arenas for large allocation
        auto arenas = manager_->getArenas();
        for (auto* arena : arenas) {
            if (arena->ownsPointer(ptr)) {
                return arena->getAllocSize(ptr);
            }
        }
        return 0;
    }

private:
    // Cross-thread deallocation handler
    void deallocateCrossThread(void* ptr) {
        // Find the arena that owns this pointer
        auto arenas = manager_->getArenas();
        for (auto* arena : arenas) {
            if (arena->ownsPointer(ptr)) {
                arena->deallocate(ptr, stats_);
                return;
            }
        }
        // Pointer not found in any arena - corrupted or already freed
#ifdef FANCY_DEBUG_VERBOSE
        fprintf(stderr, "[FANCY] Cross-thread dealloc: pointer %p not found in any arena\n", ptr);
#endif
    }

private:
    // Fast TLD lookup using allocator ID + small array (common case)
    // Falls back to map for rare case of many allocators
    static constexpr size_t TLD_FAST_SLOTS = 8;

    struct TLDCache {
        const FancyPerThreadAllocator* keys[TLD_FAST_SLOTS] = {};
        ThreadLocalData* values[TLD_FAST_SLOTS] = {};
        std::unordered_map<const FancyPerThreadAllocator*, ThreadLocalData*> overflow;

        // Destructor cleans up TLD when thread exits
        ~TLDCache() {
            for (size_t i = 0; i < TLD_FAST_SLOTS; i++) {
                if (values[i]) {
                    // Note: Arena is managed by GlobalArenaManager, don't delete it here
                    // Just delete the ThreadLocalData wrapper
                    delete values[i];
                    values[i] = nullptr;
                    keys[i] = nullptr;
                }
            }
            for (auto& [key, tld] : overflow) {
                delete tld;
            }
            overflow.clear();
        }
    };

    static thread_local TLDCache tldCache_;

    // Allocator instance ID for fast array lookup
    size_t instanceSlot_;
    static std::atomic<size_t> nextSlot_;

    __attribute__((always_inline, hot))
    ThreadLocalData* getThreadData() {
        // Fast path: check our dedicated slot first
        if (__builtin_expect(instanceSlot_ < TLD_FAST_SLOTS, 1)) {
            ThreadLocalData* tld = tldCache_.values[instanceSlot_];
            if (__builtin_expect(tld != nullptr && tldCache_.keys[instanceSlot_] == this, 1)) {
                return tld;
            }
        }
        return getThreadDataSlow();
    }

    __attribute__((noinline, cold))
    ThreadLocalData* getThreadDataSlow() {
        // Check fast slots first
        if (instanceSlot_ < TLD_FAST_SLOTS) {
            if (tldCache_.keys[instanceSlot_] == this) {
                return tldCache_.values[instanceSlot_];
            }
            // Slot available, claim it
            if (tldCache_.keys[instanceSlot_] == nullptr) {
                return initThreadData();
            }
        }

        // Fall back to overflow map
        auto it = tldCache_.overflow.find(this);
        if (it != tldCache_.overflow.end()) {
            return it->second;
        }
        return initThreadData();
    }

    __attribute__((noinline, cold))
    ThreadLocalData* initThreadData() {
        Arena* a = manager_->createArena(defaultArenaSize_);
        auto* tld = new ThreadLocalData{a, ThreadLocalSmallCache()};
        // Set owner slot for cross-thread deallocation tracking
        tld->smallCache.setOwnerSlot(static_cast<uint16_t>(instanceSlot_));

        if (instanceSlot_ < TLD_FAST_SLOTS) {
            tldCache_.keys[instanceSlot_] = this;
            tldCache_.values[instanceSlot_] = tld;
        } else {
            tldCache_.overflow[this] = tld;
        }
        return tld;
    }

    size_t defaultArenaSize_;
    std::shared_ptr<GlobalArenaManager> manager_;
    mutable AllocStats stats_;
};

thread_local FancyPerThreadAllocator::TLDCache FancyPerThreadAllocator::tldCache_;
std::atomic<size_t> FancyPerThreadAllocator::nextSlot_{0};

//-------------------------------------------------------
// 7) C API Shim Layer
//    Provides malloc/free/aligned_alloc/realloc/calloc
//    compatible interface for MLIR/LLVM integration
//-------------------------------------------------------

//-------------------------------------------------------
// Bootstrap allocator for LD_PRELOAD initialization
// Prevents infinite recursion: malloc -> init -> make_shared -> malloc
//-------------------------------------------------------
namespace bootstrap {
    // Simple bump allocator using mmap for bootstrap phase
    static constexpr size_t BOOTSTRAP_SIZE = 1024 * 1024;  // 1MB bootstrap heap

    struct BootstrapAllocator {
        char* heap = nullptr;
        size_t offset = 0;
        bool initialized = false;

        void* alloc(size_t size) {
            if (!heap) {
                heap = (char*)mmap(nullptr, BOOTSTRAP_SIZE, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (heap == MAP_FAILED) {
                    heap = nullptr;
                    return nullptr;
                }
            }
            // Align to 16 bytes
            size = (size + 15) & ~15ULL;
            if (offset + size > BOOTSTRAP_SIZE) return nullptr;
            void* ptr = heap + offset;
            offset += size;
            return ptr;
        }

        bool owns(void* ptr) {
            return heap && ptr >= heap && ptr < heap + BOOTSTRAP_SIZE;
        }
    };

    static BootstrapAllocator bootstrapAlloc;
    static thread_local bool inInit = false;
    static std::atomic<bool> mainAllocReady{false};
}

// Global allocator instance for C API
// Uses 64MB default arena size, with background reclamation disabled for performance
inline FancyPerThreadAllocator& getFancyGlobalAllocator() {
    static FancyPerThreadAllocator globalAllocator(64 * 1024 * 1024, false);
    bootstrap::mainAllocReady.store(true, std::memory_order_release);
    return globalAllocator;
}

// Check if we should use bootstrap allocator (during init)
inline bool useBootstrap() {
    return bootstrap::inInit || !bootstrap::mainAllocReady.load(std::memory_order_acquire);
}

// RAII guard for initialization
struct InitGuard {
    InitGuard() { bootstrap::inInit = true; }
    ~InitGuard() { bootstrap::inInit = false; }
};

// Safe initialization that prevents recursion
inline FancyPerThreadAllocator* getSafeAllocator() {
    if (bootstrap::inInit) return nullptr;  // Recursion detected
    InitGuard guard;
    return &getFancyGlobalAllocator();
}

// C-compatible API functions
extern "C" {

// Standard malloc replacement
inline void* fancy_malloc(size_t size) {
    return getFancyGlobalAllocator().allocate(size);
}

// Standard free replacement
inline void fancy_free(void* ptr) {
    getFancyGlobalAllocator().deallocate(ptr);
}

// Aligned allocation (C11 aligned_alloc compatible)
// alignment must be power of 2, size must be multiple of alignment
inline void* fancy_aligned_alloc(size_t alignment, size_t size) {
    return getFancyGlobalAllocator().allocateAligned(size, alignment);
}

// POSIX posix_memalign compatible
inline int fancy_posix_memalign(void** memptr, size_t alignment, size_t size) {
    if (!memptr) return EINVAL;
    if (!isPowerOfTwo(alignment) || alignment < sizeof(void*)) return EINVAL;

    void* ptr = getFancyGlobalAllocator().allocateAligned(size, alignment);
    if (!ptr) return ENOMEM;

    *memptr = ptr;
    return 0;
}

// Standard realloc replacement
inline void* fancy_realloc(void* ptr, size_t size) {
    return getFancyGlobalAllocator().reallocate(ptr, size);
}

// Standard calloc replacement
inline void* fancy_calloc(size_t nmemb, size_t size) {
    return getFancyGlobalAllocator().callocate(nmemb, size);
}

// Get allocation size (useful for debugging)
inline size_t fancy_malloc_usable_size(void* ptr) {
    return getFancyGlobalAllocator().getAllocSize(ptr);
}

// Check for memory leaks (returns true if leaks detected)
inline bool fancy_check_leaks() {
    return getFancyGlobalAllocator().checkLeaks().hasLeaks();
}

// Print detailed allocator statistics
inline void fancy_print_stats() {
    getFancyGlobalAllocator().printDetailedStats();
}

// Validate heap integrity (returns true if valid)
inline bool fancy_validate_heap() {
    return getFancyGlobalAllocator().isHeapValid();
}

} // extern "C"

//-------------------------------------------------------
// 8) Optional: LD_PRELOAD-compatible malloc replacement
//    Define FANCY_REPLACE_MALLOC to enable
//    Uses bootstrap allocator to avoid infinite recursion during init
//-------------------------------------------------------
#ifdef FANCY_REPLACE_MALLOC
extern "C" {
    void* malloc(size_t size) {
        // Use bootstrap during initialization to prevent recursion
        if (useBootstrap()) {
            return bootstrap::bootstrapAlloc.alloc(size);
        }
        InitGuard guard;
        return getFancyGlobalAllocator().allocate(size);
    }

    void free(void* ptr) {
        if (!ptr) return;
        // Bootstrap allocations are leaked (bump allocator, no individual free)
        if (bootstrap::bootstrapAlloc.owns(ptr)) return;
        if (useBootstrap()) return;  // Can't free during init
        InitGuard guard;
        getFancyGlobalAllocator().deallocate(ptr);
    }

    void* realloc(void* ptr, size_t size) {
        if (!ptr) return malloc(size);
        if (size == 0) { free(ptr); return nullptr; }
        // Bootstrap allocations: just allocate new
        if (bootstrap::bootstrapAlloc.owns(ptr)) {
            return malloc(size);
        }
        if (useBootstrap()) {
            return bootstrap::bootstrapAlloc.alloc(size);
        }
        InitGuard guard;
        return getFancyGlobalAllocator().reallocate(ptr, size);
    }

    void* calloc(size_t nmemb, size_t size) {
        size_t total = nmemb * size;
        if (size != 0 && total / size != nmemb) return nullptr;  // Overflow
        void* ptr = malloc(total);
        if (ptr) memset(ptr, 0, total);
        return ptr;
    }

    void* aligned_alloc(size_t alignment, size_t size) {
        if (useBootstrap()) {
            // Bootstrap bump allocator is 16-byte aligned
            // For larger alignment, over-allocate and adjust
            if (alignment <= 16) return bootstrap::bootstrapAlloc.alloc(size);
            void* ptr = bootstrap::bootstrapAlloc.alloc(size + alignment);
            if (!ptr) return nullptr;
            return alignPointer(ptr, alignment);
        }
        InitGuard guard;
        return getFancyGlobalAllocator().allocateAligned(size, alignment);
    }

    int posix_memalign(void** memptr, size_t alignment, size_t size) {
        if (!memptr) return EINVAL;
        if (!isPowerOfTwo(alignment) || alignment < sizeof(void*)) return EINVAL;
        void* ptr = aligned_alloc(alignment, size);
        if (!ptr) return ENOMEM;
        *memptr = ptr;
        return 0;
    }

    void* memalign(size_t alignment, size_t size) { return aligned_alloc(alignment, size); }
    void* valloc(size_t size) { return aligned_alloc(4096, size); }
    void* pvalloc(size_t size) {
        size = (size + 4095) & ~4095ULL;
        return aligned_alloc(4096, size);
    }
    size_t malloc_usable_size(void* ptr) {
        if (bootstrap::bootstrapAlloc.owns(ptr)) return 0;
        if (useBootstrap()) return 0;
        InitGuard guard;
        return getFancyGlobalAllocator().getAllocSize(ptr);
    }
}
#endif // FANCY_REPLACE_MALLOC

#endif // MEMORY_ALLOCATOR_H
