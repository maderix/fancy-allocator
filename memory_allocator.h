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
// 1) Stats
//-------------------------------------------------------
struct AllocStatsSnapshot {
    size_t totalAllocCalls;
    size_t totalFreeCalls;
    size_t currentUsedBytes;
    size_t peakUsedBytes;
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

struct SmallBlockHeader {
    size_t binIndex;
    size_t userSize;
};
struct SmallFreeBlock {
    SmallBlockHeader hdr;
    SmallFreeBlock* next;
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

public:
    ThreadLocalSmallCache() {
        for (int i=0; i<SMALL_BIN_COUNT; i++){
            freeList_[i] = nullptr;
            slabCurrent_[i] = nullptr;
            slabEnd_[i] = nullptr;
        }
        localAllocCalls_ = 0;
        localFreeCalls_ = 0;
        localBytesAllocated_ = 0;
        localBytesFreed_ = 0;
    }
    ~ThreadLocalSmallCache() {
        // Note: slabs are leaked for simplicity (or could track and munmap)
    }

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

    __attribute__((always_inline, hot))
    void* allocateSmall(size_t reqSize, AllocStats& stats) {
        int bin = findBin(reqSize);
        if (__builtin_expect(bin < 0, 0)) return nullptr;

        SmallFreeBlock* head = freeList_[bin];

        // Fast path: pop from free list (most common case)
        if (__builtin_expect(head != nullptr, 1)) {
            freeList_[bin] = head->next;
            head->hdr.userSize = reqSize;
            // Prefetch next block for future alloc
            if (head->next) __builtin_prefetch(head->next, 0, 3);
            // Lightweight stats (batch flush every 1024 ops)
            if (__builtin_expect((++localAllocCalls_ & 0x3FF) == 0, 0)) {
                localBytesAllocated_ += localAllocCalls_ * 32;  // Approximate
                flushStats(stats);
            }
            return reinterpret_cast<char*>(head) + sizeof(SmallBlockHeader);
        }

        // Slow path: allocate from slab
        size_t totalSz = sizeof(SmallBlockHeader) + SMALL_BIN_SIZE[bin];
        if (__builtin_expect(slabCurrent_[bin] + totalSz > slabEnd_[bin], 0)) {
            char* slab = (char*)allocatePages(SLAB_SIZE);
            if (!slab) return nullptr;
            slabCurrent_[bin] = slab;
            slabEnd_[bin] = slab + SLAB_SIZE;
        }

        char* block = slabCurrent_[bin];
        slabCurrent_[bin] += totalSz;

        auto* freeB = reinterpret_cast<SmallFreeBlock*>(block);
        freeB->hdr.binIndex = bin;
        freeB->hdr.userSize = reqSize;

        localAllocCalls_++;
        localBytesAllocated_ += totalSz;

        return block + sizeof(SmallBlockHeader);
    }

    __attribute__((always_inline, hot))
    void freeSmall(void* userPtr, AllocStats& stats) {
        if (__builtin_expect(!userPtr, 0)) return;

        char* blockStart = (char*)userPtr - sizeof(SmallBlockHeader);
        auto* fb = reinterpret_cast<SmallFreeBlock*>(blockStart);
        int bin = (int)fb->hdr.binIndex;

        // Safety check - corrupted block
        if (__builtin_expect(bin < 0 || bin >= SMALL_BIN_COUNT, 0)) return;

        // Push to free list (single memory write)
        fb->next = freeList_[bin];
        freeList_[bin] = fb;

        // Lightweight stats (batch flush every 1024 ops)
        if (__builtin_expect((++localFreeCalls_ & 0x3FF) == 0, 0)) {
            localBytesFreed_ += localFreeCalls_ * 32;  // Approximate
            flushStats(stats);
        }
    }

private:
    SmallFreeBlock* freeList_[SMALL_BIN_COUNT];
    // Slab allocator for each bin (avoids calling glibc)
    char* slabCurrent_[SMALL_BIN_COUNT];
    char* slabEnd_[SMALL_BIN_COUNT];
    // Thread-local stats to avoid atomic contention
    size_t localAllocCalls_;
    size_t localFreeCalls_;
    size_t localBytesAllocated_;
    size_t localBytesFreed_;
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

    struct BlockHeader {
        uint32_t magic;
        size_t   totalSize;
        size_t   userSize;
        bool     isFree;
    };
    struct BlockFooter {
        uint32_t magic;
        size_t   totalSize;
        bool     isFree;
    };
    struct FreeBlock {
        BlockHeader hdr;
        FreeBlock* next;
    };

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
    bool fullyFree() const { return usedBytes_ == 0; }

    void destroy() {
        // unmap
        if(memory_) {
            freePages(memory_, arenaSize_);
            memory_=nullptr;
        }
    }

    __attribute__((hot))
    void* allocate(size_t reqSize, size_t alignment, AllocStats& stats) {
        // No lock needed - each thread has its own Arena
        constexpr size_t overhead = sizeof(BlockHeader) + sizeof(BlockFooter);
        const size_t totalNeeded = reqSize + overhead;
        const int startBin = findLargeBin(totalNeeded);

        // Search bins from startBin upward for a fit
        for (int bin = startBin; bin < LARGE_BIN_COUNT; bin++) {
            FreeBlock* cur = freeBins_[bin];
            if (__builtin_expect(cur == nullptr, 0)) continue;

            FreeBlock* prev = nullptr;
            do {
                if (__builtin_expect(cur->hdr.totalSize >= totalNeeded, 1)) {
                    // Fast path: most allocations need no alignment padding
                    char* start = (char*)cur;
                    char* userArea = start + sizeof(BlockHeader);
                    size_t needed = totalNeeded;

                    // Remove from current bin (common: head of list)
                    if (__builtin_expect(prev == nullptr, 1))
                        freeBins_[bin] = cur->next;
                    else
                        prev->next = cur->next;

                    // Split if worthwhile
                    size_t leftover = cur->hdr.totalSize - needed;
                    if (leftover >= sizeof(FreeBlock) + overhead) {
                        char* leftoverAddr = start + needed;
                        auto* leftoverFB = (FreeBlock*)leftoverAddr;
                        leftoverFB->hdr.magic = MAGIC;
                        leftoverFB->hdr.totalSize = leftover;
                        leftoverFB->hdr.userSize = 0;
                        leftoverFB->hdr.isFree = true;
                        leftoverFB->next = nullptr;

                        auto* leftoverFoot = getFooter(&leftoverFB->hdr);
                        leftoverFoot->magic = MAGIC;
                        leftoverFoot->totalSize = leftover;
                        leftoverFoot->isFree = true;

                        insertIntoBin(leftoverFB);
                    } else {
                        needed = cur->hdr.totalSize;
                    }

                    // Mark allocated
                    cur->hdr.isFree = false;
                    cur->hdr.userSize = reqSize;
                    cur->hdr.totalSize = needed;

                    auto* foot = getFooter(&cur->hdr);
                    foot->magic = MAGIC;
                    foot->totalSize = needed;
                    foot->isFree = false;

                    usedBytes_ += needed;

                    // Batch stats every 128 ops
                    if (__builtin_expect((++localAllocCount_ & 0x7F) == 0, 0)) {
                        stats.totalAllocCalls.fetch_add(localAllocCount_, std::memory_order_relaxed);
                        stats.currentUsedBytes.fetch_add(localAllocBytes_ + needed, std::memory_order_relaxed);
                        localAllocCount_ = 0;
                        localAllocBytes_ = 0;
                    } else {
                        localAllocBytes_ += needed;
                    }

                    return userArea;
                }
                prev = cur;
                cur = cur->next;
            } while (cur);
        }
        return nullptr;
    }

    __attribute__((hot))
    void deallocate(void* userPtr, AllocStats& stats) {
        if (__builtin_expect(!userPtr, 0)) return;

        char* start = (char*)userPtr - sizeof(BlockHeader);
        auto* hdr = (BlockHeader*)start;

        // Quick validation
        if (__builtin_expect(hdr->magic != MAGIC, 0)) return;

        hdr->isFree = true;
        size_t sz = hdr->totalSize;
        usedBytes_ -= sz;

        // Batch stats every 128 ops
        if (__builtin_expect((++localFreeCount_ & 0x7F) == 0, 0)) {
            stats.totalFreeCalls.fetch_add(localFreeCount_, std::memory_order_relaxed);
            stats.currentUsedBytes.fetch_sub(localFreeBytes_ + sz, std::memory_order_relaxed);
            localFreeCount_ = 0;
            localFreeBytes_ = 0;
        } else {
            localFreeBytes_ += sz;
        }

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
    std::mutex mgrMutex_;
    std::condition_variable cv_;
};

//-------------------------------------------------------
// 6) The main facade
//-------------------------------------------------------
class FancyPerThreadAllocator {
public:
    explicit FancyPerThreadAllocator(size_t defaultArenaSize, bool enableReclamation=false)
        : defaultArenaSize_(defaultArenaSize)
    {
        manager_ = std::make_shared<GlobalArenaManager>(enableReclamation);
    }

    AllocStatsSnapshot getStatsSnapshot() const {
        return stats_.snapshot();
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
        // Check magic at offset -4 to distinguish small vs large
        // Small blocks have binIndex at offset -16, large have MAGIC at -4
        uint32_t mg = *reinterpret_cast<uint32_t*>((char*)ptr - 4);
        if (__builtin_expect(mg == Arena::MAGIC, 0)) {
            tld->arena->deallocate(ptr, stats_);
        } else {
            tld->smallCache.freeSmall(ptr, stats_);
        }
    }

private:
    static thread_local ThreadLocalData* tld_;

    __attribute__((always_inline, hot))
    ThreadLocalData* getThreadData() {
        ThreadLocalData* t = tld_;
        if (__builtin_expect(t != nullptr, 1)) {
            return t;
        }
        return initThreadData();
    }

    __attribute__((noinline, cold))
    ThreadLocalData* initThreadData() {
        Arena* a = manager_->createArena(defaultArenaSize_);
        tld_ = new ThreadLocalData{a, ThreadLocalSmallCache()};
        return tld_;
    }

    size_t defaultArenaSize_;
    std::shared_ptr<GlobalArenaManager> manager_;
    mutable AllocStats stats_;
};

thread_local ThreadLocalData* FancyPerThreadAllocator::tld_ = nullptr;

#endif // MEMORY_ALLOCATOR_H
