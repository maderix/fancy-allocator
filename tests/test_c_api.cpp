// Comprehensive tests for FancyAllocator C API
// Tests: malloc/free, aligned_alloc, realloc, calloc, cross-thread deallocation

#include "../memory_allocator.h"
#include <cassert>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>

// Test counters
static std::atomic<int> tests_passed{0};
static std::atomic<int> tests_failed{0};

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    try { \
        test_##name(); \
        tests_passed++; \
        std::cout << "PASSED\n"; \
    } catch (const std::exception& e) { \
        tests_failed++; \
        std::cout << "FAILED: " << e.what() << "\n"; \
    } catch (...) { \
        tests_failed++; \
        std::cout << "FAILED: unknown exception\n"; \
    } \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        throw std::runtime_error("Assertion failed: " #cond); \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        throw std::runtime_error("Assertion failed: " #a " == " #b); \
    } \
} while(0)

// =============================================================================
// Basic malloc/free tests
// =============================================================================

TEST(basic_malloc_free) {
    void* ptr = fancy_malloc(100);
    ASSERT(ptr != nullptr);

    // Write some data
    memset(ptr, 0xAB, 100);

    fancy_free(ptr);
}

TEST(malloc_zero_size) {
    // malloc(0) should return a valid pointer or nullptr
    void* ptr = fancy_malloc(0);
    // Either behavior is acceptable
    if (ptr) {
        fancy_free(ptr);
    }
}

TEST(free_null) {
    // free(nullptr) should be a no-op
    fancy_free(nullptr);
}

TEST(malloc_small_sizes) {
    // Test all small size bins
    size_t sizes[] = {1, 8, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512};

    for (size_t size : sizes) {
        void* ptr = fancy_malloc(size);
        ASSERT(ptr != nullptr);
        memset(ptr, 0xCD, size);
        fancy_free(ptr);
    }
}

TEST(malloc_large_sizes) {
    // Test large allocations
    size_t sizes[] = {1024, 2048, 4096, 8192, 16384, 65536, 131072, 1048576};

    for (size_t size : sizes) {
        void* ptr = fancy_malloc(size);
        ASSERT(ptr != nullptr);
        memset(ptr, 0xEF, size);
        fancy_free(ptr);
    }
}

TEST(malloc_stress) {
    std::vector<void*> ptrs;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> size_dist(1, 10000);

    // Allocate many blocks
    for (int i = 0; i < 1000; i++) {
        size_t size = size_dist(rng);
        void* ptr = fancy_malloc(size);
        ASSERT(ptr != nullptr);
        memset(ptr, 0xFF, size);
        ptrs.push_back(ptr);
    }

    // Free in random order
    std::shuffle(ptrs.begin(), ptrs.end(), rng);
    for (void* ptr : ptrs) {
        fancy_free(ptr);
    }
}

// =============================================================================
// Aligned allocation tests
// =============================================================================

TEST(aligned_alloc_16) {
    void* ptr = fancy_aligned_alloc(16, 100);
    ASSERT(ptr != nullptr);
    ASSERT(reinterpret_cast<uintptr_t>(ptr) % 16 == 0);
    memset(ptr, 0x11, 100);
    fancy_free(ptr);
}

TEST(aligned_alloc_32) {
    void* ptr = fancy_aligned_alloc(32, 256);
    ASSERT(ptr != nullptr);
    ASSERT(reinterpret_cast<uintptr_t>(ptr) % 32 == 0);
    memset(ptr, 0x22, 256);
    fancy_free(ptr);
}

TEST(aligned_alloc_64) {
    void* ptr = fancy_aligned_alloc(64, 1024);
    ASSERT(ptr != nullptr);
    ASSERT(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
    memset(ptr, 0x33, 1024);
    fancy_free(ptr);
}

TEST(aligned_alloc_4096) {
    // Page-aligned allocation
    void* ptr = fancy_aligned_alloc(4096, 8192);
    ASSERT(ptr != nullptr);
    ASSERT(reinterpret_cast<uintptr_t>(ptr) % 4096 == 0);
    memset(ptr, 0x44, 8192);
    fancy_free(ptr);
}

TEST(aligned_alloc_various) {
    size_t alignments[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (size_t align : alignments) {
        for (size_t size = 1; size <= 10000; size *= 3) {
            void* ptr = fancy_aligned_alloc(align, size);
            if (!ptr) {
                throw std::runtime_error("null ptr for align=" + std::to_string(align) +
                                        " size=" + std::to_string(size));
            }
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            if (addr % align != 0) {
                throw std::runtime_error("misaligned ptr=" + std::to_string(addr) +
                                        " align=" + std::to_string(align) +
                                        " size=" + std::to_string(size) +
                                        " mod=" + std::to_string(addr % align));
            }
            memset(ptr, 0x55, size);
            fancy_free(ptr);
        }
    }
}

TEST(posix_memalign_basic) {
    void* ptr = nullptr;
    int result = fancy_posix_memalign(&ptr, 64, 512);
    ASSERT_EQ(result, 0);
    ASSERT(ptr != nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
    fancy_free(ptr);
}

TEST(posix_memalign_invalid) {
    void* ptr = nullptr;

    // Invalid alignment (not power of 2)
    int result = fancy_posix_memalign(&ptr, 7, 100);
    ASSERT_EQ(result, EINVAL);

    // Invalid alignment (less than sizeof(void*))
    result = fancy_posix_memalign(&ptr, 4, 100);  // Assuming 64-bit system
    // This may or may not fail depending on sizeof(void*)

    // Null memptr
    result = fancy_posix_memalign(nullptr, 64, 100);
    ASSERT_EQ(result, EINVAL);
}

// =============================================================================
// Realloc tests
// =============================================================================

TEST(realloc_grow) {
    void* ptr = fancy_malloc(100);
    ASSERT(ptr != nullptr);
    memset(ptr, 0xAA, 100);

    ptr = fancy_realloc(ptr, 500);
    ASSERT(ptr != nullptr);

    // Verify first 100 bytes preserved
    unsigned char* data = (unsigned char*)ptr;
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(data[i], 0xAA);
    }

    fancy_free(ptr);
}

TEST(realloc_shrink) {
    void* ptr = fancy_malloc(500);
    ASSERT(ptr != nullptr);
    memset(ptr, 0xBB, 500);

    ptr = fancy_realloc(ptr, 100);
    ASSERT(ptr != nullptr);

    // Verify first 100 bytes preserved
    unsigned char* data = (unsigned char*)ptr;
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(data[i], 0xBB);
    }

    fancy_free(ptr);
}

TEST(realloc_null) {
    // realloc(nullptr, size) should behave like malloc(size)
    void* ptr = fancy_realloc(nullptr, 100);
    ASSERT(ptr != nullptr);
    memset(ptr, 0xCC, 100);
    fancy_free(ptr);
}

TEST(realloc_zero) {
    // realloc(ptr, 0) should free and return nullptr
    void* ptr = fancy_malloc(100);
    ASSERT(ptr != nullptr);

    ptr = fancy_realloc(ptr, 0);
    // ptr should be nullptr or a minimal allocation
}

TEST(realloc_stress) {
    void* ptr = fancy_malloc(10);
    ASSERT(ptr != nullptr);

    for (int i = 0; i < 100; i++) {
        size_t new_size = (i % 2 == 0) ? i * 100 + 10 : i * 50 + 5;
        ptr = fancy_realloc(ptr, new_size);
        ASSERT(ptr != nullptr);
    }

    fancy_free(ptr);
}

// =============================================================================
// Calloc tests
// =============================================================================

TEST(calloc_basic) {
    void* ptr = fancy_calloc(10, 100);
    ASSERT(ptr != nullptr);

    // Verify zero-initialized
    unsigned char* data = (unsigned char*)ptr;
    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(data[i], 0);
    }

    fancy_free(ptr);
}

TEST(calloc_various_sizes) {
    size_t counts[] = {1, 10, 100, 1000};
    size_t sizes[] = {1, 4, 8, 16, 64, 256};

    for (size_t count : counts) {
        for (size_t size : sizes) {
            void* ptr = fancy_calloc(count, size);
            ASSERT(ptr != nullptr);

            unsigned char* data = (unsigned char*)ptr;
            for (size_t i = 0; i < count * size; i++) {
                ASSERT_EQ(data[i], 0);
            }

            fancy_free(ptr);
        }
    }
}

TEST(calloc_overflow) {
    // This should return nullptr due to overflow
    void* ptr = fancy_calloc(SIZE_MAX, SIZE_MAX);
    ASSERT(ptr == nullptr);
}

// =============================================================================
// Cross-thread deallocation tests
// =============================================================================

TEST(cross_thread_dealloc_simple) {
    void* ptr = nullptr;

    // Allocate in main thread
    ptr = fancy_malloc(1000);
    ASSERT(ptr != nullptr);
    memset(ptr, 0xDD, 1000);

    // Free in another thread
    std::thread t([ptr]() {
        fancy_free(ptr);
    });
    t.join();
}

TEST(cross_thread_dealloc_large) {
    void* ptr = fancy_malloc(1024 * 1024);  // 1MB - definitely large allocation
    ASSERT(ptr != nullptr);
    memset(ptr, 0xEE, 1024 * 1024);

    std::thread t([ptr]() {
        fancy_free(ptr);
    });
    t.join();
}

TEST(cross_thread_dealloc_stress) {
    const int NUM_THREADS = 4;
    const int ALLOCS_PER_THREAD = 100;

    std::vector<void*> main_allocations;
    std::atomic<int> freed_count{0};

    // Main thread allocates
    for (int i = 0; i < NUM_THREADS * ALLOCS_PER_THREAD; i++) {
        void* ptr = fancy_malloc(100 + i % 1000);
        ASSERT(ptr != nullptr);
        main_allocations.push_back(ptr);
    }

    // Multiple threads free
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < ALLOCS_PER_THREAD; i++) {
                int idx = t * ALLOCS_PER_THREAD + i;
                fancy_free(main_allocations[idx]);
                freed_count++;
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    ASSERT_EQ(freed_count.load(), NUM_THREADS * ALLOCS_PER_THREAD);
}

TEST(bidirectional_cross_thread) {
    const int NUM_THREADS = 4;
    std::vector<std::vector<void*>> thread_allocations(NUM_THREADS);
    std::atomic<bool> ready{false};
    std::atomic<int> phase{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            // Phase 1: Each thread allocates
            for (int i = 0; i < 50; i++) {
                void* ptr = fancy_malloc(100 + i * 10);
                thread_allocations[t].push_back(ptr);
            }

            phase.fetch_add(1);
            while (phase.load() < NUM_THREADS) {
                std::this_thread::yield();
            }

            // Phase 2: Each thread frees allocations from a different thread
            int other = (t + 1) % NUM_THREADS;
            for (void* ptr : thread_allocations[other]) {
                fancy_free(ptr);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }
}

// =============================================================================
// malloc_usable_size tests
// =============================================================================

TEST(malloc_usable_size_basic) {
    void* ptr = fancy_malloc(100);
    ASSERT(ptr != nullptr);

    size_t usable = fancy_malloc_usable_size(ptr);
    ASSERT(usable >= 100);  // Should be at least what was requested

    fancy_free(ptr);
}

TEST(malloc_usable_size_various) {
    size_t sizes[] = {1, 10, 100, 1000, 10000, 100000};

    for (size_t size : sizes) {
        void* ptr = fancy_malloc(size);
        ASSERT(ptr != nullptr);

        size_t usable = fancy_malloc_usable_size(ptr);
        ASSERT(usable >= size);

        fancy_free(ptr);
    }
}

// =============================================================================
// SIMD alignment tests (for SimpleLang MLIR backend)
// =============================================================================

TEST(simd_sse_alignment) {
    // SSE requires 16-byte alignment for __m128
    void* ptr = fancy_aligned_alloc(16, sizeof(double) * 2);  // 128 bits
    ASSERT(ptr != nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) % 16, 0);

    // Simulate SSE vector operation
    double* vec = (double*)ptr;
    vec[0] = 1.0;
    vec[1] = 2.0;

    fancy_free(ptr);
}

TEST(simd_avx_alignment) {
    // AVX requires 32-byte alignment for __m256
    void* ptr = fancy_aligned_alloc(32, sizeof(double) * 4);  // 256 bits
    ASSERT(ptr != nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) % 32, 0);

    // Simulate AVX vector operation
    double* vec = (double*)ptr;
    for (int i = 0; i < 4; i++) {
        vec[i] = i * 1.5;
    }

    fancy_free(ptr);
}

TEST(simd_avx512_alignment) {
    // AVX-512 requires 64-byte alignment for __m512
    void* ptr = fancy_aligned_alloc(64, sizeof(double) * 8);  // 512 bits
    ASSERT(ptr != nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);

    // Simulate AVX-512 vector operation
    double* vec = (double*)ptr;
    for (int i = 0; i < 8; i++) {
        vec[i] = i * 2.5;
    }

    fancy_free(ptr);
}

TEST(simd_array_alignment) {
    // Allocate array of 64-byte aligned vectors (common in SIMD kernels)
    const int NUM_VECTORS = 100;
    void* ptr = fancy_aligned_alloc(64, NUM_VECTORS * 64);  // 100 x 512-bit vectors
    ASSERT(ptr != nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);

    // Verify each vector slot is 64-byte aligned
    double* data = (double*)ptr;
    for (int i = 0; i < NUM_VECTORS; i++) {
        double* vec = data + i * 8;  // 8 doubles per 512-bit vector
        ASSERT_EQ(reinterpret_cast<uintptr_t>(vec) % 64, 0);

        // Initialize
        for (int j = 0; j < 8; j++) {
            vec[j] = i * 8 + j;
        }
    }

    fancy_free(ptr);
}

// =============================================================================
// Leak detection and heap validation tests
// =============================================================================

TEST(leak_check_no_leaks) {
    // Allocate and free properly
    for (int i = 0; i < 100; i++) {
        void* ptr = fancy_malloc(100);
        fancy_free(ptr);
    }

    // Note: We can't check for leaks from within the test since
    // the test framework itself may have allocations
}

TEST(heap_validation) {
    // Perform many operations
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; i++) {
        ptrs.push_back(fancy_malloc(i * 10 + 1));
    }

    // Heap should be valid
    ASSERT(fancy_validate_heap());

    // Free half
    for (int i = 0; i < 50; i++) {
        fancy_free(ptrs[i]);
    }

    // Heap should still be valid
    ASSERT(fancy_validate_heap());

    // Free rest
    for (int i = 50; i < 100; i++) {
        fancy_free(ptrs[i]);
    }

    ASSERT(fancy_validate_heap());
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "FancyAllocator C API Test Suite\n";
    std::cout << "================================\n\n";

    // Basic malloc/free
    RUN_TEST(basic_malloc_free);
    RUN_TEST(malloc_zero_size);
    RUN_TEST(free_null);
    RUN_TEST(malloc_small_sizes);
    RUN_TEST(malloc_large_sizes);
    RUN_TEST(malloc_stress);

    // Aligned allocation
    RUN_TEST(aligned_alloc_16);
    RUN_TEST(aligned_alloc_32);
    RUN_TEST(aligned_alloc_64);
    RUN_TEST(aligned_alloc_4096);
    RUN_TEST(aligned_alloc_various);
    RUN_TEST(posix_memalign_basic);
    RUN_TEST(posix_memalign_invalid);

    // Realloc
    RUN_TEST(realloc_grow);
    RUN_TEST(realloc_shrink);
    RUN_TEST(realloc_null);
    RUN_TEST(realloc_zero);
    RUN_TEST(realloc_stress);

    // Calloc
    RUN_TEST(calloc_basic);
    RUN_TEST(calloc_various_sizes);
    RUN_TEST(calloc_overflow);

    // Cross-thread deallocation
    RUN_TEST(cross_thread_dealloc_simple);
    RUN_TEST(cross_thread_dealloc_large);
    RUN_TEST(cross_thread_dealloc_stress);
    RUN_TEST(bidirectional_cross_thread);

    // malloc_usable_size
    RUN_TEST(malloc_usable_size_basic);
    RUN_TEST(malloc_usable_size_various);

    // SIMD alignment (SimpleLang MLIR backend requirements)
    RUN_TEST(simd_sse_alignment);
    RUN_TEST(simd_avx_alignment);
    RUN_TEST(simd_avx512_alignment);
    RUN_TEST(simd_array_alignment);

    // Leak detection and heap validation
    RUN_TEST(leak_check_no_leaks);
    RUN_TEST(heap_validation);

    std::cout << "\n================================\n";
    std::cout << "Results: " << tests_passed.load() << " passed, "
              << tests_failed.load() << " failed\n";

    // Print allocator stats
    std::cout << "\nAllocator Statistics:\n";
    fancy_print_stats();

    return tests_failed.load() > 0 ? 1 : 0;
}
