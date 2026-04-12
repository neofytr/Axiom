/* test_memory.c — tests for arena allocator and aligned allocation */

#include "test.h"
#include "axiom/memory.h"
#include <stdint.h>

/* test arena creation and basic allocation */
static void test_arena_create(void) {
    ax_arena_t *arena = ax_arena_create(0); /* use default block size */
    AX_TEST_ASSERT(arena != NULL, "arena should be created");
    AX_TEST_ASSERT(arena->head != NULL, "arena should have a block");
    AX_TEST_ASSERT(arena->total_allocated > 0, "arena should have allocated memory");
    ax_arena_destroy(arena);
}

/* test multiple allocations from an arena */
static void test_arena_alloc(void) {
    ax_arena_t *arena = ax_arena_create(4096);

    void *p1 = ax_arena_alloc(arena, 128, 16);
    void *p2 = ax_arena_alloc(arena, 256, 16);
    void *p3 = ax_arena_alloc(arena, 64, 16);

    AX_TEST_ASSERT(p1 != NULL, "first alloc should succeed");
    AX_TEST_ASSERT(p2 != NULL, "second alloc should succeed");
    AX_TEST_ASSERT(p3 != NULL, "third alloc should succeed");

    /* verify no overlap */
    AX_TEST_ASSERT(p1 != p2, "allocations should not overlap");
    AX_TEST_ASSERT(p2 != p3, "allocations should not overlap");

    /* verify alignment */
    AX_TEST_ASSERT(((uintptr_t)p1 % 16) == 0, "p1 should be 16-byte aligned");
    AX_TEST_ASSERT(((uintptr_t)p2 % 16) == 0, "p2 should be 16-byte aligned");
    AX_TEST_ASSERT(((uintptr_t)p3 % 16) == 0, "p3 should be 16-byte aligned");

    /* verify we can write to the allocations */
    memset(p1, 0xAA, 128);
    memset(p2, 0xBB, 256);
    memset(p3, 0xCC, 64);

    AX_TEST_ASSERT(((uint8_t *)p1)[0] == 0xAA, "p1 write should persist");
    AX_TEST_ASSERT(((uint8_t *)p2)[0] == 0xBB, "p2 write should persist");
    AX_TEST_ASSERT(((uint8_t *)p3)[0] == 0xCC, "p3 write should persist");

    ax_arena_destroy(arena);
}

/* test arena growing to multiple blocks */
static void test_arena_grow(void) {
    ax_arena_t *arena = ax_arena_create(256); /* small blocks to force growth */

    /* allocate more than one block worth */
    void *ptrs[20];
    for (int i = 0; i < 20; i++) {
        ptrs[i] = ax_arena_alloc(arena, 64, 16);
        AX_TEST_ASSERT(ptrs[i] != NULL, "allocation should succeed even across blocks");
    }

    AX_TEST_ASSERT(arena->total_allocated > 256, "arena should have grown beyond initial block");
    ax_arena_destroy(arena);
}

/* test arena reset reuses memory */
static void test_arena_reset(void) {
    ax_arena_t *arena = ax_arena_create(4096);

    (void)ax_arena_alloc(arena, 128, 16);
    ax_arena_reset(arena);

    /* after reset, next alloc should reuse the same block */
    void *p2 = ax_arena_alloc(arena, 128, 16);
    AX_TEST_ASSERT(p2 != NULL, "alloc after reset should succeed");

    size_t total = arena->total_allocated;
    AX_TEST_ASSERT_EQ(total, 4096, "total allocated should not grow after reset");

    ax_arena_destroy(arena);
}

/* test large allocation that exceeds block size */
static void test_arena_large_alloc(void) {
    ax_arena_t *arena = ax_arena_create(256);

    /* allocate more than block size */
    void *p = ax_arena_alloc(arena, 1024, 16);
    AX_TEST_ASSERT(p != NULL, "large allocation should succeed");
    AX_TEST_ASSERT(((uintptr_t)p % 16) == 0, "large alloc should be aligned");

    /* write to it to verify it's usable */
    memset(p, 0xFF, 1024);

    ax_arena_destroy(arena);
}

/* test aligned alloc/free */
static void test_aligned_alloc(void) {
    void *p = ax_aligned_alloc(1024, 64);
    AX_TEST_ASSERT(p != NULL, "aligned alloc should succeed");
    AX_TEST_ASSERT(((uintptr_t)p % 64) == 0, "should be 64-byte aligned");

    /* write and read back */
    memset(p, 0x42, 1024);
    AX_TEST_ASSERT(((uint8_t *)p)[0] == 0x42, "write should persist");
    AX_TEST_ASSERT(((uint8_t *)p)[1023] == 0x42, "write should persist at end");

    ax_aligned_free(p);
}

/* test aligned alloc with various alignments */
static void test_aligned_alloc_various(void) {
    size_t alignments[] = {8, 16, 32, 64, 128, 256};
    for (int i = 0; i < 6; i++) {
        void *p = ax_aligned_alloc(512, alignments[i]);
        AX_TEST_ASSERT(p != NULL, "alloc should succeed");
        AX_TEST_ASSERT(((uintptr_t)p % alignments[i]) == 0, "should respect alignment");
        ax_aligned_free(p);
    }
}

/* test null/zero edge cases */
static void test_edge_cases(void) {
    void *p = ax_aligned_alloc(0, 64);
    AX_TEST_ASSERT(p == NULL, "zero-size alloc should return null");

    ax_aligned_free(NULL); /* should not crash */

    ax_arena_destroy(NULL); /* should not crash */
    ax_arena_reset(NULL);   /* should not crash */

    void *p2 = ax_arena_alloc(NULL, 64, 16);
    AX_TEST_ASSERT(p2 == NULL, "alloc from null arena should return null");
}

int main(void) {
    printf("=== memory tests ===\n");
    AX_RUN_TEST(test_arena_create);
    AX_RUN_TEST(test_arena_alloc);
    AX_RUN_TEST(test_arena_grow);
    AX_RUN_TEST(test_arena_reset);
    AX_RUN_TEST(test_arena_large_alloc);
    AX_RUN_TEST(test_aligned_alloc);
    AX_RUN_TEST(test_aligned_alloc_various);
    AX_RUN_TEST(test_edge_cases);
    AX_TEST_SUMMARY();
}
