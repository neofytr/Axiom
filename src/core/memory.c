/* memory.c — arena allocator and aligned allocation */

#include "axiom/memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* helpers */

/* check if value is a power of two (required for alignment) */
static inline bool is_power_of_two(size_t v) {
    return v > 0 && (v & (v - 1)) == 0;
}

/* align a value up to the nearest multiple of alignment.
   alignment must be a power of two. */
static inline size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

/* allocate a new arena block with at least min_size usable bytes */
static ax_arena_block_t *arena_block_create(size_t min_size) {
    /* check for overflow in alloc_size computation */
    if (min_size > SIZE_MAX - sizeof(ax_arena_block_t)) return NULL;
    size_t alloc_size = sizeof(ax_arena_block_t) + min_size;
    ax_arena_block_t *block = (ax_arena_block_t *)malloc(alloc_size);
    if (!block) return NULL;

    block->next = NULL;
    block->size = min_size;
    block->used = 0;
    return block;
}

/* arena allocator */

ax_arena_t *ax_arena_create(size_t block_size) {
    ax_arena_t *arena = (ax_arena_t *)malloc(sizeof(ax_arena_t));
    if (!arena) return NULL;

    if (block_size == 0) block_size = AX_ARENA_DEFAULT_BLOCK_SIZE;

    ax_arena_block_t *first = arena_block_create(block_size);
    if (!first) {
        free(arena);
        return NULL;
    }

    arena->head = first;
    arena->first = first;
    arena->block_size = block_size;
    arena->total_allocated = block_size;
    return arena;
}

void *ax_arena_alloc(ax_arena_t *arena, size_t size, size_t alignment) {
    if (!arena || size == 0) return NULL;
    if (alignment == 0) alignment = 1;
    if (!is_power_of_two(alignment)) return NULL;

    /* try the current block first; on overflow, walk the chain looking for an
       existing block with room. only allocate a new block if every block in the
       chain is exhausted. without this, ax_arena_reset() leaves older blocks
       unused and each subsequent pass grows the arena unboundedly. */
    ax_arena_block_t *blk = arena->head;
    while (blk) {
        uintptr_t base = (uintptr_t)blk->data;
        uintptr_t current = base + blk->used;
        uintptr_t aligned = (current + alignment - 1) & ~(alignment - 1);
        size_t aligned_offset = (size_t)(aligned - base);
        if (aligned_offset + size <= blk->size) {
            blk->used = aligned_offset + size;
            arena->head = blk; /* remember which block we're using */
            return (void *)aligned;
        }
        blk = blk->next;
    }

    /* every existing block is full — append a fresh block big enough for this
       request. chain grows in insertion order: first -> ... -> tail. */
    size_t new_size = arena->block_size;
    if (size > SIZE_MAX - alignment) return NULL;
    size_t needed = size + alignment;
    if (needed > new_size) new_size = needed;

    ax_arena_block_t *block = arena_block_create(new_size);
    if (!block) return NULL;

    /* append at the end of the chain (walk from first) */
    if (!arena->first) {
        arena->first = block;
    } else {
        ax_arena_block_t *tail = arena->first;
        while (tail->next) tail = tail->next;
        tail->next = block;
    }
    arena->head = block;
    arena->total_allocated += new_size;

    /* allocate from the fresh block with proper alignment */
    uintptr_t base = (uintptr_t)block->data;
    uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
    block->used = (size_t)(aligned - base) + size;
    return (void *)aligned;
}

void ax_arena_reset(ax_arena_t *arena) {
    if (!arena) return;

    /* reset used counter on every block, then rewind the active head to first.
       walks the whole chain so the next alloc can start at the oldest block. */
    ax_arena_block_t *block = arena->first;
    while (block) {
        block->used = 0;
        block = block->next;
    }
    arena->head = arena->first;
}

void ax_arena_destroy(ax_arena_t *arena) {
    if (!arena) return;

    /* free all blocks in the chain (walk from first — head may point mid-chain) */
    ax_arena_block_t *block = arena->first;
    while (block) {
        ax_arena_block_t *next = block->next;
        free(block);
        block = next;
    }
    free(arena);
}

/* aligned alloc wrappers */

#ifdef AX_COUNT_ALLOCS
#include <stdatomic.h>
static _Atomic uint64_t g_alloc_count = 0;
uint64_t ax_get_alloc_count(void) { return atomic_load(&g_alloc_count); }
void ax_reset_alloc_count(void) { atomic_store(&g_alloc_count, 0); }
#endif

void *ax_aligned_alloc(size_t size, size_t alignment) {
    if (size == 0) return NULL;
#ifdef AX_COUNT_ALLOCS
    atomic_fetch_add(&g_alloc_count, 1);
#endif
    if (alignment == 0) alignment = 1;
    /* round up to next power of two if not already */
    if (!is_power_of_two(alignment)) {
        size_t p = 1;
        while (p < alignment) p <<= 1;
        alignment = p;
    }

    /* allocate with extra space for alignment and storing the original pointer.
       check for overflow before computing total. */
    if (size > SIZE_MAX - alignment - sizeof(void *)) return NULL;
    size_t total = size + alignment + sizeof(void *);
    void *raw = malloc(total);
    if (!raw) return NULL;

    /* align the usable region, leaving room to store the raw pointer before it */
    uintptr_t raw_addr = (uintptr_t)raw + sizeof(void *);
    uintptr_t aligned_addr = (raw_addr + alignment - 1) & ~(alignment - 1);

    /* store the original pointer just before the aligned region */
    ((void **)aligned_addr)[-1] = raw;
    return (void *)aligned_addr;
}

void ax_aligned_free(void *ptr) {
    if (!ptr) return;

    /* retrieve the original malloc'd pointer stored before the aligned address */
    void *raw = ((void **)ptr)[-1];
    free(raw);
}
