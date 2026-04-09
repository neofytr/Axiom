/* memory.c — arena allocator and aligned allocation */

#include "axiom/memory.h"
#include <stdlib.h>
#include <string.h>

/* helpers */

/* align a value up to the nearest multiple of alignment */
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

    /* compute the aligned address relative to the block's data pointer.
       we need to align the actual address, not just the offset. */
    uintptr_t base = (uintptr_t)arena->head->data;
    uintptr_t current = base + arena->head->used;
    uintptr_t aligned = (current + alignment - 1) & ~(alignment - 1);
    size_t aligned_offset = (size_t)(aligned - base);

    /* check if there's room in the current block */
    if (aligned_offset + size <= arena->head->size) {
        arena->head->used = aligned_offset + size;
        return (void *)aligned;
    }

    /* need a new block — at least big enough for this allocation + alignment padding */
    size_t new_size = arena->block_size;
    if (size > SIZE_MAX - alignment) return NULL; /* overflow check */
    size_t needed = size + alignment;
    if (needed > new_size) new_size = needed;

    ax_arena_block_t *block = arena_block_create(new_size);
    if (!block) return NULL;

    /* prepend new block as the active head */
    block->next = arena->head;
    arena->head = block;
    arena->total_allocated += new_size;

    /* allocate from the fresh block with proper alignment */
    base = (uintptr_t)block->data;
    aligned = (base + alignment - 1) & ~(alignment - 1);
    block->used = (size_t)(aligned - base) + size;
    return (void *)aligned;
}

void ax_arena_reset(ax_arena_t *arena) {
    if (!arena) return;

    /* reset used counter on every block */
    ax_arena_block_t *block = arena->head;
    while (block) {
        block->used = 0;
        block = block->next;
    }
}

void ax_arena_destroy(ax_arena_t *arena) {
    if (!arena) return;

    /* free all blocks in the chain */
    ax_arena_block_t *block = arena->head;
    while (block) {
        ax_arena_block_t *next = block->next;
        free(block);
        block = next;
    }
    free(arena);
}

/* aligned alloc wrappers */

void *ax_aligned_alloc(size_t size, size_t alignment) {
    if (size == 0) return NULL;
    if (alignment == 0) alignment = 1;

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
