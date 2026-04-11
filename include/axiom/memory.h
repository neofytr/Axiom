/* axiom/memory.h — memory management primitives */

#ifndef AX_MEMORY_H
#define AX_MEMORY_H

#include "types.h"

/* default arena block size: 1 MB */
#define AX_ARENA_DEFAULT_BLOCK_SIZE (1024 * 1024)

/* default alignment for simd readiness */
#define AX_DEFAULT_ALIGNMENT 64

/* arena allocator */
/* bulk allocator for temporary buffers during forward/backward passes.
   allocations are fast (bump pointer), and freed all at once. */

/* single block in the arena's linked list */
typedef struct ax_arena_block {
    struct ax_arena_block *next;
    size_t size;         /* total usable bytes in this block */
    size_t used;         /* bytes consumed so far */
    uint8_t data[];      /* flexible array member */
} ax_arena_block_t;

/* arena: a chain of blocks with bump allocation */
typedef struct {
    ax_arena_block_t *head;       /* current block (allocations come from here) */
    ax_arena_block_t *first;      /* first block in chain (for iteration/reset) */
    size_t block_size;            /* default size for new blocks */
    size_t total_allocated;       /* total bytes allocated across all blocks */
} ax_arena_t;

/* create a new arena with the given default block size (0 = use default) */
ax_arena_t *ax_arena_create(size_t block_size);

/* allocate n bytes from the arena, aligned to the given alignment */
void *ax_arena_alloc(ax_arena_t *arena, size_t size, size_t alignment);

/* convenience: allocate with default alignment */
static inline void *ax_arena_alloc_default(ax_arena_t *arena, size_t size) {
    return ax_arena_alloc(arena, size, AX_DEFAULT_ALIGNMENT);
}

/* reset the arena — marks all memory as free without releasing it.
   subsequent allocations reuse existing blocks. */
void ax_arena_reset(ax_arena_t *arena);

/* destroy the arena — frees all blocks */
void ax_arena_destroy(ax_arena_t *arena);

/* aligned malloc/free wrappers */
/* general-purpose aligned allocation for persistent data */

void *ax_aligned_alloc(size_t size, size_t alignment);
void ax_aligned_free(void *ptr);

/* allocation counting (compile with -DAX_COUNT_ALLOCS to enable) */
#ifdef AX_COUNT_ALLOCS
#include <stdint.h>
uint64_t ax_get_alloc_count(void);
void ax_reset_alloc_count(void);
#endif

#endif /* AX_MEMORY_H */
