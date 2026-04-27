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
typedef struct ax_arena {
    ax_arena_block_t *head;       /* current block (allocations come from here) */
    ax_arena_block_t *first;      /* first block in chain (for iteration/reset) */
    size_t block_size;            /* default size for new blocks */
    size_t total_allocated;       /* total bytes allocated across all blocks */
} ax_arena_t;

/* create a new arena with the given default block size (0 = use default).
   ownership: caller owns the returned arena and must release it via
   ax_arena_destroy.
   returns NULL on allocation failure (ax_err_last_message describes why).
   thread-safety: not concurrent-safe — one thread per arena. */
ax_arena_t *ax_arena_create(size_t block_size);

/* allocate `size` bytes from the arena, aligned to `alignment` (must be
   a power of two). returns NULL if a fresh block cannot be allocated.
   ownership: pointer is owned by the arena; freed by ax_arena_reset or
   ax_arena_destroy. do not free individually.
   thread-safety: not concurrent-safe on the same arena. */
void *ax_arena_alloc(ax_arena_t *arena, size_t size, size_t alignment);

/* convenience: allocate `size` bytes aligned to AX_DEFAULT_ALIGNMENT (64).
   same ownership and thread-safety contract as ax_arena_alloc. */
static inline void *ax_arena_alloc_default(ax_arena_t *arena, size_t size) {
    return ax_arena_alloc(arena, size, AX_DEFAULT_ALIGNMENT);
}

/* reset the arena — invalidates all prior allocations but retains the
   underlying blocks for reuse. subsequent ax_arena_alloc calls bump
   from the start of the existing chain.
   thread-safety: not concurrent-safe on the same arena. */
void ax_arena_reset(ax_arena_t *arena);

/* destroy the arena — frees every block. arena pointer must not be
   dereferenced after this call.
   thread-safety: not concurrent-safe on the same arena. */
void ax_arena_destroy(ax_arena_t *arena);

/* aligned malloc/free wrappers for persistent data (tensor storage,
   long-lived buffers). general-purpose, paired calls. */

/* allocate `size` bytes aligned to `alignment` (power-of-two).
   ownership: caller owns the returned pointer and must release it via
   ax_aligned_free.
   returns NULL on allocation failure.
   thread-safety: process-wide allocator; concurrent-safe. */
void *ax_aligned_alloc(size_t size, size_t alignment);

/* free a pointer previously returned by ax_aligned_alloc. NULL is a no-op.
   thread-safety: concurrent-safe (each pointer freed by one thread). */
void ax_aligned_free(void *ptr);

/* allocation counting (compile with -DAX_COUNT_ALLOCS to enable).
   tracks every heap allocation: ax_aligned_alloc, slab misses, arena
   block growth, pool node creation. */
#ifdef AX_COUNT_ALLOCS
#include <stdint.h>
uint64_t ax_get_alloc_count(void);
void ax_reset_alloc_count(void);
void ax_alloc_count_inc(void);
#else
static inline void ax_alloc_count_inc(void) { (void)0; }
#endif

/* configurable arena block size. default 16 MB (desktop). call before
   ax_init to right-size for embedded targets. affects arenas created
   with ax_arena_create(0). */
void ax_set_arena_block_size(size_t block_size);
size_t ax_get_arena_block_size(void);

#endif /* AX_MEMORY_H */
