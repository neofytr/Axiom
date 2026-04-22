/* axiom/error.h — thread-safe error handling */

#ifndef AX_ERROR_H
#define AX_ERROR_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* max length for error messages */
#define AX_ERR_MSG_MAX 256

/* set the current thread's error state with a printf-style formatted message.
   message is truncated to AX_ERR_MSG_MAX bytes (including NUL).
   thread-safety: per-thread state, safe to call from any thread.
   ownership: fmt is borrowed; the formatted message is copied into TLS. */
void ax_err_set(ax_status_t status, const char *fmt, ...);

/* clear the current thread's error state (status -> AX_OK, message -> "").
   thread-safety: per-thread state, safe to call from any thread. */
void ax_err_clear(void);

/* return the last error status set by this thread (AX_OK if none).
   thread-safety: per-thread state, safe to call from any thread. */
ax_status_t ax_err_last_status(void);

/* return the last error message set by this thread; "" if no error.
   ownership: pointer is owned by axiom (TLS storage); valid until the
   next ax_err_set / ax_err_clear call on the same thread.
   thread-safety: per-thread state, safe to call from any thread. */
const char *ax_err_last_message(void);

/* optional user callback for errors — called on every ax_err_set.
   the cb is invoked synchronously on the thread that triggered the error.
   pass NULL to disable.
   thread-safety: process-wide setting; cb itself is invoked on the
   triggering thread and must therefore be reentrant if multi-threaded. */
typedef void (*ax_err_callback_t)(ax_status_t status, const char *msg, void *userdata);
void ax_err_set_callback(ax_err_callback_t cb, void *userdata);

/* convenience macro: return status if expression fails */
#define AX_CHECK(expr) \
    do { \
        ax_status_t _s = (expr); \
        if (_s != AX_OK) return _s; \
    } while (0)

/* convenience macro: set error and return AX_ERR_NULL_ARG if pointer is null.
   for use in functions whose return type is ax_status_t — typical signature
   `ax_status_t foo(... in arg ...)` validating a non-null requirement. */
#define AX_CHECK_NULL(ptr, msg) \
    do { \
        if (!(ptr)) { \
            ax_err_set(AX_ERR_NULL_ARG, msg); \
            return AX_ERR_NULL_ARG; \
        } \
    } while (0)

/* convenience macro: set AX_ERR_ALLOC and return NULL if pointer is null.
   for use in pointer-returning constructors (ax_*_create) right after a
   malloc / calloc / aligned_alloc. eliminates the silent-NULL anti-pattern
   where allocation failure returned NULL without telling the caller why.
   `where` is a short literal naming the function for the error message. */
#define AX_RETURN_NULL_IF_ALLOC_FAIL(ptr, where) \
    do { \
        if (!(ptr)) { \
            ax_err_set(AX_ERR_ALLOC, where ": out of memory"); \
            return NULL; \
        } \
    } while (0)

/* debug bounds assertion. active only in debug builds (NDEBUG not defined).
   use this when indexing into storage->data to catch out-of-bounds access
   during development. compiles to nothing in release builds. */
#ifndef NDEBUG
#include <stdio.h>
#include <stdlib.h>
#define AX_BOUNDS_CHECK(tensor, byte_offset) \
    do { \
        if ((tensor) && (tensor)->storage && \
            (size_t)(byte_offset) >= (tensor)->storage->size_bytes) { \
            fprintf(stderr, "axiom: bounds violation at %s:%d " \
                    "(offset %zu >= %zu bytes)\n", \
                    __FILE__, __LINE__, \
                    (size_t)(byte_offset), (tensor)->storage->size_bytes); \
            abort(); \
        } \
    } while (0)
#else
#define AX_BOUNDS_CHECK(tensor, byte_offset) ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* AX_ERROR_H */
