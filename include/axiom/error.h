/* axiom/error.h — thread-safe error handling */

#ifndef AX_ERROR_H
#define AX_ERROR_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* max length for error messages */
#define AX_ERR_MSG_MAX 256

/* set the current thread's error state with a formatted message */
void ax_err_set(ax_status_t status, const char *fmt, ...);

/* clear the current thread's error state */
void ax_err_clear(void);

/* get the last error status for this thread */
ax_status_t ax_err_last_status(void);

/* get the last error message for this thread; returns "" if no error */
const char *ax_err_last_message(void);

/* optional user callback for errors — called on every ax_err_set */
typedef void (*ax_err_callback_t)(ax_status_t status, const char *msg, void *userdata);
void ax_err_set_callback(ax_err_callback_t cb, void *userdata);

/* convenience macro: return status if expression fails */
#define AX_CHECK(expr) \
    do { \
        ax_status_t _s = (expr); \
        if (_s != AX_OK) return _s; \
    } while (0)

/* convenience macro: set error and return if pointer is null */
#define AX_CHECK_NULL(ptr, msg) \
    do { \
        if (!(ptr)) { \
            ax_err_set(AX_ERR_NULL_ARG, msg); \
            return AX_ERR_NULL_ARG; \
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
