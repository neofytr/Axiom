/* error.c — thread-local error state and reporting */

#include "axiom/error.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

/* per-thread error state */
typedef struct {
    ax_status_t status;
    char message[AX_ERR_MSG_MAX];
} ax_err_state_t;

/* thread-local storage for error state */
static _Thread_local ax_err_state_t tl_err = { AX_OK, "" };

/* global callback — set once, read from any thread */
static ax_err_callback_t g_callback = NULL;
static void *g_callback_userdata = NULL;

void ax_err_set(ax_status_t status, const char *fmt, ...) {
    tl_err.status = status;

    /* format the error message */
    va_list args;
    va_start(args, fmt);
    vsnprintf(tl_err.message, AX_ERR_MSG_MAX, fmt, args);
    va_end(args);

    /* notify callback if registered */
    if (g_callback) {
        g_callback(status, tl_err.message, g_callback_userdata);
    }
}

void ax_err_clear(void) {
    tl_err.status = AX_OK;
    tl_err.message[0] = '\0';
}

ax_status_t ax_err_last_status(void) {
    return tl_err.status;
}

const char *ax_err_last_message(void) {
    return tl_err.message;
}

void ax_err_set_callback(ax_err_callback_t cb, void *userdata) {
    g_callback = cb;
    g_callback_userdata = userdata;
}
