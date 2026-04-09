/* test_error.c — tests for error handling module */

#include "test.h"
#include "axiom/error.h"

/* test setting and retrieving errors */
static void test_error_set_get(void) {
    ax_err_clear();
    AX_TEST_ASSERT_EQ(ax_err_last_status(), AX_OK, "initial status should be ok");

    ax_err_set(AX_ERR_NULL_ARG, "test error: %s", "null pointer");
    AX_TEST_ASSERT_EQ(ax_err_last_status(), AX_ERR_NULL_ARG, "status should be null_arg");
    AX_TEST_ASSERT(strstr(ax_err_last_message(), "null pointer") != NULL,
                   "message should contain 'null pointer'");
}

/* test clearing errors */
static void test_error_clear(void) {
    ax_err_set(AX_ERR_ALLOC, "out of memory");
    ax_err_clear();
    AX_TEST_ASSERT_EQ(ax_err_last_status(), AX_OK, "status should be ok after clear");
    AX_TEST_ASSERT_EQ(strlen(ax_err_last_message()), 0, "message should be empty after clear");
}

/* test error callback */
static ax_status_t last_cb_status = AX_OK;
static char last_cb_msg[256] = "";

static void test_callback(ax_status_t status, const char *msg, void *userdata) {
    (void)userdata;
    last_cb_status = status;
    strncpy(last_cb_msg, msg, sizeof(last_cb_msg) - 1);
}

static void test_error_callback(void) {
    ax_err_set_callback(test_callback, NULL);

    ax_err_set(AX_ERR_BACKEND, "backend failure");
    AX_TEST_ASSERT_EQ(last_cb_status, AX_ERR_BACKEND, "callback should receive status");
    AX_TEST_ASSERT(strstr(last_cb_msg, "backend failure") != NULL,
                   "callback should receive message");

    /* clean up */
    ax_err_set_callback(NULL, NULL);
    ax_err_clear();
}

/* test status name lookup */
static void test_status_names(void) {
    AX_TEST_ASSERT(strcmp(ax_status_name(AX_OK), "ok") == 0, "AX_OK name");
    AX_TEST_ASSERT(strcmp(ax_status_name(AX_ERR_ALLOC), "allocation failed") == 0, "AX_ERR_ALLOC name");
    AX_TEST_ASSERT(strcmp(ax_status_name(AX_STATUS_COUNT), "unknown error") == 0, "out of range name");
}

int main(void) {
    printf("=== error tests ===\n");
    AX_RUN_TEST(test_error_set_get);
    AX_RUN_TEST(test_error_clear);
    AX_RUN_TEST(test_error_callback);
    AX_RUN_TEST(test_status_names);
    AX_TEST_SUMMARY();
}
