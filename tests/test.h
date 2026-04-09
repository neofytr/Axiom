/* test.h — minimal test harness for axiom */

#ifndef AX_TEST_H
#define AX_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* counters */
static int ax_tests_run = 0;
static int ax_tests_passed = 0;
static int ax_tests_failed = 0;

/* assert that a condition is true */
#define AX_TEST_ASSERT(cond, msg) \
    do { \
        ax_tests_run++; \
        if (!(cond)) { \
            printf("  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
            ax_tests_failed++; \
        } else { \
            ax_tests_passed++; \
        } \
    } while (0)

/* assert two ints are equal */
#define AX_TEST_ASSERT_EQ(a, b, msg) \
    do { \
        ax_tests_run++; \
        if ((a) != (b)) { \
            printf("  FAIL: %s (line %d): %s (got %ld, expected %ld)\n", \
                   __func__, __LINE__, msg, (long)(a), (long)(b)); \
            ax_tests_failed++; \
        } else { \
            ax_tests_passed++; \
        } \
    } while (0)

/* assert two floats are approximately equal */
#define AX_TEST_ASSERT_NEAR(a, b, tol, msg) \
    do { \
        ax_tests_run++; \
        if (fabsf((float)(a) - (float)(b)) > (tol)) { \
            printf("  FAIL: %s (line %d): %s (got %.6f, expected %.6f, tol %.6f)\n", \
                   __func__, __LINE__, msg, (float)(a), (float)(b), (float)(tol)); \
            ax_tests_failed++; \
        } else { \
            ax_tests_passed++; \
        } \
    } while (0)

/* run a test function and print its name */
#define AX_RUN_TEST(fn) \
    do { \
        int before = ax_tests_failed; \
        printf("[test] %s\n", #fn); \
        fn(); \
        if (ax_tests_failed == before) printf("  ok\n"); \
    } while (0)

/* print summary and return exit code */
#define AX_TEST_SUMMARY() \
    do { \
        printf("\nresults: %d passed, %d failed, %d total\n", \
               ax_tests_passed, ax_tests_failed, ax_tests_run); \
        return ax_tests_failed > 0 ? 1 : 0; \
    } while (0)

#endif /* AX_TEST_H */
