/* bench_common.h — shared helpers for all axiom benchmarks.

   every benchmark emits lines of the form:

     RESULT <suite> <case> <metric> <value>

   where metric is one of: lat_ms, gflops, gbs, acc_pct, max_err, sec.
   summarize.py joins these against the TF sidecar output to produce a
   unified comparison table. keep the format grep-friendly. */

#ifndef AX_BENCH_COMMON_H
#define AX_BENCH_COMMON_H

#include <stdio.h>
#include <time.h>
#include <string.h>

static inline double bench_now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static inline double bench_now_ms(void) { return bench_now_s() * 1000.0; }

#define BENCH_EMIT(suite, case_, metric, value) \
    printf("RESULT %s %s %s %.6f\n", (suite), (case_), (metric), (double)(value))

#define BENCH_EMIT_FLOPS(suite, case_, lat_ms, total_flops) do { \
    double _lat = (lat_ms); \
    double _flops = (total_flops); \
    BENCH_EMIT((suite), (case_), "lat_ms", _lat); \
    BENCH_EMIT((suite), (case_), "gflops", _flops / (_lat * 1e6)); \
} while (0)

#define BENCH_EMIT_BW(suite, case_, lat_ms, total_bytes) do { \
    double _lat = (lat_ms); \
    double _b = (total_bytes); \
    BENCH_EMIT((suite), (case_), "lat_ms", _lat); \
    BENCH_EMIT((suite), (case_), "gbs", _b / (_lat * 1e6)); \
} while (0)

/* run `fn(iters)` after a warmup, returning mean latency in ms. */
#define BENCH_TIME_MS(warmup, iters, expr) ({ \
    for (int _w = 0; _w < (warmup); _w++) { (void)(expr); } \
    double _t0 = bench_now_ms(); \
    for (int _i = 0; _i < (iters); _i++) { (void)(expr); } \
    (bench_now_ms() - _t0) / (iters); \
})

#endif /* AX_BENCH_COMMON_H */
