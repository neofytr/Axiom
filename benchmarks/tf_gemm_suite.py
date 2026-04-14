"""tf_gemm_suite.py — TF sidecar for bench_gemm_suite.c.

matches the C bench shape-for-shape. emits RESULT lines in the same format
so summarize.py can pair them. uses tf.function + XLA off to match what a
typical TF user would get out of keras / oneDNN."""

import os, sys, time, math
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

SUITE = "gemm"


def iters_for(M, N, K):
    flops = 2.0 * M * N * K
    it = int(3e10 / flops)
    return max(3, min(it, 500))


def time_op(fn, iters, warmup=2):
    for _ in range(warmup):
        fn().numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    out.numpy()
    return (time.perf_counter() - t0) / iters * 1000.0


def emit(case, metric, value):
    print(f"RESULT {SUITE} {case} {metric} {value:.6f}", flush=True)


def run_nn(M, N, K):
    A = tf.random.normal([M, K])
    B = tf.random.normal([K, N])
    @tf.function
    def step(): return tf.matmul(A, B)
    lat_ms = time_op(step, iters_for(M, N, K))
    emit(f"nn_{M}x{N}x{K}", "lat_ms", lat_ms)
    emit(f"nn_{M}x{N}x{K}", "gflops", 2.0 * M * N * K / lat_ms / 1e6)


def run_nt(M, N, K):
    A = tf.random.normal([M, K])
    B = tf.random.normal([N, K])
    @tf.function
    def step(): return tf.matmul(A, B, transpose_b=True)
    lat_ms = time_op(step, iters_for(M, N, K))
    emit(f"nt_{M}x{N}x{K}", "lat_ms", lat_ms)
    emit(f"nt_{M}x{N}x{K}", "gflops", 2.0 * M * N * K / lat_ms / 1e6)


def run_tn(M, N, K):
    A = tf.random.normal([K, M])
    B = tf.random.normal([K, N])
    @tf.function
    def step(): return tf.matmul(A, B, transpose_a=True)
    lat_ms = time_op(step, iters_for(M, N, K))
    emit(f"tn_{M}x{N}x{K}", "lat_ms", lat_ms)
    emit(f"tn_{M}x{N}x{K}", "gflops", 2.0 * M * N * K / lat_ms / 1e6)


# shapes MUST match bench_gemm_suite.c
SQUARES = [(64,64,64), (128,128,128), (256,256,256), (512,512,512),
           (1024,1024,1024), (2048,2048,2048), (4096,4096,4096)]
SKINNY_M = [(32,1024,1024), (64,2048,2048), (128,4096,4096), (256,4096,1024)]
SKINNY_N = [(1024,10,1024), (2048,32,2048), (4096,128,4096)]
TALL_K = [(128,128,2048), (256,256,4096)]
MLP = [(256,4096,784), (256,2048,4096), (256,1024,2048),
       (256,512,1024), (256,256,512)]
TRANS = [(1024,1024,1024), (512,2048,512), (2048,512,512)]


def main():
    for s in SQUARES + SKINNY_M + SKINNY_N + TALL_K + MLP:
        run_nn(*s)
    for s in TRANS:
        run_nt(*s)
        run_tn(*s)


if __name__ == "__main__":
    main()
