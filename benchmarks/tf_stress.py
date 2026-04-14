"""tf_stress.py — TF sidecar for bench_stress.c.

mirrors the same cases: huge GEMM, long-context SDPA, Adam at scale.
thread-scaling is measured differently in TF (set via tf.config), so
we only emit the default-threads numbers here and let the summary
report Axiom's scaling curve separately."""

import os, sys, time, math
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

SUITE = "stress"


def emit(case, metric, value):
    print(f"RESULT {SUITE} {case} {metric} {value:.6f}", flush=True)


def bench_huge_gemm(M):
    try:
        A = tf.random.normal([M, M])
        B = tf.random.normal([M, M])
        @tf.function
        def step(): return tf.matmul(A, B)
        step().numpy()
        iters = 1 if M >= 8192 else 2
        t0 = time.perf_counter()
        for _ in range(iters): step().numpy()
        dt = (time.perf_counter() - t0) / iters * 1000.0
        flops = 2.0 * M * M * M
        emit(f"huge_gemm_{M}", "lat_ms", dt)
        emit(f"huge_gemm_{M}", "gflops", flops / dt / 1e6)
    except Exception as e:
        print(f"# huge_gemm_{M} oom: {e}", file=sys.stderr)


def bench_long_ctx_sdpa(S, causal):
    BH = 8; dk = 64
    Q = tf.random.normal([BH, S, dk])
    K = tf.random.normal([BH, S, dk])
    V = tf.random.normal([BH, S, dk])
    scale = 1.0 / 8.0
    if causal:
        mask = tf.linalg.band_part(tf.ones([S, S]), -1, 0)
    @tf.function
    def step():
        scores = tf.matmul(Q, K, transpose_b=True) * scale
        if causal:
            scores = scores + (1.0 - mask) * -1e9
        attn = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(attn, V)
    step().numpy()
    iters = 1 if S >= 8192 else 2
    t0 = time.perf_counter()
    for _ in range(iters): step().numpy()
    dt = (time.perf_counter() - t0) / iters * 1000.0
    flops = (2.0 if causal else 4.0) * BH * S * S * dk
    tag = f"long_ctx_sdpa_{'causal_' if causal else ''}S{S}"
    emit(tag, "lat_ms", dt)
    emit(tag, "gflops", flops / dt / 1e6)


def bench_huge_adam():
    M = 1 << 14
    N = 512
    var = tf.Variable(tf.random.normal([M, N], stddev=0.1))
    g = tf.random.normal([M, N], stddev=0.1)
    opt = tf.keras.optimizers.Adam(1e-3)
    opt.build([var])
    @tf.function
    def step(): opt.apply_gradients([(g, var)])
    step()
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters): step()
    tf.identity(var).numpy()
    dt = (time.perf_counter() - t0) / iters * 1000.0
    n = M * N
    emit(f"adam_{n // 1000000}Mparam", "lat_ms", dt)
    emit(f"adam_{n // 1000000}Mparam", "gbs", 28.0 * n / dt / 1e6)


def main():
    bench_huge_gemm(4096)
    bench_huge_gemm(8192)
    try: bench_huge_gemm(12288)
    except Exception: pass

    bench_long_ctx_sdpa(2048, False)
    bench_long_ctx_sdpa(4096, False)
    bench_long_ctx_sdpa(8192, False)
    bench_long_ctx_sdpa(4096, True)
    bench_long_ctx_sdpa(8192, True)

    bench_huge_adam()


if __name__ == "__main__":
    main()
