"""tf_ops_suite.py — TF sidecar for bench_ops_suite.c."""

import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

SUITE = "ops"


def emit(case, metric, value):
    print(f"RESULT {SUITE} {case} {metric} {value:.6f}", flush=True)


def time_op(fn, iters, warmup=3):
    for _ in range(warmup):
        fn().numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    out.numpy()
    return (time.perf_counter() - t0) / iters * 1000.0


def bench_elemwise(name, N, bytes_per_elem, fn):
    iters_map = {"relu": 1e9, "gelu": 5e8, "add": 1e9, "sum": 1e9}
    iters = int(iters_map[name] / N)
    iters = max(10, min(iters, 10000))
    lat_ms = time_op(fn, iters)
    emit(f"{name}_{N}", "lat_ms", lat_ms)
    emit(f"{name}_{N}", "gbs", bytes_per_elem * N / lat_ms / 1e6)


def bench_softmax(rows, cols):
    x = tf.random.normal([rows, cols])
    @tf.function
    def step(): return tf.nn.softmax(x, axis=-1)
    iters = max(10, min(int(5e8 / (rows * cols)), 2000))
    lat_ms = time_op(step, iters)
    emit(f"softmax_{rows}x{cols}", "lat_ms", lat_ms)
    emit(f"softmax_{rows}x{cols}", "gbs", 8.0 * rows * cols / lat_ms / 1e6)


def bench_batchnorm(batch, feat):
    x = tf.random.normal([batch, feat])
    bn = tf.keras.layers.BatchNormalization()
    bn.build([batch, feat])
    @tf.function
    def step(): return bn(x, training=False)
    iters = max(10, min(int(3e8 / (batch * feat)), 1000))
    lat_ms = time_op(step, iters)
    emit(f"batchnorm_{batch}x{feat}", "lat_ms", lat_ms)
    emit(f"batchnorm_{batch}x{feat}", "gbs", 8.0 * batch * feat / lat_ms / 1e6)


def bench_layernorm(batch, feat):
    x = tf.random.normal([batch, feat])
    ln = tf.keras.layers.LayerNormalization()
    ln.build([batch, feat])
    @tf.function
    def step(): return ln(x)
    iters = max(10, min(int(3e8 / (batch * feat)), 1000))
    lat_ms = time_op(step, iters)
    emit(f"layernorm_{batch}x{feat}", "lat_ms", lat_ms)
    emit(f"layernorm_{batch}x{feat}", "gbs", 8.0 * batch * feat / lat_ms / 1e6)


def main():
    sizes = [1 << 14, 1 << 18, 1 << 22, 1 << 24]
    for N in sizes:
        x = tf.random.normal([N])
        y = tf.random.normal([N])
        bench_elemwise("relu", N, 8, tf.function(lambda: tf.nn.relu(x)))
        bench_elemwise("gelu", N, 8, tf.function(lambda: tf.nn.gelu(x)))
        bench_elemwise("add",  N, 12, tf.function(lambda: x + y))
        bench_elemwise("sum",  N, 4, tf.function(lambda: tf.reduce_sum(x)))

    for (r, c) in [(32, 512), (128, 1024), (256, 2048)]:
        bench_softmax(r, c)

    for (b, f) in [(256, 512), (256, 1024), (256, 4096), (64, 8192)]:
        bench_batchnorm(b, f)
        bench_layernorm(b, f)


if __name__ == "__main__":
    main()
