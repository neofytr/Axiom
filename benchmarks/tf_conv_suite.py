"""tf_conv_suite.py — TF sidecar for bench_conv_suite.c."""

import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

SUITE = "conv"


def emit(case, metric, value):
    print(f"RESULT {SUITE} {case} {metric} {value:.6f}", flush=True)


def time_op(fn, iters, warmup=2):
    for _ in range(warmup):
        fn().numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    out.numpy()
    return (time.perf_counter() - t0) / iters * 1000.0


def conv_flops(N, C_in, C_out, H_out, W_out, K):
    return 2.0 * N * C_out * H_out * W_out * C_in * K * K


def bench_conv(N, Cin, H, W, Cout, K, stride, pad):
    # tf uses NHWC by default
    x = tf.random.normal([N, H, W, Cin])
    w = tf.random.normal([K, K, Cin, Cout])
    @tf.function
    def step():
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1],
                             padding=[[0,0],[pad,pad],[pad,pad],[0,0]])
    H_out = (H + 2*pad - K) // stride + 1
    W_out = (W + 2*pad - K) // stride + 1
    flops = conv_flops(N, Cin, Cout, H_out, W_out, K)
    iters = max(2, min(int(2e10 / flops), 50))
    lat_ms = time_op(step, iters)
    cs = f"conv_{N}x{Cin}x{H}x{W}_{Cout}x{K}_k{K}_s{stride}"
    emit(cs, "lat_ms", lat_ms)
    emit(cs, "gflops", flops / lat_ms / 1e6)


def bench_cbr(N, Cin, H, W, Cout, K, stride, pad):
    x = tf.random.normal([N, H, W, Cin])
    layer = tf.keras.Sequential([
        tf.keras.layers.Conv2D(Cout, K, strides=stride, padding="same" if pad > 0 else "valid", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ])
    layer.build([N, H, W, Cin])
    @tf.function
    def step(): return layer(x, training=False)
    H_out = (H + 2*pad - K) // stride + 1
    W_out = (W + 2*pad - K) // stride + 1
    flops = conv_flops(N, Cin, Cout, H_out, W_out, K)
    iters = max(2, min(int(2e10 / flops), 50))
    lat_ms = time_op(step, iters)
    cs = f"cbr_{N}x{Cin}x{H}x{W}_{Cout}x{K}_k{K}_s{stride}"
    emit(cs, "lat_ms", lat_ms)
    emit(cs, "gflops", flops / lat_ms / 1e6)


def bench_maxpool(N, C, H, W, K, stride):
    x = tf.random.normal([N, H, W, C])
    @tf.function
    def step(): return tf.nn.max_pool2d(x, K, stride, "VALID")
    iters = max(5, min(int(1e9 / (N * C * H * W)), 500))
    lat_ms = time_op(step, iters)
    cs = f"maxpool_{N}x{C}x{H}x{W}_k{K}"
    emit(cs, "lat_ms", lat_ms)
    emit(cs, "gbs", 4.0 * N * C * H * W / lat_ms / 1e6)


def main():
    cases = [
        (32,   3, 224, 224,  64, 3, 1, 1),
        (32,  64, 112, 112, 128, 3, 1, 1),
        (32, 128,  56,  56, 256, 3, 1, 1),
        (32, 256,  28,  28, 512, 3, 1, 1),
        (32, 512,  14,  14, 512, 3, 1, 1),
        (32,  64, 112, 112, 128, 3, 2, 1),
        (32, 512,  14,  14, 512, 1, 1, 0),
    ]
    for c in cases: bench_conv(*c)
    bench_cbr(32,  64, 112, 112, 128, 3, 1, 1)
    bench_cbr(32, 256,  28,  28, 512, 3, 1, 1)
    bench_maxpool(32, 128, 112, 112, 2, 2)
    bench_maxpool(32, 256,  56,  56, 2, 2)


if __name__ == "__main__":
    main()
