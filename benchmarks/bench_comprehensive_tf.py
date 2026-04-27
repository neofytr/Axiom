"""bench_comprehensive_tf.py — exhaustive TF CPU benchmark for Axiom comparison.
   run with: OMP_NUM_THREADS=16 python3 benchmarks/bench_comprehensive_tf.py"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def stats(samples):
    s = sorted(samples)
    n = len(s)
    med = s[n // 2] if n & 1 else (s[n // 2 - 1] + s[n // 2]) * 0.5
    return min(s), med, sum(s) / n, max(s)

def emit(name, shape, samples_us, flops=0, bw_bytes=0):
    mn, med, mean, mx = stats(samples_us)
    tput = ""
    if flops > 0 and med > 0:
        tput = f"  {flops / (med * 1e3):8.1f} GFLOPS"
    elif bw_bytes > 0 and med > 0:
        tput = f"  {bw_bytes / (med * 1e3):8.1f} GB/s"
    print(f"{name:<20s} {shape:<30s} {mn:10.1f} {med:10.1f} {mean:10.1f} {mx:10.1f}{tput}")
    import sys; sys.stdout.flush()

GEMM_SHAPES = [
    (128,  128,  128,  "sq_128",     5, 60),
    (256,  256,  256,  "sq_256",     5, 60),
    (512,  512,  512,  "sq_512",     5, 50),
    (1024, 1024, 1024, "sq_1024",    5, 40),
    (2048, 2048, 2048, "sq_2048",    3, 20),
    (4096, 4096, 4096, "sq_4096",    2, 10),
    (2048, 768,  768,  "xfmr_qkv",  5, 40),
    (768,  2304, 2048, "xfmr_wgrad",5, 40),
    (768,  768,  2048, "xfmr_proj", 5, 40),
    (4096, 64,   256,  "tall_4k",   5, 50),
    (8192, 128,  64,   "tall_8k",   5, 40),
    (128,  4096, 256,  "wide",      5, 40),
    (384,  384,  384,  "non2_384",  5, 50),
    (577,  733,  512,  "non2_odd",  5, 40),
    (1,    128,  64,   "tiny_1x128",10, 80),
    (1,    256,  128,  "tiny_1x256",10, 80),
    (4,    64,   32,   "tiny_4x64", 10, 80),
    (16,   10,   64,   "tiny_cls",  10, 80),
]

def bench_gemm_nn(M, N, K, tag, warm, timed):
    A = tf.constant(tf.random.normal([M, K]))
    B = tf.constant(tf.random.normal([K, N]))
    flops = 2.0 * M * N * K
    for _ in range(warm):
        tf.matmul(A, B).numpy()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter()
        tf.matmul(A, B).numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    emit("gemm_nn", f"{M}x{N}x{K}_{tag}", samples, flops=flops)

def bench_gemm_tn(M, N, K, tag, warm, timed):
    At = tf.constant(tf.random.normal([K, M]))
    B = tf.constant(tf.random.normal([K, N]))
    flops = 2.0 * M * N * K
    for _ in range(warm):
        tf.matmul(At, B, transpose_a=True).numpy()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter()
        tf.matmul(At, B, transpose_a=True).numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    emit("gemm_tn", f"{M}x{N}x{K}_{tag}", samples, flops=flops)

CONV_SHAPES = [
    (1,  64,  64,  56, 56, 3, 1, 1, "res_s1",      5, 30),
    (1,  64,  128, 56, 56, 3, 2, 1, "res_down",     5, 30),
    (1,  128, 256, 28, 28, 3, 1, 1, "res_s3",       5, 30),
    (1,  256, 512, 14, 14, 3, 2, 1, "res_s4",       5, 30),
    (1,  256, 64,  14, 14, 1, 1, 0, "bottleneck",  10, 50),
    (1,  64,  128, 28, 28, 5, 1, 2, "k5",           5, 30),
    (1,  3,   64, 224,224, 7, 2, 3, "stem_7x7",     3, 15),
    (8,  64,  128, 28, 28, 3, 1, 1, "batch8",       3, 20),
    (32, 64,  128, 28, 28, 3, 1, 1, "batch32",      2, 10),
    (1,  64,  128, 28, 28, 3, 1, 1, "baseline",     5, 30),
    (1,  3,   16,  32, 32, 3, 1, 1, "tiny_3x16",   10, 50),
    (1,  16,  32,  16, 16, 3, 1, 1, "tiny_16x32",  10, 50),
]

def bench_conv(Nb, Cin, Cout, H, W, K, S, P, tag, warm, timed):
    x_val = tf.random.normal([Nb, H, W, Cin])
    x = tf.Variable(x_val)
    pad_str = 'same' if P > 0 else 'valid'
    conv = tf.keras.layers.Conv2D(Cout, K, strides=S, padding=pad_str, use_bias=True)
    conv.build((Nb, H, W, Cin))
    Hout = (H + 2*P - K) // S + 1
    Wout = (W + 2*P - K) // S + 1
    flops = 2.0 * 2.0 * Nb * Cout * Hout * Wout * Cin * K * K

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            y = conv(x, training=True)
            loss = tf.reduce_sum(y)
        return tape.gradient(loss, conv.trainable_variables + [x])

    for _ in range(warm):
        g = step(); g[-1].numpy()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter()
        g = step(); g[-1].numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    emit("conv_fwd_bwd",
         f"N{Nb}_C{Cin}-{Cout}_{H}x{W}_K{K}_S{S}_{tag}",
         samples, flops=flops)

BN_SHAPES = [
    (1,  64,  56, 56, "b1_c64",   10, 60),
    (8,  128, 28, 28, "b8_c128",   5, 50),
    (16, 256, 14, 14, "b16_c256",  5, 50),
    (32, 256, 14, 14, "b32_c256",  5, 40),
    (64, 512,  7,  7, "b64_c512",  5, 40),
    (1,  32,  16, 16, "tiny",     10, 60),
]

def bench_bn(Nb, C, H, W, tag, warm, timed):
    x = tf.Variable(tf.random.normal([Nb, H, W, C]))
    bn = tf.keras.layers.BatchNormalization(axis=-1)
    bn.build((Nb, H, W, C))
    bw_bytes = 2.0 * Nb * C * H * W * 4

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            y = bn(x, training=True)
            loss = tf.reduce_sum(y)
        return tape.gradient(loss, bn.trainable_variables + [x])

    for _ in range(warm):
        g = step(); g[-1].numpy()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter()
        g = step(); g[-1].numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    emit("bn_fwd_bwd", f"N{Nb}_C{C}_{H}x{W}_{tag}", samples, bw_bytes=bw_bytes)

def bench_mlp(B, d1, d2, d3, d4, tag, warm=5, timed=40):
    x = tf.Variable(tf.random.normal([B, d1]))
    labels = tf.constant(np.arange(B) % d4, dtype=tf.int64)
    fc1 = tf.keras.layers.Dense(d2, activation='relu')
    fc2 = tf.keras.layers.Dense(d3, activation='relu')
    fc3 = tf.keras.layers.Dense(d4)
    fc1.build((B, d1)); fc2.build((B, d2)); fc3.build((B, d3))
    flops = 2.0 * B * (float(d1)*d2 + float(d2)*d3 + float(d3)*d4) * 2.0

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            h = fc3(fc2(fc1(x)))
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=h))
        all_vars = fc1.trainable_variables + fc2.trainable_variables + fc3.trainable_variables + [x]
        return tape.gradient(loss, all_vars)

    for _ in range(warm):
        g = step(); g[-1].numpy()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter()
        g = step(); g[-1].numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    emit("mlp_fwd_bwd", tag, samples, flops=flops)

def bench_mha(B, S, D, H, tag, warm=3, timed=20):
    x = tf.Variable(tf.random.normal([B, S, D]))
    mha = tf.keras.layers.MultiHeadAttention(num_heads=H, key_dim=D // H)
    mha(x, x)
    flops = float(B) * S * D * D * 12.0

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            y = mha(x, x, training=True)
            loss = tf.reduce_sum(y)
        return tape.gradient(loss, mha.trainable_variables + [x])

    for _ in range(warm):
        g = step(); g[-1].numpy()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter()
        g = step(); g[-1].numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    emit("mha_train", tag, samples, flops=flops)

def bench_softmax(rows, cols, tag, warm=10, timed=60):
    x = tf.constant(tf.random.normal([rows, cols]))
    bw = 2.0 * rows * cols * 4
    for _ in range(warm):
        tf.nn.softmax(x).numpy()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter()
        tf.nn.softmax(x).numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    emit("softmax", tag, samples, bw_bytes=bw)

def bench_relu(n, tag, warm=10, timed=80):
    x = tf.constant(tf.random.normal([n]))
    bw = 2.0 * n * 4
    for _ in range(warm):
        tf.nn.relu(x).numpy()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter()
        tf.nn.relu(x).numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    emit("relu", tag, samples, bw_bytes=bw)

if __name__ == "__main__":
    nt = os.environ.get("OMP_NUM_THREADS", "?")
    print(f"TensorFlow {tf.__version__} (CPU, oneDNN, tf.function)")
    print(f"OMP_NUM_THREADS={nt}")
    hdr = f"\n{'bench':<20s} {'shape':<30s} {'min_us':>10s} {'med_us':>10s} {'mean_us':>10s} {'max_us':>10s}  throughput"
    print(hdr)

    print("\n=== GEMM NN ===")
    for s in GEMM_SHAPES:
        bench_gemm_nn(*s)

    print("\n=== GEMM TN ===")
    for s in GEMM_SHAPES:
        bench_gemm_tn(*s)

    print("\n=== CONV FWD+BWD ===")
    for s in CONV_SHAPES:
        bench_conv(*s)

    print("\n=== BATCHNORM FWD+BWD ===")
    for s in BN_SHAPES:
        bench_bn(*s)

    print("\n=== MLP FWD+BWD ===")
    bench_mlp(32,  768,  512,  256,  10,   "768-512-256-10_B32",   5, 40)
    bench_mlp(1,   784,  256,  128,  10,   "784-256-128-10_B1",   10, 60)
    bench_mlp(64,  784,  512,  256,  10,   "784-512-256-10_B64",   3, 30)
    bench_mlp(16,  1024, 1024, 1024, 1024, "1024^4_B16",           3, 20)
    bench_mlp(1,   64,   32,   16,   10,   "64-32-16-10_B1",      10, 80)

    print("\n=== MHA TRAIN ===")
    bench_mha(1,  64,  128,  4,  "B1_S64_D128_H4",    5, 40)
    bench_mha(1,  128, 256,  4,  "B1_S128_D256_H4",   5, 30)
    bench_mha(1,  256, 512,  8,  "B1_S256_D512_H8",   5, 30)
    bench_mha(1,  384, 768,  12, "B1_S384_D768_H12",  3, 20)
    bench_mha(4,  384, 768,  12, "B4_S384_D768_H12",  2, 10)
    bench_mha(8,  256, 512,  8,  "B8_S256_D512_H8",   2, 10)
    bench_mha(1,  256, 768,  12, "B1_S256_D768_H12",  3, 20)
    bench_mha(16, 128, 512,  8,  "B16_S128_D512_H8",  2, 8)

    print("\n=== SOFTMAX ===")
    bench_softmax(32,   10,    "cls_32x10")
    bench_softmax(32,   1024,  "32x1024")
    bench_softmax(512,  512,   "attn_512")
    bench_softmax(2048, 768,   "xfmr_2048x768")
    bench_softmax(1,    65536, "long_1x64k")
    bench_softmax(4096, 128,   "tall_4kx128")

    print("\n=== RELU ===")
    bench_relu(1024,     "1k")
    bench_relu(65536,    "64k")
    bench_relu(1048576,  "1M")
    bench_relu(16777216, "16M")

    print("\nDone.")
