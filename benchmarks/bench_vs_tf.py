"""bench_vs_tf.py — TensorFlow CPU benchmark matching bench_hotpaths shapes.
   tests both eager and tf.function modes. forces sync via .numpy().
   run with: OMP_NUM_THREADS=16 python3 benchmarks/bench_vs_tf.py"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

WARM = 10
TIMED = 80

def stats(samples):
    s = sorted(samples)
    n = len(s)
    med = s[n // 2] if n & 1 else (s[n // 2 - 1] + s[n // 2]) * 0.5
    return min(s), med, sum(s) / n, max(s)

def report(name, shape_tag, samples_us, flops=0, bw_bytes=0):
    mn, med, mean, mx = stats(samples_us)
    tput = ""
    if flops > 0 and med > 0:
        tput = f"\t{flops / (med * 1e3):.1f} GFLOPS"
    elif bw_bytes > 0 and med > 0:
        tput = f"\t{bw_bytes / (med * 1e3):.1f} GB/s"
    print(f"{name}\t{shape_tag}\t{mn:.1f}\t{med:.1f}\t{mean:.1f}\t{mx:.1f}{tput}")

# ---- GEMM NN ----
def bench_gemm_nn(M, N, K, tag):
    A = tf.constant(tf.random.normal([M, K]))
    B = tf.constant(tf.random.normal([K, N]))
    flops = 2.0 * M * N * K
    for _ in range(WARM):
        tf.matmul(A, B).numpy()
    samples = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        tf.matmul(A, B).numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    report("gemm_nn", f"{M}x{N}x{K}_{tag}", samples, flops=flops)

# ---- GEMM TN ----
def bench_gemm_tn(M, N, K, tag):
    At = tf.constant(tf.random.normal([K, M]))
    B = tf.constant(tf.random.normal([K, N]))
    flops = 2.0 * M * N * K
    for _ in range(WARM):
        tf.matmul(At, B, transpose_a=True).numpy()
    samples = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        tf.matmul(At, B, transpose_a=True).numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    report("gemm_tn", f"{M}x{N}x{K}_{tag}", samples, flops=flops)

# ---- Conv2d fwd+bwd ----
def bench_conv_fwd_bwd():
    Nb, Cin, Cout, H, W, K = 1, 64, 128, 28, 28, 3
    x_val = tf.random.normal([Nb, H, W, Cin])
    x = tf.Variable(x_val)
    conv = tf.keras.layers.Conv2D(Cout, K, padding='same', use_bias=True)
    conv.build((Nb, H, W, Cin))
    flops = 2.0 * 2.0 * Nb * Cout * H * W * Cin * K * K

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            y = conv(x, training=True)
            loss = tf.reduce_sum(y)
        return tape.gradient(loss, conv.trainable_variables + [x])

    for _ in range(WARM):
        g = step()
        g[-1].numpy()
    samples = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        g = step()
        g[-1].numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    report("conv_fwd_bwd", f"N{Nb}_C{Cin}-{Cout}_{H}x{W}_K{K}", samples, flops=flops)

# ---- BatchNorm fwd+bwd ----
def bench_bn_fwd_bwd():
    Nb, C, H, W = 32, 256, 14, 14
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

    for _ in range(WARM):
        g = step()
        g[-1].numpy()
    samples = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        g = step()
        g[-1].numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    report("bn_fwd_bwd", f"N{Nb}_C{C}_{H}x{W}", samples, bw_bytes=bw_bytes)

# ---- MLP 768->512->256->10 B=32 fwd+bwd ----
def bench_mlp_fwd_bwd():
    B = 32
    x = tf.Variable(tf.random.normal([B, 768]))
    labels = tf.constant(np.arange(B) % 10, dtype=tf.int64)
    fc1 = tf.keras.layers.Dense(512, activation='relu')
    fc2 = tf.keras.layers.Dense(256, activation='relu')
    fc3 = tf.keras.layers.Dense(10)
    fc1.build((B, 768)); fc2.build((B, 512)); fc3.build((B, 256))
    flops = 2.0 * B * (768*512 + 512*256 + 256*10) * 2.0

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            h = fc1(x)
            h = fc2(h)
            h = fc3(h)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=h))
        all_vars = fc1.trainable_variables + fc2.trainable_variables + fc3.trainable_variables + [x]
        return tape.gradient(loss, all_vars)

    for _ in range(WARM):
        g = step()
        g[-1].numpy()
    samples = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        g = step()
        g[-1].numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    report("mlp_fwd_bwd", "768-512-256-10_B32", samples, flops=flops)

# ---- MHA train ----
def bench_mha_train(B, S, D, H):
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

    for _ in range(WARM):
        g = step()
        g[-1].numpy()
    samples = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        g = step()
        g[-1].numpy()
        samples.append((time.perf_counter() - t0) * 1e6)
    report("mha_train", f"B{B}_S{S}_D{D}_H{H}", samples, flops=flops)

if __name__ == "__main__":
    nt = os.environ.get("OMP_NUM_THREADS", "?")
    print(f"TensorFlow {tf.__version__} (CPU, oneDNN, tf.function)")
    print(f"OMP_NUM_THREADS={nt}")
    print(f"\nbench_name\tshape\tmin_us\tmedian_us\tmean_us\tmax_us\tthroughput")
    print("-" * 72)

    bench_gemm_nn(2048, 768, 768, "dwqkv")
    bench_gemm_tn(2048, 768, 768, "dwqkv")
    bench_gemm_nn(768, 2304, 2048, "mha_wgrad")
    bench_gemm_tn(768, 2304, 2048, "mha_wgrad")
    bench_gemm_nn(768, 768, 2048, "mha_tn")
    bench_gemm_tn(768, 768, 2048, "mha_tn")
    bench_conv_fwd_bwd()
    bench_bn_fwd_bwd()
    bench_mlp_fwd_bwd()
    bench_mha_train(1, 512, 768, 12)
    bench_mha_train(4, 512, 768, 12)
