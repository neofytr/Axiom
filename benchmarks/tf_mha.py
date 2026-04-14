"""
TensorFlow MHA benchmark — mirrors benchmarks/bench_mha.c shape-for-shape.

measures:
  - SDPA forward (scaled_dot_product_attention)
  - SDPA forward, causal
  - MHA layer forward (tf.keras.layers.MultiHeadAttention)
  - MHA layer train step (forward + backward)
  - KV-style single-token attention (emulated via a 1-query SDPA call)

shapes must stay in sync with bench_mha.c. flop counts are identical so
the GFLOPS numbers are directly comparable.

run:  python3 tf_mha.py [iters]
env:
  AX_BENCH_THREADS=N   — override intra-op threads (default 8)
  TF_BENCH_GPU=1       — force GPU if one is present
"""

import os
import sys
import time
import math

NUM_THREADS = int(os.environ.get("AX_BENCH_THREADS", "8"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

USE_GPU = os.environ.get("TF_BENCH_GPU", "0") == "1"
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

if not USE_GPU:
    tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
tf.random.set_seed(42)
np.random.seed(42)

# same shapes as bench_mha.c
SHAPES = [
    dict(B=8, S=128,  D=512,  H=8),
    dict(B=4, S=512,  D=768,  H=12),
    dict(B=2, S=1024, D=768,  H=12),
    dict(B=1, S=512,  D=1024, H=16),
    dict(B=1, S=2048, D=768,  H=12),
]


def _time(fn, iters):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    # force host-side wait if GPU
    if hasattr(out, "numpy"):
        out.numpy()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def bench_sdpa(s, iters, causal):
    B, S, D, H = s["B"], s["S"], s["D"], s["H"]
    dk = D // H
    BH = B * H

    # [B, H, S, dk] — TF's SDPA expects this layout
    Q = tf.random.normal([B, H, S, dk])
    K = tf.random.normal([B, H, S, dk])
    V = tf.random.normal([B, H, S, dk])
    scale = 1.0 / math.sqrt(dk)

    @tf.function(jit_compile=True)
    def step():
        # manual SDPA: softmax(Q K^T / sqrt(dk)) V
        scores = tf.matmul(Q, K, transpose_b=True) * scale
        if causal:
            mask = tf.linalg.band_part(tf.ones([S, S]), -1, 0)
            scores = scores + (1.0 - mask) * -1e9
        attn = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(attn, V)

    elapsed = _time(step, iters)
    flops = 4.0 * BH * S * S * dk
    gflops = flops / elapsed / 1e9
    tag = "SDPA-causal" if causal else "SDPA       "
    print(f"  {tag}  BH={BH:>4d} S={S:>4d} dk={dk:>3d} :  {elapsed*1e3:7.3f} ms  {gflops:8.2f} GFLOPS")


def bench_mha_forward(s, iters):
    B, S, D, H = s["B"], s["S"], s["D"], s["H"]
    x = tf.random.normal([B, S, D])
    mha = tf.keras.layers.MultiHeadAttention(num_heads=H, key_dim=D // H)
    mha.build(query_shape=[B, S, D], value_shape=[B, S, D])

    @tf.function(jit_compile=True)
    def step():
        return mha(x, x)

    elapsed = _time(step, iters)
    flops = 4.0 * 2.0 * B * S * D * D + 4.0 * B * S * S * D
    gflops = flops / elapsed / 1e9
    print(f"  MHA-fwd      B={B:>2d} S={S:>4d} D={D:>4d} H={H:>2d} :  {elapsed*1e3:7.3f} ms  {gflops:8.2f} GFLOPS")


def bench_mha_train(s, iters):
    B, S, D, H = s["B"], s["S"], s["D"], s["H"]
    x = tf.random.normal([B, S, D])
    mha = tf.keras.layers.MultiHeadAttention(num_heads=H, key_dim=D // H)
    mha.build(query_shape=[B, S, D], value_shape=[B, S, D])

    @tf.function(jit_compile=True)
    def step():
        with tf.GradientTape() as tape:
            out = mha(x, x)
            loss = tf.reduce_sum(out)
        grads = tape.gradient(loss, mha.trainable_variables)
        return grads[0]

    elapsed = _time(step, iters)
    fwd_flops = 4.0 * 2.0 * B * S * D * D + 4.0 * B * S * S * D
    gflops = 3.0 * fwd_flops / elapsed / 1e9
    print(f"  MHA-train    B={B:>2d} S={S:>4d} D={D:>4d} H={H:>2d} :  {elapsed*1e3:7.3f} ms  {gflops:8.2f} GFLOPS (est)")


def bench_kv_cache(s, iters):
    """emulate single-token autoregressive attention:
       Q is [BH, 1, dk], K/V are [BH, S, dk]. same flop count as axiom's
       ax_kv_cache_attend so the GFLOPS numbers are comparable. """
    B, S, D, H = s["B"], s["S"], s["D"], s["H"]
    dk = D // H
    BH = B * H

    Qnew = tf.random.normal([BH, 1, dk])
    K = tf.random.normal([BH, S, dk])
    V = tf.random.normal([BH, S, dk])
    scale = 1.0 / math.sqrt(dk)

    @tf.function(jit_compile=True)
    def step():
        scores = tf.matmul(Qnew, K, transpose_b=True) * scale  # [BH, 1, S]
        attn = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(attn, V)  # [BH, 1, dk]

    elapsed = _time(step, iters)
    flops = 4.0 * BH * S * dk
    gflops = flops / elapsed / 1e9
    print(f"  KV-attend    BH={BH:>4d} S={S:>4d} dk={dk:>3d} :  {elapsed*1e3:7.3f} ms  {gflops:8.2f} GFLOPS")


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"== TF MHA / SDPA benchmarks (tf={tf.__version__}, "
          f"device={'GPU' if USE_GPU else f'CPU x{NUM_THREADS}'}, iters={iters}) ==\n")
    for i, s in enumerate(SHAPES):
        print(f"-- shape[{i}]  B={s['B']} S={s['S']} D={s['D']} H={s['H']} --")
        bench_sdpa(s, iters, False)
        bench_sdpa(s, iters, True)
        bench_mha_forward(s, iters)
        bench_mha_train(s, iters)
        bench_kv_cache(s, iters)
        print()


if __name__ == "__main__":
    main()
