"""
tensorflow wide-mlp benchmark -- mirrors benchmarks/bench_wide.c exactly.

architecture (~14.4M params):
  784 -> 4096 (bn, relu, dropout 0.3)
      -> 2048 (bn, relu, dropout 0.3)
      -> 1024 (bn, relu, dropout 0.2)
      ->  512 (relu)
      ->  256 (relu)
      ->   10

optim: adam lr=1e-3 b1=0.9 b2=0.999 eps=1e-8
batch: 256
epochs: 5
dataset: full 60000 train / 10000 test
seed: 42

modes (argv[1]):
  train    full training + eval
  infer    forward-only throughput (1000 batches of 256)
"""

import os
import struct
import time
import sys

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

N_TRAIN = 60000
N_TEST = 10000
BATCH = 256
EPOCHS = 5


def read_idx_images(path, n):
    with open(path, "rb") as f:
        struct.unpack(">IIII", f.read(16))
        raw = f.read(n * 784)
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32).reshape(n, 784) / 255.0


def read_idx_labels(path, n):
    with open(path, "rb") as f:
        f.read(8)
        raw = f.read(n)
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int32)


def build_wide():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(4096),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(2048),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(10),
    ])


def run_train():
    device_label = "gpu" if USE_GPU else "cpu"
    print(f"=== tf wide-mlp train ({device_label}) ===")
    if not USE_GPU:
        print(f"threads (intra_op): {NUM_THREADS}")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "examples", "data")

    train_x = read_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte"), N_TRAIN)
    train_y = read_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"), N_TRAIN)
    test_x = read_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte"), N_TEST)
    test_y = read_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"), N_TEST)

    model = build_wide()
    n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"model: wide-mlp  params: {n_params}")
    print(f"arch: 784 -> 4096 -> 2048 -> 1024 -> 512 -> 256 -> 10")
    print(f"batch: {BATCH}  epochs: {EPOCHS}  dataset: {N_TRAIN} train / {N_TEST} test\n")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                   clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

    # warmup
    model.train_on_batch(train_x[:BATCH], train_y[:BATCH])

    t_start = time.perf_counter()
    per_epoch = []

    for ep in range(EPOCHS):
        t_ep = time.perf_counter()
        perm = np.random.permutation(N_TRAIN)
        shuf_x, shuf_y = train_x[perm], train_y[perm]
        total_loss, n_used = 0.0, 0

        for i in range(0, N_TRAIN, BATCH):
            xb = shuf_x[i:i + BATCH]
            yb = shuf_y[i:i + BATCH]
            loss, _ = model.train_on_batch(xb, yb)
            total_loss += loss * len(xb)
            n_used += len(xb)

        _, acc = model.evaluate(test_x, test_y, verbose=0, batch_size=512)
        dt = time.perf_counter() - t_ep
        per_epoch.append(dt)
        print(f"epoch {ep + 1}/{EPOCHS}  loss={total_loss / n_used:.4f}  "
              f"test_acc={acc * 100:.2f}%  time={dt:.3f}s")

    t_total = time.perf_counter() - t_start
    _, final_acc = model.evaluate(test_x, test_y, verbose=0, batch_size=512)

    print(f"\n--- tf wide-mlp {device_label} summary ---")
    print(f"params: {n_params}")
    print(f"total training time: {t_total:.3f}s")
    print(f"mean per-epoch: {sum(per_epoch) / len(per_epoch):.3f}s")
    print(f"median per-epoch: {sorted(per_epoch)[len(per_epoch) // 2]:.3f}s")
    print(f"final test accuracy: {final_acc * 100:.2f}%")


def run_infer():
    device_label = "gpu" if USE_GPU else "cpu"
    print(f"=== tf wide-mlp infer ({device_label}) ===")
    if not USE_GPU:
        print(f"threads (intra_op): {NUM_THREADS}")

    model = build_wide()
    n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"model: wide-mlp  params: {n_params}")

    iters = 1000
    print(f"batch: {BATCH}  forward iters: {iters}\n")

    x = np.random.randn(BATCH, 784).astype(np.float32)

    # warmup
    for _ in range(10):
        model(x, training=False)

    t0 = time.perf_counter()
    for _ in range(iters):
        model(x, training=False)
    t1 = time.perf_counter()

    total = t1 - t0
    per_batch_ms = total / iters * 1000
    imgs_per_sec = iters * BATCH / total

    print(f"--- tf wide-mlp {device_label} forward summary ---")
    print(f"total forward time: {total:.3f}s")
    print(f"per-batch: {per_batch_ms:.3f} ms")
    print(f"throughput: {imgs_per_sec:.0f} images/s")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        run_train()
    elif mode == "infer":
        run_infer()
    else:
        print(f"usage: python3 {sys.argv[0]} {{train|infer}}")
        sys.exit(1)


if __name__ == "__main__":
    main()
