"""
tensorflow mnist mlp GPU benchmark — mirrors benchmarks/ax_mnist.c exactly
but forces tf to use the GPU.

model: 784 -> 128 relu -> 64 relu -> 10
optim: adam lr=1e-3 b1=0.9 b2=0.999 eps=1e-8
batch: 256, epochs: 8, subset: 10000 train / 10000 test, seed: 42
"""

import os
import struct
import time
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

# force gpu
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    print("no gpu available, aborting")
    sys.exit(1)
print(f"gpu: {gpus[0].name}")
tf.random.set_seed(42)
np.random.seed(42)


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


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "data")
    print("loading mnist...")
    train_x = read_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte"), 10000)
    train_y = read_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"), 10000)
    test_x = read_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte"), 10000)
    test_y = read_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"), 10000)
    print(f"  train: {len(train_x)}  test: {len(test_x)}")

    with tf.device("/GPU:0"):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    n_params = sum(v.numpy().size for v in model.trainable_variables)
    print(f"model: {n_params} parameters")
    n_batches = (len(train_x) + 255) // 256
    print(f"training: {n_batches} batches/epoch\n")

    epochs = 8
    batch_size = 256

    # warmup
    model.train_on_batch(train_x[:batch_size], train_y[:batch_size])

    t_start = time.perf_counter()
    per_epoch = []

    for ep in range(epochs):
        t_ep = time.perf_counter()
        perm = np.random.permutation(len(train_x))
        shuf_x, shuf_y = train_x[perm], train_y[perm]
        total_loss, n_used = 0.0, 0
        for i in range(0, len(shuf_x), batch_size):
            xb, yb = shuf_x[i:i + batch_size], shuf_y[i:i + batch_size]
            loss, _ = model.train_on_batch(xb, yb)
            total_loss += loss * len(xb)
            n_used += len(xb)
        _, acc = model.evaluate(test_x, test_y, verbose=0, batch_size=512)
        dt = time.perf_counter() - t_ep
        per_epoch.append(dt)
        print(f"epoch {ep + 1}/{epochs}  loss={total_loss / n_used:.4f}  "
              f"test_acc={acc * 100:.2f}%  time={dt:.3f}s")

    t_total = time.perf_counter() - t_start
    _, acc_final = model.evaluate(test_x, test_y, verbose=0, batch_size=512)

    print(f"\n--- tf-gpu summary ---")
    print(f"total training time: {t_total:.3f}s")
    print(f"mean per-epoch: {sum(per_epoch) / len(per_epoch):.3f}s")
    print(f"median per-epoch: {sorted(per_epoch)[len(per_epoch) // 2]:.3f}s")
    print(f"final test accuracy: {acc_final * 100:.2f}%")


if __name__ == "__main__":
    main()
