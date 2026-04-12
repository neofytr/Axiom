"""
tensorflow large-mlp mnist benchmark — mirrors benchmarks/bench_large.c.

model: 784 -> 1024 relu -> 512 relu -> 256 relu -> 10
optim: adam, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8
batch: 256
epochs: 5
dataset: full 60000 train / 10000 test
seed: 42
"""

import os
import struct
import time
import sys

NUM_THREADS = int(os.environ.get("AX_BENCH_THREADS", "8"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
# force cpu for fair comparison with axiom cpu_opt
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.random.set_seed(42)
np.random.seed(42)


def read_idx_images(path, n):
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x00000803
        raw = f.read(n * 784)
    return np.frombuffer(raw, dtype=np.uint8).astype(np.float32).reshape(n, 784) / 255.0


def read_idx_labels(path, n):
    with open(path, "rb") as f:
        f.read(8)
        raw = f.read(n)
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int32)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "examples", "data")
    print(f"=== tensorflow {mode} ===")
    print(f"threads (intra_op): {NUM_THREADS}")
    print(f"model: 784 -> 1024 -> 512 -> 256 -> 10")

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(10),
    ])

    n_params = sum(v.numpy().size for v in model.trainable_variables)
    print(f"params: {n_params}")

    if mode == "train":
        train_x = read_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte"), 60000)
        train_y = read_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"), 60000)
        test_x = read_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte"), 10000)
        test_y = read_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"), 10000)
        print(f"loaded mnist: train={len(train_x)} test={len(test_x)}")

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9,
                                       beta_2=0.999, epsilon=1e-8)
        model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

        batch_size = 256
        epochs = 5
        n_batches = (len(train_x) + batch_size - 1) // batch_size
        print(f"training: {epochs} epochs, {n_batches} batches/epoch\n")

        # warm up the graph
        model.train_on_batch(train_x[:batch_size], train_y[:batch_size])

        t_start = time.perf_counter()
        per_epoch = []
        for ep in range(epochs):
            t_ep = time.perf_counter()
            perm = np.random.permutation(len(train_x))
            shuf_x, shuf_y = train_x[perm], train_y[perm]
            total_loss, n_used = 0.0, 0
            for i in range(0, len(shuf_x), batch_size):
                xb = shuf_x[i:i + batch_size]
                yb = shuf_y[i:i + batch_size]
                loss, _acc = model.train_on_batch(xb, yb)
                total_loss += loss * len(xb)
                n_used += len(xb)
            _, acc = model.evaluate(test_x, test_y, verbose=0, batch_size=512)
            dt = time.perf_counter() - t_ep
            per_epoch.append(dt)
            print(f"epoch {ep+1}/{epochs}  loss={total_loss/n_used:.4f}  "
                  f"test_acc={acc*100:.2f}%  time={dt:.3f}s")

        t_total = time.perf_counter() - t_start
        _, final_acc = model.evaluate(test_x, test_y, verbose=0, batch_size=512)
        median = sorted(per_epoch)[len(per_epoch) // 2]
        mean = sum(per_epoch) / len(per_epoch)

        print(f"\n--- tf training summary ---")
        print(f"total training time: {t_total:.3f}s")
        print(f"mean per-epoch: {mean:.3f}s")
        print(f"median per-epoch: {median:.3f}s")
        print(f"final test accuracy: {final_acc*100:.2f}%")

    elif mode == "infer":
        batch_size = 256
        iters = 1000
        x = np.random.randn(batch_size, 784).astype(np.float32)
        print(f"batch: {batch_size}  forward iters: {iters}\n")

        # warm up
        for _ in range(10):
            model(x, training=False)

        t0 = time.perf_counter()
        for _ in range(iters):
            model(x, training=False)
        t1 = time.perf_counter()
        total = t1 - t0
        per_batch_ms = total / iters * 1000
        imgs_per_sec = iters * batch_size / total

        print(f"--- tf forward summary ---")
        print(f"total forward time: {total:.3f}s")
        print(f"per-batch:   {per_batch_ms:.3f} ms")
        print(f"throughput:  {imgs_per_sec:.0f} images/s")
    else:
        print(f"usage: python3 {sys.argv[0]} {{train|infer}}")
        sys.exit(1)


if __name__ == "__main__":
    main()
