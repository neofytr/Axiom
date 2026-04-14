"""tf_transformer.py — TF sidecar for bench_transformer.c.

mirrors shape-for-shape: B=8, S=128, d=256, heads=4, FFN=1024, 6 layers."""

import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

B = 8
S = 128
D_MODEL = 256
N_HEADS = 4
D_FF = 1024
N_LAYERS = 6
VOCAB = 1024
STEPS = 20

SUITE = "transformer"


def emit(case, metric, value):
    print(f"RESULT {SUITE} {case} {metric} {value:.6f}", flush=True)


def build_block():
    return tf.keras.Sequential([
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=D_MODEL // N_HEADS),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(D_FF),
        tf.keras.layers.Activation("gelu"),
        tf.keras.layers.Dense(D_MODEL),
    ])


# a Sequential containing MultiHeadAttention won't auto-feed (q,k,v);
# build an explicit Model instead.
class Block(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=D_MODEL // N_HEADS)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.ff1 = tf.keras.layers.Dense(D_FF)
        self.gelu = tf.keras.layers.Activation("gelu")
        self.ff2 = tf.keras.layers.Dense(D_MODEL)

    def call(self, x, training=False):
        h = self.ln1(x)
        h = self.mha(h, h, training=training)
        h = self.ln2(h)
        h = self.ff1(h); h = self.gelu(h); h = self.ff2(h)
        return h


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.blocks = [Block() for _ in range(N_LAYERS)]
        self.ln_final = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(VOCAB)

    def call(self, x, training=False):
        for b in self.blocks:
            x = b(x, training=training)
        x = self.ln_final(x)
        return self.head(x)


def main():
    model = Model()
    dummy = tf.random.normal([B, S, D_MODEL])
    model(dummy)  # build
    n_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"# transformer: B={B} S={S} d={D_MODEL} H={N_HEADS} ff={D_FF} layers={N_LAYERS} params={n_params}")

    x = tf.random.normal([B, S, D_MODEL])
    labels = tf.constant([i % VOCAB for i in range(B * S)])
    opt = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            flat = tf.reshape(logits, [B * S, VOCAB])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=flat)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # warmup
    step().numpy()

    t0 = time.perf_counter()
    for _ in range(STEPS):
        step()
    step().numpy()  # force sync on last
    dt = time.perf_counter() - t0

    case = f"encoder_B{B}_S{S}_d{D_MODEL}_L{N_LAYERS}"
    emit(case, "sec", dt)
    emit(case, "tokens_per_sec", B * S * STEPS / dt)
    emit(case, "ms_per_step", dt * 1000.0 / STEPS)


if __name__ == "__main__":
    main()
