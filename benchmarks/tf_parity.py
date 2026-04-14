"""tf_parity.py — numerical parity checker.

loads the .axt inputs / outputs Axiom wrote in bench_parity.c, runs the
same op in TF, and reports max absolute difference per op.  gates at
1e-3 — anything larger is a numerical correctness regression.

axt format (from src/core/serialize.c):
  u32 magic (0x41585430 "AXT0")
  u32 dtype
  u32 ndim
  i64 shape[ndim]
  raw data
"""

import os, sys, struct
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

TENSOR_MAGIC = 0x41585430
AX_FLOAT32 = 0

SUITE = "parity"


def emit(case, metric, value):
    print(f"RESULT {SUITE} {case} {metric} {value:.6e}", flush=True)


def load_axt(path):
    with open(path, "rb") as f:
        magic, = struct.unpack("<I", f.read(4))
        if magic != TENSOR_MAGIC:
            raise ValueError(f"{path}: bad magic {magic:#x}")
        dtype, ndim = struct.unpack("<II", f.read(8))
        if dtype != AX_FLOAT32:
            raise ValueError(f"{path}: unsupported dtype {dtype}")
        shape = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
        n = 1
        for d in shape: n *= d
        raw = f.read(4 * n)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        return arr.copy()


def check(name, tf_out, ax_out):
    max_err = float(np.max(np.abs(tf_out - ax_out)))
    mean_err = float(np.mean(np.abs(tf_out - ax_out)))
    emit(name, "max_err", max_err)
    emit(name, "mean_err", mean_err)
    threshold = 1e-3
    status = "OK" if max_err < threshold else "FAIL"
    print(f"# parity {name}: max_err={max_err:.3e} mean={mean_err:.3e} {status}", flush=True)
    return max_err < threshold


def parity_matmul():
    A = load_axt("build/parity_matmul_in0.axt")
    B = load_axt("build/parity_matmul_in1.axt")
    ax_out = load_axt("build/parity_matmul_out.axt")
    tf_out = (A @ B).astype(np.float32)
    return check("matmul", tf_out, ax_out)


def parity_softmax():
    X = load_axt("build/parity_softmax_in0.axt")
    ax_out = load_axt("build/parity_softmax_out.axt")
    tf_out = tf.nn.softmax(X, axis=-1).numpy()
    return check("softmax", tf_out, ax_out)


def parity_gelu():
    X = load_axt("build/parity_gelu_in0.axt")
    ax_out = load_axt("build/parity_gelu_out.axt")
    # use the tanh-approx flavor — Axiom's ax_gelu uses the same one
    tf_out = tf.nn.gelu(X, approximate=True).numpy()
    return check("gelu", tf_out, ax_out)


def parity_layernorm():
    X = load_axt("build/parity_layernorm_in0.axt")
    ax_out = load_axt("build/parity_layernorm_out.axt")
    # match Axiom's default (gamma=1, beta=0, eps=1e-5) init
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    tf_out = (X - mean) / np.sqrt(var + 1e-5)
    return check("layernorm", tf_out, ax_out.astype(np.float32))


def parity_sdpa():
    Q = load_axt("build/parity_sdpa_in0.axt")
    K = load_axt("build/parity_sdpa_in1.axt")
    V = load_axt("build/parity_sdpa_in2.axt")
    ax_out = load_axt("build/parity_sdpa_out.axt")
    BH, S, dk = Q.shape
    scale = 1.0 / np.sqrt(dk)
    scores = np.einsum("bij,bkj->bik", Q, K) * scale
    mx = np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(scores - mx)
    p = exp / np.sum(exp, axis=-1, keepdims=True)
    tf_out = np.einsum("bik,bkj->bij", p, V).astype(np.float32)
    return check("sdpa", tf_out, ax_out)


def main():
    ok = True
    for fn in (parity_matmul, parity_softmax, parity_gelu, parity_layernorm, parity_sdpa):
        try:
            ok &= fn()
        except Exception as e:
            print(f"# parity error: {e}", file=sys.stderr)
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
