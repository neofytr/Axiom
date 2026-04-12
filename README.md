<p align="center">
  <h1 align="center">Axiom</h1>
  <p align="center">
    A deep learning framework built from scratch in C.<br>
    No Python. No BLAS. No dependencies. Faster than TensorFlow.
  </p>
</p>

<p align="center">
  <a href="#performance"><img src="https://img.shields.io/badge/vs_TensorFlow-1.21x_faster-brightgreen" alt="Performance"></a>
  <a href="#building"><img src="https://img.shields.io/badge/dependencies-zero-blue" alt="Zero deps"></a>
  <a href="#build-profiles"><img src="https://img.shields.io/badge/binary-under_100KB-orange" alt="Small binary"></a>
  <a href="#testing"><img src="https://img.shields.io/badge/tests-22_binaries-informational" alt="Tests"></a>
</p>

---

Axiom started as a question: how fast can a neural network train if you strip away every abstraction layer? Turns out, faster than TensorFlow on the same hardware. No hand-tuned assembly, no vendor math libraries. Just C, SIMD intrinsics, and careful engineering.

## Performance

Benchmarked head-to-head against TensorFlow (oneDNN/MKL backend) on identical CI hardware. Same model, same data, same thread count. No cherry-picking.

### x86_64 (AMD EPYC 9V74, 4 threads)

```
Model: 784 -> 4096 -> 2048 -> 512 -> 10   (12.6M parameters)
Dataset: MNIST 60K, batch 256, 2 epochs

  Axiom        35.0s
  TensorFlow   42.4s
                        ─── Axiom is 1.21x faster ───
```

### ARM (Graviton3 Neoverse-N2, 4 threads)

```
Model: 784 -> 1024 -> 512 -> 256 -> 10   (1.46M parameters)
Dataset: MNIST 60K, batch 256, 5 epochs

  Axiom        3.10s
  TensorFlow   3.76s
                        ─── Axiom is 1.21x faster ───
```

The advantage grows with model size. At 1.46M params on x86 the two frameworks are neck-and-neck; at 12.6M params Axiom pulls ahead by 21%. This is because Axiom's tiled GEMM kernels have better cache utilization on large matrices where TensorFlow's overhead starts to show.

## Features

```
Tensors          arbitrary dims, views, slicing, broadcasting
Autograd         reverse-mode, computation graph, automatic cleanup
Layers           Dense, Conv2D, BatchNorm, LayerNorm, Dropout,
                 MaxPool, AvgPool, GlobalAvgPool, Flatten
Activations      ReLU, Sigmoid, Tanh, GELU, Swish, LeakyReLU, ELU, Softmax
Losses           MSE, Cross-Entropy
Optimizers       SGD (momentum + nesterov), Adam, AdamW, RMSprop, Adagrad
LR Schedulers    Step decay, Exponential, Cosine annealing, Warmup + Cosine
Serialization    compact binary format, model save/load
Data             batching, shuffling, train/test splits
GPU              CUDA backend with cuBLAS, fused kernels, explicit device memory
```

## Quick Start

```c
#include "axiom/axiom.h"

int main(void) {
    ax_init();

    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(784, 128, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(128, 10, true));

    ax_model_t *model = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(
        model->params, model->n_params,
        1e-3f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(model, opt, ax_cross_entropy_loss);

    for (int i = 0; i < 1000; i++)
        ax_model_train_step(model, train_x, train_y);

    ax_model_save(model, "model.axm");
    ax_model_destroy(model);
    ax_shutdown();
}
```

See `examples/` for complete programs: XOR, MNIST (MLP + CNN), deep MLP.

## Building

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### Build Profiles

| Profile | Flag | Use case |
|---------|------|----------|
| **Desktop** | _(default)_ | Full framework, OpenMP, all features |
| **ISA Dispatch** | `-DAX_CPU_ISA_DISPATCH=ON` | Builds AVX-512 + AVX2 + scalar, picks best at runtime |
| **CUDA** | `-DAX_CUDA=ON` | GPU backend (needs CUDA toolkit) |
| **Inference Only** | `-DAX_INFERENCE_ONLY=ON` | Strips training code, under 100KB on ARM |
| **Embedded Linux** | `-DAX_PROFILE=embedded-linux` | Smaller buffers, no unnecessary features |
| **Baremetal** | `-DAX_PROFILE=embedded-baremetal` | No stdio, no threads, no dynamic allocation |

## Why It's Fast

No magic. Just doing the obvious things without ten layers of abstraction in the way.

**Compute**
- BLIS-style tiled GEMM with architecture-specific micro-kernels: 14x32 (AVX-512), 6x16 (AVX2), 8x12 (NEON)
- Fused Dense+ReLU: one kernel call, output stays cache-hot through matmul and activation
- Runtime ISA dispatch: builds multiple SIMD variants, picks the fastest at startup
- Cache auto-tuning: reads L1d/L2 sizes from the OS, adjusts tile dimensions to fit

**Autograd**
- Direct-write backward: fresh gradients are written directly, no temp allocation + accumulate
- Pack-b caching with generation counter invalidation across batches
- Thread-local slab allocator for grad_fn nodes (eliminates per-op malloc)

**Threading**
- Adaptive JC tiling: auto-shrinks tile width when tiles < threads to eliminate idle cores
- HT/SMT dedup: detects hyperthreading siblings, pins to physical cores
- Per-thread pack buffers: no lock contention in the GEMM hot path

All general-purpose. No microarchitecture-specific tuning. The same binary runs on Zen3, Ice Lake, Graviton, or a Raspberry Pi.

## Project Structure

```
include/axiom/       public API headers
src/core/            tensors, autograd, ops, layers, optimizers, losses
src/compute/         dispatch layer + backends
  backends/          cpu_naive.c, cpu_opt.c (SIMD), simd_defs.h, cuda/
tests/               22 test binaries, ~5700 lines
examples/            xor, mnist, mnist_cnn, deep_mlp
benchmarks/          axiom vs tensorflow comparisons
```

~19K lines of C total. The whole framework.

## Testing

22 test binaries, CI on x86_64 (AVX-512), ARM64 (NEON), with ASan and inference-only builds.

Tests cover: tensor ops and broadcasting, every activation forward + backward, every optimizer on XOR, SIMD vs naive cross-checks, conv/batchnorm/layernorm/dropout gradient flow, large model stress (up to 12.6M params), fused op correctness, numerical gradient verification (finite differences vs analytical), odd-dimension edge cases, memory pressure, model serialization roundtrip.

## How It Differs From TensorFlow / PyTorch

Axiom is not trying to replace TensorFlow. Different tool, different constraints.

| | Axiom | TensorFlow |
|---|---|---|
| **Dependencies** | None. One C library. | Python, protobuf, BLAS, 500MB+ install |
| **Binary size** | < 100KB (inference, ARM) | Hundreds of MB |
| **Embedded** | Baremetal Cortex-M, no heap | Not an option |
| **Ops** | ~30 core ops | Thousands |
| **Distributed** | Single machine | Multi-node, TPU pods |
| **Ecosystem** | Just the framework | Massive |

If you need ResNet-152 on a TPU pod, use TensorFlow. If you need a 100KB inference engine on a microcontroller, or you want to understand exactly what your training loop is doing down to the SIMD instruction, Axiom is the tool.

## Roadmap

- [ ] INT8 quantization with calibration
- [ ] Attention / transformer layers
- [ ] ONNX model import
- [ ] Multi-GPU data parallelism
- [ ] Depthwise separable and dilated convolutions
- [ ] RISC-V vector extension support

## License

MIT
