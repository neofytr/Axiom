# Axiom — Lesson Plan & Project Roadmap

Security-first. Embedded-efficient. Production-hardened. Open-source ready.

---

## Lesson Plan

Units 1–12 exist in `lessons/`. Units 13 onwards are the forward plan.

### Existing Units (lessons/01–12)

| Unit | Topic |
|------|-------|
| 1  | Vectors, Matrices, and Tensors |
| 2  | Calculus for Deep Learning |
| 3  | The Neuron |
| 4  | Loss Functions |
| 5  | Gradient Descent and Backpropagation |
| 6  | Optimizers |
| 7  | Weight Initialization |
| 8  | The Layer System and Model API |
| 9  | Code Map |
| 10 | Serialization and Data Pipeline |
| 11 | Convolutions |
| 12 | Security Hardening |

---

### Unit 13 — Broadcasting: The Hidden Engine

What it covers: how `src/core/broadcast.c` makes `add(tensor[64,1], tensor[1,10])` work.
Strides, implicit dimension expansion, the contiguous-copy rule. Why broadcasting is
essential for bias addition in every dense layer.

Why now: every lesson before this assumes it works. You need to understand why it works
before normalization (which broadcasts mean/variance over batch).

Codebase: `src/core/broadcast.c`, `ax_tensor_broadcast_to()`, stride rules in `tensor.c`.

---

### Unit 14 — The Full Training Workflow

What it covers: walk through `examples/regression.c` line by line. Full-batch vs mini-batch.
Why regression uses MSE + tanh and not cross-entropy + softmax. What "no dataloader needed"
means for memory. The eval pattern: `ax_layer_eval` -> `ax_no_grad` -> forward ->
`ax_layer_train` -> `ax_enable_grad`.

Why now: you've seen training loops in theory. Now you read a complete, working one
from first principles.

Codebase: `examples/regression.c`, `src/core/losses.c` (`ax_mse_loss`),
`src/core/layer.c` (eval/train mode).

---

### Unit 15 — Non-Linear Decision Boundaries (Spiral Example)

What it covers: why linear models fail on spirals. What a non-linearly separable dataset
looks like. The noise problem: why NOISE=0.12 causes overlapping arms and conflicting
gradients. The overfit sanity test methodology. Reading `examples/spiral.c`.

Why now: the spiral example is the most instructive debugging story in the project.
Understanding why it failed before it worked teaches you more than any working example.

Codebase: `examples/spiral.c`, `src/core/rng.c` (xoshiro256**, Box-Muller).

---

### Unit 16 — Normalization Layers

What it covers:
- BatchNorm: normalize per-feature across the batch, learn gamma/beta scale+shift.
  Why it stabilizes training (reduces internal covariate shift). Running mean/variance
  for inference mode. Bessel correction.
- LayerNorm: normalize per-sample across features. Why transformers use LayerNorm
  not BatchNorm.
- Dropout: randomly zero activations at rate p during training, scale by 1/(1-p).
  Why it regularizes (prevents co-adaptation).

Why now: these are in `src/core/norm.c` fully implemented. You cannot understand
ResNets or transformers without them.

Codebase: `src/core/norm.c`, `include/axiom/norm.h`, `tests/test_norm.c`.

---

### Unit 17 — Learning Rate Schedulers

What it covers: why constant LR is rarely optimal. Four scheduler types in
`src/core/lr_scheduler.c`:
- Step decay: halve LR every N epochs.
- Exponential decay: lr *= gamma each epoch.
- Cosine annealing: LR follows half-cosine from lr_max to lr_min.
- Warmup + cosine: linear ramp for first K steps, then cosine. Standard in large
  model training.

When to use each. How to wire `ax_lr_scheduler_step()` into the training loop.

Codebase: `src/core/lr_scheduler.c`, `include/axiom/lr_scheduler.h`,
`tests/test_lr_sched.c`.

---

### Unit 18 — The Compute Backend

What it covers: how `src/compute/dispatch.c` selects between `cpu_naive.c` and
`cpu_opt.c` at runtime. What SIMD means (AVX2: 8 floats per instruction vs 1).
Why the same high-level API works on both. How to add a new backend. The
`AX_BACKEND_CPU_SIMD` flag.

Why now: the optimized backend (cpu_opt.c) is being built now. Understanding the
dispatch layer lets you read and contribute to that work.

Codebase: `src/compute/dispatch.c`, `src/compute/backends/cpu_naive.c`,
`src/compute/backends/cpu_opt.c`, `include/axiom/compute.h`.

---

### Unit 19 — Convolutional Networks (Deep Dive)

What it covers: beyond the intro in Unit 11. Receptive fields, stride, padding,
dilation. Why convolution is translation-equivariant. MaxPool vs AvgPool. The NCHW
memory layout. Reading the CNN integration test.

Codebase: `src/core/conv.c`, `tests/test_integration.c` (CNN path),
`tests/test_conv.c`.

---

### Unit 20 — Serialization and Deployment

What it covers: how `src/core/serialize.c` saves and loads model weights. The v3
format with buffers. Why versioned binary formats matter for long-lived models.
Security considerations: never deserialize untrusted data without bounds checking —
how the library enforces this with 14 specific validation checks.

Codebase: `src/core/serialize.c`, `tests/test_serialize_all.c`.

---

### Unit 21 — The Attention Mechanism

What it covers: scaled dot-product attention from first principles. Q, K, V matrices.
Why `1/sqrt(d_k)` scaling prevents gradient vanishing in softmax. Multi-head attention.
The connection to memory retrieval (query looks up keys, retrieves values).

Status: not yet implemented — this unit ships when `ax_attention_layer_create()` lands.

---

### Unit 22 — Transformer Architecture

What it covers: building a transformer block — attention + LayerNorm + FFN + residual
connections. Why residual connections allow very deep networks. Position encodings.
The GPT-style (decoder-only) vs BERT-style (encoder) distinction.

Status: depends on Unit 21.

---

### Unit 23 — Quantization and Efficiency

What it covers: FP16 vs INT8 quantization. Post-training quantization (PTQ) vs
quantization-aware training (QAT). Why embedded targets need INT8 (memory bandwidth,
no FPU on some MCUs). The accuracy-efficiency tradeoff.

Status: ships when quantization lands in the library.

---

### Unit 24 — Security Hardening for Open Source

What it covers: the full security design of the library. `safe_mul_i64` prevents
integer overflow in shape math. Iterative DFS prevents stack overflow on deep graphs.
No `exit()` anywhere — the library never terminates the caller's process. ASan+UBSan
clean. `_FORTIFY_SOURCE=2` and stack protector in release builds. How to audit a C
library before open-sourcing. CVE-class bugs in ML libraries (pickle deserialization
in Python torch, shape confusion in TensorFlow).

Codebase: `src/core/error.c`, `src/core/memory.c`, `CMakeLists.txt` (sanitizer
flags), `src/core/tensor.c` (`safe_mul_i64`).

---

### Unit 25 — Embedded Deployment

What it covers: running inference-only on a microcontroller. Stripping autograd,
eliminating heap allocation with arena allocators, ROM-ing weights via `const` arrays,
static tensor descriptors. The embedded build: no mmap, no posix_memalign,
`AXIOM_EMBEDDED` define to cut dead code. Stack and RAM budgets for STM32/RP2040
class devices.

Status: ships when the embedded build target lands.

---

## Project Roadmap

### Phase 1 — Backend Integration (current)

- `cpu_opt.c`: AVX2 GEMM, vectorized activation functions, fused ops for inference
- Runtime dispatch: CPUID detection selects backend at startup, zero API change
- Benchmark suite: `examples/bench_matmul.c`, throughput in GFLOPS vs naive baseline
- Milestone: 4-8x speedup on dense layers vs cpu_naive on x86-64

### Phase 2 — Attention + Transformer

- `ax_attention_layer_create(n_heads, d_model, d_k, causal)` — scaled dot-product,
  multi-head
- `ax_transformer_block_create()` — attention + LayerNorm + FFN + residuals
- Causal masking for autoregressive models
- New example: character-level language model on tiny dataset
- Milestone: GPT-nano trains on Shakespeare, generates coherent text

### Phase 3 — Production Hardening for Open Source

- Fuzzing harness: `tests/fuzz_tensor.c`, `tests/fuzz_serialize.c` — libFuzzer
  targets for all public-facing parsing (serialization, data loading)
- Sanitizer CI: every PR runs `-fsanitize=address,undefined` and
  `-fsanitize=memory` (separate run)
- No-undefined-behavior audit: all integer arithmetic through `safe_mul_i64`
  pattern, all pointer arithmetic bounds-checked
- Documentation pass: API reference (Doxygen), contribution guide
- License selection: MIT or Apache 2.0 (both allow embedded commercial use without
  copyleft)
- Milestone: zero sanitizer findings, fuzzer runs 1M iterations without crash

### Phase 4 — Embedded Targets

- `AXIOM_EMBEDDED` compile flag: disables mmap, heap allocators, file I/O,
  posix_memalign
- Static arena allocator: caller provides a byte buffer, library uses it
  exclusively — zero malloc
- ROM-able weights: `ax_model_from_const_array(const uint8_t *data, size_t len)`
  for flash-stored models
- Minimal footprint build: strip autograd, LR schedulers, data loaders ->
  inference-only ~15-30KB code size
- Port test: RP2040 (264KB RAM) running MNIST inference at >95% accuracy
- Milestone: MNIST inference on bare-metal RP2040, no OS, no heap, from flash

### Phase 5 — Quantization

- FP16 storage type (`AX_FLOAT16`): half-precision weights, FP32 accumulation
- INT8 post-training quantization: per-channel scale+zero-point, calibration pass
- Quantization-aware training hooks: fake-quantize nodes in forward pass
- New backend: `cpu_opt_int8.c` with VNNI/SDOT where available
- Milestone: MobileNet-class model, INT8, <2% accuracy drop vs FP32, 4x smaller
  weights

### Phase 6 — General Market Expansion

- Python bindings (ctypes — no C extension module, keeps embedded build clean)
- CUDA backend: `ax_backend_cuda.c`, cuBLAS for GEMM, custom kernels for
  activations
- Device placement: `AX_DEVICE_CPU` vs `AX_DEVICE_CUDA`
- Model format: versioned `.axm`, cryptographically signed (SHA-256 manifest)
- Milestone: Python `import axiom` installs via pip, trains CIFAR-10 ResNet-20
  to >91%

---

## Priority rationale

Security and embedded come first. Once open-sourced, a CVE in a C library is
highly visible and hard to recover from. Embedded API decisions made late are
breaking changes. The ordering above puts Phase 3 (hardening) and Phase 4
(embedded) before GPU/Python (Phase 6). The optimized CPU backend (Phase 1) is
already in progress and unblocks everything else — SIMD-optimized GEMM is the
foundation that makes quantization, transformers, and benchmark comparisons
meaningful.
