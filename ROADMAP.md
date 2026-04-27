# Axiom Roadmap

Core values: security, zero memory leaks, best general performance without per-arch tuning.

## v0.2 — Hardening

- [ ] Fuzz testing for all public APIs (tensor create/destroy, serialization parse)
- [ ] Valgrind/ASan CI gate — zero leaks, zero UB on every commit
- [ ] Bounds-checked storage access in debug builds
- [ ] Integer overflow guards on shape arithmetic (size_t wraps on large tensors)
- [ ] Secure model format: magic number validation, version field, checksum
- [ ] Stack buffer overflow protection in conv scratch allocation
- [ ] Thread-safety audit: verify all TLS usage, no data races under TSan

## v0.3 — Compute

- [ ] Multi-head attention layer (needed for transformers)
- [ ] Depthwise separable conv (needed for MobileNet-class models on embedded)
- [ ] Dilated convolution
- [ ] Transposed convolution (decoders, generators)
- [ ] GroupNorm (works with any batch size, better than BatchNorm for small batches)
- [ ] Residual connection helper (skip connections without manual tensor management)

## v0.4 — Training

- [ ] Gradient accumulation (effective batch > memory batch)
- [ ] Mixed precision training (FP16 storage, FP32 accumulation on GPU)
- [ ] ONNX export (interop with other inference runtimes)
- [ ] Checkpoint/resume (save optimizer state + LR scheduler state alongside weights)
- [ ] Early stopping callback
- [ ] Training metrics logging (loss/accuracy history accessible from C)

## v0.5 — GPU

- [ ] Multi-stream pipeline (H2D/compute/D2H overlap, ~15% throughput boost)
- [ ] CUDA graph capture for training step (eliminate per-op launch overhead)
- [ ] cuDNN conv integration (faster than im2col+cuBLAS for large convolutions)
- [ ] Tensor core FP16 GEMM (significant speedup on Ampere+)
- [ ] GPU memory pool tuning (per-stream arenas, reduce fragmentation)

## v0.6 — Embedded & Quantization

- [ ] INT8 post-training quantization with calibration
- [ ] Weight pruning (structured, for inference acceleration)
- [ ] CMSIS-NN backend for Cortex-M
- [ ] Binary model compression (smaller flash footprint)
- [ ] Fixed-point inference path (no FPU required)

## Ongoing (every release)

- Benchmark against TF on CPU (x86 + ARM) and GPU, track regressions
- Zero memory leaks verified by CI (ASan + Valgrind)
- All tests pass on x86 (AVX-512), ARM (NEON), and CUDA
- Documentation site updated with new APIs and architecture changes
- No per-microarchitecture tuning — all optimizations must be ISA-general
