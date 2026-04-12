# GPU Backend Completion Plan (Phases B-H)

Phase A (MLP training on GPU) is done and working. This document details
the remaining phases for the agent that picks up this work. Each phase
is self-contained with exact file paths, line numbers, function names,
and code patterns. The CPU backend must NOT be touched — every change
is an `if (device != CPU) { gpu_path(); return; }` early-exit.

Current state: MNIST MLP trains end-to-end on GPU at 2.7x vs CPU.
cuBLAS GEMM with TF32, fused Adam kernel, device-aware cross-entropy,
broadcast gradient accumulation — all working. 21/21 CPU tests, 113/113
CUDA kernel tests passing.


## Phase B: CNN Training on GPU

### B1: CUDA col2im kernel

**File**: `src/compute/backends/cuda/ops_conv.cu`

The im2col kernel already exists (`k_im2col_sample` at line 29). The
inverse (col2im) is needed for conv backward's input gradient path.

**Kernel signature**:
```cuda
__global__ static void k_col2im_sample(
    const float *col, float *img,
    int64_t C_in, int64_t H, int64_t W,
    int kh, int kw, int sh, int sw, int ph, int pw,
    int64_t out_h, int64_t out_w)
```

**Algorithm**: one thread per col-matrix element. For each (r, c) in
col[K, M], compute the input position (channel, in_y, in_x) using the
same index math as im2col but in reverse. Use `atomicAdd` to scatter
into the output image since multiple col positions map to the same
input pixel (overlapping receptive fields).

**Grid**: same 2D layout as im2col: `dim3 block(16,16)`,
`dim3 grid((M+15)/16, (K+15)/16)`.

**Registration**: not a vtable slot — it's a static helper called from
the device-aware conv backward in conv.c.

### B2: Device-aware conv forward

**File**: `src/core/conv.c`, function `conv2d_forward` (line ~735)

**Pattern**: add at the TOP of the function, after input validation:
```c
if (inp->storage->device != AX_DEVICE_CPU) {
    return conv2d_forward_gpu(conv, inp, input, output, ...);
}
/* existing CPU path unchanged below */
```

`conv2d_forward_gpu` does per-sample:
1. Build `ax_conv_params_t` with `params.input` pointing into the GPU
   tensor's storage (already a device pointer — this is correct because
   `cuda_conv_gemm` in ops_conv.cu expects device pointers).
2. Call `ax_compute_conv_gemm(w2d, &params, res)` — already works on GPU
   (tested in test_cuda: 113/113).
3. Bias: call `ax_compute_bias_add(res, bias_tensor, 0, out_slice)` or
   manually broadcast-add using the existing `cuda_bias_add` kernel.
4. Copy result into output tensor's per-sample slice.

**Scratch**: the CPU path uses `struct ax_conv_scratch` with per-thread
buffers. For GPU, allocate a single `res` tensor on CUDA once and reuse
across samples (no threading on GPU — one sample at a time through the
CUDA stream). Use `ax_tensor_cuda_zeros` for allocation.

**Key detail**: the CPU path's `ensure_scratch` allocates CPU tensors
(`col_bufs`, `res_bufs`, `w2d`). The GPU path must NOT use these — it
needs its own GPU scratch. Either:
- Add `gpu_res` / `gpu_w2d` fields to `ax_conv_scratch`, OR
- Allocate locally in `conv2d_forward_gpu` and cache via a static.

The simpler approach: allocate GPU scratch locally. The persistent conv
col buffer in ops_conv.cu already caches the im2col scratch on GPU.

### B3: Device-aware conv backward

**File**: `src/core/conv.c`, function `conv2d_backward` (line ~470)

Same early-exit pattern. `conv2d_backward_gpu` does per-sample:

**Weight gradient (dW)**:
```
dW_sample = grad_out_sample @ col^T
```
Use `ax_compute_gemm_nt(go_gpu, col_gpu, dw_sample_gpu)`.
The `col_gpu` is the im2col output from forward (OR re-run im2col on
GPU via `k_im2col_sample`). The existing `cuda_conv_gemm` already runs
im2col internally — but for backward we need the col matrix separately.
Run im2col on GPU, then gemm_nt, then accumulate dW across samples.

**Input gradient (dX)**:
```
dcol = W^T @ grad_out_sample
img_grad = col2im(dcol)
```
Use `ax_compute_gemm_tn(w2d_gpu, go_gpu, dcol_gpu)` for the first
step. Then call `k_col2im_sample<<<...>>>(dcol_gpu, img_grad_gpu, ...)`
for the scatter-add into the input gradient.

**Bias gradient**:
`dbias[c] = sum over (n, h, w) of grad_out[n, c, h, w]`
Use `ax_compute_sum` with the appropriate axis. On a 4D grad_out
[N, C, H, W], summing over axes 0, 2, 3 gives [C]. The existing
CUDA sum kernel handles axis reduction.

**Accumulation**: weight gradient must be summed across samples.
Allocate a per-sample `dw_sample` on GPU, compute each sample's
contribution, then `ax_compute_axpy(dw_sample, 1.0f, weight->grad)`.


## Phase C: BatchNorm/LayerNorm on GPU

### C1: CUDA batchnorm forward kernel

**New file**: `src/compute/backends/cuda/ops_norm.cu`

**Algorithm**: 2-pass fused kernel, one block per feature channel.

Pass 1 (mean + variance): block-parallel reduction over batch*spatial
elements for channel c. Use shared memory + warp shuffle (same pattern
as `ops_reduce.cu`'s full reduction kernel).

Pass 2 (normalize + affine + running stats): for each element in
channel c, compute `x_hat = (x - mean) / sqrt(var + eps)`, then
`out = gamma * x_hat + beta`. Update running mean/var with exponential
moving average.

**Vtable slot** (add to `backend_ops.h`):
```c
ax_status_t (*batchnorm_fwd)(
    const ax_tensor_t *input,    /* [N, C, H, W] or [N, C] */
    const ax_tensor_t *gamma,    /* [C] */
    const ax_tensor_t *beta,     /* [C] */
    ax_tensor_t *running_mean,   /* [C], updated in-place */
    ax_tensor_t *running_var,    /* [C], updated in-place */
    float eps, float momentum, bool training,
    ax_tensor_t *output,         /* [N, C, H, W] */
    ax_tensor_t *save_mean,      /* [C], for backward */
    ax_tensor_t *save_invstd);   /* [C], for backward */
```

### C2: CUDA batchnorm backward kernel

**Same file**: `ops_norm.cu`

**Algorithm**: per-channel block-parallel. For each channel c:
1. Compute `dgamma = sum(grad_out * x_hat)` over batch*spatial
2. Compute `dbeta = sum(grad_out)` over batch*spatial
3. Compute `dx = (1/sqrt(var+eps)) * (grad_out - mean(grad_out) - x_hat * mean(grad_out * x_hat))`

All three use block-parallel reductions.

**Vtable slot**:
```c
ax_status_t (*batchnorm_bwd)(
    const ax_tensor_t *grad_out, const ax_tensor_t *input,
    const ax_tensor_t *save_mean, const ax_tensor_t *save_invstd,
    const ax_tensor_t *gamma,
    ax_tensor_t *grad_input, ax_tensor_t *dgamma, ax_tensor_t *dbeta);
```

### C3: Device check in norm.c

**File**: `src/core/norm.c`

In `batchnorm_forward` (line ~130) and the backward fn (line ~27), add:
```c
if (inp->storage->device != AX_DEVICE_CPU) {
    return batchnorm_forward_gpu(bn, inp, ...);
}
```

The GPU variant calls `ax_compute_batchnorm_fwd` / `bwd` which dispatch
to the CUDA kernels. CPU path is character-for-character unchanged.

**LayerNorm**: same pattern. One block per (batch, spatial) position
reducing over the feature dimension. Simpler than batchnorm because
there's no running stats.

### C4: CMakeLists.txt

Add `src/compute/backends/cuda/ops_norm.cu` to `AX_CUDA_SOURCES`.


## Phase D: Extended Activation Kernels

### D1: CUDA activation kernels

**New file**: `src/compute/backends/cuda/ops_activations.cu`

Forward kernels (one thread per element, same pattern as
`ops_elementwise.cu`'s UNOP_KERNEL macro):

| Activation | Formula |
|---|---|
| leaky_relu | `x > 0 ? x : alpha * x` |
| elu | `x > 0 ? x : alpha * (expf(x) - 1)` |
| selu | `lambda * (x > 0 ? x : alpha * (expf(x) - 1))` with hardcoded lambda=1.0507, alpha=1.6733 |
| gelu | `0.5 * x * (1 + tanhf(sqrt(2/pi) * (x + 0.044715 * x^3)))` |
| swish | `x / (1 + expf(-x))` |
| softplus | `x > 20 ? x : logf(1 + expf(x))` (numerically stable) |
| mish | `x * tanhf(softplus(x))` |

Backward kernels (each takes input, grad_out, writes grad_in):

| Activation | Derivative |
|---|---|
| leaky_relu_bwd | `grad * (x > 0 ? 1 : alpha)` |
| elu_bwd | `grad * (x > 0 ? 1 : output + alpha)` |
| selu_bwd | `grad * (x > 0 ? lambda : lambda * alpha * expf(x))` |
| gelu_bwd | `grad * (0.5 * (1 + tanh) + 0.5 * x * sech^2 * sqrt(2/pi) * (1 + 3*0.044715*x^2))` |
| swish_bwd | `grad * (sigmoid + x * sigmoid * (1 - sigmoid))` |
| softplus_bwd | `grad * sigmoid(x)` |
| mish_bwd | complex — precompute softplus, tanh, then combine |

### D2: Vtable slots

Add to `backend_ops.h` after the existing `tanh_op` slot:
```c
ax_status_t (*leaky_relu)(const ax_tensor_t *in, float alpha, ax_tensor_t *out);
ax_status_t (*elu_op)(const ax_tensor_t *in, float alpha, ax_tensor_t *out);
ax_status_t (*selu_op)(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t (*gelu_op)(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t (*swish_op)(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t (*softplus_op)(const ax_tensor_t *in, ax_tensor_t *out);
ax_status_t (*mish_op)(const ax_tensor_t *in, ax_tensor_t *out);
```

Add corresponding `ax_compute_*` dispatch wrappers in `compute.h` and
`dispatch.c`. CPU backends leave these NULL.

### D3: Device check in activations.c

**File**: `src/core/activations.c`

For each of the 7 activations, add at the top:
```c
ax_tensor_t *ax_leaky_relu(ax_tensor_t *a, float alpha) {
    if (!a) return NULL;
    if (a->storage->device != AX_DEVICE_CPU) {
        /* gpu dispatch */
        ax_tensor_t *out = ax_tensor_zeros(a->shape, a->ndim, a->dtype);
        if (!out) return NULL;
        ax_status_t s = ax_compute_leaky_relu(a, alpha, out);
        if (s != AX_OK) { ax_tensor_destroy(out); return NULL; }
        if (ax_grad_enabled() && a->requires_grad) {
            out->requires_grad = true;
            /* backward also dispatches — use a device-aware grad_fn */
            ax_grad_fn_t *gf = ax_grad_fn_create(leaky_relu_backward_device);
            gf->inputs[0] = a; gf->n_inputs = 1;
            gf->saved[0] = a; gf->saved_owned[0] = false;
            gf->n_saved = 1; gf->scalar_ctx = (double)alpha;
            out->grad_fn = gf;
        }
        return out;
    }
    /* existing CPU path unchanged */
    ...
}
```

The `*_backward_device` functions use `ax_compute_*_bwd` dispatch
instead of raw pointer loops. Pattern is identical to `ce_backward_device`
in losses.c (Phase A3).

### D4: CMakeLists.txt

Add `src/compute/backends/cuda/ops_activations.cu` to `AX_CUDA_SOURCES`.


## Phase E: Float4 Vectorized Elementwise

**File**: `src/compute/backends/cuda/ops_elementwise.cu`

### Approach

For unary ops where `n % 4 == 0` and both input and output offsets are
0 (aligned), launch a float4 kernel variant that processes 4 elements
per thread via a single 16-byte load + 16-byte store.

**Macro for vectorized unary kernel**:
```cuda
#define UNOP_KERNEL_VEC4(kname, expr)                                   \
__global__ static void kname##_vec4(                                    \
        const float4 *in, float4 *out, int64_t n4) {                   \
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;       \
    if (i >= n4) return;                                                \
    float4 val = in[i];                                                 \
    float4 res;                                                         \
    { float v = val.x; res.x = (expr); }                               \
    { float v = val.y; res.y = (expr); }                               \
    { float v = val.z; res.z = (expr); }                               \
    { float v = val.w; res.w = (expr); }                               \
    out[i] = res;                                                       \
}
```

**Dispatch in `run_unop`**: check `n % 4 == 0 && in->offset == 0 && out->offset == 0`,
dispatch to vec4 variant with grid `(n/4 + BLOCK-1)/BLOCK`, else
fall back to existing scalar kernel.

Same pattern for binary ops (same-shape, no broadcast).


## Phase F: Block-Parallel Axis Reductions

**File**: `src/compute/backends/cuda/ops_reduce.cu`

### Current state

Axis reduction (line ~78) uses one thread per output element with a
serial inner loop over the reduced dimension. This wastes warps for
large axis lengths (e.g., axis=1 on [batch, 4096]).

### Fix

Replace with one BLOCK per output element. Each block does a
collaborative parallel reduction over the axis dimension using shared
memory + warp shuffle (same algorithm as the full-reduction kernel at
line 34).

**Kernel signature**:
```cuda
__global__ static void k_reduce_axis_block(
    const float *in, float *out,
    int64_t outer, int64_t axis_len, int64_t inner,
    int op)  /* 0=sum, 1=max, 2=min */
```

**Grid**: `<<<outer * inner, AX_CUDA_BLOCK>>>` — one block per output
element. Each block's threads cooperatively reduce `axis_len` elements.

**Threshold**: use block-parallel when `axis_len >= 64` (enough work
per block to justify the shared-memory overhead). Below that, the
current one-thread-per-output kernel is fine.


## Phase G: Softmax Multi-Block

**File**: `src/compute/backends/cuda/ops_fused.cu`

### Current state

`k_softmax_row` (line ~47) uses one block per row with `AX_CUDA_BLOCK`
threads. Rows wider than `AX_SOFTMAX_ROW_MAX_COLS` (1024) return
`AX_ERR_NOT_IMPLEMENTED`.

### Fix: two-pass multi-block softmax

For rows wider than the block size:
1. Compute row max via a separate block-parallel reduction (reuse the
   axis-reduction kernel from Phase F with op=max).
2. Compute exp(x - max) and row sum via another block-parallel reduction.
3. Normalize: divide each element by the row sum.

Each step is a separate kernel launch. This handles arbitrary row
widths at the cost of 3 kernel launches instead of 1.

**Alternatively**: use cooperative groups to synchronize across
multiple blocks working on the same row. More complex but single-launch.

The 3-launch approach is simpler and still fast for the bandwidth-bound
softmax workload.

### Changes

Remove the `AX_SOFTMAX_ROW_MAX_COLS` limit. Add a `cols > AX_CUDA_BLOCK`
branch that dispatches to the multi-pass implementation. Keep the
single-block fast path for `cols <= AX_CUDA_BLOCK`.


## Phase H: Multi-Stream H2D/Compute Overlap

**File**: `src/compute/backends/cuda/backend.cu` + callers

### Design

Create a small pool of CUDA streams (e.g., 3: one for H2D, one for
compute, one for D2H). Assign streams to operations based on their type.

**Changes needed**:
1. `backend.cu`: allocate stream pool in `cuda_init_hook`, destroy in
   `cuda_shutdown_hook`. Add `ax_cuda_get_stream(int idx)`.
2. `memory.cu`: `cuda_memcpy_h2d_hook` and `cuda_memcpy_d2h_hook` use
   `cudaMemcpyAsync` on their assigned stream.
3. All kernel launches: add the stream parameter.
4. cuBLAS: `cublasSetStream(handle, stream)` before each gemm call.
5. Synchronization: insert `cudaStreamSynchronize` at points where the
   host reads back results (loss scalar read, eval accuracy read).

**Risk**: the current scratch arena is not stream-safe (multiple
concurrent ops might clobber each other's scratch). Either:
- Use one scratch arena per stream, OR
- Serialize scratch-using ops on a single stream.

The compute stream handles all kernel launches (serialized anyway on
one stream), so scratch is safe. H2D/D2H run on separate streams but
don't use scratch.

### Expected impact

On a training loop where each batch does: H2D(input) → forward →
backward → H2D(next_input), the H2D of the next batch can overlap with
the backward of the current batch. Saves ~0.1ms per batch at typical
sizes, or ~10-15% total.


## Build + Test Verification

After each phase, verify:
1. `cd build && cmake --build . && ctest` — 21/21 CPU tests unchanged
2. `cd build-cuda && cmake --build . && ./test_cuda` — 113+ CUDA tests
3. `./build-cuda/ax_mnist_gpu` — GPU MLP trains, accuracy ~93-94%
4. Phase B adds: GPU CNN training test (train mnist_cnn on gpu)
5. Phase C adds: GPU batchnorm correctness test
6. Phase D adds: GPU activation correctness tests


## File Summary

| Phase | Modified files | New files |
|---|---|---|
| B | `conv.c`, `ops_conv.cu` | — |
| C | `norm.c`, `backend_ops.h`, `compute.h`, `dispatch.c`, `backend.cu`, `internal.h` | `ops_norm.cu` |
| D | `activations.c`, `backend_ops.h`, `compute.h`, `dispatch.c`, `backend.cu`, `internal.h` | `ops_activations.cu` |
| E | `ops_elementwise.cu` | — |
| F | `ops_reduce.cu` | — |
| G | `ops_fused.cu` | — |
| H | `backend.cu`, `memory.cu`, `ops_gemm.cu`, `ops_conv.cu`, `ops_fused.cu` | — |


## Critical Constraint

**Nothing in the CPU hot path changes.** Every device-aware function in
core/ uses this exact pattern:

```c
if (tensor->storage->device != AX_DEVICE_CPU) {
    return gpu_variant(...);
}
/* character-for-character identical CPU code below */
```

The branch predictor learns "not taken" for CPU tensors in ~2 iterations.
The CPU backend stays at its current peak throughput.
