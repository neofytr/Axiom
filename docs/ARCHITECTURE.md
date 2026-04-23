# axiom — internal architecture

a guided tour of the codebase for new contributors. covers how a forward
pass actually flows through the machinery, what the backend abstraction
looks like, how memory is owned, how autograd records and replays
operations, and what changes when you add a new op.

axiom is a pure-c deep learning framework, ~27k loc, zero runtime
dependencies. every symbol below maps to real code in `src/` or
`include/`; file:line references are inlined where they help.

last updated against v0.10.0 (the first tagged pre-1.0 release; baselines
the public abi).

## 1. overview

a single forward+backward call from user code goes through five layers:

```
   user code
       |  ax_dense_create / ax_sequential_forward / ax_backward
       v
   layers              src/core/layer.c, src/core/{conv,attention}/
       |  build a grad_fn, call into ops
       v
   ops                 src/core/ops.c, src/core/autograd_ops.c
       |  record on the autograd tape, dispatch math
       v
   compute dispatch    src/compute/dispatch.c
       |  vtable lookup on the active backend
       v
   backend             src/compute/backends/{cpu_naive,cpu_opt,cuda}/
       |  isa-specialised kernels, optionally jit-emitted
       v
   kernel              opt_gemm / cuda kernels / jit-emitted micro-kernels
```

call chain example for a single dense layer:

```
ax_sequential_forward
  -> dense_forward                            src/core/layer.c:48
       -> ax_matmul_bias                      src/core/ops.c:295
            -> ax_compute_gemm                src/compute/dispatch.c:375
                 -> active_ops->gemm          (vtable slot)
                      -> opt_gemm             src/compute/backends/cpu_opt.c
                           -> jit micro-kernel src/compute/backends/jit_gemm_avx2.c
       -> attach grad_fn (matmul_bias_backward)
```

backward starts at the loss tensor, walks the recorded grad_fn graph in
reverse topological order, and dispatches input gradients through the
same `ax_compute_*` wrappers — the backend is unaware of forward vs
backward; both are just tensor ops.

the public api lives under `include/axiom/*.h` (master include is
`axiom/axiom.h`). internal-only contracts (vtable struct, parallelism
thresholds, cuda extension table) live under `include/axiom/internal/`
and carry no abi promise — see the conventions block at the top of
`include/axiom/axiom.h:1` for the full contract.

## 2. backend dispatch

the entire compute layer is one vtable. every backend fills in a
`struct ax_backend_ops` with function pointers for every op it
implements; null slots map to "not implemented" at the dispatch site.

### the vtable

`include/axiom/internal/backend_ops.h:30` defines `struct ax_backend_ops`.
~60 function-pointer slots organised by op family:

  - element-wise binary  — add, sub, mul, div_op
  - element-wise unary   — neg, abs_op, exp_op, log_op, sqrt_op, square
  - scalar               — add_scalar, mul_scalar
  - matrix               — gemm, gemm_nt, gemm_tn, gemm_relu, gemm_ex
  - fused                — add_relu, axpy, softmax_rowwise, bias_add
  - conv                 — conv_gemm (implicit-im2col fast path)
  - reductions           — sum, mean, max_op, min_op, argmax
  - comparisons          — equal, greater
  - data movement        — fill, copy
  - activations          — relu, sigmoid, tanh_op, leaky_relu, elu_op,
                           gelu_op, swish_op
  - fused optimiser      — adam_update, sgd_update
  - device-owner hooks   — device, init, shutdown, synchronize,
                           device_count, storage_alloc, storage_free,
                           memcpy_h2d, memcpy_d2h, memcpy_d2d

every op slot may be NULL. dispatch wrappers in `src/compute/dispatch.c`
check for null and either fall back, return `AX_ERR_NOT_IMPLEMENTED`,
or expose a `ax_compute_has_<op>()` predicate so callers can branch.

ownership of the vtable: backends export a `const ax_backend_ops_t`
symbol with file scope. `dispatch.c` declares the externs at
`src/compute/dispatch.c:34-56` and registers them at init.

### registration and selection

`ax_compute_init` (`src/compute/dispatch.c:108`) does five things:

  1. registers the cpu_naive vtable unconditionally (always available).
  2. registers a cpu_opt vtable. under `AX_CPU_ISA_DISPATCH` the cpu_opt
     source is compiled three times (avx-512, avx2, scalar) producing
     three distinct vtable symbols, and runtime probes
     `__builtin_cpu_supports("avx512f"|"avx2")` pick one
     (`dispatch.c:114-120`).
  3. registers cuda when `AX_HAVE_CUDA` is set.
  4. fires `register_device_owner` for each backend whose
     `ops->device != AX_DEVICE_COUNT`, populating a separate
     `device_backends[]` table (`dispatch.c:65,82-92`). this is how core
     code finds out who owns memory for `AX_DEVICE_CUDA` without
     `#ifdef`s.
  5. runs the autotune sequence: hybrid-cpu thread classification
     (`ax_autotune_threads`), omp fork/join overhead probe
     (`ax_calibrate_thresholds`), gemm tile sweep (opt-in via
     `AX_GEMM_CALIBRATE=1`), per-thread speed measurement.

the active backend is a single static pointer (`active_ops` at
`dispatch.c:70`). `ax_compute_set_backend(id)` overrides it
(`dispatch.c:226`). every dispatch wrapper begins with `ensure_compute_init()`
so user code that forgets to call `ax_init()` still works.

### the dispatch macro pattern

dispatch wrappers follow a uniform shape: validate, call, touch storage
generation. example (`dispatch.c:317-329`):

```c
#define DISPATCH_BINOP(op, a, b, out) \
    do { \
        ensure_compute_init(); \
        if (!active_ops) { ax_err_set(...); return AX_ERR_BACKEND; } \
        if (!active_ops->op) { ax_err_set(...); return AX_ERR_NOT_IMPLEMENTED; } \
        return dispatch_touch_on_ok((out), active_ops->op(a, b, out)); \
    } while (0)
```

the `dispatch_touch_on_ok` wrapper bumps `out->storage->generation` on
success (`dispatch.c:308`). this is how cpu_opt's `pack_b` cache (keyed
on `(storage_ptr, generation)`) detects in-place mutations between
calls and invalidates stale packed-b tiles.

### the cuda extension registry

cuda has ops the cpu vtable can't represent — sdpa fwd/bwd, head
interleave, batched conv gemm, qkv cache rebuilds, winograd. they live
in the cuda backend tu and their signatures don't fit the generic
backend_ops_t. cpu code paths still need to call them when running on
cuda tensors (e.g. `attention.c` running on a cuda input must dispatch
the layout transform to a cuda kernel).

before v0.10.0 this was a constellation of `__attribute__((weak))`
externs in cpu code, plus a `__attribute__((used))` force-link table in
`backend.cu` to keep them alive in the static archive. error-prone to
keep in sync.

K.4 replaced that with a single registry
(`include/axiom/internal/cuda_extension.h`):

  - `ax_cuda_extension_t` — struct of function pointers for every cuda-
    only op (10 entries: head interleave, sdpa fwd/bwd, qkv cache
    builds, conv gemm batched, winograd).
  - `ax_compute_register_cuda_extension(&table)` — called once from the
    cuda backend's `init` hook (`backend.cu:89`).
  - `ax_compute_get_cuda_extension()` — returns the registered table
    or NULL. cpu paths call this and dispatch through the struct.

clearing the registration with `NULL` is used by `ax_compute_shutdown`
for clean teardown (`dispatch.c:251`).

## 3. tensor lifecycle

axiom uses four distinct ownership patterns. picking the right one is
how you avoid both leaks and double-frees. the contract is documented
in `include/axiom/axiom.h:17-29` and `include/axiom/tensor.h:1-16`.

### pattern 1: heap

the default. `ax_tensor_create`, `ax_tensor_zeros`, `ax_tensor_ones`,
`ax_tensor_from_array`, every `ax_*_view`/`reshape`/`transpose` — all
return a heap tensor owned by the caller. release with
`ax_tensor_destroy`. storage is ref-counted (`ax_storage_t.refcount` at
`tensor.h:32`); `ax_tensor_destroy` decrements and frees when it hits
zero.

use for: parameters, persistent activations, anything outside a tight
forward/backward scope.

### pattern 2: arena

`ax_arena_t` (`include/axiom/memory.h:14-32`) is a chain of
bump-allocated blocks. `ax_arena_alloc(arena, size, align)` is a
pointer bump; `ax_arena_reset(arena)` retires every allocation in bulk
without freeing the underlying blocks; `ax_arena_destroy(arena)` frees
the blocks.

axiom keeps two thread-local arenas in the autograd module
(`src/core/autograd.c:20-49`):

  - `ax_backward_arena()` — reset at the end of every `ax_backward()`
    call. backward fns use it for input-gradient scratch that doesn't
    need to outlive the backward walk.
  - `ax_forward_arena()` — reset at `ax_graph_cleanup()`, which is
    called per training step *after* backward. survives the entire
    forward+backward of one step. used for tensors the forward pass
    must save for backward (e.g. batchnorm `x_hat` / `inv_std`).

`ax_tensor_arena_zeros(arena, ...)` (`tensor.h:110`) allocates the
metadata, the storage struct, and the data buffer all from the arena.
`ax_tensor_destroy` on it is a safe no-op. **never** store an arena
tensor in `grad_fn->saved[]` past the next reset boundary — it
becomes stale.

use for: short-lived per-call scratch in hot kernels. zero malloc
traffic per training step.

### pattern 3: pool

the persistent storage pool (managed inside `tensor.c`) recycles slots
across training steps. callers do not interact with it directly —
tensor lifecycle drives it. `ax_graph_cleanup` releases pool-resident
storage back to the slab so the next step's tensors can reuse it
without going through the system allocator.

use for: framework-internal recycling. user code rarely sees this.

### pattern 4: view

a view is a heap tensor that aliases another tensor's storage with
different shape/strides/offset. `ax_tensor_view`, `ax_tensor_reshape`
(when contiguous), `ax_tensor_transpose`, `ax_tensor_squeeze`,
`ax_tensor_unsqueeze` all produce views.

  - the view holds a refcount on the parent's `ax_storage_t`.
  - destroying the view drops one ref but does not free the data
    until the parent (and every other view) is also released.
  - the parent must outlive any use of the view's data; axiom does not
    detect dangling views.

use for: zero-copy reshape (e.g. `[B, S, K]` -> `[B*S, K]` before a
gemm — `ax_matmul_bias` does this at `src/core/ops.c:330`).

### the storage generation counter

`ax_storage_t.generation` (`tensor.h:41`) is bumped on every in-place
write via `ax_storage_touch`. caches keyed on a raw pointer use it to
detect mutations: cpu_opt's `pack_b` cache stores
`(storage_ptr, generation)` and invalidates when the generation
changes. the optimizer step rewrites weight buffers without changing
the pointer, so this is the only signal that the cache is stale.

## 4. autograd

reverse-mode autodiff via a dynamically-recorded tape. the tape is just
the chain of `ax_grad_fn_t` pointers attached to tensors via
`ax_tensor_t.grad_fn` (`tensor.h:63`).

### the grad_fn struct

`include/axiom/autograd.h:48-75`:

```c
struct ax_grad_fn {
    ax_backward_fn_t backward;       /* function pointer */
    ax_tensor_t *inputs[2];          /* parents in the graph */
    int n_inputs;
    ax_tensor_t *saved[4];           /* tensors saved for backward */
    bool saved_owned[4];             /* did we malloc them? */
    bool saved_retained[4];          /* did we ax_storage_retain them? */
    int n_saved;
    double scalar_ctx; int int_ctx;  /* op-specific scalars (axis, alpha, ...) */
    void *ctx; void (*ctx_cleanup)(void *);  /* opaque per-op extra context */
};
```

`AX_GRAD_MAX_INPUTS = 2` (binary ops are the worst case — most are
unary). `AX_GRAD_MAX_SAVED = 4`.

allocation: thread-local slab free-list (`autograd.c:66-97`). avoids
per-op calloc; freed nodes are pushed back onto the free-list and
reused on the next forward.

### forward recording

every differentiable op in `autograd_ops.c` follows the pattern:

  1. compute the forward result via `ax_compute_*`.
  2. if `ax_grad_enabled()` and any input requires grad:
     - allocate a grad_fn via `ax_grad_fn_create(backward_fn)`.
     - record `inputs[]` (raw pointers — no retain).
     - save what backward needs in `saved[]`. ownership rules:
       * if you `ax_tensor_zeros` a temp specifically for backward,
         set `saved_owned[i] = true` so cleanup destroys it.
       * if you save a graph node or user tensor, set
         `saved_retained[i] = true` and call
         `ax_storage_retain(t->storage)` — keeps the storage alive
         even if the original wrapper is destroyed early.
     - attach to output: `out->grad_fn = grad_fn; out->requires_grad = true`.

example: `matmul_bias` saves the lhs and rhs tensors so
`matmul_bias_backward` can compute `dX = dY @ W^T` and `dW = X^T @ dY`.

### backward walk

`ax_backward(loss)` (`autograd.c:325`):

  1. seeds `loss->grad = ones`.
  2. iterative dfs from `loss` down through `grad_fn->inputs`,
     producing reverse-topological order in `tl_order`. open-addressing
     hash set (`ptr_set_t` at `autograd.c:118`) gives O(1) visited
     lookups instead of O(n) linear scan.
  3. iterates `tl_order` from leaf-side to root, calling each node's
     `gf->backward(gf, node->grad)` exactly once. each backward
     accumulates input grads via `accumulate_grad` (`autograd_ops.c:59`),
     which handles broadcast reductions and the device-aware case
     (cuda backend uses `axpy`, cpu backend uses simd add).
  4. resets `backward_arena` in bulk.

### graph cleanup

`ax_graph_cleanup(loss)` (`autograd.c:386`) runs the same dfs to gather
the graph then does a two-pass cleanup:

  - pass 1: release saved-tensor refs, free `grad_fn` structs, mark
    intermediate tensor structs with a tombstone sentinel
    (`AX_GRAD_FN_TOMBSTONE = (void*)1` at `autograd.c:406`).
  - pass 2: destroy tombstoned intermediates. the two-pass split
    avoids a use-after-free where node A's storage is dropped before
    a later node B's saved-ref to A is released.

leaf tensors (parameters with no grad_fn) and the root tensor are not
destroyed — caller owns them. typical loop:

```c
ax_backward(loss);
ax_optimizer_step(opt);
ax_graph_cleanup(loss);   /* frees intermediates, detaches root */
ax_tensor_destroy(loss);  /* frees the root scalar */
```

### grad-disabled fast path

`ax_no_grad()` / `ax_enable_grad()` are nestable (depth counter at
`autograd.c:55-61`). `ax_grad_enabled()` is a per-thread bool. forward
ops branch on it; when grad is off, no `grad_fn` is allocated and the
forward call is just compute.

inference-only builds (`-DAX_INFERENCE_ONLY`) define `ax_grad_enabled`
as a static-inline `false` (`autograd.h:122`). the compiler
constant-folds the branch and dce's every grad-recording path. the
`losses.h` / `optim.h` / `data.h` / `lr_scheduler.h` headers and their
`.c` files are excluded from the build entirely (`axiom.h:93-98`).

## 5. cpu optimizations

`src/compute/backends/cpu_opt.c` is one large translation unit
(~5,500 loc) that ships every fast cpu path. the file has a TOC at the
top (`cpu_opt.c:1-34`) listing its 10 sections. a planned per-file
split is documented but not yet committed.

### the four isa flavours

under `-DAX_CPU_ISA_DISPATCH=ON` the same source compiles three times
with three different intrinsic widths:

  - `cpu_opt.avx512.o` — built with `-mavx512f -mfma`, exports
    `ax_cpu_opt_ops_avx512`.
  - `cpu_opt.avx2.o`   — built with `-mavx2 -mfma`, exports
    `ax_cpu_opt_ops_avx2`.
  - `cpu_opt.scalar.o` — built with no simd flags, exports
    `ax_cpu_opt_ops_scalar`.

the `AX_CPU_OPT_SUFFIX` macro (`cpu_opt.c:45-51`) suffixes every
externally-visible symbol so the three .o files coexist in one
archive. dispatch.c picks one at init time via `__builtin_cpu_supports`
(`dispatch.c:114-120`).

a fourth flavour is built unconditionally on aarch64: NEON intrinsics
selected via `simd_defs.h`. NEON does not need the runtime probe — the
compiler either targets aarch64 or it doesn't.

single-build mode (the default; ci uses dispatch mode) compiles
`cpu_opt.c` once with whatever `-march=native` inferred and exports
`ax_cpu_opt_ops` directly.

### jit micro-kernels

the gemm inner kernel is jit-emitted at runtime instead of compiled
ahead-of-time. rationale: the optimal kc value (k-block size) is shape-
and isa-specific, but baking it into a compile-time constant means
every off-by-one pad in the tail loop costs a branch mispredict.

  - `jit_x64.c` / `jit_x64.h` — minimal AVX2 instruction encoder
    (vfmadd231ps, vbroadcastss, vmovaps, vxorps + gpr arithmetic +
    bne label patching). emits raw machine bytes into an mmap'd RW
    page, then flips it to RX.
  - `jit_arm64.c` / `jit_arm64.h` — same shape for A64. emits NEON
    fmla-by-lane + gpr arithmetic + bne. uses
    `__builtin___clear_cache` (mandatory on aarch64; x86 is coherent).
  - `jit_gemm_avx2.c` — emits a 6x16 inner kernel specialised for the
    chosen kc. ~500 loc.
  - `jit_gemm_avx512.c` — 14x32 kernel for AVX-512. also includes the
    I.1.a strided-A variant for sdpa_bwd.
  - `jit_gemm_neon.c` — 8x12 kernel for aarch64. 146 loc.

emitted kernels are cached per (mc, nc, kc, isa) tuple in the cpu_opt
tu. the jit pages are leaked at process exit — they're persistent for
the whole run.

baremetal targets that lack mmap fall back to the (slower) generic
gemm via `simd_defs.h` macros; the jit path is gated on
`AX_HAVE_MMAP`.

### the auto-tuner

three calibration phases run from `ax_compute_init`:

  1. **omp fork/join overhead probe** (`dispatch.c:1055`,
     `ax_calibrate_thresholds`). measures the mean cost of entering
     and leaving an empty `#pragma omp parallel` block, then derives
     `ax_par_threshold_elems` / `_light` / `_heavy` / `_batch` /
     `_flops` (`compute_internal.h:39-48`). kernels with per-call
     work below the threshold skip omp; the fork-join would dominate.
  2. **hybrid cpu classification** (`dispatch.c:730`,
     `ax_autotune_threads`). on linux, classifies cores by sysfs
     `base_frequency` (or falls back to a 200 ms per-core probe loop).
     marks cores within 85% of the fastest as "fast"
     (`dispatch.c:662-702`). on hybrid systems (alder lake p+e, big.LITTLE),
     exports `ax_gemm_fast_threads` (p-cores) and `ax_gemm_all_threads`
     (p+e). large gemms use all; small ops use fast only — avoids
     omp barrier stalls where p-cores wait on slow e-cores. also
     deduplicates SMT siblings on machines with >= 4 physical cores
     (sharing fma units halves throughput).
  3. **gemm tile sweep** (opt-in via `AX_GEMM_CALIBRATE=1`,
     `dispatch.c:1135`). per-isa probe of (mc, nc, kc) candidates on
     a representative 1024^3 gemm and on mha-shaped probes; updates
     the per-isa GEMM_MC/NC/KC statics in cpu_opt.c. ~500 ms cost.
     a multi-shape probe table replaces the old single-shape sweep
     (commit 9bacfe0).

every calibration is a no-op under `AX_NO_AUTOTUNE` (the embedded
profile sets this) or when omp is off.

## 6. cuda backend

`src/compute/backends/cuda/` is one .h plus 11 .cu files (~3,500 loc).
the directory was split out of the previous monolithic `cuda_backend.cu`
in K.2/K.4.

### file map

  - `internal.h` — module prologue. std::atomic compat shims so c11
    `<stdatomic.h>` types compile under nvcc, then pulls every public
    axiom header, then declares cross-tu function externs.
  - `backend.cu` — vtable struct, lifecycle hooks (`cuda_init_hook`,
    `cuda_shutdown_hook`, `cuda_synchronize_hook`,
    `cuda_device_count_hook`), cublas handle, persistent device-side
    scratch arena (64 KB).
  - `memory.cu` — `storage_alloc` / `storage_free` /
    `memcpy_h2d` / `memcpy_d2h` / `memcpy_d2d`.
  - `ops_elementwise.cu` — add/sub/mul/div + unary + scalar.
  - `ops_gemm.cu` — gemm via cublasSgemm with TF32 enabled on sm_80+
    (set in `cuda_init_hook` at `backend.cu:74`).
  - `ops_reduce.cu` — sum/mean/max/min/argmax.
  - `ops_activations.cu` — relu/sigmoid/tanh/leaky_relu/elu/gelu/swish.
  - `ops_fused.cu` — gemm_relu, gemm_ex, add_relu, axpy, bias_add.
  - `ops_optim.cu` — adam_update / sgd_update fused on-device.
  - `ops_conv.cu` — conv_gemm_batched (one cublas stridedBatched per
    batch) + winograd dispatch heuristic.
  - `ops_winograd.cu` — Winograd F(2,3) for 3x3 stride-1 convs.
  - `ops_attention.cu` — sdpa_fwd / sdpa_bwd (FA-2 fused variant
    behind `AX_SDPA_FUSED=1`) + head_interleave / qkv split / qkv
    cache rebuilds.

### lifecycle

`cuda_init_hook` runs once on first dispatch (fired from
`register_device_owner` in `dispatch.c:88`). it:

  1. creates the cublas handle.
  2. enables TF32 tensor-core math on sm_80+ (~2x gemm at 19-bit
     mantissa, fp32-equivalent for training).
  3. allocates the device-side scratch arena.
  4. registers the cuda extension table via
     `ax_compute_register_cuda_extension(&ax_cuda_ext_table)`.

`cuda_shutdown_hook` clears the registration first (so any in-flight
cpu code falls back), then frees the scratch arena and destroys the
cublas handle.

### cuda-only ops

ops the cpu vtable can't represent (sdpa, layout transforms, batched
gemm, winograd, qkv cache builds) are reached through the cuda
extension table — see section 2 for the registry pattern. cpu code
that needs to dispatch on a cuda tensor calls
`ax_compute_get_cuda_extension()`, branches on NULL, then dispatches
through the returned struct.

## 7. adding a new op

minimum viable op contribution. example: a hypothetical `ax_compute_clip`
that clamps values to `[lo, hi]`.

  1. **declare in the public api**. add to `include/axiom/compute.h`:
     ```c
     ax_status_t ax_compute_clip(const ax_tensor_t *in, float lo, float hi,
                                  ax_tensor_t *out);
     int ax_compute_has_clip(void);
     ```
  2. **add the vtable slot**. in
     `include/axiom/internal/backend_ops.h`, add to the `ax_backend_ops`
     struct:
     ```c
     ax_status_t (*clip_op)(const ax_tensor_t *in, float lo, float hi,
                             ax_tensor_t *out);
     ```
     keep the slot optional (NULL means not implemented) unless every
     backend will support it.
  3. **implement the dispatch wrapper**. in `src/compute/dispatch.c`,
     follow the pattern of `ax_compute_axpy` (`dispatch.c:449`):
     validate active_ops, check the slot is non-null, call,
     `dispatch_touch_on_ok` the result. add an
     `ax_compute_has_clip()` predicate.
  4. **implement the cpu_naive reference**. in
     `src/compute/backends/cpu_naive.c`, write the obvious correct
     loop (no simd, no parallelism, no fast paths). add to the
     `ax_cpu_naive_ops` initialiser at the bottom of the file.
  5. **implement the cpu_opt fast path**. in
     `src/compute/backends/cpu_opt.c`, write a contiguous-fast-path
     version using the `ax_vf32_*` macros from `simd_defs.h`. fall
     back to the cpu_naive version for strided/broadcast cases. add
     to the `ax_cpu_opt_ops` vtable initialiser. remember the
     `AX_SYM(name)` macro if the symbol is externally visible — it
     suffixes for the multi-isa build.
  6. **(optional) implement cuda**. add a kernel + wrapper in
     `src/compute/backends/cuda/ops_<family>.cu`, declare the wrapper
     in `cuda/internal.h`, add to `ax_cuda_ops` in `backend.cu`.
  7. **write tests**. add to the relevant `tests/test_*.c` (probably
     `test_ops.c` or `test_compute.c`). cover: contiguous vs strided
     inputs, broadcast (if applicable), boundary values, the
     "no backend implements it" path. autograd ops also need a
     `ax_grad_check` numerical-vs-analytical comparison.
  8. **(optional) wire into autograd**. if the op is differentiable,
     add a `clip_backward` and `make_clip_grad_fn` in
     `src/core/autograd_ops.c`. expose at the ops layer in
     `src/core/ops.c`.

  9. **build + run ctest**. `cmake --build build && ctest --test-dir build`
     should still pass. for the multi-isa build, also run the
     `_avx512` / `_avx2` / `_scalar` ctest suites.

## v0.10.0 module splits

the K.1-K.5 phases (commit `216fcd5`) reorganised what was previously
a few oversized translation units:

  - **K.1**: cpu_opt.c gained a TOC and split-plan; the actual file
    split is staged for v0.11.0.
  - **K.2**: `src/core/conv.c` (3591 loc) split into `src/core/conv/`
    with eight files (`forward.c`, `backward.c`, `im2col.c`,
    `direct.c`, `winograd.c`, `path_selection.c`, `conv_bn_relu.c`,
    plus `internal.h`). `pool.c` was extracted earlier.
  - **K.3**: `src/core/attention.c` (1097 loc) split into
    `src/core/attention/` with `layout.c` (head interleave/split),
    `cache.c` (Wqkv + bqkv panel cache), and the layer thinned to
    just create / forward / backward / destroy.
  - **K.4**: cuda extension registry. removes 10 weak externs and
    the force-link table. see section 2 above.
  - **K.5**: documents the four tensor lifecycle patterns in
    `axiom.h` and `tensor.h`. see section 3 above.

the J.1-J.5 phases in the same commit hardened the public api: moved
internal-only declarations under `include/axiom/internal/`, normalised
header guards to `AX_<NAME>_H`, added `AX_VERSION_*` constants and the
`AX_DEPRECATED` / `AX_ABI_STABLE_SINCE` macros, and applied
`AX_RETURN_NULL_IF_ALLOC_FAIL` across every public `ax_*_create`
constructor.

## further reading

  - `include/axiom/axiom.h` — conventions block (errors, ownership,
    thread safety, naming, abi stability).
  - `CHANGELOG.md` — release notes; v0.10.0 establishes the abi
    baseline.
  - `docs/PERF_REPORT.md` — measured perf numbers per phase.
  - `docs/PRODUCTION_PLAN.md` — the J/K/L/M/N production hardening
    plan that produced v0.10.0.
  - `PERF_PLAN.md` — the I.* perf phases (sdpa fused kernel,
    winograd, JIT-emitted strided-A kernel).
  - `ROADMAP.md` — what's next.
