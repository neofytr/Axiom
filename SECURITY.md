# Axiom Security Policy

This document defines the security norms, conventions, and practices for the
Axiom codebase. Every contributor must follow these. No exceptions.


## 1. Memory Safety

**Rule: every allocation must have a matching free on every code path.**

- After `malloc`/`calloc`/`ax_tensor_create`/`ax_tensor_zeros`: check for NULL immediately.
- On error paths (early returns, gotos): free everything allocated before the error.
- Use `ax_graph_cleanup` after backward to free computation graph intermediates.
- In inference mode, sequential forward frees intermediates between layers.
- `ax_tensor_destroy` frees the tensor, its grad, its grad_fn (including owned saved
  tensors and ctx), and releases storage.
- `grad_fn->saved_owned[i]` marks tensors that the grad_fn is responsible for freeing.
  Set this when saving a tensor that was created specifically for backward (not a user tensor).
- `grad_fn->ctx_cleanup` is called to free the ctx struct if backward never ran.
  Always set this for malloc'd ctx structs.

**Rule: manual training loops MUST call ax_graph_cleanup.**

Composed ops (e.g., `ax_mse_loss` = sub + square + mean) create intermediate tensors
that live until `ax_graph_cleanup(loss)` is called. The `ax_model_train_step` function
handles this automatically. If you write a manual training loop:

```c
ax_optimizer_zero_grad(opt);
ax_tensor_t *pred = ax_layer_forward(net, input);
ax_tensor_t *loss = ax_mse_loss(pred, target);
ax_backward(loss);
ax_optimizer_step(opt);
ax_graph_cleanup(loss);   /* frees intermediates */
ax_tensor_destroy(loss);  /* frees the loss tensor itself */
```

Omitting `ax_graph_cleanup` causes memory to grow every training step.

**Rule: no use-after-free.**

- After destroying a tensor, never access it again.
- Autograd `saved[]` tensors must outlive the backward pass. Don't destroy inputs
  that backward still needs.
- `ax_graph_cleanup` only frees intermediates (tensors with grad_fn), never leaf
  tensors (parameters, user-created tensors).

**Validation: build with ASan + UBSan before every release.**

```
cmake .. -DCMAKE_BUILD_TYPE=Debug -DAX_SANITIZE=ON
make -j$(nproc)
ctest --output-on-failure
```

Zero leaks in examples. Zero errors in tests.


## 2. Integer Overflow Protection

**Rule: all shape arithmetic must use checked multiplication.**

The product of tensor dimensions can overflow int64_t. A tensor with shape
[INT64_MAX, 2] would silently overflow to a small number, allocate a tiny
buffer, and cause a heap buffer overflow on writes.

- Use `safe_mul_i64(a, b, &result)` which returns false on overflow.
- `compute_numel()` returns -1 on overflow; callers must check.
- `ax_tensor_create()` validates `numel * elem_size` doesn't overflow `size_t`.
- All serialization code validates shapes from files before using them.


## 3. Input Validation

**Rule: validate everything that comes from outside the library.**

External inputs: files (.axm, .axt, .csv), function arguments, data arrays.

For file parsing (`serialize.c`):
- Validate magic bytes
- Validate version number
- Validate dtype < AX_DTYPE_COUNT
- Validate ndim <= AX_MAX_DIMS
- Validate all shape dimensions > 0
- Validate numel doesn't overflow
- Validate numel * elem_size doesn't overflow size_t
- Validate n_layers <= AX_SEQ_MAX_LAYERS
- Validate n_params <= AX_LAYER_MAX_PARAMS
- Validate loaded tensor element count matches expected parameter size
- Check fread return values (detect truncated files)

For CSV parsing (`data.c`):
- Detect line truncation (line fills buffer without newline)
- Bounds-check field count against array size
- Bounds-check column indices against actual field count
- Bounds-check dataset index in get_item

For function arguments:
- NULL checks on all public API entry points
- Shape validation (positive dimensions, reasonable ndim)
- Dtype validation


## 4. Bounds Checking

**Rule: tensor element access is bounds-checked in debug builds.**

`ax_tensor_get_f32` and `ax_tensor_set_f32` check that each index is within
the shape when `AX_DEBUG` is defined or `NDEBUG` is not defined.

In release builds, bounds checks are compiled out for performance. This is the
standard approach (same as Python's `-O` flag removing asserts).


## 5. No Undefined Behavior

**Rule: no UB in any code path.**

- No type punning through incompatible pointer types. Use `void *ctx` in
  grad_fn instead of casting unrelated types.
- No signed integer overflow (it's UB in C). Use unsigned or checked arithmetic.
- No out-of-bounds array access (see bounds checking above).
- No null pointer dereference (validate before use).
- No use of uninitialized memory (`calloc` over `malloc` where practical).
- Build with `-fsanitize=undefined` to catch UB at runtime.


## 6. Stack Safety

**Rule: no unbounded recursion.**

The autograd backward pass uses iterative DFS with an explicit stack array,
not recursion. A deeply nested computation graph (thousands of chained ops)
would blow the call stack with recursive DFS.

`MAX_GRAPH_NODES` (4096) limits the graph size. Operations that could
produce unbounded recursion must be converted to iterative form.


## 7. Compiler Hardening

Always compile with:
- `-Wall -Wextra -Wpedantic` (catch mistakes)
- `-Wformat=2 -Wformat-security` (format string safety)
- `-Wshadow` (catch variable shadowing bugs)
- `-Wstrict-prototypes` (enforce proper C prototypes)
- `-fstack-protector-strong` (detect stack buffer overflows at runtime)
- `-D_FORTIFY_SOURCE=2` in release builds (runtime buffer overflow detection)


## 8. Thread Safety

**Current status: single-threaded only.**

Global mutable state exists in:
- `grad_tracking` (autograd.c) — modified by `ax_no_grad()` / `ax_enable_grad()`
- `rng_seeded` (tensor.c, init.c) — modified once on first use
- `active_ops` (dispatch.c) — set during init, read-only after

If multi-threading is added later:
- `grad_tracking` needs to be thread-local (`_Thread_local`)
- `active_ops` needs atomic access or per-thread dispatch
- Tensor storage refcount needs atomic increment/decrement


## 9. Secure Defaults

- No `exit()` calls in library code. Return error codes instead.
- No `printf` of user-controlled strings as format arguments.
- Error messages use `ax_err_set` with explicit format strings.
- All file operations check return values (fopen, fread, fwrite).
- Default alignment is 64 bytes (SIMD-ready, cache-friendly).
- Default arena block size is 1MB (reasonable for both desktop and embedded).


## 10. Testing Requirements

Every module must have:
- Unit tests for normal operation
- Edge case tests (empty tensors, single-element tensors, max dimensions)
- Error path tests (null inputs, invalid shapes, file not found)
- Gradient correctness tests (for differentiable operations)

The test suite runs under ASan + UBSan in CI. Tests that intentionally
leak memory (test code, not library code) suppress leak detection via
`ASAN_OPTIONS=detect_leaks=0`.

Examples run with full leak detection enabled and must show zero leaks.
