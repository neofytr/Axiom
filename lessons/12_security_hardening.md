# Unit 12: Security Hardening

## Why This Matters

A neural network library handles untrusted data: model files from the internet,
CSV files from users, adversarial inputs designed to crash your program. A single
unchecked integer overflow can turn a shape mismatch into a heap buffer overflow.
This unit covers every security measure in the Axiom codebase and explains the
threat model behind each one.


## 12.1 Threat Model

What can go wrong in a C-based ML library?

1. **Malicious model files**: a crafted .axm file with extreme shape values
   could trigger integer overflow, allocate a tiny buffer, and cause heap corruption.
2. **Malicious data files**: a CSV with extremely long lines or unexpected field
   counts could cause buffer overflows or out-of-bounds reads.
3. **API misuse**: null pointers, invalid shapes, wrong dtypes passed to functions.
4. **Memory leaks**: training loops that don't clean up computation graphs.
5. **Stack overflow**: deeply nested computation graphs processed recursively.

Axiom defends against all of these.


## 12.2 Integer Overflow Protection

### The Attack

Consider a tensor with shape `[INT64_MAX, 2]`. The product overflows:

    INT64_MAX * 2 = -2  (signed overflow, undefined behavior in C!)

If you pass this to `malloc(-2)` wrapped as `size_t`, it becomes a huge allocation
that fails. But if the overflow wraps to a small positive number, malloc succeeds
with a tiny buffer, and subsequent writes corrupt the heap.

### The Defense

Every shape multiplication in Axiom uses checked arithmetic:

```c
static bool safe_mul_i64(int64_t a, int64_t b, int64_t *result) {
    if (a == 0 || b == 0) { *result = 0; return true; }
    if (a > 0 && b > 0 && a > INT64_MAX / b) return false;     // positive overflow
    if (a > 0 && b < 0 && b < INT64_MIN / a) return false;     // negative overflow
    if (a < 0 && b > 0 && a < INT64_MIN / b) return false;     // negative overflow
    if (a < 0 && b < 0 && a < INT64_MAX / b) return false;     // positive overflow
    *result = a * b;
    return true;
}
```

`compute_numel` uses this for every dimension multiplication:

```c
static int64_t compute_numel(const int64_t *shape, int ndim) {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return -1;          // non-positive dimension
        if (!safe_mul_i64(n, shape[i], &n))
            return -1;                          // overflow
    }
    return n;
}
```

And `ax_tensor_create` also checks the `numel * elem_size` multiplication:

```c
if (elem_size > 0 && (size_t)n > SIZE_MAX / elem_size) {
    ax_err_set(AX_ERR_INVALID_SHAPE,
               "allocation size overflow: %ld elements * %zu bytes", n, elem_size);
    free(t);
    return NULL;
}
```

### Where Overflow Checks Appear

- `tensor.c`: `compute_numel`, `compute_strides`, `ax_tensor_create`
- `serialize.c`: `ser_safe_mul` for file-parsed shapes, `numel * dtype_size` check
- `data.c`: `n_samples * sizeof(int64_t)` check in dataloader creation
- `memory.c`: `size + alignment + sizeof(void *)` overflow check in `ax_aligned_alloc`,
  `min_size + sizeof(ax_arena_block_t)` check in `arena_block_create`


## 12.3 Bounds Checking

### Tensor Element Access

`ax_tensor_get_f32` and `ax_tensor_set_f32` validate every index:

```c
float ax_tensor_get_f32(const ax_tensor_t *t, const int64_t *indices) {
    for (int i = 0; i < t->ndim; i++) {
        if (indices[i] < 0 || indices[i] >= t->shape[i]) {
            ax_err_set(AX_ERR_OUT_OF_BOUNDS,
                       "get_f32: index %ld out of bounds for dim %d (size %ld)",
                       indices[i], i, t->shape[i]);
            return 0.0f;
        }
    }
    // ... compute offset and access
}
```

This is always-on (not gated by NDEBUG) because the cost of a few comparisons
is negligible compared to the cost of a heap corruption bug.

### Dataset Index Access

Both tensor and CSV datasets validate indices in `get_item`:

```c
if (idx < 0 || idx >= ds->n_samples) {
    ax_err_set(AX_ERR_OUT_OF_BOUNDS,
               "dataset index %ld out of range [0, %ld)", idx, ds->n_samples);
    *input = NULL;
    *target = NULL;
    return;
}
```

### Shape Dimension Validation

`tensor_alloc_meta` validates all shape dimensions are positive:

```c
for (int i = 0; i < ndim; i++) {
    if (shape[i] <= 0) {
        ax_err_set(AX_ERR_INVALID_SHAPE,
                   "shape dimension %d must be positive, got %ld", i, shape[i]);
        return NULL;
    }
}
```


## 12.4 File Parsing Validation

### Model Files (.axm)

The model loader validates every field read from the file:

1. **Magic bytes**: `if (magic != AX_MAGIC)` -- rejects non-Axiom files.

2. **Version**: `if (version > AX_FORMAT_VERSION)` -- rejects files from
   newer versions that might have incompatible format changes.

3. **Layer count**: `if (n_layers == 0 || n_layers > AX_SEQ_MAX_LAYERS)` --
   prevents zero-layer models and allocation of enormous layer arrays.

4. **Layer type**: the switch statement in model load has a `default` case that
   rejects unknown layer types.

5. **Params per layer**: `if (descs[i].n_params > AX_LAYER_MAX_PARAMS)` --
   prevents reading past the fixed-size params array.

6. **Feature dimensions**: for dense layers, validates `in_f > 0 && out_f > 0`.

7. **Tensor metadata from file**: delegates to `read_tensor` which validates:
   - `dtype < AX_DTYPE_COUNT`
   - `ndim <= AX_MAX_DIMS`
   - Every shape dimension > 0
   - No overflow in numel computation
   - No overflow in `numel * dtype_size`

8. **Parameter shape match**: after loading, compares element count of loaded
   tensor vs expected parameter shape:
   ```c
   if (n_existing != n_loaded) {
       ax_err_set(AX_ERR_SHAPE_MISMATCH,
                  "layer %u param %d: expected %ld elements, file has %ld",
                  i, p, n_existing, n_loaded);
   }
   ```

9. **fread return values**: every `fread`/`fwrite` return value is checked.
   A truncated file is detected immediately.

### Tensor Files (.axt)

Same validation as model tensors, plus magic byte check (`TENSOR_MAGIC = 0x41585430`).

### CSV Files

1. **Line truncation**: if a line fills the buffer without a newline character,
   it's detected and skipped:
   ```c
   if (line_len == CSV_LINE_MAX - 1 && line[line_len - 1] != '\n') {
       ax_err_set(AX_ERR_INTERNAL, "csv line %ld exceeds maximum length");
       // Skip rest of truncated line
       while ((ch = fgetc(f)) != '\n' && ch != EOF) {}
       continue;
   }
   ```

2. **Field count limit**: parsing stops at `CSV_FIELDS_MAX` (256) fields per line.

3. **Column index validation**: before accessing `fields[col]`, checks
   `col >= 0 && col < n_fields`.


## 12.5 Null Pointer Protection

Every public API entry point validates its arguments:

```c
ax_status_t ax_tensor_save(ax_tensor_t *t, const char *path) {
    if (!t || !path) {
        ax_err_set(AX_ERR_NULL_ARG, "null tensor or path");
        return AX_ERR_NULL_ARG;
    }
    // ...
}

ax_status_t ax_model_save(ax_model_t *model, const char *path) {
    if (!model || !model->net || !path) {
        ax_err_set(AX_ERR_NULL_ARG, "null model or path");
        return AX_ERR_NULL_ARG;
    }
    // ...
}
```

Destroy functions are null-safe:
```c
void ax_tensor_destroy(ax_tensor_t *t) {
    if (!t) return;
    // ...
}
```

This prevents crashes from double-free or destroy-after-null patterns.


## 12.6 Graph Cleanup and Memory Safety

### The Problem

Each training step creates dozens of intermediate tensors in the computation graph.
If not cleaned up, they leak.

### The Solution

`ax_graph_cleanup(root)`:
1. Traverses the entire computation graph from the root
2. Destroys all intermediate nodes (tensors with a `grad_fn`)
3. Preserves leaf nodes (parameters, user-created tensors with no `grad_fn`)
4. Detaches the root from the graph (frees its `grad_fn`)

```c
void ax_graph_cleanup(ax_tensor_t *root) {
    // Collect all nodes
    topo_list_t visited = {0}, order = {0};
    topo_sort_dfs(root, &visited, &order);

    // Destroy intermediates
    for (int i = 0; i < order.count; i++) {
        ax_tensor_t *node = order.nodes[i];
        if (node != root && node->grad_fn)
            ax_tensor_destroy(node);
    }

    // Detach root
    if (root->grad_fn) {
        free(root->grad_fn);
        root->grad_fn = NULL;
    }
}
```

The detach step is critical: without it, destroying the root would try to access
already-freed intermediate nodes through the `grad_fn` pointers.

### Storage Reference Counting

The refcounting mechanism in `ax_storage_t` prevents double-frees:

```c
void ax_storage_release(ax_storage_t *s) {
    if (!s) return;
    if (s->refcount <= 0) return;  // already freed or corrupted
    s->refcount--;
    if (s->refcount == 0) {
        ax_aligned_free(s->data);
        free(s);
    }
}
```

The `refcount <= 0` check is a defensive measure against corruption.


## 12.7 Stack Safety

### The Problem

The autograd backward pass walks a computation graph. A naive recursive DFS would
blow the stack on deep graphs (thousands of chained operations).

### The Solution

Iterative DFS with an explicit stack array:

```c
#define MAX_GRAPH_NODES 4096

typedef struct {
    ax_tensor_t *node;
    int child_idx;
} dfs_frame_t;

static void topo_sort_dfs(...) {
    dfs_frame_t stack[MAX_GRAPH_NODES];
    int stack_top = 0;
    // ... iterative traversal using stack_top
}
```

`MAX_GRAPH_NODES` (4096) limits the graph size. This is sufficient for any
practical network (a 100-layer ResNet with 3 ops per layer is 300 nodes).


## 12.8 No Undefined Behavior

Axiom avoids all forms of C undefined behavior:

1. **No signed integer overflow**: all shape arithmetic uses checked multiplication.
   Unsigned overflow is well-defined in C; signed overflow is UB.

2. **No type punning through incompatible pointers**: the `grad_fn->ctx` field is
   `void *`, cast to the correct type. The layer polymorphism uses the first-field
   embedding pattern, which is well-defined by the C standard.

3. **No null pointer dereference**: every function validates pointers before use.

4. **No use of uninitialized memory**: `calloc` over `malloc` where practical.
   `ax_tensor_zeros` uses `memset(data, 0, size)`.

5. **No out-of-bounds array access**: bounds-checked element access, validated
   loop bounds.


## 12.9 Compiler Hardening Flags

The build system enables defensive compiler options:

```
-Wall -Wextra -Wpedantic       Catch common mistakes
-Wformat=2 -Wformat-security   Format string safety
-Wshadow                       Detect variable shadowing
-Wstrict-prototypes            Enforce proper C prototypes
-fstack-protector-strong       Runtime stack overflow detection
-D_FORTIFY_SOURCE=2            Runtime buffer overflow detection (release)
```

These catch entire categories of bugs at compile time or crash early at runtime
rather than silently corrupting memory.


## 12.10 Sanitizer Support

For development and testing, Axiom supports AddressSanitizer (ASan) and
UndefinedBehaviorSanitizer (UBSan):

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DAX_SANITIZE=ON
make -j$(nproc)
ctest --output-on-failure
```

What they catch:
- **ASan**: heap buffer overflow, stack buffer overflow, use-after-free,
  double-free, memory leaks.
- **UBSan**: signed integer overflow, null pointer dereference, shift overflow,
  out-of-bounds array access, misaligned pointer access.

The test suite runs under both sanitizers. Examples run with full leak detection.
The policy: **zero leaks in examples, zero errors in tests**.


## 12.11 Error Handling: No exit() in Library Code

Library code never calls `exit()`. Instead:
- Functions return error codes (`ax_status_t`) or NULL on failure.
- Before returning an error, `ax_err_set()` stores a human-readable message.
- The application decides how to handle errors.

```c
ax_err_set(AX_ERR_INVALID_SHAPE,
           "shape[%u] = %ld is non-positive in file", i, shape[i]);
return NULL;
```

This is critical for embedded systems where crashing is unacceptable, and for
libraries where the caller might want to handle the error gracefully.

Error messages use explicit format strings — never passing user-controlled
strings as the format argument (prevents format string attacks).


## 12.12 Thread Safety (Current Status)

Axiom is currently single-threaded. Global mutable state exists in:

- `grad_tracking` (autograd.c): modified by `ax_no_grad()` / `ax_enable_grad()`
- `rng_seeded` (tensor.c, init.c): modified once on first use
- `active_ops` (dispatch.c): set during init, read-only after

If multi-threading is added:
- `grad_tracking` needs `_Thread_local`
- `active_ops` needs atomic access
- Storage refcount needs atomic increment/decrement


## 12.13 Security Checklist for New Code

When adding new features to Axiom:

1. **Validate all external inputs** (files, function args, data arrays).
2. **Use `safe_mul_i64` for any shape arithmetic** that involves multiplication.
3. **Check `compute_numel` return value** (-1 means overflow).
4. **Check all allocation return values** (malloc, calloc, tensor_create can fail).
5. **Free everything on error paths** (use goto cleanup pattern if needed).
6. **Never call exit()** — return an error code.
7. **Run ASan + UBSan** before merging.
8. **Add tests for error cases**, not just happy paths.


## Key Takeaways

1. Integer overflow in shape arithmetic is the most dangerous vulnerability class.
   Checked multiplication everywhere.
2. File parsing validates every field: magic, version, dtype, ndim, shape, numel.
3. Bounds checking on tensor access and dataset indexing prevents out-of-bounds reads.
4. Graph cleanup + storage refcounting prevents memory leaks and use-after-free.
5. Iterative (not recursive) graph traversal prevents stack overflow.
6. ASan + UBSan in CI catches bugs that static analysis misses.
7. No `exit()`, no format string vulnerabilities, no undefined behavior.
