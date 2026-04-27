# Unit 10: Serialization and the Data Pipeline

## Why This Matters

A trained model is useless if you can't save it. A model can't train if you can't
feed it data efficiently. This unit covers both: Axiom's binary serialization format
(designed for embedded deployment) and its dataset/dataloader/transform pipeline.


## Part A: Serialization

### 10.1 Design Goals

Axiom's serialization format was designed for embedded systems:

- **No external dependencies**: no protobuf, no JSON, no XML parsers.
- **Fixed-size headers**: no variable-length strings or complex nesting.
- **Little-endian throughout**: standard on ARM and x86.
- **Incrementally readable**: no need to load the entire file into memory.
- **Simple enough for a microcontroller** with just `fread()`.

Two file formats exist:
- `.axt` for individual tensors
- `.axm` for complete models


### 10.2 Tensor File Format (.axt)

```
Offset  Size    Type    Field
0       4       u32     magic (0x41585430 = "AXT0")
4       4       u32     dtype (enum value)
8       4       u32     ndim
12      8*ndim  i64[]   shape[0..ndim-1]
varies  varies  raw     data (numel * dtype_size bytes)
```

Reading a tensor:
1. Read and validate the magic bytes
2. Read dtype and validate it's < AX_DTYPE_COUNT
3. Read ndim and validate it's <= AX_MAX_DIMS
4. Read each shape dimension, validating > 0 and checking for overflow in numel
5. Allocate tensor
6. Read raw data bytes directly into storage


### 10.3 Model File Format (.axm)

```
Offset  Size           Type    Field
0       4              u32     magic (0x41584F4E = "AXON")
4       4              u32     version
8       4              u32     n_layers

--- Layer Descriptors (repeated n_layers times) ---
+0      4              u32     layer_type (enum)
+4      4              u32     n_params_in_layer
+8      8              i64     input_features
+16     8              i64     output_features
+24     4              f32     extra (alpha, axis, etc.)
+28     1              u8      flags (bit 0 = use_bias)

--- Parameter Data (all layers' params, flattened) ---
For each param tensor:
  +0    4              u32     dtype
  +4    4              u32     ndim
  +8    8*ndim         i64[]   shape
  +varies varies       raw     data bytes
```

The key insight: architecture and weights are stored separately. The layer
descriptors tell you the network structure; the parameter data contains the
learned values. On load, the architecture is reconstructed first, then weights
are copied into the freshly created layers.


### 10.4 Save Flow

```c
ax_status_t ax_model_save(ax_model_t *model, const char *path);
```

1. Validate inputs (null check, must be sequential model)
2. Write header: magic, version, n_layers
3. For each layer: write descriptor (type, n_params, features, extra, flags)
4. For each layer's parameters: write tensor metadata + raw data

The `write_tensor` helper handles non-contiguous tensors by making a contiguous
copy first:

```c
if (ax_tensor_is_contiguous(t) && t->offset == 0) {
    fwrite(t->storage->data, 1, bytes, f);  // fast path
} else {
    ax_tensor_t *c = ax_tensor_contiguous(t);  // slow path: copy first
    fwrite(c->storage->data, 1, bytes, f);
    ax_tensor_destroy(c);
}
```


### 10.5 Load Flow

```c
ax_model_t *ax_model_load(const char *path);
```

1. Read and validate header (magic, version, n_layers)
2. Read all layer descriptors into a temporary array
3. Reconstruct each layer (`ax_dense_create`, `ax_relu_layer_create`, etc.)
4. Add them to a new sequential model
5. For each parameter: read tensor from file, validate shape matches, copy data
6. Wrap in an `ax_model_t` and return

The load validates everything from the file because it's untrusted input.
See Unit 12 for the full security analysis.


### 10.6 Non-Contiguous Tensor Handling

Transposed or sliced tensors have non-standard strides. The serialization format
stores raw contiguous data, so saving requires making a contiguous copy. Loading
always creates contiguous tensors.

This means: `save(transpose(t))` then `load()` gives you a contiguous tensor
with the transposed shape and data, but the strides will be standard row-major.


---


## Part B: The Data Pipeline

### 10.7 The Dataset Interface

The dataset is an abstract interface (vtable pattern, same as layers):

```c
typedef struct {
    void (*get_item)(ax_dataset_t *self, int64_t index,
                     ax_tensor_t **input, ax_tensor_t **target);
    int64_t (*length)(ax_dataset_t *self);
    void (*destroy)(ax_dataset_t *self);
} ax_dataset_ops_t;
```

Any struct that embeds `ax_dataset_t` as its first field and fills in the ops
is a valid dataset. Axiom provides two concrete implementations.


### 10.8 Tensor Dataset

Wraps two existing tensors where dimension 0 is the sample axis:

```c
ax_dataset_t *ax_tensor_dataset_create(ax_tensor_t *inputs, ax_tensor_t *targets);
```

- `inputs`: shape `[n_samples, ...]`
- `targets`: shape `[n_samples, ...]`

`get_item` returns views (slices) into the original tensors. Views are lazily
created and cached — the first call allocates them, subsequent calls return
the cached pointer.

The `make_row_view` helper creates a view of a single sample:

```c
static ax_tensor_t *make_row_view(ax_tensor_t *t, int64_t row) {
    // For 1D: return a copy of a single scalar
    // For 2D+: return a view sharing storage, with adjusted offset
    v->offset = t->offset + row * t->strides[0];
    // Copy shape[1:] and strides[1:] into the view
}
```

The dataset does NOT own the input/target tensors (it doesn't free them on
destroy). The caller manages their lifetime.


### 10.9 CSV Dataset

Loads data from a CSV file:

```c
ax_dataset_t *ax_csv_dataset_load(
    const char *path,
    const int *feature_cols, int n_features,    // which columns are inputs
    const int *target_cols, int n_targets,      // which columns are targets
    bool has_header);                           // skip first line?
```

On load:
1. Count lines in the file
2. Allocate `[n_rows, n_features]` and `[n_rows, n_targets]` tensors
3. Parse the file line by line, tokenizing by comma
4. Fill the tensors with parsed float values

Security considerations:
- Line truncation detection (if line fills buffer without newline)
- Column bounds checking (field count vs column index)
- Index bounds checking in `get_item`


### 10.10 The Dataloader

Iterates over a dataset in shuffled batches:

```c
ax_dataloader_t *ax_dataloader_create(ax_dataset_t *dataset,
                                       int64_t batch_size,
                                       bool shuffle);
```

Internal state:
```c
typedef struct {
    ax_dataset_t *dataset;
    int64_t batch_size;
    bool shuffle;
    int64_t *indices;       // shuffled index array
    int64_t n_samples;
    int64_t current_pos;    // progress through the epoch
} ax_dataloader_t;
```

The iteration protocol:

```c
ax_batch_t batch;
while (ax_dataloader_next(dl, &batch)) {
    // batch.input:  [batch_size, ...input_shape]
    // batch.target: [batch_size, ...target_shape]
    // batch.batch_size: actual size (last batch may be smaller)

    float loss = ax_model_train_step(model, batch.input, batch.target);

    ax_tensor_destroy(batch.input);
    ax_tensor_destroy(batch.target);
}
ax_dataloader_reset(dl);  // start next epoch (reshuffles if shuffle=true)
```

**Shuffling**: uses Fisher-Yates shuffle on an index array. Each epoch gets a
different random permutation:

```c
static void shuffle_indices(int64_t *arr, int64_t n) {
    for (int64_t i = n - 1; i > 0; i--) {
        int64_t j = rand() % (i + 1);
        int64_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
```

**Batch assembly**: `ax_dataloader_next` allocates new batch tensors, queries
the dataset for each sample in the batch, and copies them into the batch tensor.
The caller owns the batch tensors and must destroy them.


### 10.11 Transforms

Preprocessing functions that return new tensors:

**Normalize** (z-score normalization per feature):

    x_normalized = (x - mean) / (std + 1e-7)

```c
ax_tensor_t *ax_transform_normalize(ax_tensor_t *t);
```

Computes mean and standard deviation along axis 0 (per feature), then normalizes.
The epsilon (1e-7) prevents division by zero for constant features.

**Normalize with known statistics** (for applying training stats to test data):

```c
ax_tensor_t *ax_transform_normalize_with(ax_tensor_t *t, ax_tensor_t *mean, ax_tensor_t *std);
```

**One-hot encoding**:

```c
ax_tensor_t *ax_transform_one_hot(ax_tensor_t *labels, int num_classes);
// Input:  [n] of integer class labels
// Output: [n, num_classes] of float 0/1
```

**Min-max scaling** (to [0, 1] range):

    x_scaled = (x - min) / (max - min)

```c
ax_tensor_t *ax_transform_minmax_scale(ax_tensor_t *t);
```


### 10.12 Putting It All Together

A complete training pipeline:

```c
// 1. Load data
ax_dataset_t *train_ds = ax_csv_dataset_load("data.csv",
    (int[]){0, 1, 2, 3}, 4,    // 4 feature columns
    (int[]){4}, 1,              // 1 target column
    true);                      // has header

// 2. Create dataloader
ax_dataloader_t *dl = ax_dataloader_create(train_ds, 32, true);

// 3. Build and compile model
ax_layer_t *net = ax_sequential_create();
ax_sequential_add(net, ax_dense_create(4, 16, true));
ax_sequential_add(net, ax_relu_layer_create());
ax_sequential_add(net, ax_dense_create(16, 1, true));
ax_model_t *model = ax_model_create(net);
ax_model_compile(model, ax_adam_create(...), ax_mse_loss);

// 4. Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    ax_batch_t batch;
    while (ax_dataloader_next(dl, &batch)) {
        ax_model_train_step(model, batch.input, batch.target);
        ax_tensor_destroy(batch.input);
        ax_tensor_destroy(batch.target);
    }
    ax_dataloader_reset(dl);
}

// 5. Save model
ax_model_save(model, "trained.axm");

// 6. Later: load and predict
ax_model_t *loaded = ax_model_load("trained.axm");
ax_tensor_t *pred = ax_model_predict(loaded, test_input);
```


## Key Takeaways

1. The binary format (.axm) stores architecture descriptors then raw parameter data.
2. No dependencies: just magic bytes, fixed-size fields, and raw data.
3. The dataset interface uses the same vtable pattern as layers.
4. Dataloaders handle batching and shuffling via Fisher-Yates on an index array.
5. Transforms (normalize, one-hot, min-max) return new tensors.
6. The full pipeline: CSV -> dataset -> dataloader -> model -> serialize.
