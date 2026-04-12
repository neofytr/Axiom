import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/api/data')({
  component: DataApiPage,
})

function DataApiPage() {
  return (
    <>
      <h1>Data Loading API</h1>
      <p>
        Dataset interface, data loaders, and preprocessing transforms. Provides batching,
        shuffling, and basic transforms with no external dependencies.
      </p>
      <p>Header: <code>axiom/data.h</code></p>

      <h2>Dataset interface</h2>
      <p>
        The dataset is a vtable-based interface. Concrete datasets embed{' '}
        <code>ax_dataset_t</code> as their first field (same C polymorphism pattern as layers).
      </p>
      <pre><code className="language-c">{`typedef struct {
    void (*get_item)(ax_dataset_t *self, int64_t index,
                     ax_tensor_t **input, ax_tensor_t **target);
    int64_t (*length)(ax_dataset_t *self);
    void (*destroy)(ax_dataset_t *self);
} ax_dataset_ops_t;`}</code></pre>

      <h3>Convenience wrappers</h3>
      <pre><code className="language-c">{`void ax_dataset_get_item(ax_dataset_t *ds, int64_t idx,
                         ax_tensor_t **input, ax_tensor_t **target);
int64_t ax_dataset_length(ax_dataset_t *ds);
void ax_dataset_destroy(ax_dataset_t *ds);`}</code></pre>

      <h2>Tensor dataset</h2>
      <pre><code className="language-c">{`ax_dataset_t *ax_tensor_dataset_create(ax_tensor_t *inputs, ax_tensor_t *targets);`}</code></pre>
      <p>
        Wraps a pair of tensors where the first dimension is the sample dimension.{' '}
        <code>get_item</code> returns a view (slice) of each tensor at the given index.
        No data copying.
      </p>
      <pre><code className="language-c">{`// inputs: [60000, 784], targets: [60000, 10]
ax_dataset_t *ds = ax_tensor_dataset_create(train_x, train_y);`}</code></pre>

      <h2>CSV dataset</h2>
      <pre><code className="language-c">{`ax_dataset_t *ax_csv_dataset_load(const char *path,
                                   const int *feature_cols, int n_features,
                                   const int *target_cols, int n_targets,
                                   bool has_header);`}</code></pre>
      <p>
        Loads a CSV file where each row is a sample. You specify which columns are features
        and which are targets. All values are parsed as float32.
      </p>
      <pre><code className="language-c">{`ax_dataset_t *ds = ax_csv_dataset_load("data.csv",
    (int[]){0, 1, 2}, 3,     // feature columns 0, 1, 2
    (int[]){3}, 1,            // target column 3
    true);                    // has header row`}</code></pre>

      <h2>Dataloader</h2>
      <pre><code className="language-c">{`ax_dataloader_t *ax_dataloader_create(ax_dataset_t *dataset,
                                       int64_t batch_size,
                                       bool shuffle);`}</code></pre>
      <p>
        Iterates over a dataset in batches. When <code>shuffle = true</code>, the sample indices
        are shuffled at the start of each epoch using Fisher-Yates.
      </p>

      <h3>Iteration</h3>
      <pre><code className="language-c">{`typedef struct {
    ax_tensor_t *input;   // [batch_size, ...]
    ax_tensor_t *target;  // [batch_size, ...]
    int64_t batch_size;   // actual size (last batch may be smaller)
} ax_batch_t;

// get next batch; returns false when epoch is done
bool ax_dataloader_next(ax_dataloader_t *dl, ax_batch_t *batch);

// reset to start (reshuffles if shuffle=true)
void ax_dataloader_reset(ax_dataloader_t *dl);

// number of batches per epoch
int64_t ax_dataloader_num_batches(ax_dataloader_t *dl);`}</code></pre>
      <p>
        Batch tensors are freshly allocated on each call to <code>ax_dataloader_next</code>.
        The caller is responsible for destroying them after use.
      </p>

      <h3>Full example</h3>
      <pre><code className="language-c">{`ax_dataset_t *ds = ax_tensor_dataset_create(train_x, train_y);
ax_dataloader_t *dl = ax_dataloader_create(ds, 64, true);

for (int epoch = 0; epoch < 10; epoch++) {
    ax_dataloader_reset(dl);
    ax_batch_t batch;
    while (ax_dataloader_next(dl, &batch)) {
        float loss = ax_model_train_step(model, batch.input, batch.target);
        ax_tensor_destroy(batch.input);
        ax_tensor_destroy(batch.target);
    }
}

ax_dataloader_destroy(dl);
ax_dataset_destroy(ds);`}</code></pre>

      <h2>Transforms</h2>

      <h3>Normalize</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_transform_normalize(ax_tensor_t *t);
ax_tensor_t *ax_transform_normalize_with(ax_tensor_t *t,
                                          ax_tensor_t *mean,
                                          ax_tensor_t *std);`}</code></pre>
      <p>
        Normalizes to zero mean and unit variance along axis 0 (per-feature). The{' '}
        <code>_with</code> variant applies known statistics (from training) to test data.
        Returns a new tensor.
      </p>

      <h3>One-hot encoding</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_transform_one_hot(ax_tensor_t *labels, int num_classes);`}</code></pre>
      <p>
        Converts an integer label tensor [n] to a float one-hot tensor [n, num_classes].
        Labels must be in [0, num_classes).
      </p>

      <h3>Min-max scaling</h3>
      <pre><code className="language-c">{`ax_tensor_t *ax_transform_minmax_scale(ax_tensor_t *t);`}</code></pre>
      <p>Scales features to [0, 1] range along axis 0.</p>

      <h2>Cleanup</h2>
      <pre><code className="language-c">{`void ax_dataloader_destroy(ax_dataloader_t *dl);  // does NOT free the dataset
void ax_dataset_destroy(ax_dataset_t *ds);`}</code></pre>
    </>
  )
}
