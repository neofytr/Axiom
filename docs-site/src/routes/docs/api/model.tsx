import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/api/model')({
  component: ModelApiPage,
})

function ModelApiPage() {
  return (
    <>
      <h1>Model API</h1>
      <p>
        The model container bundles a network layer, optimizer, and loss function into a
        single object with a simple train/predict interface. Designed for embedded: uses
        fixed-size parameter buffers, no dynamic arrays.
      </p>
      <p>Header: <code>axiom/model.h</code></p>

      <h2>Creating a model</h2>
      <pre><code className="language-c">{`ax_model_t *ax_model_create(ax_layer_t *net);`}</code></pre>
      <p>
        Creates a model from a network layer (usually a sequential). Collects all trainable
        parameter pointers from the layer tree into a flat array. Maximum{' '}
        <code>AX_MODEL_MAX_PARAMS</code> (256) parameters.
      </p>

      <h2>Compiling</h2>
      <pre><code className="language-c">{`void ax_model_compile(ax_model_t *model, ax_optimizer_t *opt, ax_loss_fn_t loss_fn);`}</code></pre>
      <p>
        Attaches an optimizer and loss function. Call this before training. The model takes
        ownership of the optimizer (it will be freed when the model is destroyed).
      </p>
      <p>
        The loss function is a function pointer matching:{' '}
        <code>ax_tensor_t *(*ax_loss_fn_t)(ax_tensor_t *pred, ax_tensor_t *target)</code>
      </p>
      <pre><code className="language-c">{`ax_model_t *m = ax_model_create(net);
ax_optimizer_t *opt = ax_adam_create(
    m&gt;params, m&gt;n_params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0);
ax_model_compile(m, opt, ax_cross_entropy_loss);`}</code></pre>

      <h2>Training</h2>

      <h3>ax_model_train_step</h3>
      <pre><code className="language-c">{`float ax_model_train_step(ax_model_t *model, ax_tensor_t *input, ax_tensor_t *target);`}</code></pre>
      <p>
        A single training step: forward pass, compute loss, backward pass, optimizer step,
        graph cleanup. Returns the loss value as a float. This is the simplest way to train;
        it handles all the boilerplate internally.
      </p>

      <h3>ax_model_fit</h3>
      <pre><code className="language-c">{`void ax_model_fit(ax_model_t *model,
                  ax_tensor_t *train_x, ax_tensor_t *train_y,
                  int epochs, int print_every);`}</code></pre>
      <p>
        Train for multiple epochs on a full dataset. Prints the loss every{' '}
        <code>print_every</code> epochs. Simple but gets the job done for small datasets that
        fit in memory. For larger datasets, use the dataloader with a manual training loop.
      </p>
      <pre><code className="language-c">{`// train for 100 epochs, print every 10
ax_model_fit(model, train_x, train_y, 100, 10);`}</code></pre>

      <h2>Inference</h2>
      <pre><code className="language-c">{`ax_tensor_t *ax_model_predict(ax_model_t *model, ax_tensor_t *input);`}</code></pre>
      <p>
        Forward pass only. Switches the network to eval mode (disables dropout, uses batchnorm
        running stats), runs forward, then switches back. Does not track gradients.
        Returns the output tensor.
      </p>
      <pre><code className="language-c">{`ax_tensor_t *pred = ax_model_predict(model, test_x);
ax_tensor_print(pred);
ax_tensor_destroy(pred);`}</code></pre>

      <h2>Serialization</h2>
      <p>Header: <code>axiom/serialize.h</code></p>
      <pre><code className="language-c">{`// save model to binary file
ax_status_t ax_model_save(ax_model_t *model, const char *path);

// load model from binary file
ax_model_t *ax_model_load(const char *path);

// save/load individual tensors
ax_status_t ax_tensor_save(ax_tensor_t *t, const char *path);
ax_tensor_t *ax_tensor_load(const char *path);`}</code></pre>
      <p>
        The binary format is designed for embedded deployment: no external dependencies
        (no protobuf, no JSON), fixed-size header, little-endian throughout, can be read
        incrementally (no need to mmap the entire file).
      </p>
      <p>Format: <code>[header (magic "AXON" + version)] [layer descriptors] [parameter data]</code></p>
      <pre><code className="language-c">{`// save after training
ax_model_save(model, "classifier.axm");

// load on another machine (or embedded device)
ax_model_t *m = ax_model_load("classifier.axm");
ax_tensor_t *pred = ax_model_predict(m, input);`}</code></pre>

      <h2>Diagnostics</h2>
      <pre><code className="language-c">{`void ax_model_summary(ax_model_t *model);`}</code></pre>
      <p>
        Prints a summary of the model: layer types, shapes, and parameter counts. Useful for
        verifying the architecture.
      </p>

      <h2>Cleanup</h2>
      <pre><code className="language-c">{`void ax_model_destroy(ax_model_t *model);`}</code></pre>
      <p>
        Frees the model, its optimizer, the network layer tree, and all owned tensors.
        One call cleans up everything.
      </p>

      <h2>Inference-only builds</h2>
      <p>
        When compiled with <code>AX_INFERENCE_ONLY</code>, <code>ax_model_compile</code>,{' '}
        <code>ax_model_train_step</code>, and <code>ax_model_fit</code> are not available. The
        model can only be loaded and used for prediction. This strips all training code from the
        binary.
      </p>
    </>
  )
}
