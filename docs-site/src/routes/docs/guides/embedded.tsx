import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/guides/embedded')({
  component: EmbeddedGuidePage,
})

function EmbeddedGuidePage() {
  return (
    <>
      <h1>Embedded Deployment</h1>
      <p>
        Axiom is designed to run on embedded and edge devices. This guide covers how to
        build for embedded targets, minimize the binary, deploy inference-only models,
        and handle the constraints of resource-limited platforms.
      </p>

      <h2>Build profiles</h2>

      <h3>Inference-only</h3>
      <pre><code className="language-bash">{`cmake -DAX_INFERENCE_ONLY=ON -DCMAKE_BUILD_TYPE=Release ..`}</code></pre>
      <p>
        Strips all training code: the autograd engine, optimizers, losses, backward functions,
        and gradient tracking. Only forward-pass operations and serialization remain.
        The binary is under 100KB on ARM.
      </p>
      <p>
        What gets removed:
      </p>
      <ul>
        <li><code>ax_backward()</code>, <code>ax_graph_cleanup()</code> — undefined (link error)</li>
        <li>All optimizer code — <code>ax_optimizer_t</code> is forward-declared but empty</li>
        <li>All loss functions</li>
        <li>Slab allocator for grad_fn — not compiled</li>
        <li>Backward arena and forward arena — not created</li>
        <li><code>ax_grad_enabled()</code> returns false (inline, constant-folded)</li>
        <li>All grad_fn setup in forward ops is dead-code-eliminated</li>
      </ul>

      <h3>Embedded Linux</h3>
      <pre><code className="language-bash">{`cmake -DAX_INFERENCE_ONLY=ON -DAX_PROFILE=embedded-linux ..`}</code></pre>
      <p>
        Smaller buffer defaults, trimmed for embedded Linux targets (Raspberry Pi, Jetson Nano,
        BeagleBone). Still uses stdio and pthreads, so it works on any Linux with a C library.
      </p>

      <h3>Baremetal</h3>
      <pre><code className="language-bash">{`cmake -DAX_INFERENCE_ONLY=ON -DAX_PROFILE=embedded-baremetal ..`}</code></pre>
      <p>
        The most constrained profile:
      </p>
      <ul>
        <li>No stdio — <code>AX_NO_STDIO</code> is defined, all logging macros expand to no-ops</li>
        <li>No heap allocation — designed for static memory or a fixed pool</li>
        <li>No threads — <code>AX_SINGLE_THREADED</code> removes all <code>_Thread_local</code> qualifiers and OpenMP pragmas</li>
        <li>No pthreads dependency</li>
      </ul>
      <p>
        This profile targets Cortex-M class microcontrollers with as little as 256KB of flash
        and 64KB of RAM.
      </p>

      <h2>Workflow: train on host, deploy on device</h2>
      <ol>
        <li>
          <strong>Train on your workstation</strong> with the full build (all features enabled).
          Use GPU if available.
        </li>
        <li>
          <strong>Save the model:</strong>
          <pre><code className="language-c">{`ax_model_save(model, "model.axm");`}</code></pre>
        </li>
        <li>
          <strong>Cross-compile Axiom</strong> for the target with inference-only:
          <pre><code className="language-bash">{`cmake -DCMAKE_TOOLCHAIN_FILE=arm-none-eabi.cmake \\
      -DAX_INFERENCE_ONLY=ON \\
      -DAX_PROFILE=embedded-baremetal \\
      -DCMAKE_BUILD_TYPE=Release ..`}</code></pre>
        </li>
        <li>
          <strong>Load and run on device:</strong>
          <pre><code className="language-c">{`ax_model_t *m = ax_model_load("model.axm");
ax_tensor_t *pred = ax_model_predict(m, input);
// use predictions
ax_tensor_destroy(pred);
ax_model_destroy(m);`}</code></pre>
        </li>
      </ol>

      <h2>Binary format</h2>
      <p>
        The <code>.axm</code> format is designed for embedded:
      </p>
      <ul>
        <li>No external dependencies (no protobuf, no JSON, no XML)</li>
        <li>Fixed-size header with a magic number ("AXON") and version</li>
        <li>Little-endian throughout</li>
        <li>Can be read incrementally (no need to load the entire file into memory)</li>
        <li>Layer descriptors followed by densely packed parameter data</li>
      </ul>
      <p>
        On devices with a filesystem, <code>ax_model_load()</code> reads the file. On baremetal
        devices, you can link the model data into flash and parse it from a memory-mapped address.
      </p>

      <h2>Memory considerations</h2>
      <ul>
        <li>
          <strong>Tensor storage is refcounted.</strong> When an inference pass creates temporary
          tensors, they're freed as soon as they're no longer referenced. There's no graph to
          clean up (no autograd in inference-only mode).
        </li>
        <li>
          <strong>Views share storage.</strong> Reshape, transpose, and slice operations don't
          allocate new buffers. This is important when memory is tight.
        </li>
        <li>
          <strong>The sequential forward pass reuses intermediate tensors.</strong> After each
          layer completes, the input to that layer can be freed (its refcount drops when the
          output is computed). Peak memory usage is roughly the size of the two largest
          consecutive layer outputs.
        </li>
      </ul>

      <h2>Model size reduction</h2>
      <p>
        For extremely constrained targets, consider:
      </p>
      <ul>
        <li>
          <strong>Smaller architectures.</strong> A 784-32-10 MLP for MNIST digit recognition
          has ~25K parameters (100KB of weights). That fits in 256KB of flash with room for
          the inference code.
        </li>
        <li>
          <strong>Skip batchnorm at inference.</strong> Batchnorm can be folded into the preceding
          conv/dense layer's weights and bias before deployment, eliminating the batchnorm layer
          entirely.
        </li>
        <li>
          <strong>INT8 quantization</strong> (planned): will reduce model size by 4x and speed
          up inference on targets with 8-bit SIMD.
        </li>
      </ul>

      <h2>Example: Cortex-M deployment</h2>
      <pre><code className="language-c">{`// baremetal main — no stdio, no heap
#include "axiom/axiom.h"

// model data linked from flash
extern const unsigned char model_data[];
extern const unsigned int model_data_len;

// static buffer for input
static float input_buf[784];

void run_inference(void) {
    ax_init();

    ax_model_t *m = ax_model_load_from_memory(model_data, model_data_len);

    // fill input_buf from sensor/camera...
    ax_tensor_t *input = ax_tensor_from_array(
        input_buf, (int64_t[]){1, 784}, 2, AX_FLOAT32);

    ax_tensor_t *pred = ax_model_predict(m, input);

    // read prediction
    int best_class = 0;
    float best_score = -1e30f;
    for (int c = 0; c < 10; c++) {
        float score = ax_tensor_get_f32(pred, (int64_t[]){0, c});
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }

    // use best_class...

    ax_tensor_destroy(pred);
    ax_tensor_destroy(input);
    ax_model_destroy(m);
}`}</code></pre>
    </>
  )
}
