import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/docs/quickstart')({
  component: QuickstartPage,
})

function QuickstartPage() {
  return (
    <>
      <h1>Quick Start</h1>
      <p>
        Get Axiom built and train your first model in under five minutes.
      </p>

      <h2>Build from source</h2>
      <pre><code className="language-bash">{`git clone https://github.com/neofytr/Axiom.git
cd Axiom
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
ctest --output-on-failure`}</code></pre>
      <p>
        That's it. No package manager, no pip, no downloading pretrained weights.
        The C compiler is the only dependency.
      </p>

      <h2>Your first program</h2>
      <p>
        Create a file called <code>hello.c</code> next to the build directory:
      </p>
      <pre><code className="language-c">{`#include "axiom/axiom.h"
#include <stdio.h>

int main(void) {
    ax_init();

    // create a 2x3 tensor filled with ones
    int64_t shape[] = {2, 3};
    ax_tensor_t *a = ax_tensor_ones(shape, 2, AX_FLOAT32);
    ax_tensor_t *b = ax_tensor_full(shape, 2, AX_FLOAT32, 2.0);

    ax_tensor_t *c = ax_add(a, b);
    ax_tensor_print(c);  // [2, 3]: 3.0 3.0 3.0 3.0 3.0 3.0

    ax_tensor_destroy(a);
    ax_tensor_destroy(b);
    ax_tensor_destroy(c);
    ax_shutdown();
    return 0;
}`}</code></pre>
      <p>Compile and run:</p>
      <pre><code className="language-bash">{`gcc -o hello hello.c -Iinclude -Lbuild -laxiom -lm -lpthread
./hello`}</code></pre>

      <h2>Train XOR</h2>
      <p>
        The classic "can your framework learn XOR" test. A 2-layer network with 4 hidden units
        is enough to learn this non-linear function.
      </p>
      <pre><code className="language-c">{`#include "axiom/axiom.h"
#include <stdio.h>

int main(void) {
    ax_init();

    // XOR dataset
    float x_data[] = {0,0, 0,1, 1,0, 1,1};
    float y_data[] = {0, 1, 1, 0};
    ax_tensor_t *X = ax_tensor_from_array(x_data, (int64_t[]){4, 2}, 2, AX_FLOAT32);
    ax_tensor_t *Y = ax_tensor_from_array(y_data, (int64_t[]){4, 1}, 2, AX_FLOAT32);

    // 2 &gt; 4 &gt; 1 network
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(2, 4, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(4, 1, true));
    ax_sequential_add(net, ax_sigmoid_layer_create());

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(
        m&gt;params, m&gt;n_params, 0.01f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_mse_loss);

    // train
    for (int epoch = 0; epoch < 2000; epoch++) {
        float loss = ax_model_train_step(m, X, Y);
        if (epoch % 500 == 0)
            printf("epoch %d  loss %.6f\\n", epoch, loss);
    }

    // predict
    ax_tensor_t *pred = ax_model_predict(m, X);
    ax_tensor_print(pred);

    ax_tensor_destroy(pred);
    ax_tensor_destroy(X);
    ax_tensor_destroy(Y);
    ax_model_destroy(m);
    ax_shutdown();
    return 0;
}`}</code></pre>
      <p>
        After ~1000 epochs the loss drops below 0.001 and the predictions converge to
        the correct XOR outputs (0, 1, 1, 0).
      </p>

      <h2>Next steps</h2>
      <ul>
        <li><a href="/docs/building">Build options</a> for CUDA, embedded profiles, and ISA dispatch</li>
        <li><a href="/docs/architecture/overview">Architecture overview</a> to understand how the pieces fit together</li>
        <li><a href="/docs/api/tensor">Tensor API</a> for the full reference</li>
        <li><a href="/docs/guides/training">Training guide</a> for a complete training loop with batching, scheduling, and evaluation</li>
      </ul>
    </>
  )
}
