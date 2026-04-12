import { createFileRoute, Link } from '@tanstack/react-router'

export const Route = createFileRoute('/')({
  component: LandingPage,
})

function LandingPage() {
  return (
    <main className="page-wrap px-4 pb-12 pt-14">
      {/* hero */}
      <section className="island-shell rise-in relative overflow-hidden rounded-[2rem] px-6 py-12 sm:px-10 sm:py-16">
        <div className="pointer-events-none absolute -left-20 -top-24 h-56 w-56 rounded-full bg-[radial-gradient(circle,rgba(79,184,178,0.32),transparent_66%)]" />
        <div className="pointer-events-none absolute -bottom-20 -right-20 h-56 w-56 rounded-full bg-[radial-gradient(circle,rgba(47,106,74,0.18),transparent_66%)]" />
        <p className="island-kicker mb-3">Deep Learning in C</p>
        <h1 className="display-title mb-5 max-w-3xl text-4xl leading-[1.05] font-bold tracking-tight text-[var(--sea-ink)] sm:text-6xl">
          Axiom
        </h1>
        <p className="mb-8 max-w-2xl text-base leading-relaxed text-[var(--sea-ink-soft)] sm:text-lg">
          A deep learning framework written in C from scratch. No dependencies, no Python runtime,
          no BLAS library. Trains neural networks on everything from cloud servers to microcontrollers.
          Faster than TensorFlow on CPU.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link
            to="/docs/quickstart"
            className="rounded-full border border-[rgba(50,143,151,0.3)] bg-[rgba(79,184,178,0.14)] px-5 py-2.5 text-sm font-semibold text-[var(--lagoon-deep)] no-underline transition hover:-translate-y-0.5 hover:bg-[rgba(79,184,178,0.24)]"
          >
            Get Started
          </Link>
          <a
            href="https://github.com/neofytr/Axiom"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-[rgba(23,58,64,0.2)] bg-white/50 px-5 py-2.5 text-sm font-semibold text-[var(--sea-ink)] no-underline transition hover:-translate-y-0.5 hover:border-[rgba(23,58,64,0.35)]"
          >
            GitHub
          </a>
        </div>
      </section>

      {/* benchmark table */}
      <section className="island-shell rise-in mt-8 rounded-2xl p-6 sm:p-8" style={{ animationDelay: '80ms' }}>
        <p className="island-kicker mb-2">Benchmarks</p>
        <h2 className="mb-1 text-lg font-bold text-[var(--sea-ink)]">Training (MNIST, 60K samples, Adam, batch 256)</h2>
        <p className="mb-4 text-sm text-[var(--sea-ink-soft)]">Same model, same dataset, same thread count, same CI machine.</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--line)] text-left text-xs uppercase tracking-wider text-[var(--sea-ink-soft)]">
                <th className="pb-2 pr-4">Model</th>
                <th className="pb-2 pr-4">x86 (EPYC, 4T)</th>
                <th className="pb-2">ARM (Graviton3, 4T)</th>
              </tr>
            </thead>
            <tbody className="text-[var(--sea-ink)]">
              <tr className="border-b border-[var(--line)]">
                <td className="py-2 pr-4 font-medium">12.6M params</td>
                <td className="py-2 pr-4">Axiom <strong>69s</strong> / TF 87s</td>
                <td className="py-2"></td>
              </tr>
              <tr className="border-b border-[var(--line)]">
                <td className="py-2 pr-4 font-medium">4.2M params</td>
                <td className="py-2 pr-4">Axiom <strong>34s</strong> / TF 36s</td>
                <td className="py-2">Axiom <strong>26s</strong> / TF 30s</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-medium">1.46M params</td>
                <td className="py-2 pr-4">Axiom <strong>16s</strong> / TF 17s</td>
                <td className="py-2">Axiom <strong>16s</strong> / TF 20s</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h2 className="mb-1 mt-6 text-lg font-bold text-[var(--sea-ink)]">Inference (1.46M params, batch 256, 1000 passes)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--line)] text-left text-xs uppercase tracking-wider text-[var(--sea-ink-soft)]">
                <th className="pb-2 pr-4">Metric</th>
                <th className="pb-2 pr-4">Axiom</th>
                <th className="pb-2">TensorFlow</th>
              </tr>
            </thead>
            <tbody className="text-[var(--sea-ink)]">
              <tr className="border-b border-[var(--line)]">
                <td className="py-2 pr-4 font-medium">Throughput</td>
                <td className="py-2 pr-4 font-bold">51,226 img/s</td>
                <td className="py-2">22,185 img/s</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-medium">Per batch</td>
                <td className="py-2 pr-4 font-bold">4.99ms</td>
                <td className="py-2">11.54ms</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-[var(--sea-ink-soft)]">2.3x faster inference. The gap comes from TF's Python dispatch overhead and graph executor overhead per op.</p>
      </section>

      {/* feature cards */}
      <section className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {[
          ['Zero Dependencies', 'No BLAS, no Python, no package manager. The C compiler is the only requirement. Builds in seconds.'],
          ['Reverse-Mode Autograd', 'Records a computation graph during the forward pass, then backpropagates through it. Slab-allocated grad nodes with no malloc per op.'],
          ['BLIS-Style GEMM', 'Tiled matrix multiply with micro-kernels sized to the register file. 14x32 on AVX-512, 6x16 on AVX2, 8x12 on NEON. Cache-blocked with adaptive tile sizes.'],
          ['Op Fusion', 'Dense+ReLU fuses into a single kernel. Cross-entropy backward computes softmax-target in one SIMD pass. Bias add is folded into the GEMM writeback.'],
          ['Embedded Ready', 'Inference-only builds strip all training code. Binary under 100KB on ARM. Baremetal profile runs on Cortex-M with no heap, no stdio, no threads.'],
          ['CUDA Backend', 'GPU support with cuBLAS GEMM, fused element-wise kernels, explicit device memory management. Same training code, just flip a flag.'],
        ].map(([title, desc], i) => (
          <article
            key={title}
            className="island-shell feature-card rise-in rounded-2xl p-5"
            style={{ animationDelay: `${i * 70 + 160}ms` }}
          >
            <h2 className="mb-2 text-base font-semibold text-[var(--sea-ink)]">{title}</h2>
            <p className="m-0 text-sm leading-relaxed text-[var(--sea-ink-soft)]">{desc}</p>
          </article>
        ))}
      </section>

      {/* code example */}
      <section className="island-shell rise-in mt-8 rounded-2xl p-6 sm:p-8" style={{ animationDelay: '400ms' }}>
        <p className="island-kicker mb-2">Example</p>
        <h2 className="mb-4 text-lg font-bold text-[var(--sea-ink)]">Train an MNIST classifier in ~20 lines</h2>
        <pre className="overflow-x-auto rounded-lg bg-[var(--foam)] p-4 text-sm leading-relaxed"><code className="language-c">{`#include "axiom/axiom.h"

int main(void) {
    ax_init();

    // 784 -> 128 -> 10 classifier
    ax_layer_t *net = ax_sequential_create();
    ax_sequential_add(net, ax_dense_create(784, 128, true));
    ax_sequential_add(net, ax_relu_layer_create());
    ax_sequential_add(net, ax_dense_create(128, 10, true));

    ax_model_t *m = ax_model_create(net);
    ax_optimizer_t *opt = ax_adam_create(
        m->params, m->n_params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0);
    ax_model_compile(m, opt, ax_cross_entropy_loss);

    for (int i = 0; i < 1000; i++)
        ax_model_train_step(m, train_x, train_y);

    ax_model_save(m, "model.axm");
    ax_model_destroy(m);
    ax_shutdown();
}`}</code></pre>
      </section>

      {/* what's included */}
      <section className="island-shell rise-in mt-8 rounded-2xl p-6 sm:p-8" style={{ animationDelay: '480ms' }}>
        <p className="island-kicker mb-2">What's Included</p>
        <div className="grid gap-4 text-sm text-[var(--sea-ink-soft)] sm:grid-cols-2">
          <div>
            <h3 className="mb-1 font-semibold text-[var(--sea-ink)]">Layers</h3>
            <p className="m-0">Dense, Conv2D, BatchNorm, LayerNorm, Dropout, MaxPool, AvgPool, GlobalAvgPool, Flatten, Sequential</p>
          </div>
          <div>
            <h3 className="mb-1 font-semibold text-[var(--sea-ink)]">Activations</h3>
            <p className="m-0">ReLU, Sigmoid, Tanh, GELU, Swish, LeakyReLU, ELU, SELU, Mish, Softplus, Softmax</p>
          </div>
          <div>
            <h3 className="mb-1 font-semibold text-[var(--sea-ink)]">Optimizers</h3>
            <p className="m-0">SGD (momentum + nesterov), Adam, AdamW, RMSprop, Adagrad. LR scheduling: cosine, step decay, exponential, warmup+cosine.</p>
          </div>
          <div>
            <h3 className="mb-1 font-semibold text-[var(--sea-ink)]">Losses</h3>
            <p className="m-0">MSE, MAE, Cross-Entropy, BCE with logits, Huber. All return scalar tensors ready for ax_backward().</p>
          </div>
        </div>
      </section>
    </main>
  )
}
