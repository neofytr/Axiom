import { Outlet, createRootRoute, Link, useRouterState } from '@tanstack/react-router'
import '../styles.css'

const navSections = [
  {
    title: 'Getting Started',
    links: [
      { to: '/', label: 'Introduction' },
      { to: '/docs/quickstart', label: 'Quick Start' },
      { to: '/docs/building', label: 'Building' },
    ],
  },
  {
    title: 'Architecture',
    links: [
      { to: '/docs/architecture/overview', label: 'Overview' },
      { to: '/docs/architecture/tensors', label: 'Tensors & Storage' },
      { to: '/docs/architecture/autograd', label: 'Autograd Engine' },
      { to: '/docs/architecture/compute', label: 'Compute Dispatch' },
      { to: '/docs/architecture/gemm', label: 'GEMM Kernels' },
      { to: '/docs/architecture/cuda', label: 'CUDA Backend' },
    ],
  },
  {
    title: 'API Reference',
    links: [
      { to: '/docs/api/tensor', label: 'Tensor' },
      { to: '/docs/api/layers', label: 'Layers' },
      { to: '/docs/api/activations', label: 'Activations' },
      { to: '/docs/api/losses', label: 'Losses' },
      { to: '/docs/api/optimizers', label: 'Optimizers' },
      { to: '/docs/api/model', label: 'Model' },
      { to: '/docs/api/autograd', label: 'Autograd' },
      { to: '/docs/api/data', label: 'Data Loading' },
    ],
  },
  {
    title: 'Guides',
    links: [
      { to: '/docs/guides/training', label: 'Training a Model' },
      { to: '/docs/guides/embedded', label: 'Embedded Deployment' },
      { to: '/docs/guides/performance', label: 'Performance Tuning' },
    ],
  },
]

export const Route = createRootRoute({
  component: RootComponent,
  notFoundComponent: NotFound,
})

function NotFound() {
  return (
    <div style={{ padding: '2rem', textAlign: 'center' }}>
      <h2 style={{ color: 'var(--sea-ink)', marginBottom: '0.5rem' }}>Page not found</h2>
      <p style={{ color: 'var(--sea-ink-soft)' }}>
        <Link to="/" style={{ color: 'var(--lagoon-deep)' }}>Back to home</Link>
      </p>
    </div>
  )
}

function RootComponent() {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const isHome = pathname === '/'

  return (
    <>
      {/* header */}
      <header className="sticky top-0 z-50 border-b border-[var(--line)] backdrop-blur" style={{ background: 'var(--header-bg)' }}>
        <div className="page-wrap flex items-center justify-between py-3">
          <Link to="/" className="text-xl font-bold tracking-tight no-underline" style={{ color: 'var(--lagoon)' }}>
            Axiom
          </Link>
          <nav className="flex items-center gap-5 text-sm font-medium">
            <Link to="/docs/quickstart" className="nav-link">Docs</Link>
            <Link to="/docs/api/tensor" className="nav-link">API</Link>
            <a href="https://github.com/neofytr/Axiom" target="_blank" rel="noopener noreferrer" className="nav-link">GitHub</a>
          </nav>
        </div>
      </header>

      {isHome ? (
        <LandingPage />
      ) : (
        <div className="page-wrap" style={{ width: 'min(1280px, calc(100% - 2rem))' }}>
          <div className="flex gap-8 pt-6 pb-16">
            <aside className="hidden w-56 shrink-0 lg:block">
              <nav className="sticky top-16 max-h-[calc(100vh-5rem)] overflow-y-auto pr-4">
                {navSections.map((section) => (
                  <div key={section.title} className="mb-5">
                    <h4 className="island-kicker mb-2" style={{ fontSize: '0.65rem' }}>{section.title}</h4>
                    <ul className="space-y-0.5">
                      {section.links.map((link) => (
                        <li key={link.to}>
                          <Link
                            to={link.to}
                            className="nav-link block rounded-md px-2.5 py-1.5 text-[0.82rem] no-underline"
                            activeProps={{ className: 'nav-link is-active block rounded-md px-2.5 py-1.5 text-[0.82rem] no-underline' }}
                            style={{ textDecoration: 'none' }}
                          >
                            {link.label}
                          </Link>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </nav>
            </aside>
            <main className="min-w-0 flex-1">
              <article className="island-shell rounded-2xl px-8 py-8 sm:px-10 sm:py-10">
                <div className="docs-prose">
                  <Outlet />
                </div>
              </article>
            </main>
          </div>
        </div>
      )}
    </>
  )
}

function LandingPage() {
  return (
    <main className="page-wrap px-4 pb-12 pt-14">
      <section className="island-shell rise-in relative overflow-hidden rounded-[2rem] px-6 py-12 sm:px-10 sm:py-16">
        <div className="pointer-events-none absolute -left-20 -top-24 h-56 w-56 rounded-full bg-[radial-gradient(circle,rgba(79,184,178,0.32),transparent_66%)]" />
        <div className="pointer-events-none absolute -bottom-20 -right-20 h-56 w-56 rounded-full bg-[radial-gradient(circle,rgba(47,106,74,0.18),transparent_66%)]" />
        <p className="island-kicker mb-3">Deep Learning in C</p>
        <h1 className="display-title mb-5 max-w-3xl text-4xl leading-[1.05] font-bold tracking-tight text-[var(--sea-ink)] sm:text-6xl">
          Axiom
        </h1>
        <p className="mb-8 max-w-2xl text-base leading-relaxed text-[var(--sea-ink-soft)] sm:text-lg">
          A deep learning framework written from scratch in C. No dependencies, no Python runtime,
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
            className="rounded-full border border-[var(--chip-line)] px-5 py-2.5 text-sm font-semibold text-[var(--sea-ink)] no-underline transition hover:-translate-y-0.5"
            style={{ background: 'var(--chip-bg)' }}
          >
            GitHub
          </a>
        </div>
      </section>

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
                <td className="py-2 text-[var(--sea-ink-soft)]">—</td>
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
                <td className="py-2 pr-4 font-bold" style={{ color: 'var(--lagoon)' }}>51,226 img/s</td>
                <td className="py-2">22,185 img/s</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-medium">Per batch</td>
                <td className="py-2 pr-4 font-bold" style={{ color: 'var(--lagoon)' }}>4.99ms</td>
                <td className="py-2">11.54ms</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-[var(--sea-ink-soft)]">2.3x faster inference. The gap is TensorFlow's Python dispatch and graph executor overhead per op.</p>
      </section>

      <section className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {([
          ['Zero Dependencies', 'No BLAS, no Python, no package manager. The C compiler is the only requirement.'],
          ['Reverse-Mode Autograd', 'Records a computation graph during forward, backpropagates through it. Slab-allocated grad nodes, zero malloc per op.'],
          ['BLIS-Style GEMM', 'Tiled matmul with micro-kernels sized to the register file. 14x32 AVX-512, 6x16 AVX2, 8x12 NEON.'],
          ['Op Fusion', 'Dense+ReLU fuses into one kernel. Cross-entropy backward is a single SIMD pass. Bias folded into GEMM writeback.'],
          ['Embedded Ready', 'Inference-only strips training code. Under 100KB on ARM. Baremetal runs on Cortex-M, no heap, no stdio.'],
          ['CUDA Backend', 'GPU with cuBLAS GEMM, fused kernels, explicit device memory. Same code, just flip a build flag.'],
        ] as const).map(([title, desc], i) => (
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

      <section className="island-shell rise-in mt-8 rounded-2xl p-6 sm:p-8" style={{ animationDelay: '400ms' }}>
        <p className="island-kicker mb-2">Example</p>
        <h2 className="mb-4 text-lg font-bold text-[var(--sea-ink)]">Train an MNIST classifier</h2>
        <pre><code>{`#include "axiom/axiom.h"

int main(void) {
    ax_init();

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
    </main>
  )
}
