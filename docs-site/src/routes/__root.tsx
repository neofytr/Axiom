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
})

function RootComponent() {
  const location = useRouterState({ select: (s) => s.location })
  const isHome = location.pathname === '/'

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
        <Outlet />
      ) : (
        <div className="page-wrap" style={{ width: 'min(1280px, calc(100% - 2rem))' }}>
          <div className="flex gap-8 pt-6 pb-16">
            {/* sidebar */}
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

            {/* content */}
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
