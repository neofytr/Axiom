import { Outlet, createRootRoute, Link, useMatches } from '@tanstack/react-router'
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
  const matches = useMatches()
  const currentPath = matches[matches.length - 1]?.fullPath || '/'
  const isHome = currentPath === '/'

  return (
    <div className="min-h-screen bg-white text-gray-900">
      <header className="sticky top-0 z-50 border-b border-gray-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <Link to="/" className="flex items-center gap-2 text-xl font-bold tracking-tight text-gray-900">
            Axiom
          </Link>
          <nav className="hidden items-center gap-6 text-sm font-medium md:flex">
            <Link to="/docs/quickstart" className="text-gray-600 hover:text-gray-900 [&.active]:text-blue-600">
              Docs
            </Link>
            <Link to="/docs/api/tensor" className="text-gray-600 hover:text-gray-900 [&.active]:text-blue-600">
              API
            </Link>
            <a
              href="https://github.com/neofytr/Axiom"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-600 hover:text-gray-900"
            >
              GitHub
            </a>
          </nav>
        </div>
      </header>

      {isHome ? (
        <Outlet />
      ) : (
        <div className="mx-auto flex max-w-7xl">
          <aside className="sticky top-14 hidden h-[calc(100vh-3.5rem)] w-64 shrink-0 overflow-y-auto border-r border-gray-200 px-4 py-6 lg:block">
            {navSections.map((section) => (
              <div key={section.title} className="mb-6">
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400">
                  {section.title}
                </h3>
                <ul className="space-y-1">
                  {section.links.map((link) => (
                    <li key={link.to}>
                      <Link
                        to={link.to}
                        className="block rounded px-2 py-1 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 [&.active]:bg-blue-50 [&.active]:font-medium [&.active]:text-blue-700"
                      >
                        {link.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </aside>
          <main className="min-w-0 flex-1 px-8 py-8">
            <div className="prose prose-gray max-w-3xl prose-headings:scroll-mt-20 prose-code:before:content-none prose-code:after:content-none">
              <Outlet />
            </div>
          </main>
        </div>
      )}
    </div>
  )
}
