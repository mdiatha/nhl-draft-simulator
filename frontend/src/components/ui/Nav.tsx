import { Link, useLocation } from 'react-router-dom'
import { clsx } from 'clsx'

const links = [
  { to: '/lottery', label: 'Lottery', live: true },
  { to: '/scenario', label: 'What-If' },
  { to: '/prospects', label: 'Prospects' },
  { to: '/scout', label: 'Ask the Scout' },
]

export default function Nav() {
  const location = useLocation()
  return (
    <nav className="border-b border-border-subtle bg-bg-secondary sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 flex items-center h-14 gap-8">
        <Link to="/" className="font-bold text-lg tracking-tight text-white flex-shrink-0">
          <span className="text-accent-blue">NHL</span> Draft Sim
        </Link>
        <div className="flex items-center gap-1">
          {links.map(link => (
            <Link
              key={link.to}
              to={link.to}
              className={clsx(
                'px-3 py-1.5 rounded text-sm font-medium transition-colors flex items-center gap-1.5',
                location.pathname === link.to
                  ? 'bg-accent-blue/20 text-accent-blue'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
              )}
            >
              {link.label}
              {link.live && (
                <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 leading-none">
                  LIVE
                </span>
              )}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
