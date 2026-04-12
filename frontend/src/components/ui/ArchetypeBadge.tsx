import { clsx } from 'clsx'

const ARCHETYPE_CONFIG: Record<string, { style: string; label: string; tooltip: string }> = {
  'BPA': {
    style: 'bg-blue-500/15 text-blue-300 border-blue-500/30',
    label: 'BPA',
    tooltip: 'Best Player Available — takes the highest-ranked prospect regardless of position',
  },
  'need-based': {
    style: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    label: 'Need-Based',
    tooltip: 'Fills roster gaps — concentrates heavily at 1-2 positions',
  },
  'safe': {
    style: 'bg-slate-500/15 text-slate-300 border-slate-500/30',
    label: 'Safe',
    tooltip: 'North American bias — strongly prefers CHL/NCAA prospects over Europeans',
  },
  'euro-scout': {
    style: 'bg-violet-500/15 text-violet-300 border-violet-500/30',
    label: 'Euro Scout',
    tooltip: 'Actively targets European talent — above-average international picks',
  },
  'analytics': {
    style: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30',
    label: 'Analytics',
    tooltip: 'Balanced positional spread + willing to take undervalued Europeans',
  },
}

const FALLBACK = {
  style: 'bg-gray-500/15 text-gray-400 border-gray-500/30',
  label: 'Unknown',
  tooltip: '',
}

interface Props {
  archetype?: string | null
  size?: 'sm' | 'md' | 'lg'
  showTooltip?: boolean
}

export default function ArchetypeBadge({ archetype, size = 'sm', showTooltip = true }: Props) {
  if (!archetype) return null
  const config = ARCHETYPE_CONFIG[archetype] ?? { ...FALLBACK, label: archetype }

  return (
    <span
      title={showTooltip ? config.tooltip : undefined}
      className={clsx(
        'inline-flex items-center border rounded-full font-medium cursor-default',
        config.style,
        size === 'sm' && 'px-2 py-0.5 text-xs',
        size === 'md' && 'px-2.5 py-1 text-xs',
        size === 'lg' && 'px-3 py-1 text-sm',
      )}
    >
      {config.label}
    </span>
  )
}
