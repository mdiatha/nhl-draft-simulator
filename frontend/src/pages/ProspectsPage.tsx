import { useState, useMemo, useEffect, Fragment } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { draftApi } from '../lib/api'
import type { Prospect } from '../types'

const POSITIONS = ['All', 'C', 'LW', 'RW', 'D', 'G'] as const
const LEAGUES = ['All', 'OHL', 'WHL', 'QMJHL', 'NCAA', 'SHL', 'Liiga', 'Other'] as const
const NATIONALITIES = ['All', 'CAN', 'USA', 'EUR'] as const

const POS_COLORS: Record<string, string> = {
  C: '#3b82f6', LW: '#10b981', RW: '#06b6d4', D: '#ef4444', G: '#f59e0b',
}

const NAT_FLAGS: Record<string, string> = {
  CAN: '🇨🇦', USA: '🇺🇸', SWE: '🇸🇪', FIN: '🇫🇮', RUS: '🇷🇺',
  CZE: '🇨🇿', SVK: '🇸🇰', GER: '🇩🇪', SUI: '🇨🇭', DEN: '🇩🇰',
}

function getNatFlag(nat?: string): string {
  if (!nat) return '🌍'
  return NAT_FLAGS[nat] ?? '🌍'
}

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return debounced
}

type SortField = 'css_ranking' | 'name' | 'age_at_draft' | 'points' | 'goals' | 'assists' | 'games_played' | 'points_per_game'

export default function ProspectsPage() {
  const [posFilter, setPosFilter] = useState<string>('All')
  const [leagueFilter, setLeagueFilter] = useState<string>('All')
  const [natFilter, setNatFilter] = useState<string>('All')
  const [search, setSearch] = useState('')
  const [sortField, setSortField] = useState<SortField>('css_ranking')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')
  const [page, setPage] = useState(0)
  const [expanded, setExpanded] = useState<number | null>(null)
  const PAGE_SIZE = 50

  const debouncedSearch = useDebounce(search, 300)

  const { data, isLoading } = useQuery({
    queryKey: ['prospects', posFilter, natFilter, debouncedSearch],
    queryFn: () => draftApi.getProspects({
      ...(posFilter !== 'All' && { position: posFilter }),
      ...(natFilter !== 'All' && { nationality: natFilter }),
      ...(debouncedSearch && { search: debouncedSearch }),
      limit: 500,
    }),
    staleTime: 1000 * 60 * 5,
  })

  const allProspects: Prospect[] = data?.prospects ?? []

  // Client-side league filter
  const filtered = useMemo(() => {
    let result = allProspects
    if (leagueFilter !== 'All') {
      result = result.filter(p => {
        const league = p.draft_league ?? ''
        if (leagueFilter === 'Other') {
          return !['OHL', 'WHL', 'QMJHL', 'NCAA', 'SHL', 'Liiga'].some(l => league.includes(l))
        }
        return league.includes(leagueFilter)
      })
    }
    return result
  }, [allProspects, leagueFilter])

  // Sort
  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const av = a[sortField] ?? (sortField === 'css_ranking' ? 9999 : 0)
      const bv = b[sortField] ?? (sortField === 'css_ranking' ? 9999 : 0)
      if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv as string) : (bv as string).localeCompare(av)
      return sortDir === 'asc' ? (av as number) - (bv as number) : (bv as number) - (av as number)
    })
  }, [filtered, sortField, sortDir])

  const paginated = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const totalPages = Math.ceil(sorted.length / PAGE_SIZE)

  function toggleSort(field: SortField) {
    if (sortField === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortField(field); setSortDir(field === 'css_ranking' ? 'asc' : 'desc') }
    setPage(0)
  }

  const SortIcon = ({ field }: { field: SortField }) => (
    <span className="ml-1 text-xs">{sortField === field ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}</span>
  )

  return (
    <main className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-1">2025 Draft Class</h1>
        <p className="text-text-secondary">
          {data?.total ?? '—'} prospects
          {sorted.length !== (data?.total ?? 0) && ` · ${sorted.length} shown`}
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-5">
        {/* Search */}
        <input
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(0) }}
          placeholder="Search player..."
          className="px-3 py-1.5 bg-bg-card border border-border-default rounded-lg text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-accent-blue w-48"
        />

        {/* Position */}
        <div className="flex gap-1">
          {POSITIONS.map(pos => (
            <button
              key={pos}
              onClick={() => { setPosFilter(pos); setPage(0) }}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${posFilter === pos ? 'bg-accent-blue text-white' : 'bg-bg-card border border-border-default text-text-secondary hover:text-white'}`}
            >
              {pos}
            </button>
          ))}
        </div>

        {/* League */}
        <div className="flex gap-1 flex-wrap">
          {LEAGUES.map(l => (
            <button
              key={l}
              onClick={() => { setLeagueFilter(l); setPage(0) }}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${leagueFilter === l ? 'bg-purple-600/80 text-white' : 'bg-bg-card border border-border-default text-text-secondary hover:text-white'}`}
            >
              {l}
            </button>
          ))}
        </div>

        {/* Nationality */}
        <div className="flex gap-1">
          {NATIONALITIES.map(n => (
            <button
              key={n}
              onClick={() => { setNatFilter(n); setPage(0) }}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${natFilter === n ? 'bg-green-600/80 text-white' : 'bg-bg-card border border-border-default text-text-secondary hover:text-white'}`}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="bg-bg-card border border-border-subtle rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-border-subtle text-text-secondary text-xs uppercase tracking-wide">
              <tr>
                <th className="px-3 py-3 text-left w-12 cursor-pointer hover:text-white" onClick={() => toggleSort('css_ranking')}>CSS<SortIcon field="css_ranking" /></th>
                <th className="px-3 py-3 text-left cursor-pointer hover:text-white" onClick={() => toggleSort('name')}>Player<SortIcon field="name" /></th>
                <th className="px-3 py-3 text-left">Pos</th>
                <th className="px-3 py-3 text-left cursor-pointer hover:text-white hidden sm:table-cell" onClick={() => toggleSort('age_at_draft')}>Age<SortIcon field="age_at_draft" /></th>
                <th className="px-3 py-3 text-left hidden md:table-cell">League</th>
                <th className="px-3 py-3 text-center hidden sm:table-cell">Flag</th>
                <th className="px-3 py-3 text-right cursor-pointer hover:text-white" onClick={() => toggleSort('games_played')}>GP<SortIcon field="games_played" /></th>
                <th className="px-3 py-3 text-right cursor-pointer hover:text-white" onClick={() => toggleSort('goals')}>G<SortIcon field="goals" /></th>
                <th className="px-3 py-3 text-right cursor-pointer hover:text-white" onClick={() => toggleSort('assists')}>A<SortIcon field="assists" /></th>
                <th className="px-3 py-3 text-right cursor-pointer hover:text-white" onClick={() => toggleSort('points')}>PTS<SortIcon field="points" /></th>
                <th className="px-3 py-3 text-right cursor-pointer hover:text-white hidden lg:table-cell" onClick={() => toggleSort('points_per_game')}>PTS/GP<SortIcon field="points_per_game" /></th>
              </tr>
            </thead>
            <tbody>
              {isLoading
                ? Array.from({ length: 20 }).map((_, i) => (
                    <tr key={i} className="border-b border-border-subtle">
                      {Array.from({ length: 11 }).map((_, j) => (
                        <td key={j} className="px-3 py-3"><div className="skeleton h-4 w-full" /></td>
                      ))}
                    </tr>
                  ))
                : paginated.map((p) => {
                    return (
                      <Fragment key={p.id}>
                        <tr
                          onClick={() => setExpanded(expanded === p.id ? null : p.id)}
                          className="border-b border-border-subtle hover:bg-bg-hover cursor-pointer transition-colors"
                        >
                          <td className="px-3 py-2.5 font-mono text-text-muted text-xs">{p.css_ranking ?? '—'}</td>
                          <td className="px-3 py-2.5 font-semibold text-white">{p.name}</td>
                          <td className="px-3 py-2.5">
                            {p.position && (
                              <span className="text-xs font-bold px-1.5 py-0.5 rounded" style={{ color: POS_COLORS[p.position] ?? '#8892a4', background: (POS_COLORS[p.position] ?? '#8892a4') + '22' }}>
                                {p.position}
                              </span>
                            )}
                          </td>
                          <td className="px-3 py-2.5 text-text-secondary hidden sm:table-cell">{p.age_at_draft?.toFixed(1) ?? '—'}</td>
                          <td className="px-3 py-2.5 text-text-secondary text-xs hidden md:table-cell">{p.draft_league ?? '—'}</td>
                          <td className="px-3 py-2.5 text-center hidden sm:table-cell">{getNatFlag(p.nationality)}</td>
                          <td className="px-3 py-2.5 text-right text-text-secondary font-mono">{p.games_played ?? '—'}</td>
                          <td className="px-3 py-2.5 text-right text-text-secondary font-mono">{p.goals ?? '—'}</td>
                          <td className="px-3 py-2.5 text-right text-text-secondary font-mono">{p.assists ?? '—'}</td>
                          <td className="px-3 py-2.5 text-right font-bold text-white font-mono">{p.points ?? '—'}</td>
                          <td className="px-3 py-2.5 text-right text-text-secondary font-mono hidden lg:table-cell">{p.points_per_game?.toFixed(2) ?? '—'}</td>
                        </tr>
                        <AnimatePresence>
                          {expanded === p.id && (
                            <motion.tr
                              key={`exp-${p.id}`}
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                            >
                              <td colSpan={11} className="bg-bg-secondary px-6 py-4 border-b border-border-subtle">
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                                  <div>
                                    <div className="text-text-muted text-xs mb-1">Height</div>
                                    <div className="text-white">{p.height_cm ? `${p.height_cm}cm` : '—'}</div>
                                  </div>
                                  <div>
                                    <div className="text-text-muted text-xs mb-1">Weight</div>
                                    <div className="text-white">{p.weight_kg ? `${p.weight_kg}kg` : '—'}</div>
                                  </div>
                                  <div>
                                    <div className="text-text-muted text-xs mb-1">CSS Category</div>
                                    <div className="text-white">{p.css_category ?? '—'}</div>
                                  </div>
                                  <div>
                                    <div className="text-text-muted text-xs mb-1">League Tier</div>
                                    <div className="text-white">{p.draft_league_tier ? `Tier ${p.draft_league_tier}` : '—'}</div>
                                  </div>
                                </div>
                              </td>
                            </motion.tr>
                          )}
                        </AnimatePresence>
                      </Fragment>
                    )
                  })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-border-subtle">
            <span className="text-text-muted text-xs">
              {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, sorted.length)} of {sorted.length}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-3 py-1 text-xs bg-bg-hover border border-border-default rounded disabled:opacity-30 hover:border-accent-blue transition-colors"
              >
                ←
              </button>
              <button
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="px-3 py-1 text-xs bg-bg-hover border border-border-default rounded disabled:opacity-30 hover:border-accent-blue transition-colors"
              >
                →
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
