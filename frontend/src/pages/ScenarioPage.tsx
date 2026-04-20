import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { draftApi, standingsApi } from '../lib/api'
import { getTeamColors, getTeamLogo } from '../lib/teamColors'
import { DRAFT_YEAR } from '../lib/config'
import type { DraftSimulationPick } from '../types'

interface TeamSlot {
  team_id: number
  team_name: string
  abbreviation: string
}

const POSITION_COLORS: Record<string, string> = {
  C: '#3b82f6', LW: '#10b981', RW: '#06b6d4', D: '#ef4444', G: '#f59e0b',
}

export default function ScenarioPage() {
  const [teams, setTeams] = useState<TeamSlot[]>([])
  const [loadingTeams, setLoadingTeams] = useState(true)
  const [picks, setPicks] = useState<DraftSimulationPick[]>([])
  const [simulating, setSimulating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // drag state
  const dragIdx = useRef<number | null>(null)
  const dragOverIdx = useRef<number | null>(null)

  useEffect(() => {
    loadTeams()
  }, [])

  async function loadTeams() {
    setLoadingTeams(true)
    try {
      // Load all 32 teams from live standings: lottery teams first (worst→best), then playoff teams
      const liveData = await standingsApi.getLive()
      const live: { team_id: number; team_name: string; abbreviation: string; points: number; row: number; in_playoffs: boolean }[] = liveData?.standings ?? []
      const lottery = [...live].filter(t => !t.in_playoffs).sort((a, b) => a.points - b.points || a.row - b.row)
      const playoff = [...live].filter(t => t.in_playoffs).sort((a, b) => a.points - b.points || a.row - b.row)
      const sorted = [...lottery, ...playoff]
      setTeams(sorted.map(t => ({
        team_id: t.team_id,
        team_name: t.team_name,
        abbreviation: t.abbreviation,
      })))
    } catch {
      setError('Failed to load teams. Make sure the backend is running.')
    } finally {
      setLoadingTeams(false)
    }
  }

  function handleDragStart(idx: number) {
    dragIdx.current = idx
  }

  function handleDragEnter(idx: number) {
    dragOverIdx.current = idx
  }

  function handleDragEnd() {
    const from = dragIdx.current
    const to = dragOverIdx.current
    if (from === null || to === null || from === to) {
      dragIdx.current = null
      dragOverIdx.current = null
      return
    }
    setTeams(prev => {
      const next = [...prev]
      const [moved] = next.splice(from, 1)
      next.splice(to, 0, moved)
      return next
    })
    dragIdx.current = null
    dragOverIdx.current = null
    // Clear results when order changes
    setPicks([])
  }

  function moveUp(idx: number) {
    if (idx === 0) return
    setTeams(prev => {
      const next = [...prev]
      ;[next[idx - 1], next[idx]] = [next[idx], next[idx - 1]]
      return next
    })
    setPicks([])
  }

  function moveDown(idx: number) {
    if (idx === teams.length - 1) return
    setTeams(prev => {
      const next = [...prev]
      ;[next[idx], next[idx + 1]] = [next[idx + 1], next[idx]]
      return next
    })
    setPicks([])
  }

  async function runSimulation() {
    if (teams.length === 0) return
    setSimulating(true)
    setError(null)
    setPicks([])
    try {
      const data = await draftApi.simulate(teams.map(t => t.team_id))
      setPicks(data.picks ?? [])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Simulation failed')
    } finally {
      setSimulating(false)
    }
  }

  function resetOrder() {
    setPicks([])
    loadTeams()
  }

  return (
    <main className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white">{DRAFT_YEAR} What-If Simulator</h1>
        <p className="text-text-secondary text-sm mt-1">
          Drag teams into any pick order, then simulate — the model drafts for every team.
        </p>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-500/20 border border-red-500/30 text-red-400 text-sm">
          {error}
        </div>
      )}

      <div className="flex gap-8 flex-col xl:flex-row">
        {/* ── Left panel: order editor ─────────────────────────────────────── */}
        <div className="xl:w-80 flex-shrink-0">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-semibold text-white">Pick Order</span>
            <button
              onClick={resetOrder}
              className="text-xs text-text-muted hover:text-white transition-colors"
            >
              Reset
            </button>
          </div>

          {loadingTeams ? (
            <div className="flex items-center justify-center h-40">
              <div className="w-6 h-6 border-2 border-accent-blue border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <div className="space-y-1 select-none">
              {teams.map((team, idx) => {
                const colors = getTeamColors(team.abbreviation)
                return (
                  <div
                    key={team.team_id}
                    draggable
                    onDragStart={() => handleDragStart(idx)}
                    onDragEnter={() => handleDragEnter(idx)}
                    onDragEnd={handleDragEnd}
                    onDragOver={e => e.preventDefault()}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-bg-card border border-border-subtle hover:border-border-default cursor-grab active:cursor-grabbing transition-colors group"
                    style={{ borderLeft: `3px solid ${colors.primary}` }}
                  >
                    <span className="text-text-muted text-xs w-5 text-right flex-shrink-0 font-mono">
                      {idx + 1}
                    </span>
                    <div
                      className="w-6 h-6 rounded flex items-center justify-center flex-shrink-0"
                      style={{ backgroundColor: colors.primary + '22' }}
                    >
                      <img
                        src={getTeamLogo(team.abbreviation)}
                        alt={team.abbreviation}
                        className="w-5 h-5 object-contain"
                        onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
                      />
                    </div>
                    <span className="text-white text-xs font-medium flex-1 truncate">
                      {team.team_name}
                    </span>
                    <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => moveUp(idx)}
                        disabled={idx === 0}
                        className="p-0.5 rounded hover:bg-bg-hover text-text-muted hover:text-white disabled:opacity-20 transition-colors"
                        title="Move up"
                      >
                        ▲
                      </button>
                      <button
                        onClick={() => moveDown(idx)}
                        disabled={idx === teams.length - 1}
                        className="p-0.5 rounded hover:bg-bg-hover text-text-muted hover:text-white disabled:opacity-20 transition-colors"
                        title="Move down"
                      >
                        ▼
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          <button
            onClick={runSimulation}
            disabled={simulating || loadingTeams || teams.length === 0}
            className="mt-4 w-full px-4 py-2.5 bg-accent-blue hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-semibold rounded-lg transition-colors"
          >
            {simulating ? 'Simulating…' : 'Simulate Draft'}
          </button>
        </div>

        {/* ── Right panel: results ─────────────────────────────────────────── */}
        <div className="flex-1 min-w-0">
          {simulating && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <div className="w-10 h-10 border-2 border-accent-blue border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                <p className="text-text-secondary text-sm">Simulating draft…</p>
              </div>
            </div>
          )}

          {!simulating && picks.length === 0 && (
            <div className="flex items-center justify-center h-64 text-center">
              <div>
                <p className="text-text-secondary text-sm mb-1">
                  Set your pick order on the left, then click <strong className="text-white">Simulate Draft</strong>.
                </p>
                <p className="text-text-muted text-xs">
                  Drag rows or use the arrows to reorder teams.
                </p>
              </div>
            </div>
          )}

          <AnimatePresence>
            {picks.length > 0 && (() => {
              const byRound = new Map<number, typeof picks>()
              for (const pick of picks) {
                const r = pick.round ?? 1
                if (!byRound.has(r)) byRound.set(r, [])
                byRound.get(r)!.push(pick)
              }
              const rounds = Array.from(byRound.entries()).sort(([a], [b]) => a - b)
              let globalIdx = 0
              return (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                  {rounds.map(([roundNum, roundPicks]) => (
                    <div key={roundNum}>
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-bold uppercase tracking-widest text-text-muted">Round {roundNum}</span>
                        <div className="flex-1 h-px bg-border-subtle" />
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {roundPicks.map((pick) => {
                          const colors = getTeamColors(pick.abbreviation)
                          const posColor = POSITION_COLORS[pick.position ?? ''] ?? '#8892a4'
                          const idx = globalIdx++
                          return (
                            <motion.div
                              key={pick.pick}
                              initial={{ opacity: 0, y: 8 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: Math.min(idx * 0.01, 0.5) }}
                              className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-bg-card border border-border-subtle"
                              style={{ borderLeft: `3px solid ${colors.primary}` }}
                            >
                              <span className="text-base font-black text-white w-7 text-right flex-shrink-0">
                                {pick.pick_in_round ?? pick.pick}
                              </span>
                              <div
                                className="w-8 h-8 rounded flex items-center justify-center flex-shrink-0"
                                style={{ backgroundColor: colors.primary + '22' }}
                              >
                                <img
                                  src={getTeamLogo(pick.abbreviation)}
                                  alt={pick.abbreviation}
                                  className="w-6 h-6 object-contain"
                                  onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
                                />
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="text-white font-semibold text-sm truncate">
                                  {pick.prospect_name}
                                </div>
                                <div className="text-text-muted text-xs truncate">
                                  {pick.team_name}
                                  {pick.css_rank != null && <span className="ml-1">· CSS #{pick.css_rank}</span>}
                                  {pick.points_per_game != null && <span className="ml-1">· {pick.points_per_game.toFixed(2)} PPG</span>}
                                </div>
                              </div>
                              {pick.position && (
                                <span
                                  className="text-xs font-bold px-1.5 py-0.5 rounded flex-shrink-0"
                                  style={{ color: posColor, background: posColor + '22' }}
                                >
                                  {pick.position}
                                </span>
                              )}
                            </motion.div>
                          )
                        })}
                      </div>
                    </div>
                  ))}
                </motion.div>
              )
            })()}
          </AnimatePresence>
        </div>
      </div>
    </main>
  )
}
