import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { standingsApi } from '../lib/api'
import LotteryDraw from '../components/lottery/LotteryDraw'
import { getTeamColors, getTeamLogo } from '../lib/teamColors'
import { useLotteryStore } from '../stores/lotteryStore'
import { useDraftStore } from '../stores/draftStore'
import { DRAFT_YEAR } from '../lib/config'
import type { LotteryPick, LotterySimulationTeam } from '../types'

interface LiveTeam {
  team_id: number | null
  abbreviation: string
  team_name: string
  conference: string
  division: string
  wins: number
  losses: number
  otl: number
  games_played: number
  points: number
  row: number
  overall_rank: number
  in_playoffs: boolean
  lottery_slot: number | null
  lottery_odds_pct: number
  playoff_status: string
}

type SortKey = 'points' | 'lottery_odds_pct' | 'division'

const CONF_ORDER: Record<string, number> = { Eastern: 0, Western: 1 }

export default function LotteryPage() {
  const navigate = useNavigate()
  const { reset } = useLotteryStore()
  const { setLotteryResult } = useDraftStore()
  const [seed] = useState(() => Math.floor(Math.random() * 99999))
  const [resetKey, setResetKey] = useState(0)
  const [completed, setCompleted] = useState(false)
  const [sortBy, setSortBy] = useState<SortKey>('points')
  const [showPlayoffs, setShowPlayoffs] = useState(true)
  const [lotteryOrder, setLotteryOrder] = useState<LiveTeam[]>([])

  const { data, isLoading, error, dataUpdatedAt } = useQuery({
    queryKey: ['standings-live'],
    queryFn: standingsApi.getLive,
    refetchInterval: 5 * 60 * 1000,
    staleTime: 5 * 60 * 1000,
  })

  const allTeams: LiveTeam[] = data?.standings ?? []
  const nonPlayoff = allTeams.filter(t => !t.in_playoffs)
  const lotteryDraws: number = data?.lottery_draws ?? 2

  // Teams sorted for the sidebar table
  const sidebarTeams = (() => {
    const list = showPlayoffs ? allTeams : nonPlayoff
    if (sortBy === 'lottery_odds_pct') {
      return [...list].sort((a, b) => b.lottery_odds_pct - a.lottery_odds_pct || b.points - a.points)
    }
    if (sortBy === 'division') {
      return [...list].sort((a, b) =>
        (CONF_ORDER[a.conference] ?? 2) - (CONF_ORDER[b.conference] ?? 2) ||
        a.division.localeCompare(b.division) ||
        b.points - a.points
      )
    }
    // default: points ASC (worst → best, matching lottery seeding)
    return [...list].sort((a, b) => a.points - b.points || a.row - b.row)
  })()

  const liveSimulationTeams: LotterySimulationTeam[] = allTeams.map(team => ({
    team_id: team.team_id ?? 0,
    team_name: team.team_name,
    abbreviation: team.abbreviation,
    odds_pct: team.lottery_odds_pct,
    standing: team.lottery_slot ?? team.overall_rank,
    overall_rank: team.overall_rank,
    in_playoffs: team.in_playoffs,
    lottery_slot: team.lottery_slot,
    wins: team.wins,
    losses: team.losses,
    otl: team.otl,
    points: team.points,
  }))

  function handleReset() {
    reset()
    setCompleted(false)
    setLotteryOrder([])
    setResetKey(k => k + 1)
  }

  function handleComplete(result: LotteryPick[], completedSeed: number) {
    setCompleted(true)
    // Map lottery result back to LiveTeam order for display
    const byAbbrev = Object.fromEntries(allTeams.map(t => [t.abbreviation, t]))
    const ordered = result
      .map(p => byAbbrev[p.abbreviation])
      .filter(Boolean) as LiveTeam[]
    setLotteryOrder(ordered)

    // Push lottery result into draft store so DraftPage can pick it up
    setLotteryResult(result, completedSeed)
  }

  const lastUpdated = dataUpdatedAt
    ? new Date(dataUpdatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null

  return (
    <main className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold text-white">Draft Lottery</h1>
          <p className="text-text-secondary mt-1 text-sm">
            {DRAFT_YEAR} · {nonPlayoff.length} lottery-eligible teams · {lotteryDraws} draws · odds based on live standings
            {lastUpdated && <span className="ml-2 text-text-muted">· updated {lastUpdated}</span>}
          </p>
        </div>
        <div className="flex gap-2">
          {completed && (
            <button
              onClick={() => navigate('/draft')}
              className="px-4 py-2 text-sm bg-accent-blue hover:bg-blue-500 text-white font-semibold rounded-lg transition-colors"
            >
              Simulate Draft →
            </button>
          )}
          <button
            onClick={handleReset}
            className="px-4 py-2 text-sm bg-bg-card border border-border-default rounded-lg text-text-secondary hover:text-white hover:border-accent-blue transition-colors"
          >
            {completed ? 'Run Again' : 'Reset'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          Could not load live standings. Check your connection or try again.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Lottery draw */}
        <div className="lg:col-span-3 flex flex-col gap-4">
          {isLoading ? (
            <div className="h-80 bg-bg-card rounded-xl animate-pulse border border-border-subtle" />
          ) : (
            <LotteryDraw
              key={resetKey}
              onComplete={handleComplete}
              seed={seed}
              teams={liveSimulationTeams}
            />
          )}

        </div>

        {/* Live standings sidebar */}
        <div className="lg:col-span-2">
          <div className="bg-bg-card border border-border-subtle rounded-xl overflow-hidden">
            {/* Controls */}
            <div className="px-4 py-3 border-b border-border-subtle flex items-center justify-between gap-2">
              <h3 className="font-semibold text-white text-sm flex-shrink-0">Live Standings</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowPlayoffs(p => !p)}
                  className={`text-xs px-2 py-1 rounded-full border transition-colors ${showPlayoffs ? 'border-accent-blue/50 text-accent-blue bg-accent-blue/10' : 'border-border-subtle text-text-muted'}`}
                >
                  {showPlayoffs ? 'All 32' : 'Lottery only'}
                </button>
                <select
                  value={sortBy}
                  onChange={e => setSortBy(e.target.value as SortKey)}
                  className="text-xs bg-bg-secondary border border-border-subtle rounded px-2 py-1 text-text-secondary focus:outline-none"
                >
                  <option value="points">By points</option>
                  <option value="lottery_odds_pct">By odds</option>
                  <option value="division">By division</option>
                </select>
              </div>
            </div>

            {/* Column headers */}
            <div className="px-4 py-2 grid grid-cols-[1.5rem_1fr_4rem_3.5rem_3rem] gap-1 text-[10px] text-text-muted uppercase tracking-wide border-b border-border-subtle">
              <span>#</span>
              <span>Team</span>
              <span className="text-right">W-L-OT</span>
              <span className="text-right">PTS</span>
              <span className="text-right">Odds</span>
            </div>

            <div className="divide-y divide-border-subtle max-h-[600px] overflow-y-auto">
              {isLoading
                ? Array.from({ length: 16 }).map((_, i) => (
                    <div key={i} className="px-4 py-2.5 flex items-center gap-3 animate-pulse">
                      <div className="w-4 h-3 rounded bg-bg-secondary" />
                      <div className="w-24 h-3 rounded bg-bg-secondary" />
                      <div className="ml-auto w-10 h-3 rounded bg-bg-secondary" />
                    </div>
                  ))
                : sidebarTeams.map(team => {
                    const colors = getTeamColors(team.abbreviation)
                    const isLottery = !team.in_playoffs
                    return (
                      <div
                        key={team.abbreviation}
                        className={`px-3 py-2 grid grid-cols-[1.5rem_1fr_4rem_3.5rem_3rem] gap-1 items-center transition-colors ${isLottery ? 'hover:bg-bg-hover' : 'opacity-50'}`}
                        style={{ borderLeft: `3px solid ${isLottery ? colors.primary : 'transparent'}` }}
                      >
                        {/* Rank / slot */}
                        <span className="text-[11px] text-text-muted">
                          {isLottery ? team.lottery_slot : '—'}
                        </span>

                        {/* Logo + name */}
                        <div className="flex items-center gap-2 min-w-0">
                          <img
                            src={getTeamLogo(team.abbreviation)}
                            alt={team.abbreviation}
                            className="w-5 h-5 object-contain flex-shrink-0"
                            onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
                          />
                          <span className={`text-xs truncate ${isLottery ? 'text-white' : 'text-text-muted'}`}>
                            {team.team_name}
                          </span>
                          {team.playoff_status === 'x' || team.playoff_status === 'y' || team.playoff_status === 'z' ? (
                            <span className="text-[9px] text-emerald-400 flex-shrink-0">✓</span>
                          ) : team.playoff_status === 'e' ? (
                            <span className="text-[9px] text-red-400 flex-shrink-0">✕</span>
                          ) : null}
                        </div>

                        {/* W-L-OT */}
                        <span className="text-[11px] text-text-muted text-right">
                          {team.wins}-{team.losses}-{team.otl}
                        </span>

                        {/* Points */}
                        <span className={`text-xs font-semibold text-right ${isLottery ? 'text-white' : 'text-text-muted'}`}>
                          {team.points}
                        </span>

                        {/* Lottery odds */}
                        <span className={`text-xs text-right font-semibold ${isLottery ? 'text-accent-blue' : 'text-text-muted'}`}>
                          {isLottery ? `${team.lottery_odds_pct.toFixed(1)}%` : '—'}
                        </span>
                      </div>
                    )
                  })}
            </div>

            {/* Legend */}
            <div className="px-4 py-2.5 border-t border-border-subtle flex gap-4 text-[10px] text-text-muted">
              <span><span className="text-emerald-400">✓</span> Clinched</span>
              <span><span className="text-red-400">✕</span> Eliminated</span>
              <span className="text-accent-blue">Blue = lottery eligible</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
