import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { draftApi, lotteryApi } from '../lib/api'
import { useDraftStore } from '../stores/draftStore'
import { useLotteryStore } from '../stores/lotteryStore'
import { getTeamColors, getTeamLogo } from '../lib/teamColors'
import type { DraftSimulationPick, Team, LotteryPick } from '../types'

type AppPhase = 'idle' | 'drawing' | 'lottery-done' | 'drafting' | 'draft-done'

const delay = (ms: number) => new Promise<void>(res => setTimeout(res, ms))

const POS_COLORS: Record<string, string> = {
  C: '#3b82f6', LW: '#10b981', RW: '#06b6d4', D: '#ef4444', G: '#f59e0b',
}

export default function HomePage() {
  const navigate = useNavigate()
  const { setLotteryResult, setDraftResult, draftResult, resetDraft } = useDraftStore()
  const { setPhase: setStorePhase, setCurrentDraw, setAnimating } = useLotteryStore()

  const [phase, setPhase] = useState<AppPhase>('idle')
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 99999))
  const [, setLotteryOrder] = useState<LotteryPick[]>([])
  const [revealedPicks, setRevealedPicks] = useState<LotteryPick[]>([])
  const [revealing, setRevealing] = useState<LotteryPick | null>(null)
  const [error, setError] = useState<string | null>(null)

  const { data } = useQuery({
    queryKey: ['lottery-odds'],
    queryFn: lotteryApi.getOdds,
  })
  const teams: Team[] = data?.teams ?? []


  async function runLottery(newSeed = seed) {
    setPhase('drawing')
    setStorePhase('drawing')
    setAnimating(true)
    setRevealedPicks([])
    setRevealing(null)
    setError(null)
    resetDraft()

    try {
      const data = await lotteryApi.simulate(newSeed)
      const picks: LotteryPick[] = data.pick_order ?? []
      setLotteryOrder(picks)
      setLotteryResult(picks, data.seed)

      for (let i = 0; i < 2; i++) {
        const pick = picks[i]
        if (!pick) continue
        setCurrentDraw(i + 1)
        await delay(1000)
        setRevealing(pick)
        await delay(2000)
        setRevealedPicks(prev => [...prev, pick])
        setRevealing(null)
        await delay(500)
      }

      await delay(600)
      setStorePhase('complete')
      setAnimating(false)

      // Auto-run draft immediately after lottery
      setPhase('drafting')
      const lotteryOrderIds = picks.map(p => p.team_id)
      const draftData = await draftApi.simulate(lotteryOrderIds, newSeed)
      setDraftResult(draftData.picks ?? [])
      setPhase('draft-done')
    } catch {
      setError('Lottery simulation failed')
      setPhase('idle')
      setAnimating(false)
    }
  }

  async function resimulate() {
    const newSeed = Math.floor(Math.random() * 99999)
    setSeed(newSeed)
    setLotteryOrder([])
    await runLottery(newSeed)
  }

  const picks: DraftSimulationPick[] = draftResult

  return (
    <div className="max-w-5xl mx-auto px-4 py-6">

          {/* Header */}
          <div className="flex items-center justify-between mb-5">
            <div>
              <h1 className="text-2xl font-bold text-white">2025 NHL Draft Simulator</h1>
              <p className="text-text-secondary text-sm mt-0.5">
                GM tendency · positional need · ML prediction
              </p>
            </div>
            <div className="flex gap-2">
              {phase === 'idle' ? (
                <button
                  onClick={() => runLottery()}
                  className="px-5 py-2 text-sm bg-accent-blue hover:bg-blue-500 text-white font-bold rounded-lg transition-colors shadow-lg shadow-blue-500/20"
                >
                  Simulate Draft
                </button>
              ) : (
                <button
                  onClick={resimulate}
                  disabled={phase === 'drawing' || phase === 'drafting'}
                  className="px-4 py-2 text-sm bg-bg-card border border-border-default rounded-lg text-text-secondary hover:text-white hover:border-accent-blue transition-colors disabled:opacity-40"
                >
                  Re-simulate
                </button>
              )}
            </div>
          </div>

          {error && (
            <div className="mb-4 p-3 rounded-lg bg-red-500/20 border border-red-500/30 text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* ── Idle: teams ── */}
          {phase === 'idle' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="bg-bg-card border border-border-subtle rounded-xl overflow-hidden">
                <div className="px-4 py-3 border-b border-border-subtle flex items-center justify-between">
                  <span className="text-sm font-semibold text-white">Lottery Teams</span>
                  <span className="text-text-muted text-xs">{teams.length} teams</span>
                </div>
                <div>
                  {teams.map((team) => {
                    const colors = getTeamColors(team.abbreviation)
                    const record = (team.wins != null && team.losses != null && team.otl != null)
                      ? `${team.wins}-${team.losses}-${team.otl}`
                      : null
                    return (
                      <div
                        key={team.team_id}
                        onClick={() => navigate(`/teams/${team.team_id}`)}
                        className="flex items-center gap-3 px-4 py-2.5 border-b border-border-subtle hover:bg-bg-hover cursor-pointer transition-colors"
                        style={{ borderLeft: `3px solid ${colors.primary}44` }}
                      >
                        <img src={getTeamLogo(team.abbreviation)} alt="" className="w-6 h-6 object-contain flex-shrink-0"
                          onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }} />
                        <span className="text-text-primary text-xs flex-1 truncate">{team.team_name}</span>
                        {record && <span className="text-text-muted font-mono text-xs flex-shrink-0">{record}</span>}
                        <span className="text-white font-semibold text-xs flex-shrink-0 w-10 text-right">{team.lottery_odds_pct?.toFixed(1)}%</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            </motion.div>
          )}

          {/* ── Drawing: lottery animation ── */}
          {phase === 'drawing' && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-4">
                {[1, 2].map(pickNum => {
                  const revealed = revealedPicks.find(p => p.pick === pickNum)
                  const isRevealing = revealing?.pick === pickNum
                  const colors = revealed ? getTeamColors(revealed.abbreviation) : null

                  return (
                    <div
                      key={pickNum}
                      className="relative rounded-xl border overflow-hidden"
                      style={{
                        borderColor: revealed ? (colors!.primary + '60') : '#1e2a3a',
                        background: revealed ? (colors!.primary + '12') : '#0d1520',
                        minHeight: 110,
                      }}
                    >
                      <div className="px-4 pt-3 pb-1">
                        <span className="text-xs font-semibold text-text-muted tracking-widest uppercase">
                          Pick #{pickNum}
                        </span>
                      </div>
                      <AnimatePresence mode="wait">
                        {isRevealing && (
                          <motion.div
                            key="revealing"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0 flex items-center justify-center"
                          >
                            <div className="w-3 h-3 rounded-full bg-accent-blue animate-ping" />
                          </motion.div>
                        )}
                        {revealed && !isRevealing && (
                          <motion.div
                            key="revealed"
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="px-4 pb-4 flex items-center gap-3"
                          >
                            <div
                              className="w-12 h-12 rounded-xl flex items-center justify-center overflow-hidden flex-shrink-0"
                              style={{ background: colors!.primary + '25' }}
                            >
                              <img src={getTeamLogo(revealed.abbreviation)} alt="" className="w-10 h-10 object-contain" onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }} />
                            </div>
                            <div>
                              <div className="font-bold text-white text-sm">{revealed.team_name}</div>
                              <div className="text-xs text-text-muted">{revealed.lottery_odds_pct?.toFixed(1)}% odds</div>
                            </div>
                          </motion.div>
                        )}
                        {!revealed && !isRevealing && (
                          <motion.div key="empty" className="px-4 pb-4 flex items-center gap-3">
                            <div className="w-12 h-12 rounded-xl bg-bg-secondary border border-border-subtle flex items-center justify-center">
                              <span className="text-text-muted text-xl font-black">?</span>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )
                })}
              </div>

              <AnimatePresence>
                {revealing && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-center py-8 rounded-2xl border border-accent-blue/30 bg-accent-blue/5"
                  >
                    <p className="text-text-secondary text-xs mb-2 tracking-widest uppercase">Pick #{revealing.pick} goes to</p>
                    <div className="flex items-center justify-center gap-4">
                      <img src={getTeamLogo(revealing.abbreviation)} alt="" className="w-14 h-14 object-contain" onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }} />
                      <div className="text-left">
                        <div className="text-3xl font-black text-white">{revealing.team_name}</div>
                        <div className="text-accent-blue text-sm font-semibold mt-0.5">{revealing.lottery_odds_pct?.toFixed(1)}% odds</div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {!revealing && (
                <div className="flex items-center justify-center gap-2 text-text-secondary text-sm py-2">
                  <span className="w-2 h-2 rounded-full bg-accent-blue animate-pulse" />
                  Drawing...
                </div>
              )}
            </div>
          )}

          {/* ── Drafting: spinner ── */}
          {phase === 'drafting' && (
            <div className="flex items-center justify-center gap-2 text-text-secondary text-sm py-16">
              <div className="w-5 h-5 border-2 border-accent-blue border-t-transparent rounded-full animate-spin" />
              Simulating draft...
            </div>
          )}

          {/* ── Draft done: picks grid ── */}
          {phase === 'draft-done' && (
            <div>
              <p className="text-xs text-text-muted uppercase tracking-widest mb-3">Draft Results · Seed {seed}</p>
              <div className="flex flex-col gap-1.5">
                {picks.map((pick, i) => {
                  const colors = getTeamColors(pick.abbreviation)
                  return (
                    <motion.div
                      key={pick.pick}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.02 }}
                      onClick={() => navigate(`/teams/${pick.team_id}`)}
                      className="flex items-center gap-2 p-2.5 rounded-lg bg-bg-card border border-border-subtle hover:bg-bg-hover cursor-pointer transition-colors"
                      style={{ borderLeft: `3px solid ${colors.primary}` }}
                    >
                      <span className="text-base font-black text-white w-7 text-right flex-shrink-0">{pick.pick}</span>
                      <div className="w-8 h-8 rounded flex items-center justify-center flex-shrink-0 overflow-hidden" style={{ background: colors.primary + '22' }}>
                        <img src={getTeamLogo(pick.abbreviation)} alt="" className="w-6 h-6 object-contain" onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-white font-semibold text-xs truncate">{pick.prospect_name ?? 'TBD'}</div>
                        <div className="text-text-muted text-xs truncate">
                          {pick.team_name}{pick.css_rank ? ` · CSS #${pick.css_rank}` : ''}
                        </div>
                      </div>
                      {pick.position && (
                        <span className="text-xs font-bold px-1.5 py-0.5 rounded flex-shrink-0"
                          style={{ color: POS_COLORS[pick.position] ?? '#8892a4', background: (POS_COLORS[pick.position] ?? '#8892a4') + '22' }}>
                          {pick.position}
                        </span>
                      )}
                    </motion.div>
                  )
                })}
              </div>
            </div>
          )}
    </div>
  )
}
