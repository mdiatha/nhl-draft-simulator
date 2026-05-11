import { useEffect, useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { buildApiUrl, draftApi, lotteryApi } from '../lib/api'
import { useDraftStore } from '../stores/draftStore'
import { getTeamColors, getTeamLogo } from '../lib/teamColors'
import { DRAFT_YEAR } from '../lib/config'
import type { DraftSimulationPick, LotteryPick } from '../types/index'

const POS_COLORS: Record<string, string> = {
  C: '#3b82f6', LW: '#10b981', RW: '#06b6d4', D: '#ef4444', G: '#f59e0b',
}

function getConfidenceBadge(pick: DraftSimulationPick): { label: string; className: string } | null {
  if (pick.in_prediction_set === false) {
    return { label: 'Volatile pick', className: 'bg-amber-500/10 text-amber-300 border border-amber-500/30' }
  }
  if (typeof pick.confidence === 'number') {
    return {
      label: pick.confidence >= 0.9 ? 'High confidence' : 'In model range',
      className: pick.confidence >= 0.9
        ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
        : 'bg-blue-500/10 text-blue-300 border border-blue-500/30',
    }
  }
  return null
}

function PosBadge({ pos }: { pos?: string | null }) {
  if (!pos) return null
  return (
    <span
      className="text-xs font-bold px-1.5 py-0.5 rounded flex-shrink-0"
      style={{ color: POS_COLORS[pos] ?? '#8892a4', background: (POS_COLORS[pos] ?? '#8892a4') + '22' }}
    >
      {pos}
    </span>
  )
}

function PickDetailPanel({ pick, onClose }: { pick: DraftSimulationPick; onClose: () => void }) {
  const colors = getTeamColors(pick.abbreviation)
  const confidenceBadge = getConfidenceBadge(pick)

  // Normalise scores for the bar chart: chosen + alternatives
  const allScores = [
    { label: pick.prospect_name, score: pick.ml_score ?? 0, chosen: true, pos: pick.position, cssRank: pick.css_rank },
    ...(pick.alternatives ?? []).map(a => ({
      label: a.prospect_name, score: a.ml_score, chosen: false, pos: a.position ?? undefined, cssRank: a.css_rank ?? undefined,
    })),
  ]
  const maxScore = Math.max(...allScores.map(s => s.score), 0.001)

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      transition={{ duration: 0.18 }}
      className="col-span-1 md:col-span-2 xl:col-span-3 rounded-xl bg-bg-card border border-border-default overflow-hidden"
      style={{ borderTop: `3px solid ${colors.primary}` }}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div
              className="w-10 h-10 rounded flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: colors.primary + '22' }}
            >
              <img
                src={getTeamLogo(pick.abbreviation)}
                alt={pick.abbreviation}
                className="w-8 h-8 object-contain"
                onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
              />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-white font-bold text-base">{pick.prospect_name}</span>
                <PosBadge pos={pick.position} />
              </div>
              <div className="text-text-muted text-xs">
                Pick #{pick.pick} · {pick.team_name}
                {pick.css_rank ? ` · CSS #${pick.css_rank}` : ''}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-white transition-colors text-xl leading-none px-1"
          >
            ×
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left: pick facts */}
          <div className="space-y-3">
            <div className="text-xs font-semibold uppercase tracking-widest text-text-muted mb-2">Pick details</div>

            <div className="grid grid-cols-2 gap-2">
              {pick.draft_league && (
                <div className="bg-bg-primary rounded-lg p-2.5">
                  <div className="text-text-muted text-[10px] uppercase tracking-wide mb-0.5">League</div>
                  <div className="text-white text-sm font-medium">{pick.draft_league}</div>
                </div>
              )}
              {pick.nationality && (
                <div className="bg-bg-primary rounded-lg p-2.5">
                  <div className="text-text-muted text-[10px] uppercase tracking-wide mb-0.5">Nationality</div>
                  <div className="text-white text-sm font-medium">{pick.nationality}</div>
                </div>
              )}
              {pick.points_per_game != null && (
                <div className="bg-bg-primary rounded-lg p-2.5">
                  <div className="text-text-muted text-[10px] uppercase tracking-wide mb-0.5">PPG (pre-draft)</div>
                  <div className="text-white text-sm font-medium">{pick.points_per_game.toFixed(2)}</div>
                </div>
              )}
              {pick.ml_score != null && (
                <div className="bg-bg-primary rounded-lg p-2.5">
                  <div className="text-text-muted text-[10px] uppercase tracking-wide mb-0.5">Model score</div>
                  <div className="text-white text-sm font-medium">{pick.ml_score.toFixed(3)}</div>
                </div>
              )}
            </div>

            {confidenceBadge && (
              <div className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold ${confidenceBadge.className}`}>
                {confidenceBadge.label === 'Volatile pick' ? '⚠ ' : '✓ '}
                {confidenceBadge.label}
              </div>
            )}

            {pick.in_prediction_set === false && (
              <p className="text-text-muted text-xs leading-relaxed">
                This pick fell outside the model's 90% prediction set — meaning it was a statistically
                surprising choice relative to the GM's historical tendencies and CSS consensus.
              </p>
            )}
          </div>

          {/* Right: model score comparison */}
          {allScores.length > 1 && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-widest text-text-muted mb-3">Model considered</div>
              <div className="space-y-2">
                {allScores.map((s, i) => (
                  <div key={i} className="space-y-0.5">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <PosBadge pos={s.pos} />
                        <span className={`truncate ${s.chosen ? 'text-white font-semibold' : 'text-text-secondary'}`}>
                          {s.label}
                        </span>
                        {s.cssRank && <span className="text-text-muted flex-shrink-0">#{s.cssRank}</span>}
                      </div>
                      <span className={`ml-2 flex-shrink-0 ${s.chosen ? 'text-accent-blue' : 'text-text-muted'}`}>
                        {s.score.toFixed(3)}
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-bg-primary overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${(s.score / maxScore) * 100}%` }}
                        transition={{ duration: 0.4, delay: i * 0.06 }}
                        className="h-full rounded-full"
                        style={{ background: s.chosen ? colors.primary : '#374151' }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-text-muted text-[10px] mt-3 leading-relaxed">
                Scores are the raw ML model output for this GM at this pick slot. CSS rank is one of the model's input features alongside GM tendency, position need, and prospect quality.
              </p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

export default function DraftPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const seedParam = searchParams.get('seed')
  const orderParam = searchParams.get('order')
  const { lotteryResult, lotterySeed, draftResult, setDraftResult, resetDraft } = useDraftStore()
  const [isSimulating, setIsSimulating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [summary, setSummary] = useState<string | null>(null)
  const [isSummarizing, setIsSummarizing] = useState(false)
  const [selectedPick, setSelectedPick] = useState<number | null>(null)

  const sharedOrderIds = (orderParam ?? '')
    .split(',')
    .map(part => Number(part.trim()))
    .filter(id => Number.isInteger(id) && id > 0)
  const activeSeed = seedParam ? Number(seedParam) : lotterySeed
  const picks: DraftSimulationPick[] = draftResult

  useEffect(() => {
    if (seedParam && !orderParam && lotteryResult.length === 0) {
      lotteryApi.simulate(Number(seedParam))
        .then(data => {
          const { setLotteryResult } = useDraftStore.getState()
          setLotteryResult(data.pick_order ?? [], data.seed)
        })
        .catch((err: Error) => setError(err?.message ?? 'Failed to refresh lottery'))
    }
  }, [seedParam, orderParam, lotteryResult.length])

  async function simulateDraft(rerunLottery = false) {
    setIsSimulating(true)
    setError(null)
    setSelectedPick(null)
    resetDraft()
    try {
      let lotteryOrderIds = lotteryResult.length > 0
        ? lotteryResult.map((p: LotteryPick) => p.team_id)
        : sharedOrderIds
      let currentSeed = activeSeed

      if (rerunLottery || lotteryOrderIds.length === 0) {
        const newSeed = Math.floor(Math.random() * 99999)
        const lotteryData = await lotteryApi.simulate(newSeed)
        const { setLotteryResult } = useDraftStore.getState()
        setLotteryResult(lotteryData.pick_order ?? [], lotteryData.seed)
        lotteryOrderIds = (lotteryData.pick_order ?? []).map((p: LotteryPick) => p.team_id)
        currentSeed = lotteryData.seed
      }

      const data = await draftApi.simulate(lotteryOrderIds, currentSeed)
      setDraftResult(data.picks ?? [])
      setSummary(null)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Simulation failed')
    } finally {
      setIsSimulating(false)
    }
  }

  async function generateSummary() {
    if (!picks.length) return
    setIsSummarizing(true)
    setSummary('')
    try {
      const res = await fetch(buildApiUrl('/api/draft/summary/stream'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ picks }),
      })
      if (!res.ok || !res.body) throw new Error('Stream unavailable')
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const payload = JSON.parse(line.slice(6))
            if (payload.token) setSummary(prev => (prev ?? '') + payload.token)
            if (payload.done) setIsSummarizing(false)
            if (payload.error) { setSummary(payload.error); setIsSummarizing(false) }
          } catch { /* skip malformed line */ }
        }
      }
    } catch {
      setSummary('Failed to generate summary.')
    } finally {
      setIsSummarizing(false)
    }
  }

  if (lotteryResult.length === 0 && !seedParam && sharedOrderIds.length === 0) {
    return (
      <main className="max-w-7xl mx-auto px-4 py-16 text-center">
        <h1 className="text-3xl font-bold text-white mb-4">Draft Simulator</h1>
        <p className="text-text-secondary mb-6">You need to run the lottery first to get a pick order.</p>
        <button
          onClick={() => navigate('/lottery')}
          className="px-6 py-3 bg-accent-blue hover:bg-blue-500 text-white font-semibold rounded-lg transition-colors"
        >
          Go to Lottery →
        </button>
      </main>
    )
  }

  return (
    <main className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold text-white">{DRAFT_YEAR} Draft Board</h1>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => simulateDraft(picks.length > 0)}
            disabled={isSimulating}
            className="px-4 py-2 text-sm bg-accent-blue hover:bg-blue-500 text-white font-semibold rounded-lg transition-colors disabled:opacity-50"
          >
            {picks.length > 0 ? 'Re-simulate' : 'Simulate Draft'}
          </button>
          {picks.length > 0 && (
            <button
              onClick={generateSummary}
              disabled={isSummarizing}
              className="px-4 py-2 text-sm bg-bg-card border border-border-default hover:border-accent-blue/60 text-text-secondary hover:text-white rounded-lg transition-colors disabled:opacity-50"
            >
              {isSummarizing ? 'Analyzing…' : 'AI Analysis'}
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-500/20 border border-red-500/30 text-red-400 text-sm">
          {error}
        </div>
      )}

      {isSimulating && (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="w-10 h-10 border-2 border-accent-blue border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-text-secondary">Simulating draft...</p>
          </div>
        </div>
      )}

      {picks.length === 0 && !isSimulating && (
        <div className="text-center py-20 text-text-secondary">
          <p className="mb-4">Pick order loaded. Click <strong className="text-white">Simulate Draft</strong> to run all 7 rounds.</p>
          {lotteryResult.length > 0 ? (
            <div className="max-w-md mx-auto space-y-1">
              {lotteryResult.slice(0, 5).map((p: LotteryPick) => {
                const colors = getTeamColors(p.abbreviation)
                return (
                  <div key={p.pick} className="flex items-center gap-3 text-left px-4 py-2 rounded-lg bg-bg-card border border-border-subtle" style={{ borderLeft: `3px solid ${colors.primary}` }}>
                    <span className="font-bold text-white w-6">#{p.pick}</span>
                    <span className="text-white text-sm">{p.team_name}</span>
                  </div>
                )
              })}
              {lotteryResult.length > 5 && <p className="text-text-muted text-xs pt-1">+{lotteryResult.length - 5} more teams</p>}
            </div>
          ) : (
            <p className="text-text-muted text-sm">Shared draft order loaded from the URL.</p>
          )}
        </div>
      )}

      {(summary !== null || isSummarizing) && (
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-5 p-4 rounded-xl bg-bg-card border border-accent-blue/30"
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="w-5 h-5 rounded-full bg-accent-blue/20 flex items-center justify-center text-xs text-accent-blue font-bold">S</span>
            <span className="text-xs text-accent-blue font-semibold uppercase tracking-wide">Scout's Analysis</span>
            {isSummarizing && (
              <div className="flex gap-1 items-center ml-1">
                {[0,1,2].map(i => (
                  <span key={i} className="w-1 h-1 rounded-full bg-accent-blue/60 animate-bounce" style={{ animationDelay: `${i*0.12}s` }} />
                ))}
              </div>
            )}
          </div>
          <p className="text-text-secondary text-sm leading-relaxed whitespace-pre-wrap">{summary}</p>
        </motion.div>
      )}

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
          <div className="space-y-8">
            {rounds.map(([roundNum, roundPicks]) => (
              <div key={roundNum}>
                <div className="flex items-center gap-3 mb-3">
                  <span className="text-xs font-bold uppercase tracking-widest text-text-muted">Round {roundNum}</span>
                  <div className="flex-1 h-px bg-border-subtle" />
                  <span className="text-xs text-text-muted">{roundPicks.length} picks</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-2">
                  {roundPicks.map((pick) => {
                    const colors = getTeamColors(pick.abbreviation)
                    const confidenceBadge = getConfidenceBadge(pick)
                    const isSelected = selectedPick === pick.pick
                    const idx = globalIdx++
                    return (
                      <>
                        <motion.div
                          key={pick.pick}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: Math.min(idx * 0.01, 0.5) }}
                          className={`flex flex-col p-3 rounded-lg bg-bg-card border transition-colors cursor-pointer ${
                            isSelected
                              ? 'border-accent-blue/60 bg-bg-hover'
                              : 'border-border-subtle hover:bg-bg-hover'
                          }`}
                          style={{ borderLeft: `3px solid ${colors.primary}` }}
                          onClick={() => setSelectedPick(isSelected ? null : pick.pick)}
                        >
                          <div className="flex items-center gap-3">
                            <div className="text-right flex-shrink-0 w-8">
                              <div className="text-base font-black text-white leading-tight">{pick.pick}</div>
                            </div>
                            <div
                              className="w-9 h-9 rounded flex items-center justify-center flex-shrink-0 overflow-hidden"
                              style={{ backgroundColor: colors.primary + '22' }}
                            >
                              <img
                                src={getTeamLogo(pick.abbreviation)}
                                alt={pick.abbreviation}
                                className="w-7 h-7 object-contain"
                                onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
                              />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="text-white font-semibold text-sm truncate">
                                {pick.prospect_name ?? 'TBD'}
                              </div>
                              <div className="text-text-muted text-xs truncate">
                                {pick.team_name}
                                {pick.css_rank && <span className="ml-1">· CSS #{pick.css_rank}</span>}
                              </div>
                              {confidenceBadge && (
                                <div className="mt-1">
                                  <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold ${confidenceBadge.className}`}>
                                    {confidenceBadge.label}
                                  </span>
                                </div>
                              )}
                            </div>
                            <PosBadge pos={pick.position} />
                          </div>
                        </motion.div>

                        <AnimatePresence>
                          {isSelected && (
                            <PickDetailPanel
                              key={`detail-${pick.pick}`}
                              pick={pick}
                              onClose={() => setSelectedPick(null)}
                            />
                          )}
                        </AnimatePresence>
                      </>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        )
      })()}
    </main>
  )
}
