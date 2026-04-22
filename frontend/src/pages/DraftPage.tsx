import { useEffect, useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { buildApiUrl, draftApi, lotteryApi } from '../lib/api'
import { useDraftStore } from '../stores/draftStore'
import { getTeamColors, getTeamLogo } from '../lib/teamColors'
import { DRAFT_YEAR } from '../lib/config'
import type { DraftSimulationPick, LotteryPick } from '../types/index'

function getConfidenceBadge(pick: DraftSimulationPick): { label: string; className: string } | null {
  if (pick.in_prediction_set === false) {
    return {
      label: 'Volatile pick',
      className: 'bg-amber-500/10 text-amber-300 border border-amber-500/30',
    }
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

  const sharedOrderIds = (orderParam ?? '')
    .split(',')
    .map(part => Number(part.trim()))
    .filter(id => Number.isInteger(id) && id > 0)
  const activeSeed = seedParam ? Number(seedParam) : lotterySeed
  const picks: DraftSimulationPick[] = draftResult

  useEffect(() => {
    // If a shared seed is present without explicit order data, fetch the lottery result.
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
    resetDraft()
    try {
      let lotteryOrderIds = lotteryResult.length > 0
        ? lotteryResult.map((p: LotteryPick) => p.team_id)
        : sharedOrderIds
      let currentSeed = activeSeed

      // Re-simulate re-runs the lottery with a fresh seed first
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
        // Group by round
        const byRound = new Map<number, typeof picks>()
        for (const pick of picks) {
          const r = pick.round ?? 1
          if (!byRound.has(r)) byRound.set(r, [])
          byRound.get(r)!.push(pick)
        }
        const rounds = Array.from(byRound.entries()).sort(([a], [b]) => a - b)
        const POS_COLORS: Record<string, string> = { C: '#3b82f6', LW: '#10b981', RW: '#06b6d4', D: '#ef4444', G: '#f59e0b' }
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
                    const idx = globalIdx++
                    return (
                      <motion.div
                        key={pick.pick}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: Math.min(idx * 0.01, 0.5) }}
                        className="flex flex-col p-3 rounded-lg bg-bg-card border border-border-subtle hover:bg-bg-hover transition-colors"
                        style={{ borderLeft: `3px solid ${colors.primary}` }}
                      >
                        <div
                          className="flex items-center gap-3 cursor-pointer"
                          onClick={() => navigate(`/teams/${pick.team_id}`)}
                        >
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
                          {pick.position && (
                            <span className="text-xs font-bold px-1.5 py-0.5 rounded flex-shrink-0" style={{ color: (POS_COLORS[pick.position] ?? '#8892a4'), background: ((POS_COLORS[pick.position] ?? '#8892a4') + '22') }}>
                              {pick.position}
                            </span>
                          )}
                        </div>
                      </motion.div>
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
