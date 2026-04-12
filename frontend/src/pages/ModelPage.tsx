import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts'
import { mlApi } from '../lib/api'

// ── Types ─────────────────────────────────────────────────────────────────────

interface RoundBreakdown {
  n: number
  top1: number
  top3: number
  top5: number
  top1_pct: number
  top3_pct: number
  top5_pct: number
}

interface ArchetypeBreakdown {
  n: number
  top1: number
  top3: number
  top1_pct: number
  top3_pct: number
}

interface WorstMiss {
  pick: number
  round: number
  team: string
  actual: string
  predicted: string
  actual_rank: number
  pool_size: number
}

interface BacktestMetrics {
  test_year: number
  train_cutoff: number
  picks_evaluated: number
  top1_accuracy: number
  top3_accuracy: number
  top5_accuracy: number
  mrr: number
  by_round: Record<string, RoundBreakdown>
  by_archetype: Record<string, ArchetypeBreakdown>
  worst_misses: WorstMiss[]
  baselines: Record<string, BaselineMetrics>
  lift_vs_baselines: Record<string, BaselineLift>
}

interface BaselineMetrics {
  label: string
  top1_accuracy: number
  top3_accuracy: number
  top5_accuracy: number
  mrr: number
}

interface BaselineLift {
  top1_accuracy: number
  top3_accuracy: number
  top5_accuracy: number
  mrr: number
}

interface AggregateMetric {
  mean: number
  std: number
  min: number
  max: number
  n: number
}

interface MultiYearResult {
  years_evaluated: number[]
  errors: string[]
  per_year: BacktestMetrics[]
  aggregate: Record<string, AggregateMetric>
  baseline_aggregate: Record<string, {
    label: string
    top1_accuracy: AggregateMetric
    top3_accuracy: AggregateMetric
    top5_accuracy: AggregateMetric
    mrr: AggregateMetric
  }>
}

interface MLStatus {
  model_loaded: boolean
  meta: {
    trained_at?: string
    validation_auc?: number
    mode?: string
    training_samples?: number
    feature_importances?: Record<string, number>
  } | null
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function pct(v: number) { return `${(v * 100).toFixed(1)}%` }
function deltaPct(v: number) { return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)} pts` }

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-bg-card border border-border-subtle rounded-xl p-4 flex flex-col gap-1">
      <p className="text-text-muted text-xs uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-black text-white">{value}</p>
      {sub && <p className="text-text-muted text-xs">{sub}</p>}
    </div>
  )
}

const ARCHETYPE_COLORS: Record<string, string> = {
  BPA: '#3b82f6',
  'need-based': '#10b981',
  safe: '#f59e0b',
  'boom-bust': '#ef4444',
  'system-fit': '#8b5cf6',
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ModelPage() {
  const [metrics, setMetrics] = useState<BacktestMetrics | null>(null)
  const [multiYear, setMultiYear] = useState<MultiYearResult | null>(null)
  const [status, setStatus] = useState<MLStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [multiLoading, setMultiLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [testYear, setTestYear] = useState(2024)
  const [trainCutoff, setTrainCutoff] = useState(2023)

  async function loadStatus() {
    try {
      setStatus(await mlApi.status())
    } catch { /* ignore */ }
  }

  async function runBacktest() {
    setLoading(true)
    setError(null)
    setMetrics(null)
    if (!status) await loadStatus()
    try {
      const data = await mlApi.backtest(testYear, trainCutoff)
      setMetrics(data.metrics)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Backtest failed')
    } finally {
      setLoading(false)
    }
  }

  async function runMultiYearBacktest() {
    setMultiLoading(true)
    setError(null)
    try {
      const data = await mlApi.backtestMultiYear()
      setMultiYear(data.result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Multi-year backtest failed')
    } finally {
      setMultiLoading(false)
    }
  }

  // load status on mount
  useEffect(() => { loadStatus() }, [])

  // ── Derived chart data ──────────────────────────────────────────────────────

  const roundData = metrics
    ? Object.entries(metrics.by_round)
        .sort(([a], [b]) => parseInt(a.split('_')[1]) - parseInt(b.split('_')[1]))
        .map(([key, v]) => ({
          round: `R${key.split('_')[1]}`,
          'Top-1': +(v.top1_pct * 100).toFixed(1),
          'Top-3': +(v.top3_pct * 100).toFixed(1),
          'Top-5': +(v.top5_pct * 100).toFixed(1),
          n: v.n,
        }))
    : []

  const archetypeRadar = metrics
    ? Object.entries(metrics.by_archetype).map(([arch, v]) => ({
        archetype: arch,
        'Top-1 %': +(v.top1_pct * 100).toFixed(1),
        'Top-3 %': +(v.top3_pct * 100).toFixed(1),
      }))
    : []

  const featureData = status?.meta?.feature_importances
    ? Object.entries(status.meta.feature_importances)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 10)
        .map(([name, score]) => ({
          name: name.replace(/_/g, ' '),
          score: +score.toFixed(4),
        }))
    : []

  const baselineRows = metrics ? Object.entries(metrics.baselines) : []
  const perYearRows = multiYear?.per_year ?? []

  return (
    <main className="max-w-5xl mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Model Performance</h1>
        <p className="text-text-secondary text-sm mt-1">
          Temporal backtest: train on picks ≤ cutoff year, evaluate on held-out draft year.
        </p>
      </div>

      {/* Model status strip */}
      {status && (
        <div className="mb-6 flex flex-wrap gap-3">
          <div className={`px-3 py-1.5 rounded-full text-xs font-semibold border ${status.model_loaded ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-red-500/10 border-red-500/30 text-red-400'}`}>
            {status.model_loaded ? 'Model loaded' : 'No model loaded'}
          </div>
          {status.meta?.trained_at && (
            <div className="px-3 py-1.5 rounded-full text-xs border border-border-subtle text-text-muted">
              Trained {new Date(status.meta.trained_at).toLocaleDateString()}
            </div>
          )}
          {status.meta?.validation_auc && (
            <div className="px-3 py-1.5 rounded-full text-xs border border-border-subtle text-text-muted">
              Validation AUC <span className="text-white font-semibold">{status.meta.validation_auc.toFixed(3)}</span>
            </div>
          )}
          {status.meta?.training_samples && (
            <div className="px-3 py-1.5 rounded-full text-xs border border-border-subtle text-text-muted">
              {status.meta.training_samples.toLocaleString()} training rows
            </div>
          )}
          {status.meta?.mode && (
            <div className="px-3 py-1.5 rounded-full text-xs border border-border-subtle text-text-muted capitalize">
              {status.meta.mode} mode
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-3 mb-6">
        <div>
          <label className="block text-xs text-text-muted mb-1">Test year</label>
          <select
            value={testYear}
            onChange={e => setTestYear(+e.target.value)}
            className="bg-bg-card border border-border-subtle rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-blue/60"
          >
            {[2024, 2023, 2022, 2021, 2020].map(y => <option key={y} value={y}>{y}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">Train cutoff</label>
          <select
            value={trainCutoff}
            onChange={e => setTrainCutoff(+e.target.value)}
            className="bg-bg-card border border-border-subtle rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-blue/60"
          >
            {[2023, 2022, 2021, 2020, 2019].map(y => <option key={y} value={y}>{y}</option>)}
          </select>
        </div>
        <button
          onClick={runBacktest}
          disabled={loading}
          className="px-5 py-2 bg-accent-blue hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-semibold rounded-lg transition-colors"
        >
          {loading ? 'Running backtest…' : 'Run Backtest'}
        </button>
        <button
          onClick={runMultiYearBacktest}
          disabled={multiLoading}
          className="px-5 py-2 bg-bg-card border border-border-default hover:border-accent-blue/60 text-text-secondary hover:text-white disabled:opacity-50 text-sm font-semibold rounded-lg transition-colors"
        >
          {multiLoading ? 'Running multi-year…' : 'Run Multi-Year'}
        </button>
        {loading && (
          <p className="text-text-muted text-xs self-center">
            Training a held-out model — this takes ~30 seconds
          </p>
        )}
        {multiLoading && (
          <p className="text-text-muted text-xs self-center">
            Rolling year-by-year evaluation — this takes a few minutes
          </p>
        )}
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {metrics && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">

          {/* Top-line stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatCard label="Top-1 Accuracy" value={pct(metrics.top1_accuracy)} sub="Actual pick was model's #1 choice" />
            <StatCard label="Top-3 Accuracy" value={pct(metrics.top3_accuracy)} sub="Actual pick in model's top 3" />
            <StatCard label="Top-5 Accuracy" value={pct(metrics.top5_accuracy)} sub="Actual pick in model's top 5" />
            <StatCard label="Mean Reciprocal Rank" value={metrics.mrr.toFixed(3)} sub={`${metrics.picks_evaluated} picks evaluated`} />
          </div>

          {baselineRows.length > 0 && (
            <div className="bg-bg-card border border-border-subtle rounded-xl p-5">
              <h2 className="text-sm font-semibold text-white mb-1">Baseline Comparisons</h2>
              <p className="text-xs text-text-muted mb-4">
                The model is benchmarked against simple heuristics so we can measure actual lift, not just raw accuracy.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm min-w-[560px]">
                  <thead>
                    <tr className="text-text-muted text-xs">
                      <th className="text-left pb-2">Approach</th>
                      <th className="text-right pb-2">Top-1</th>
                      <th className="text-right pb-2">Top-3</th>
                      <th className="text-right pb-2">Top-5</th>
                      <th className="text-right pb-2">MRR</th>
                      <th className="text-right pb-2">Lift vs model</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-t border-border-subtle">
                      <td className="py-2 text-white font-semibold">XGBoost ranker</td>
                      <td className="py-2 text-right text-white">{pct(metrics.top1_accuracy)}</td>
                      <td className="py-2 text-right text-white">{pct(metrics.top3_accuracy)}</td>
                      <td className="py-2 text-right text-white">{pct(metrics.top5_accuracy)}</td>
                      <td className="py-2 text-right text-white">{metrics.mrr.toFixed(3)}</td>
                      <td className="py-2 text-right text-emerald-400 font-semibold">Baseline</td>
                    </tr>
                    {baselineRows.map(([key, baseline]) => {
                      const lift = metrics.lift_vs_baselines[key]
                      return (
                        <tr key={key} className="border-t border-border-subtle">
                          <td className="py-2 text-text-secondary">{baseline.label}</td>
                          <td className="py-2 text-right text-text-secondary">{pct(baseline.top1_accuracy)}</td>
                          <td className="py-2 text-right text-text-secondary">{pct(baseline.top3_accuracy)}</td>
                          <td className="py-2 text-right text-text-secondary">{pct(baseline.top5_accuracy)}</td>
                          <td className="py-2 text-right text-text-secondary">{baseline.mrr.toFixed(3)}</td>
                          <td className="py-2 text-right text-emerald-400 font-semibold">{deltaPct(lift.top1_accuracy)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* By-round bar chart */}
          {roundData.length > 0 && (
            <div className="bg-bg-card border border-border-subtle rounded-xl p-5">
              <h2 className="text-sm font-semibold text-white mb-4">Accuracy by Round</h2>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={roundData} barGap={2}>
                  <XAxis dataKey="round" tick={{ fill: '#8892a4', fontSize: 12 }} axisLine={false} tickLine={false} />
                  <YAxis tickFormatter={v => `${v}%`} tick={{ fill: '#8892a4', fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ background: '#1a2035', border: '1px solid #2a3550', borderRadius: 8 }}
                    labelStyle={{ color: '#fff' }}
                    formatter={(v: number) => [`${v}%`]}
                  />
                  <Bar dataKey="Top-1" fill="#3b82f6" radius={[3,3,0,0]} />
                  <Bar dataKey="Top-3" fill="#10b981" radius={[3,3,0,0]} />
                  <Bar dataKey="Top-5" fill="#8b5cf6" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="flex gap-4 mt-2 justify-center">
                {[['Top-1','#3b82f6'],['Top-3','#10b981'],['Top-5','#8b5cf6']].map(([l,c]) => (
                  <div key={l} className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-sm" style={{ background: c }} />
                    <span className="text-xs text-text-muted">{l}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* By-archetype table + radar */}
          {archetypeRadar.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-bg-card border border-border-subtle rounded-xl p-5">
                <h2 className="text-sm font-semibold text-white mb-4">Accuracy by GM Archetype</h2>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-text-muted text-xs">
                      <th className="text-left pb-2">Archetype</th>
                      <th className="text-right pb-2">n</th>
                      <th className="text-right pb-2">Top-1</th>
                      <th className="text-right pb-2">Top-3</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics.by_archetype).map(([arch, v]) => (
                      <tr key={arch} className="border-t border-border-subtle">
                        <td className="py-2 flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: ARCHETYPE_COLORS[arch] ?? '#8892a4' }} />
                          <span className="text-white">{arch}</span>
                        </td>
                        <td className="py-2 text-right text-text-muted">{v.n}</td>
                        <td className="py-2 text-right text-white font-semibold">{pct(v.top1_pct)}</td>
                        <td className="py-2 text-right text-text-muted">{pct(v.top3_pct)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="bg-bg-card border border-border-subtle rounded-xl p-5">
                <h2 className="text-sm font-semibold text-white mb-2">Top-1 % by Archetype</h2>
                <ResponsiveContainer width="100%" height={200}>
                  <RadarChart data={archetypeRadar}>
                    <PolarGrid stroke="#2a3550" />
                    <PolarAngleAxis dataKey="archetype" tick={{ fill: '#8892a4', fontSize: 11 }} />
                    <Radar dataKey="Top-1 %" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Feature importance */}
          {featureData.length > 0 && (
            <div className="bg-bg-card border border-border-subtle rounded-xl p-5">
              <h2 className="text-sm font-semibold text-white mb-4">Top-10 Feature Importances</h2>
              <div className="space-y-2">
                {featureData.map((f, i) => {
                  const pctBar = (f.score / featureData[0].score) * 100
                  return (
                    <div key={f.name} className="flex items-center gap-3">
                      <span className="text-text-muted text-xs w-4 text-right flex-shrink-0">{i+1}</span>
                      <span className="text-xs text-text-secondary w-36 flex-shrink-0 truncate capitalize">{f.name}</span>
                      <div className="flex-1 bg-bg-secondary rounded-full h-2">
                        <div className="h-2 rounded-full bg-accent-blue" style={{ width: `${pctBar}%` }} />
                      </div>
                      <span className="text-xs text-text-muted w-12 text-right flex-shrink-0">{f.score.toFixed(4)}</span>
                    </div>
                  )
                })}
              </div>
              <p className="text-xs text-text-muted mt-3">XGBoost gain-based importance from current loaded model.</p>
            </div>
          )}

          {/* Worst misses */}
          {metrics.worst_misses.length > 0 && (
            <div className="bg-bg-card border border-border-subtle rounded-xl p-5">
              <h2 className="text-sm font-semibold text-white mb-4">Biggest Misses</h2>
              <p className="text-xs text-text-muted mb-3">Picks where the model ranked the actual selection furthest from #1.</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm min-w-[520px]">
                  <thead>
                    <tr className="text-text-muted text-xs">
                      <th className="text-left pb-2">Pick</th>
                      <th className="text-left pb-2">Team</th>
                      <th className="text-left pb-2">Actual pick</th>
                      <th className="text-left pb-2">Model's #1</th>
                      <th className="text-right pb-2">Model rank</th>
                      <th className="text-right pb-2">Pool</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.worst_misses.map((m, i) => (
                      <tr key={i} className="border-t border-border-subtle">
                        <td className="py-2 text-text-muted">#{m.pick} R{m.round}</td>
                        <td className="py-2 text-white font-medium">{m.team}</td>
                        <td className="py-2 text-white">{m.actual}</td>
                        <td className="py-2 text-text-muted">{m.predicted}</td>
                        <td className="py-2 text-right">
                          <span className="text-red-400 font-semibold">#{m.actual_rank}</span>
                        </td>
                        <td className="py-2 text-right text-text-muted">{m.pool_size}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <p className="text-xs text-text-muted">
            Train cutoff: ≤{metrics.train_cutoff} · Test year: {metrics.test_year} · {metrics.picks_evaluated} picks evaluated
          </p>
        </motion.div>
      )}

      {multiYear && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-8 space-y-6">
          <div className="bg-bg-card border border-border-subtle rounded-xl p-5">
            <h2 className="text-sm font-semibold text-white mb-1">Multi-Year Backtest</h2>
            <p className="text-xs text-text-muted mb-4">
              Held-out evaluation across {multiYear.years_evaluated.join(', ')} to show whether performance is stable year to year.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
              <StatCard
                label="Top-1 Mean"
                value={pct(multiYear.aggregate.top1_accuracy.mean)}
                sub={`± ${(multiYear.aggregate.top1_accuracy.std * 100).toFixed(1)} pts across ${multiYear.aggregate.top1_accuracy.n} years`}
              />
              <StatCard
                label="Top-3 Mean"
                value={pct(multiYear.aggregate.top3_accuracy.mean)}
                sub={`Range ${pct(multiYear.aggregate.top3_accuracy.min)}–${pct(multiYear.aggregate.top3_accuracy.max)}`}
              />
              <StatCard
                label="Top-5 Mean"
                value={pct(multiYear.aggregate.top5_accuracy.mean)}
                sub={`Range ${pct(multiYear.aggregate.top5_accuracy.min)}–${pct(multiYear.aggregate.top5_accuracy.max)}`}
              />
              <StatCard
                label="MRR Mean"
                value={multiYear.aggregate.mrr.mean.toFixed(3)}
                sub={`± ${multiYear.aggregate.mrr.std.toFixed(3)}`}
              />
            </div>
          </div>

          {perYearRows.length > 0 && (
            <div className="bg-bg-card border border-border-subtle rounded-xl p-5">
              <h2 className="text-sm font-semibold text-white mb-4">Per-Year Results</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm min-w-[680px]">
                  <thead>
                    <tr className="text-text-muted text-xs">
                      <th className="text-left pb-2">Year</th>
                      <th className="text-right pb-2">Top-1</th>
                      <th className="text-right pb-2">Top-3</th>
                      <th className="text-right pb-2">Top-5</th>
                      <th className="text-right pb-2">MRR</th>
                      <th className="text-right pb-2">CSS BPA</th>
                      <th className="text-right pb-2">PPG BPA</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perYearRows.map((row) => (
                      <tr key={row.test_year} className="border-t border-border-subtle">
                        <td className="py-2 text-white font-semibold">{row.test_year}</td>
                        <td className="py-2 text-right text-white">{pct(row.top1_accuracy)}</td>
                        <td className="py-2 text-right text-text-secondary">{pct(row.top3_accuracy)}</td>
                        <td className="py-2 text-right text-text-secondary">{pct(row.top5_accuracy)}</td>
                        <td className="py-2 text-right text-text-secondary">{row.mrr.toFixed(3)}</td>
                        <td className="py-2 text-right text-text-muted">{pct(row.baselines.consensus_css.top1_accuracy)}</td>
                        <td className="py-2 text-right text-text-muted">{pct(row.baselines.production_ppg.top1_accuracy)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {!metrics && !loading && (
        <div className="text-center py-20 text-text-secondary">
          <p className="mb-2 text-white font-medium">No backtest results yet</p>
          <p className="text-sm">Click <strong className="text-white">Run Backtest</strong> to train a held-out model and compare it against simple baselines, or <strong className="text-white">Run Multi-Year</strong> to see year-by-year stability.</p>
        </div>
      )}
    </main>
  )
}
