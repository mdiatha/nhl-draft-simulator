import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { motion } from 'framer-motion'
import { teamsApi } from '../lib/api'
import { getTeamColors, getTeamLogo } from '../lib/teamColors'
import type { TeamTendency, DraftPickRecord } from '../types'

const POS_COLORS: Record<string, string> = {
  C: '#3b82f6', LW: '#10b981', RW: '#06b6d4', D: '#ef4444', G: '#f59e0b',
}

const NAT_FLAGS: Record<string, string> = {
  CAN: '🇨🇦', USA: '🇺🇸', SWE: '🇸🇪', FIN: '🇫🇮', RUS: '🇷🇺',
  CZE: '🇨🇿', SVK: '🇸🇰', DEU: '🇩🇪',
  CHE: '🇨🇭', SUI: '🇨🇭',  // Switzerland (both ISO codes used)
  DNK: '🇩🇰', NOR: '🇳🇴', AUT: '🇦🇹', BLR: '🇧🇾', LVA: '🇱🇻', UKR: '🇺🇦',
  GBR: '🇬🇧', FRA: '🇫🇷', SVN: '🇸🇮', HUN: '🇭🇺',
}

// Display code for nationality (maps DB codes to readable abbreviations)
const NAT_DISPLAY: Record<string, string> = {
  CHE: 'SUI', SUI: 'SUI',
}

const NAT_COLORS: Record<string, string> = {
  CAN: '#ef4444', USA: '#3b82f6', SWE: '#f59e0b', FIN: '#06b6d4',
  RUS: '#ef4444', CZE: '#8b5cf6', SVK: '#10b981', DEU: '#f59e0b',
  CHE: '#ef4444', SUI: '#ef4444', DNK: '#ef4444', default: '#8892a4',
}



function SectionTitle({ children, subtitle }: { children: React.ReactNode; subtitle?: string }) {
  return (
    <div className="mb-4">
      <h2 className="text-base font-semibold text-white">{children}</h2>
      {subtitle && <p className="text-text-muted text-xs mt-0.5">{subtitle}</p>}
    </div>
  )
}

export default function TeamPage() {
  const { teamId } = useParams<{ teamId: string }>()
  const navigate = useNavigate()

  const { data, isLoading, error } = useQuery<TeamTendency>({
    queryKey: ['team-tendency', teamId],
    queryFn: () => teamsApi.getTendency(Number(teamId)),
    enabled: !!teamId,
  })

  if (isLoading) return (
    <main className="max-w-5xl mx-auto px-4 py-8 space-y-4">
      {Array.from({ length: 4 }).map((_, i) => <div key={i} className="skeleton h-32 w-full rounded-xl" />)}
    </main>
  )

  if (error || !data) return (
    <main className="max-w-5xl mx-auto px-4 py-8">
      <p className="text-red-400">Failed to load team data.</p>
      <button onClick={() => navigate(-1)} className="mt-4 text-accent-blue hover:underline text-sm">← Back</button>
    </main>
  )

  const colors = getTeamColors(data.abbreviation)
  const gmStart = data.gm?.start_date ? new Date(data.gm.start_date) : null

  // Filter draft history to current GM's tenure
  const gmPicks: DraftPickRecord[] = (data.draft_history ?? []).filter(p => {
    if (!gmStart) return true
    return p.year >= gmStart.getFullYear()
  }).sort((a, b) => b.year - a.year || a.overall_pick - b.overall_pick)

  const totalGmPicks = gmPicks.length
  const gmDrafts = new Set(gmPicks.map(p => p.year)).size

  // Position breakdown from GM picks
  const posCounts: Record<string, number> = {}
  for (const p of gmPicks) {
    const pos = p.position?.split('/')[0] ?? 'UNK'
    posCounts[pos] = (posCounts[pos] ?? 0) + 1
  }
  const posData = Object.entries(posCounts)
    .map(([pos, count]) => ({ pos, pct: Math.round(count / totalGmPicks * 100), count }))
    .sort((a, b) => b.pct - a.pct)

  // Nationality breakdown from GM picks — per country
  const natCounts: Record<string, number> = {}
  for (const p of gmPicks) {
    if (p.nationality) natCounts[p.nationality] = (natCounts[p.nationality] ?? 0) + 1
  }
  const natData = Object.entries(natCounts)
    .map(([nat, count]) => ({ nat, count, pct: Math.round(count / totalGmPicks * 100), flag: NAT_FLAGS[nat] ?? '🌍', color: NAT_COLORS[nat] ?? NAT_COLORS.default }))
    .sort((a, b) => b.count - a.count)

  // League breakdown
  const leagueCounts: Record<string, number> = {}
  for (const p of gmPicks) {
    const l = p.draft_league ?? 'Unknown'
    leagueCounts[l] = (leagueCounts[l] ?? 0) + 1
  }
  const leagueData = Object.entries(leagueCounts)
    .map(([league, count]) => ({ league, pct: Math.round(count / totalGmPicks * 100), count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 7)

  // Round breakdown
  const byRound: Record<number, number> = {}
  for (const p of gmPicks) byRound[p.round] = (byRound[p.round] ?? 0) + 1

  return (
    <main className="max-w-5xl mx-auto px-4 py-8 space-y-8">

      {/* ── Header ── */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="flex items-start gap-5">
        <div
          className="w-16 h-16 rounded-xl flex items-center justify-center font-black text-lg flex-shrink-0 overflow-hidden"
          style={{ background: colors.primary + '22', border: `2px solid ${colors.primary}44` }}
        >
          <img src={getTeamLogo(data.abbreviation)} alt="" className="w-12 h-12 object-contain"
            onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }} />
        </div>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">{data.team_name}</h1>
          <p className="text-text-secondary text-sm mt-1">
            GM: <span className="text-white font-medium">{data.gm?.name ?? '—'}</span>
            {data.gm?.start_date && <span className="text-text-muted ml-2">since {data.gm.start_date}</span>}
          </p>
          <div className="flex gap-4 mt-2 text-xs text-text-secondary">
            <span><span className="text-white font-semibold">{gmDrafts}</span> drafts under this GM</span>
          </div>
        </div>
      </motion.div>


      {/* ── Drafting Tendencies ── */}
      {totalGmPicks > 0 && (
        <section>
          <SectionTitle subtitle={`Based on ${totalGmPicks} picks since ${data.gm?.start_date?.slice(0,4) ?? '—'}. Weights show % of picks at each value.`}>
            Drafting Tendencies
          </SectionTitle>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

            {/* Position */}
            <div className="bg-bg-card border border-border-subtle rounded-xl p-4">
              <p className="text-sm font-medium text-white mb-1">By Position</p>
              <p className="text-xs text-text-muted mb-3">Which positions this GM targets most in the draft</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={posData} layout="vertical" margin={{ left: 0, right: 20 }}>
                  <XAxis type="number" hide domain={[0, 100]} />
                  <YAxis type="category" dataKey="pos" tick={{ fontSize: 11, fill: '#8892a4' }} width={28} />
                  <Tooltip
                    formatter={(v: number, _: string, props: { payload?: { count?: number } }) => [`${v}% (${props?.payload?.count} picks)`, '']}
                    contentStyle={{ background: '#141929', border: '1px solid #2a3a52', borderRadius: 6, fontSize: 12 }}
                    labelStyle={{ color: '#e8eaf0' }}
                    itemStyle={{ color: '#e8eaf0' }}
                  />
                  <Bar dataKey="pct" radius={[0, 4, 4, 0]}>
                    {posData.map(e => <Cell key={e.pos} fill={POS_COLORS[e.pos] ?? '#8892a4'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* League */}
            <div className="bg-bg-card border border-border-subtle rounded-xl p-4">
              <p className="text-sm font-medium text-white mb-1">By League</p>
              <p className="text-xs text-text-muted mb-3">Which leagues this GM scouts and drafts from most</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={leagueData} layout="vertical" margin={{ left: 0, right: 20 }}>
                  <XAxis type="number" hide domain={[0, 100]} />
                  <YAxis type="category" dataKey="league" tick={{ fontSize: 10, fill: '#8892a4' }} width={60} />
                  <Tooltip
                    formatter={(v: number, _: string, props: { payload?: { count?: number } }) => [`${v}% (${props?.payload?.count} picks)`, '']}
                    contentStyle={{ background: '#141929', border: '1px solid #2a3a52', borderRadius: 6, fontSize: 12 }}
                    labelStyle={{ color: '#e8eaf0' }}
                    itemStyle={{ color: '#e8eaf0' }}
                  />
                  <Bar dataKey="pct" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Nationality */}
            <div className="bg-bg-card border border-border-subtle rounded-xl p-4">
              <p className="text-sm font-medium text-white mb-1">By Nationality</p>
              <p className="text-xs text-text-muted mb-3">Countries this GM drafts from, by % of total picks</p>
              <div className="space-y-2 mt-1">
                {natData.map(({ nat, pct, count, flag, color }) => (
                  <div key={nat} className="flex items-center gap-2">
                    <span className="text-base w-6">{flag}</span>
                    <span className="text-xs text-text-secondary w-8">{NAT_DISPLAY[nat] ?? nat}</span>
                    <div className="flex-1 bg-bg-secondary rounded-full h-2 overflow-hidden">
                      <div className="h-2 rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
                    </div>
                    <span className="text-xs font-semibold text-white w-8 text-right">{pct}%</span>
                    <span className="text-xs text-text-muted w-10 text-right">{count} picks</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ── Picks Under This GM ── */}
      <section>
        <SectionTitle subtitle={`All draft picks made by ${data.gm?.name ?? 'this GM'} — sorted newest first`}>
          Picks Under This GM
        </SectionTitle>
        <div className="bg-bg-card border border-border-subtle rounded-xl overflow-hidden">
          <div className="overflow-x-auto max-h-[520px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-bg-card border-b border-border-subtle">
                <tr className="text-text-muted text-xs uppercase tracking-wide">
                  <th className="px-3 py-2.5 text-left">Year</th>
                  <th className="px-3 py-2.5 text-left">Rd</th>
                  <th className="px-3 py-2.5 text-left">OA</th>
                  <th className="px-3 py-2.5 text-left">Player</th>
                  <th className="px-3 py-2.5 text-left">Pos</th>
                  <th className="px-3 py-2.5 text-left">League</th>
                  <th className="px-3 py-2.5 text-left">Nat</th>
                </tr>
              </thead>
              <tbody>
                {gmPicks.length === 0 ? (
                  <tr><td colSpan={7} className="px-4 py-8 text-center text-text-muted text-sm">No picks on record for this GM</td></tr>
                ) : gmPicks.map((pick, i) => (
                  <tr key={i} className="border-b border-border-subtle hover:bg-bg-hover transition-colors">
                    <td className="px-3 py-2 text-text-secondary font-mono text-xs">{pick.year}</td>
                    <td className="px-3 py-2 text-text-muted text-xs">{pick.round}</td>
                    <td className="px-3 py-2 text-text-muted font-mono text-xs">{pick.overall_pick}</td>
                    <td className="px-3 py-2 text-white font-medium text-sm">{pick.player_name ?? '—'}</td>
                    <td className="px-3 py-2">
                      {pick.position && (
                        <span className="text-xs font-bold px-1.5 py-0.5 rounded"
                          style={{ color: POS_COLORS[pick.position?.split('/')[0]] ?? '#8892a4', background: (POS_COLORS[pick.position?.split('/')[0]] ?? '#8892a4') + '22' }}>
                          {pick.position}
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-text-secondary text-xs">{pick.draft_league ?? '—'}</td>
                    <td className="px-3 py-2 text-sm">
                      {pick.nationality ? (
                        <span title={pick.nationality}>{NAT_FLAGS[pick.nationality] ?? '🌍'} <span className="text-text-muted text-xs">{NAT_DISPLAY[pick.nationality] ?? pick.nationality}</span></span>
                      ) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Round summary footer */}
          {totalGmPicks > 0 && (
            <div className="px-4 py-2.5 border-t border-border-subtle flex gap-4 flex-wrap">
              {Object.entries(byRound).sort(([a], [b]) => Number(a) - Number(b)).map(([rnd, count]) => (
                <span key={rnd} className="text-xs text-text-muted">
                  R{rnd}: <span className="text-white font-medium">{count}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      </section>

    </main>
  )
}
