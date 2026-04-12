/**
 * ShapDrawer — inline "Why this pick?" SHAP explanation panel.
 *
 * Renders a horizontal bar chart of the top SHAP features for a single
 * (prospect, team, pick_number) combination. Fetches from GET /api/ml/explain/:id.
 *
 * Usage:
 *   <ShapDrawer prospectId={42} teamId={7} pickNumber={3} />
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ShapFeature {
  feature: string
  shap_value: number
  feature_value: number
}

interface ShapResult {
  prospect_name: string
  team_abbreviation: string
  predicted_probability: number
  top_features: ShapFeature[]
  base_value: number
}

// Human-readable labels for raw feature names
const FEATURE_LABELS: Record<string, string> = {
  css_rank_norm:           'CSS Rank',
  points_per_game:         'Points / Game',
  gm_pos_weight:           'GM Position Fit',
  gm_league_weight:        'GM League Fit',
  gm_nat_weight:           'GM Nationality Fit',
  overall_pick:            'Pick Slot',
  age_at_draft:            'Age at Draft',
  ppg_league_norm:         'PPG vs League Avg',
  pos_taken_before_norm:   'Position Scarcity',
  pos_remaining_norm:      'Position Available',
  team_drafted_this_pos:   'Team Pos Depth',
  draft_league_tier:       'League Tier',
  ppg_trend:               'PPG Trend YoY',
  has_prev_season:         'Has Prior Season',
  height_cm:               'Height',
  weight_kg:               'Weight',
}

function label(feature: string): string {
  // Handle one-hot encoded features like position_C, nationality_CAN
  if (feature.startsWith('position_')) return `Position: ${feature.slice(9)}`
  if (feature.startsWith('nationality_')) return `Nationality: ${feature.slice(12)}`
  if (feature.startsWith('league_')) return `League: ${feature.slice(7)}`
  return FEATURE_LABELS[feature] ?? feature.replace(/_/g, ' ')
}

interface Props {
  prospectId: number
  teamId: number
  pickNumber: number
}

export default function ShapDrawer({ prospectId, teamId, pickNumber }: Props) {
  const [open, setOpen] = useState(false)
  const [data, setData] = useState<ShapResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function fetchShap() {
    if (data) { setOpen(o => !o); return }
    setOpen(true)
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(
        `/api/ml/explain/${prospectId}?team_id=${teamId}&pick_number=${pickNumber}`
      )
      if (!res.ok) throw new Error('Explanation unavailable')
      const json = await res.json()
      setData(json.explanation)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }

  const maxAbs = data
    ? Math.max(...data.top_features.slice(0, 6).map(f => Math.abs(f.shap_value)), 0.001)
    : 1

  return (
    <div className="w-full">
      <button
        onClick={e => { e.stopPropagation(); fetchShap() }}
        className="text-xs text-text-muted hover:text-accent-blue transition-colors mt-1 flex items-center gap-1"
      >
        <span className="text-[10px]">{open ? '▲' : '▼'}</span>
        Why this pick?
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
            onClick={e => e.stopPropagation()}
          >
            <div className="mt-2 pt-2 border-t border-border-subtle">
              {loading && (
                <div className="flex gap-1 items-center py-1">
                  {[0,1,2].map(i => (
                    <span key={i} className="w-1 h-1 rounded-full bg-text-muted animate-bounce" style={{ animationDelay: `${i*0.12}s` }} />
                  ))}
                </div>
              )}
              {error && <p className="text-xs text-red-400">{error}</p>}
              {data && (
                <>
                  <p className="text-[10px] text-text-muted mb-2">
                    Model confidence: <span className="text-white font-semibold">{(data.predicted_probability * 100).toFixed(1)}%</span>
                  </p>
                  <div className="space-y-1.5">
                    {data.top_features.slice(0, 6).map(f => {
                      const pct = Math.abs(f.shap_value) / maxAbs * 100
                      const positive = f.shap_value > 0
                      return (
                        <div key={f.feature} className="flex items-center gap-2">
                          <span className="text-[10px] text-text-muted w-28 flex-shrink-0 truncate text-right">
                            {label(f.feature)}
                          </span>
                          <div className="flex-1 flex items-center gap-1 h-3">
                            {positive ? (
                              <>
                                <div className="w-1/2 flex justify-end">
                                  <div className="h-2 rounded-sm bg-transparent" style={{ width: '100%' }} />
                                </div>
                                <div className="w-1/2 flex justify-start">
                                  <div
                                    className="h-2 rounded-sm bg-emerald-500/70"
                                    style={{ width: `${pct}%` }}
                                  />
                                </div>
                              </>
                            ) : (
                              <>
                                <div className="w-1/2 flex justify-end">
                                  <div
                                    className="h-2 rounded-sm bg-red-500/70"
                                    style={{ width: `${pct}%` }}
                                  />
                                </div>
                                <div className="w-1/2 flex justify-start">
                                  <div className="h-2 rounded-sm bg-transparent" style={{ width: '100%' }} />
                                </div>
                              </>
                            )}
                          </div>
                          <span className={`text-[10px] w-10 text-right flex-shrink-0 ${positive ? 'text-emerald-400' : 'text-red-400'}`}>
                            {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(3)}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                  <p className="text-[9px] text-text-muted mt-2">
                    Green = pushes toward pick · Red = pushes against
                  </p>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
