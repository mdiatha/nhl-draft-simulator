import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { lotteryApi } from '../../lib/api'
import { useDraftStore } from '../../stores/draftStore'
import { useLotteryStore } from '../../stores/lotteryStore'
import { getTeamColors, getTeamLogo } from '../../lib/teamColors'
import type { LotteryPick, LotterySimulationTeam } from '../../types'

interface Props {
  onComplete: (result: LotteryPick[], seed: number) => void
  seed?: number
  teams?: LotterySimulationTeam[]
}

const delay = (ms: number) => new Promise<void>(res => setTimeout(res, ms))

export default function LotteryDraw({ onComplete, seed, teams }: Props) {
  const navigate = useNavigate()
  const { setLotteryResult } = useDraftStore()
  const { phase, setPhase, currentDraw, setCurrentDraw, isAnimating, setAnimating } = useLotteryStore()

  const [lotteryResult, setLotteryResultLocal] = useState<LotteryPick[]>([])
  const [revealedPicks, setRevealedPicks] = useState<LotteryPick[]>([])
  const [revealing, setRevealing] = useState<LotteryPick | null>(null)

  const startLottery = async () => {
    setPhase('drawing')
    setAnimating(true)

    try {
      const data = await lotteryApi.simulate(seed, teams)
      const picks: LotteryPick[] = data.pick_order ?? []
      setLotteryResultLocal(picks)

      // Reveal the 2 lottery picks dramatically, then show full order
      for (let i = 0; i < 2; i++) {
        const pick = picks[i]
        if (!pick) continue
        setCurrentDraw(i + 1)

        // Suspense pause before reveal
        await delay(1200)
        setRevealing(pick)
        await delay(2200)
        setRevealedPicks(prev => [...prev, pick])
        setRevealing(null)
        await delay(600)
      }

      // Brief pause, then show complete order
      await delay(800)
      setPhase('complete')
      setAnimating(false)
      setLotteryResult(picks, data.seed)
      onComplete(picks, data.seed)
    } catch (err) {
      console.error('Lottery simulation failed:', err)
      setPhase('idle')
      setAnimating(false)
    }
  }

  // ── Complete view: full pick order ────────────────────────────────────────
  if (phase === 'complete') {
    return (
      <div className="space-y-3">
        <h2 className="text-xl font-bold text-white mb-4">2025 Draft Order</h2>
        {lotteryResult.map((pick, idx) => {
          const colors = getTeamColors(pick.abbreviation)
          const isLotteryWinner = pick.pick <= 2 && (pick.original_standing ?? 0) > pick.pick
          const isTop2 = pick.pick <= 2
          return (
            <motion.div
              key={pick.pick}
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.03, duration: 0.25 }}
              className="flex items-center gap-3 px-4 py-3 rounded-xl bg-bg-card border border-border-subtle"
              style={{ borderLeft: `3px solid ${isTop2 ? colors.primary : colors.primary + '66'}` }}
            >
              {/* Pick number */}
              <span
                className="text-lg font-black w-7 text-right flex-shrink-0"
                style={{ color: isTop2 ? colors.primary : '#8892a4' }}
              >
                {pick.pick}
              </span>

              {/* Logo */}
              <div
                className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 overflow-hidden"
                style={{ background: colors.primary + '20' }}
              >
                <img
                  src={getTeamLogo(pick.abbreviation)}
                  alt={pick.abbreviation}
                  className="w-7 h-7 object-contain"
                  onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
                />
              </div>

              {/* Team name */}
              <div className="flex-1 min-w-0">
                <span className={`font-semibold text-sm truncate block ${isTop2 ? 'text-white' : 'text-text-primary'}`}>
                  {pick.team_name}
                </span>
                {pick.lottery_odds_pct != null && (
                  <span className="text-xs text-text-muted">
                    {pick.lottery_odds_pct.toFixed(1)}% odds · was #{pick.original_standing}
                  </span>
                )}
              </div>

              {/* Lottery winner badge */}
              {isLotteryWinner && (
                <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-yellow-400/15 text-yellow-400 border border-yellow-400/30 flex-shrink-0">
                  ↑ Lottery Winner
                </span>
              )}
            </motion.div>
          )
        })}

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="pt-2"
        >
          <button
            onClick={() => navigate('/draft')}
            className="w-full py-3 bg-accent-blue hover:bg-blue-500 text-white font-bold rounded-xl transition-colors text-base"
          >
            Simulate Draft →
          </button>
        </motion.div>
      </div>
    )
  }

  // ── Drawing view ──────────────────────────────────────────────────────────
  return (
    <div className="space-y-6">
      {/* Pick slots */}
      <div className="grid grid-cols-2 gap-3">
        {[1, 2].map(pickNum => {
          const revealed = revealedPicks.find(p => p.pick === pickNum)
          const isCurrentlyRevealing = revealing?.pick === pickNum
          const colors = revealed ? getTeamColors(revealed.abbreviation) : null

          return (
            <div
              key={pickNum}
              className="relative rounded-xl border overflow-hidden"
              style={{
                borderColor: revealed ? (colors?.primary + '60') : '#1e2a3a',
                background: revealed ? (colors?.primary + '12') : '#0d1520',
                minHeight: 120,
              }}
            >
              {/* Pick label */}
              <div className="px-4 pt-3 pb-1">
                <span className="text-xs font-semibold text-text-muted tracking-widest uppercase">
                  Pick #{pickNum}
                </span>
              </div>

              <AnimatePresence mode="wait">
                {isCurrentlyRevealing && (
                  <motion.div
                    key="revealing"
                    initial={{ opacity: 0, scale: 0.85 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 flex flex-col items-center justify-center"
                  >
                    <div className="w-3 h-3 rounded-full bg-accent-blue animate-ping" />
                  </motion.div>
                )}

                {revealed && !isCurrentlyRevealing && (
                  <motion.div
                    key="revealed"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="px-4 pb-4 flex flex-col gap-2"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="w-12 h-12 rounded-xl flex items-center justify-center overflow-hidden flex-shrink-0"
                        style={{ background: colors!.primary + '25' }}
                      >
                        <img
                          src={getTeamLogo(revealed.abbreviation)}
                          alt={revealed.abbreviation}
                          className="w-10 h-10 object-contain"
                          onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
                        />
                      </div>
                      <div>
                        <div className="font-bold text-white text-sm leading-tight">{revealed.team_name}</div>
                        <div className="text-xs text-text-muted mt-0.5">
                          {revealed.lottery_odds_pct?.toFixed(1)}% odds
                        </div>
                      </div>
                    </div>
                    {(revealed.original_standing ?? 0) > pickNum && (
                      <span className="self-start text-xs font-semibold px-2 py-0.5 rounded-full bg-yellow-400/15 text-yellow-400 border border-yellow-400/30">
                        ↑ Lottery Winner · was #{revealed.original_standing}
                      </span>
                    )}
                  </motion.div>
                )}

                {!revealed && !isCurrentlyRevealing && (
                  <motion.div
                    key="empty"
                    className="px-4 pb-4 flex items-center gap-3"
                  >
                    <div className="w-12 h-12 rounded-xl bg-bg-secondary border border-border-subtle flex items-center justify-center">
                      <span className="text-text-muted text-xl font-black">?</span>
                    </div>
                    <span className="text-text-muted text-sm">
                      {phase === 'idle' ? 'Not drawn' : currentDraw === pickNum ? 'Drawing...' : 'Pending…'}
                    </span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )
        })}
      </div>

      {/* Suspense countdown / current pick reveal */}
      <AnimatePresence>
        {revealing && (
          <motion.div
            key="spotlight"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="text-center py-8 rounded-2xl border border-accent-blue/30 bg-accent-blue/5"
          >
            <p className="text-text-secondary text-sm mb-2 tracking-widest uppercase">Pick #{revealing.pick} goes to</p>
            <div className="flex items-center justify-center gap-4">
              <img
                src={getTeamLogo(revealing.abbreviation)}
                alt={revealing.abbreviation}
                className="w-16 h-16 object-contain"
                onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
              />
              <div className="text-left">
                <div className="text-3xl font-black text-white">{revealing.team_name}</div>
                <div className="text-accent-blue text-sm font-semibold mt-0.5">
                  {revealing.lottery_odds_pct?.toFixed(1)}% odds
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Already revealed (small) */}
      {revealedPicks.length > 0 && !revealing && (
        <div className="space-y-2">
          <p className="text-xs text-text-muted uppercase tracking-widest">Drawn</p>
          {revealedPicks.map(pick => {
            const colors = getTeamColors(pick.abbreviation)
            return (
              <div
                key={pick.pick}
                className="flex items-center gap-3 px-3 py-2 rounded-lg bg-bg-card border border-border-subtle"
                style={{ borderLeft: `3px solid ${colors.primary}` }}
              >
                <span className="font-bold text-white text-sm w-5">#{pick.pick}</span>
                <img
                  src={getTeamLogo(pick.abbreviation)}
                  alt={pick.abbreviation}
                  className="w-5 h-5 object-contain"
                  onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none' }}
                />
                <span className="text-white text-sm font-medium">{pick.team_name}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* Draw button */}
      {phase === 'idle' && (
        <button
          onClick={startLottery}
          className="w-full py-4 bg-accent-blue hover:bg-blue-500 text-white font-bold rounded-xl text-lg transition-colors shadow-lg shadow-blue-500/20"
        >
          Run Draft Lottery
        </button>
      )}

      {isAnimating && phase === 'drawing' && !revealing && (
        <div className="flex items-center justify-center gap-2 text-text-secondary text-sm py-2">
          <span className="w-2 h-2 rounded-full bg-accent-blue animate-pulse" />
          Drawing pick #{currentDraw}...
        </div>
      )}
    </div>
  )
}
