import { create } from 'zustand'

type LotteryPhase = 'idle' | 'mixing' | 'drawing' | 'complete'

interface LotteryStore {
  phase: LotteryPhase
  currentDraw: number  // 1, 2, or 3
  ballCombination: number[]
  isAnimating: boolean
  error: string | null
  setPhase: (phase: LotteryPhase) => void
  setCurrentDraw: (n: number) => void
  setBallCombination: (balls: number[]) => void
  setAnimating: (v: boolean) => void
  setError: (msg: string | null) => void
  reset: () => void
}

export const useLotteryStore = create<LotteryStore>((set) => ({
  phase: 'idle',
  currentDraw: 0,
  ballCombination: [],
  isAnimating: false,
  error: null,
  setPhase: (phase) => set({ phase }),
  setCurrentDraw: (n) => set({ currentDraw: n }),
  setBallCombination: (balls) => set({ ballCombination: balls }),
  setAnimating: (v) => set({ isAnimating: v }),
  setError: (msg) => set({ error: msg }),
  reset: () => set({ phase: 'idle', currentDraw: 0, ballCombination: [], isAnimating: false, error: null }),
}))
