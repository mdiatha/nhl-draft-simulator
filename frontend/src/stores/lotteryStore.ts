import { create } from 'zustand'

type LotteryPhase = 'idle' | 'mixing' | 'drawing' | 'complete'

interface LotteryStore {
  phase: LotteryPhase
  currentDraw: number  // 1, 2, or 3
  ballCombination: number[]
  isAnimating: boolean
  setPhase: (phase: LotteryPhase) => void
  setCurrentDraw: (n: number) => void
  setBallCombination: (balls: number[]) => void
  setAnimating: (v: boolean) => void
  reset: () => void
}

export const useLotteryStore = create<LotteryStore>((set) => ({
  phase: 'idle',
  currentDraw: 0,
  ballCombination: [],
  isAnimating: false,
  setPhase: (phase) => set({ phase }),
  setCurrentDraw: (n) => set({ currentDraw: n }),
  setBallCombination: (balls) => set({ ballCombination: balls }),
  setAnimating: (v) => set({ isAnimating: v }),
  reset: () => set({ phase: 'idle', currentDraw: 0, ballCombination: [], isAnimating: false }),
}))
