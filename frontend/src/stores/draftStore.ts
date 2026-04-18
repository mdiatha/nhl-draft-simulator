import { create } from 'zustand'
import type { DraftSimulationPick, LotteryPick } from '../types'

interface DraftStore {
  lotteryResult: LotteryPick[]
  lotterySeed: number | undefined
  draftResult: DraftSimulationPick[]
  isLoading: boolean
  error: string | null
  setLotteryResult: (result: LotteryPick[], seed?: number) => void
  setDraftResult: (picks: DraftSimulationPick[]) => void
  setLoading: (v: boolean) => void
  setError: (msg: string | null) => void
  resetDraft: () => void
}

export const useDraftStore = create<DraftStore>((set) => ({
  lotteryResult: [],
  lotterySeed: undefined,
  draftResult: [],
  isLoading: false,
  error: null,
  setLotteryResult: (result, seed) => set({ lotteryResult: result, lotterySeed: seed }),
  setDraftResult: (picks) => set({ draftResult: picks }),
  setLoading: (v) => set({ isLoading: v }),
  setError: (msg) => set({ error: msg }),
  resetDraft: () => set({ draftResult: [], error: null }),
}))
