import { create } from 'zustand'
import type { DraftSimulationPick, LotteryPick } from '../types'

interface DraftStore {
  lotteryResult: LotteryPick[]
  lotterySeed: number | undefined
  draftResult: DraftSimulationPick[]
  setLotteryResult: (result: LotteryPick[], seed?: number) => void
  setDraftResult: (picks: DraftSimulationPick[]) => void
  resetDraft: () => void
}

export const useDraftStore = create<DraftStore>((set) => ({
  lotteryResult: [],
  lotterySeed: undefined,
  draftResult: [],
  setLotteryResult: (result, seed) => set({ lotteryResult: result, lotterySeed: seed }),
  setDraftResult: (picks) => set({ draftResult: picks }),
  resetDraft: () => set({ draftResult: [] }),
}))
