import axios, { type AxiosRequestConfig } from 'axios'
import type { LotterySimulationTeam } from '../types'

const rawBaseUrl = (import.meta.env.VITE_API_URL ?? '').trim()
const normalizedBaseUrl = rawBaseUrl.replace(/\/$/, '')

function adminConfig(apiKey?: string): AxiosRequestConfig | undefined {
  if (!apiKey) return undefined
  return {
    headers: {
      'X-Admin-Key': apiKey,
    },
  }
}

export function buildApiUrl(path: string): string {
  if (!normalizedBaseUrl) return path
  return `${normalizedBaseUrl}${path.startsWith('/') ? path : `/${path}`}`
}

const api = axios.create({
  baseURL: normalizedBaseUrl || '',
  timeout: 30000,
})

api.interceptors.response.use(
  res => res,
  err => {
    console.error('API Error:', err.response?.status, err.config?.url)
    return Promise.reject(err)
  }
)

export default api

// Typed API calls
export const lotteryApi = {
  getOdds: () => api.get('/api/lottery/odds').then(r => r.data),
  simulate: (seed?: number, teams?: LotterySimulationTeam[]) =>
    api.post('/api/lottery/simulate', { seed, ...(teams ? { teams } : {}) }).then(r => r.data),
  simulateMany: (n: number, seed?: number) => api.post('/api/lottery/simulate-many', { n, seed }).then(r => r.data),
}

export const draftApi = {
  getProspects: (params?: Record<string, string | number>) =>
    api.get('/api/draft/prospects', { params }).then(r => r.data),
  simulate: (lotteryResult: number[], seed?: number) =>
    api.post('/api/draft/simulate', { lottery_result: lotteryResult, seed }).then(r => r.data),
}

export const teamsApi = {
  getTendency: (teamId: number) => api.get(`/api/teams/${teamId}/tendency`).then(r => r.data),
}

export const mlApi = {
  getScores: () => api.get('/api/ml/scores').then(r => r.data),
  train: (apiKey?: string) => api.post('/api/ml/train', undefined, adminConfig(apiKey)).then(r => r.data),
  status: () => api.get('/api/ml/status').then(r => r.data),
  backtest: (testYear: number, trainCutoff: number) =>
    api.post(`/api/ml/backtest?test_year=${testYear}&train_cutoff=${trainCutoff}`).then(r => r.data),
  backtestMultiYear: () => api.post('/api/ml/backtest/multi-year').then(r => r.data),
}

export const adminApi = {
  ingest: (apiKey?: string) => api.post('/api/admin/ingest', undefined, adminConfig(apiKey)).then(r => r.data),
  seedProspects: (apiKey?: string) => api.post('/api/admin/seed-prospects', undefined, adminConfig(apiKey)).then(r => r.data),
  computeTendencies: (apiKey?: string) => api.post('/api/admin/compute-tendencies', undefined, adminConfig(apiKey)).then(r => r.data),
  updateCareerStats: (apiKey?: string) => api.post('/api/admin/fetch-prospect-stats', undefined, adminConfig(apiKey)).then(r => r.data),
  fetchCssRankings: (apiKey?: string) => api.post('/api/admin/fetch-css-rankings', undefined, adminConfig(apiKey)).then(r => r.data),
  trainML: (apiKey?: string) => api.post('/api/ml/train', undefined, adminConfig(apiKey)).then(r => r.data),
}

export const standingsApi = {
  getLive: () => api.get('/api/standings/live').then(r => r.data),
}

export const agentApi = {
  chat: (message: string, sessionId: string) =>
    api.post('/api/agent/chat', { message, session_id: sessionId }).then(r => r.data),
  buildIndex: () => api.post('/api/agent/index').then(r => r.data),
  draftSummary: (picks: unknown[]) =>
    api.post('/api/draft/summary', { picks }).then(r => r.data),
}
