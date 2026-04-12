import { useState } from 'react'
import { adminApi, mlApi } from '../lib/api'

type TaskStatus = 'idle' | 'running' | 'success' | 'error'

interface TaskState {
  status: TaskStatus
  message: string
}

const initialTask = (): TaskState => ({ status: 'idle', message: '' })

export default function AdminPage() {
  const [apiKey, setApiKey] = useState('')
  const [ingest, setIngest] = useState<TaskState>(initialTask())
  const [seedProspects, setSeedProspects] = useState<TaskState>(initialTask())
  const [computeTendencies, setComputeTendencies] = useState<TaskState>(initialTask())
  const [fetchStats, setFetchStats] = useState<TaskState>(initialTask())
  const [fetchCssRankings, setFetchCssRankings] = useState<TaskState>(initialTask())
  const [trainML, setTrainML] = useState<TaskState>(initialTask())

  async function run(
    setter: (s: TaskState) => void,
    fn: () => Promise<unknown>
  ) {
    setter({ status: 'running', message: '' })
    try {
      const data = await fn()
      setter({ status: 'success', message: JSON.stringify(data, null, 2) })
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        (err as Error)?.message ||
        'Unknown error'
      setter({ status: 'error', message: msg })
    }
  }

  const statusColor: Record<TaskStatus, string> = {
    idle: 'text-text-secondary',
    running: 'text-accent-blue',
    success: 'text-green-400',
    error: 'text-red-400',
  }

  function TaskCard({
    title,
    description,
    state,
    onRun,
  }: {
    title: string
    description: string
    state: TaskState
    onRun: () => void
  }) {
    return (
      <div className="bg-bg-secondary border border-border-subtle rounded-lg p-5 flex flex-col gap-3">
        <div>
          <h3 className="font-semibold text-white">{title}</h3>
          <p className="text-sm text-text-secondary mt-0.5">{description}</p>
        </div>
        <button
          onClick={onRun}
          disabled={state.status === 'running'}
          className="self-start px-4 py-1.5 rounded bg-accent-blue text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {state.status === 'running' ? 'Running…' : 'Run'}
        </button>
        {state.status !== 'idle' && (
          <pre
            className={`text-xs whitespace-pre-wrap break-all font-mono rounded bg-bg-primary p-3 ${statusColor[state.status]}`}
          >
            {state.status === 'running' ? 'Started — check /health for progress…' : state.message}
          </pre>
        )}
      </div>
    )
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-bold text-white mb-1">Admin Dashboard</h1>
      <p className="text-text-secondary text-sm mb-8">
        Run data ingestion, ML training, and maintenance tasks. Requires an admin API key if one is configured.
      </p>

      <div className="mb-8">
        <label className="block text-sm font-medium text-text-secondary mb-1.5">
          Admin API Key <span className="text-text-secondary font-normal">(leave blank if not set)</span>
        </label>
        <input
          type="password"
          value={apiKey}
          onChange={e => setApiKey(e.target.value)}
          placeholder="Enter ADMIN_API_KEY…"
          className="w-full max-w-sm bg-bg-primary border border-border-subtle rounded px-3 py-2 text-sm text-white placeholder-text-secondary focus:outline-none focus:border-accent-blue"
        />
      </div>

      <div className="grid gap-4">
        <TaskCard
          title="1. Full Ingestion"
          description="Fetch teams, GMs, draft history (2000–2024), lottery standings, and 2025 prospects from the NHL API. Runs in the background (~2–5 min)."
          state={ingest}
          onRun={() =>
            run(setIngest, () =>
              adminApi.ingest(apiKey)
            )
          }
        />

        <TaskCard
          title="2. Seed 2025 Prospects"
          description="Fetch and upsert the 2025 draft class into the prospects table. Safe to re-run."
          state={seedProspects}
          onRun={() => run(setSeedProspects, () => adminApi.seedProspects(apiKey))}
        />

        <TaskCard
          title="3. Compute GM Tendencies"
          description="Build or rebuild tendency profiles for all active GMs from historical draft data. Required before ML training."
          state={computeTendencies}
          onRun={() => run(setComputeTendencies, () => adminApi.computeTendencies(apiKey))}
        />

        <TaskCard
          title="4. Fetch Prospect Stats"
          description="Pull current-season stats (PPG, goals, assists) for all 2025 prospects. Runs in the background."
          state={fetchStats}
          onRun={() => run(setFetchStats, () => adminApi.updateCareerStats(apiKey))}
        />

        <TaskCard
          title="5. Fetch Historical CSS Rankings"
          description="Pull NHL Central Scouting final rankings for 2008–2024 and store on historical picks. Replaces the circular training proxy with real pre-draft consensus ranks. Run after draft history is ingested, before training."
          state={fetchCssRankings}
          onRun={() => run(setFetchCssRankings, () => adminApi.fetchCssRankings(apiKey))}
        />

        <TaskCard
          title="6. Train ML Model"
          description="Train the XGBoost draft-pick classifier on historical data (evaluation mode: train ≤2019, validate 2020–2024). Hot-swaps the in-memory model on completion."
          state={trainML}
          onRun={() => run(setTrainML, () => mlApi.train(apiKey))}
        />
      </div>

      <div className="mt-8 p-4 rounded-lg bg-bg-secondary border border-border-subtle text-sm text-text-secondary">
        <span className="font-medium text-white">Recommended order:</span> Run steps 1 → 3 → 4 → 5 → 6 on a fresh database. Step 5 (CSS rankings) is the key training data quality step — skip it and the model trains on circular data. Step 2 is included in Step 1 but can be run standalone.
      </div>
    </div>
  )
}
