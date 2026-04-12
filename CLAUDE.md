# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend
```bash
cd backend

# Start dependencies (requires pgvector/pgvector:pg16, not postgres:16-alpine)
docker compose up -d db redis

# Migrations
alembic upgrade head
alembic downgrade -1   # always test downgrade path

# Run tests
pytest tests/ -v --tb=short --cov=app --cov-fail-under=70          # unit
pytest tests/integration/ -v --tb=short                            # integration (requires Docker)
pytest tests/test_features.py::test_name -v                        # single test

# ML pipeline CLI
python -m app.ml.train_model                       # eval mode (train ≤2019, val 2020–2024)
python -m app.ml.train_model --cutoff 2023         # custom cutoff
```

### Frontend
```bash
cd frontend
npm run dev      # dev server at http://localhost:5173
npm run build    # tsc + vite build
npm run lint     # eslint, zero warnings allowed
```

### API calls (admin key required for write endpoints)
```bash
curl -X POST http://localhost:8000/api/ml/train -H "X-Admin-Key: your-key"
curl -X POST http://localhost:8000/api/ml/calibrate -H "X-Admin-Key: your-key"
curl -X POST http://localhost:8000/api/agent/index          # rebuild RAG index
curl http://localhost:8000/health                           # DB + Redis + model status
```

## Architecture

### Request flow
1. Frontend → REST/SSE → FastAPI (`backend/app/api/`)
2. API routes call engines/ML/agent modules
3. Results cached in Redis (draft simulations keyed by seed + lottery order, TTL 3600s)

### ML pipeline (`backend/app/ml/`)
- **train.py** — XGBoost LambdaMART (`rank:ndcg`). Two modes: `final=False` uses a temporal split (train ≤ `TRAIN_CUTOFF_YEAR=2019`, val ≥ `VAL_START_YEAR=2020`) and runs conformal calibration automatically; `final=True` trains on all data with fixed `FINAL_N_ESTIMATORS=173`.
- **features.py** — 33-feature engineering. `build_training_dataset(db)` pulls historical picks with negative sampling (`NEGATIVE_WINDOW=31` picks ahead). Feature normalization is fixed-denominator (e.g. `CSS_RANK_DENOM=450`) — must match identically in both training and inference or predictions break.
- **predict.py** — Inference path. Scores prospects via `score_pool_for_team()`. Has a feature store cache path (pre-materialized features for each prospect); if cache column schema is stale it falls back gracefully.
- **registry.py** — Module-level singleton `registry`. Loaded at startup, hot-swapped via `POST /api/ml/reload`. Holds XGBoost model + `calibration` (conformal prediction quantiles).
- **calibration.py** — Split conformal prediction at α=0.10/0.15/0.20. `apply_intervals()` attaches `in_prediction_set` and `nc_score` to every scored prospect. Called automatically in eval-mode training; can be re-run standalone via `POST /api/ml/calibrate`.

### Agent / RAG (`backend/app/agent/`)
- **scout.py** — Tool-use loop (up to `MAX_TOOL_ROUNDS=10`, exits on `end_turn`). Prompt caching via `CACHED_SYSTEM` + `_cached_tools()` (Anthropic ephemeral cache). MCP routing via `_create_message()` — uses `client.beta.messages` when `MCP_SERVERS` config is set.
- **tools.py** — 6 tools: `get_gm_profile`, `get_top_prospects`, `get_team_needs`, `search_prospects`, `semantic_prospect_search`, `get_draft_history`. All inputs validated through `_str_input()` / `_int_input()` before DB access.
- **embeddings.py** — Indexes GM profiles, top-200 prospects (enriched with stat history), and team draft histories (2020–2024) into `scout_embeddings` via voyage-3-lite (1024-dim). `build_index(db)` is called via `POST /api/agent/index`.
- **memory.py** — Compresses conversation history when session exceeds 20 messages (Claude summarizes oldest turns into bullets, keeps 6 most recent verbatim).

### GM tendency engine (`backend/app/engines/tendency_engine.py`)
Bayesian shrinkage (K=30) + recency decay (0.85/year) over a GM's historical picks. Classifies GMs into 5 archetypes. Must be recomputed after new draft history is ingested (`POST /api/admin/compute-tendencies`).

### Auth
`require_admin_key` in `middleware/auth.py`: silent pass-through when `ADMIN_API_KEY=""` **only** when `APP_ENV=development`. In any other environment, an unset key returns 503. Key comparison uses `hmac.compare_digest`.

### Database
PostgreSQL 16 with pgvector extension (`pgvector/pgvector:pg16` Docker image — **not** `postgres:16-alpine`). 13 Alembic migrations in `backend/alembic/versions/`. The `scout_embeddings` table uses a native `vector(1024)` column with an IVFFlat cosine index (migration 011).

### Streaming (SSE)
Both `/api/agent/chat/stream` and `/api/draft/summary/stream` use Server-Sent Events. Tool calls execute synchronously between streaming turns in `_run_tool_rounds_silent()` — only the final text response streams token-by-token. Frontend consumes `data: {"token": "..."}` events, terminates on `data: {"done": true}`.

## Key constants to know before editing

| Constant | Location | Value | Impact if changed |
|---|---|---|---|
| `TRAIN_CUTOFF_YEAR` | `ml/train.py:46` | 2019 | Shifts train/val split |
| `FINAL_N_ESTIMATORS` | `ml/train.py:49` | 200 | Must match `best_iteration` from eval run; reset to 200 after feature set change |
| `NEGATIVE_WINDOW` | `ml/features.py` | 31 | Changes training set size and model behavior |
| `CSS_RANK_DENOM` | `ml/features.py` | 450 | Train/inference normalization — must stay identical |
| `MAX_TOOL_ROUNDS` | `agent/scout.py:53` | 10 | Claude tool call depth limit |
| `EMBED_DIM` | `agent/embeddings.py:26` | 1024 | Must match voyage-3-lite output |

## Data pipeline order (fresh setup)
```
POST /api/admin/ingest              # teams, GMs, draft history 2000–2024, prospects
POST /api/admin/fetch-prospect-stats
POST /api/admin/compute-tendencies
POST /api/ml/train                  # eval mode — runs calibration automatically
POST /api/ml/reload                 # inject trained model
POST /api/agent/index               # build RAG embeddings
POST /api/admin/feature-store/refresh   # warm inference cache
```
