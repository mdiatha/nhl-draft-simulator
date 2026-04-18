# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend
```bash
cd backend

# Start dependencies (requires pgvector/pgvector:pg16, not postgres:16-alpine)
docker compose up -d db redis ollama

# Pull embedding model (one-time, persists in ollama_data Docker volume)
docker compose exec ollama ollama pull nomic-embed-text

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
- **tools.py** — 10 tools: `get_gm_profile`, `get_top_prospects`, `get_team_needs`, `search_prospects`, `semantic_prospect_search`, `get_draft_history`, `get_ml_ranking`, `compare_prospects`, `get_prospect_detail`, `get_nhl_comp`. All inputs validated through Pydantic models before DB access.
- **embeddings.py** — Indexes GM profiles, top-200 prospects (one rich prose document per prospect combining facts, style, and production trend), and team draft histories (2020–2024) into `scout_embeddings` via Ollama nomic-embed-text (768-dim). `build_index(db)` is called via `POST /api/agent/index`.
- **memory.py** — Compresses conversation history when session exceeds 20 messages (Claude summarizes oldest turns into bullets, keeps 6 most recent verbatim).

### GM tendency engine (`backend/app/engines/tendency_engine.py`)
Bayesian shrinkage (K=30) + recency decay (0.85/year) over a GM's historical picks. Classifies GMs into 5 archetypes. Must be recomputed after new draft history is ingested (`POST /api/admin/compute-tendencies`).

### Auth
`require_admin_key` in `middleware/auth.py`: silent pass-through when `ADMIN_API_KEY=""` **only** when `APP_ENV=development`. In any other environment, an unset key returns 503. Key comparison uses `hmac.compare_digest`.

### Database
PostgreSQL 16 with pgvector extension (`pgvector/pgvector:pg16` Docker image — **not** `postgres:16-alpine`). 15 Alembic migrations in `backend/alembic/versions/`. The `scout_embeddings` table uses a native `vector(768)` column with an IVFFlat cosine index (migration 015).

**Migration deployment note:** `alembic/` is excluded from the Docker image (see `.dockerignore`) — migrations are never baked in. To run migrations against a live container:
```bash
# Copy any new migration files into the running container first
docker cp backend/alembic/versions/<new_migration>.py <api-container>:/app/alembic/versions/
# Then upgrade
docker compose exec api alembic upgrade head
```
For CI/CD, mount the `alembic/` directory as a volume or run migrations from the host with `DATABASE_URL` pointed at the container.

### Streaming (SSE)
Both `/api/agent/chat/stream` and `/api/draft/summary/stream` use Server-Sent Events. Tool calls execute synchronously between streaming turns in `_run_tool_rounds_silent()` — only the final text response streams token-by-token. Frontend consumes `data: {"token": "..."}` events, terminates on `data: {"done": true}`.

## Known limitations

This section is intentionally honest. Interviewers respect self-awareness over overselling.

### ML model
- **CSS rank is the dominant feature.** The model learns that GMs largely follow consensus rankings. This means it mostly replicates the CSS board with team-specific adjustments. The GM tendency features add signal at the margin, especially for GMs with strong positional preferences (goalies, defensemen).
- **Training data covers CSS era (2008–present).** Pre-2008 picks use a pick-order proxy for CSS rank, which is noisier. Including them adds rows but dilutes signal. The constant `CSS_ERA_START=2008` documents this decision.
- **Round 1 only in simulation.** The simulation engine runs 32 picks. Rounds 2–7 involve significantly more variance and idiosyncratic GM behavior that the model is less equipped to capture.
- **No trade simulation.** Real drafts include pick trades before and during the draft. The simulation assumes fixed pick order after the lottery.
### Data
- **GM identity is hand-maintained.** `gms.json` must be updated manually when GMs change. Stale data produces incorrect tendency profiles for teams with recent GM transitions.
- **CSS rankings for the 2025 class are seeded manually.** The NHL Records API only has historical data. Current-year rankings must be entered via the admin endpoint or CSV import.
- **Pre-draft PPG covers the season before the draft.** Stats from earlier seasons (ppg_prev_season) are fetched on a best-effort basis via NHL player IDs. Some players, especially Europeans in lower leagues, may have NULL values.

### Infrastructure
- **Ollama runs locally.** In the AWS deployment, Ollama for embeddings must run as a sidecar or separate service. The current CDK does not provision this — RAG embeddings silently degrade to unavailable if Ollama is unreachable.
- **No multi-round simulation.** The draft simulation covers round 1 only. Real draft strategy involves round-2 considerations that influence round-1 behavior (e.g., trading down to accumulate picks).

## Key constants to know before editing

| Constant | Location | Value | Impact if changed |
|---|---|---|---|
| `TRAIN_CUTOFF_YEAR` | `ml/train.py:46` | 2019 | Shifts train/val split |
| `FINAL_N_ESTIMATORS` | `ml/train.py:49` | 200 | Must match `best_iteration` from eval run; reset to 200 after feature set change |
| `NEGATIVE_WINDOW` | `ml/features.py` | 31 | Changes training set size and model behavior |
| `CSS_RANK_DENOM` | `ml/features.py` | 450 | Train/inference normalization — must stay identical |
| `MAX_TOOL_ROUNDS` | `agent/scout.py:53` | 10 | Claude tool call depth limit |
| `EMBED_DIM` | `agent/embeddings.py:26` | 768 | Must match nomic-embed-text output — changing requires a new migration to resize vector(768) column |
| `OLLAMA_MODEL` | `agent/embeddings.py:27` | nomic-embed-text | Ollama model used for RAG embeddings — must be pulled before indexing |

## Data sources — exactly where each input comes from

This section answers the most common ML interview question: **"Where did your training data come from?"**

### Pre-draft player stats (PPG, GP, age)
- **Source:** NHL Records API (`records.nhl.com/site/api/draft`) + NHL Stats API (`api-web.nhle.com/v1`)
- **What's fetched:** For each historically drafted player (by NHL player ID), we call the NHL Stats API to retrieve their season stats from the year **before** their draft. This gives us the pre-draft PPG that is used as a training feature.
- **Temporal integrity:** The fetch is keyed by `(nhl_player_id, season_year - 1)` so we only ever retrieve stats from the season ending before June of the draft year. No post-draft stats contaminate training features.
- **Code path:** `backend/app/ingestion/stat_ingestion.py` → `fetch_predraft_stats()` → stored in `draft_picks_historical.points_per_game`

### CSS rankings (consensus scouting rank)
- **Source:** NHL Central Scouting Service rankings, fetched via the NHL Records API draft endpoint (`records.nhl.com/site/api/draft?cayenneExp=draftYear=YYYY`).
- **What's available:** CSS rankings are available from **2008 onward** in the NHL Records API response. Pre-2008 picks use a pick-order proxy (`overall_pick / total_picks_that_year`) as a fallback. This is documented in `features.py::_compute_predraft_quality()` and in the CSS_ERA_START=2008 constant.
- **What CSS rank means:** CSS ranks NA skaters, EUR skaters, and goalies in separate lists. The raw rank is normalised with square-root scaling (`_css_norm()`) so the gap between #1 and #2 is amplified vs. the gap between #100 and #101.
- **Limitation:** CSS rankings are published mid-season and finalized in May. The API returns the final rankings. For the 2025 class (inference), CSS rankings are manually seeded via `POST /api/admin/seed-prospects`.

### GM identity and tenure
- **Source:** `backend/app/ingestion/gms.json` — manually maintained JSON file with each current GM's name, team, and start date.
- **Why manual:** The NHL API does not expose GM data. This file must be updated when GMs are hired or fired. Last verified: April 2026.
- **Historical GM attribution:** Draft picks from before a GM's tenure are assigned to their predecessor using the pick year vs. GM start_date. This matters for tendency profiles — a GM shouldn't be credited with picks made before they joined.

## Data pipeline order (fresh setup)
```
POST /api/admin/ingest                  # teams, GMs, draft history 2000–2024, prospects
POST /api/admin/fetch-prospect-stats    # pre-draft PPG for 2025 class
POST /api/admin/compute-tendencies      # GM tendency profiles (Bayesian shrinkage)
POST /api/ml/train                      # eval mode — runs calibration automatically
POST /api/ml/reload                     # inject trained model into registry
POST /api/agent/index                   # build RAG embeddings (Ollama must be running)
POST /api/admin/feature-store/refresh   # warm inference cache
```
