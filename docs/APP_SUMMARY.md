# NHL Draft Simulator — Comprehensive Project Summary

## One-Line Description

A full-stack NHL draft analytics platform that simulates the 2026 NHL Entry Draft pick-by-pick using an XGBoost learning-to-rank model, Bayesian GM tendency profiles, conformal prediction uncertainty, a tool-use AI Scout agent with RAG, and production AWS infrastructure — all accessible through a polished React product.

---

## What This Project Is

The NHL Draft Simulator is not a static mock-draft generator or a notebook exercise. It is a software product that answers two questions:

1. **What is a team most likely to do at a given draft slot?** (behavioral prediction)
2. **Why does that pick make sense given the board, the model, and the team's history?** (explainability)

It runs a weighted draft lottery, simulates all 7 rounds pick-by-pick using an ML model that scores every available prospect for every team, displays results in a polished React UI, and lets users ask an AI scout backed by live application data any draft question they have. The entire system is deployed on AWS and ships through a GitHub Actions CI/CD pipeline.

---

## Goals and Scope

| Goal | What Was Built |
|------|---------------|
| Simulate the 2026 draft realistically | 7-round, 224-pick simulation engine with team-specific behavior |
| Predict which prospect each team selects | XGBoost LambdaMART ranker trained on 16 years of draft data |
| Model GM-specific drafting behavior | Bayesian shrinkage + recency decay tendency profiles for every current GM |
| Surface model uncertainty | Split conformal prediction: statistically valid 90%/85%/80% coverage sets |
| Explain individual picks | SHAP TreeExplainer with per-pick feature attributions in the UI |
| Let users explore counterfactuals | What-If scenario editor and pick-swap counterfactual API |
| Answer natural-language questions | Claude tool-use agent grounded in live DB data, 10 DB-backed tools, hybrid RAG |
| Make data fresh | Automated daily ingestion via EventBridge + Lambda; resumable pipeline with checkpointing |
| Ship like a real product | AWS CDK infrastructure, 17 Alembic migrations, 70%+ test coverage CI gate |

---

## Architecture Overview

```
Browser
  └── CloudFront → S3  (React SPA, hashed asset caching)
  └── EC2 (FastAPI behind Nginx)
        ├── PostgreSQL 16 + pgvector  (main data + vector embeddings)
        ├── Redis                      (simulation cache, SSE pub/sub)
        └── Anthropic API              (Claude 3 for Scout + draft analysis)

EventBridge (daily cron)
  └── Lambda → POST /api/admin/ingest  (standings, prospects, stats refresh)
```

The backend is a single FastAPI application (~6,500 lines across APIs, ML, agent, engines, and ingestion). The frontend is a React + Vite SPA with 9 pages. Infrastructure is defined as AWS CDK in TypeScript.

---

## Frontend

**Stack:** React 18, TypeScript, Tailwind CSS, Vite, React Router, React Query, Zustand, Recharts, Framer Motion

### Pages

| Page | What It Does |
|------|-------------|
| `HomePage.tsx` | Guided entry: runs the lottery with animated reveal, auto-runs the draft, shows round-grouped results |
| `LotteryPage.tsx` | Live standings, per-team odds, seeded lottery simulation |
| `DraftPage.tsx` | Full 7-round draft board grouped by round; shareable URLs via seed+order params; AI analysis via SSE; per-pick confidence badges; SHAP drawer for round-1 picks |
| `ProspectsPage.tsx` | Prospect browsing with position/league/nationality filters |
| `TeamPage.tsx` | GM profile, archetype badge, position tendency chart, draft history table |
| `ScoutPage.tsx` | Streaming AI Scout chat; SSE-consuming token-by-token render |
| `ModelPage.tsx` | Model status, NDCG@1 eval results, multi-year backtest charts, SHAP feature importance, baseline comparisons, calibration coverage |
| `ScenarioPage.tsx` | What-If simulator: drag-and-drop pick order editor, simulate any team ordering |
| `AdminPage.tsx` | Ingestion triggers, model training, tendency recompute, index rebuild |

### Notable Frontend Engineering

- **SSE consumption**: `DraftPage` and `ScoutPage` both consume server-sent event streams, managing partial JSON parsing and token accumulation in React state
- **Zustand cross-page state**: `useDraftStore` and `useLotteryStore` persist lottery result and draft picks across navigation so the full flow (lottery → draft board → team page) stays coherent
- **Shareable URLs**: `DraftPage` encodes `seed` and `order` query params so any simulated draft can be shared and reproduced exactly
- **SHAP drawer**: inline expandable component per pick that fetches and renders SHAP feature attributions without leaving the draft board
- **Confidence badges**: `in_prediction_set` and `confidence` fields from the API drive "High confidence / In model range / Volatile pick" badges per card
- **Drag-and-drop reorder**: `ScenarioPage` implements HTML5 drag-and-drop with fallback arrow buttons for touch, clearing simulation results on order change

---

## Backend

**Stack:** FastAPI, SQLAlchemy 2, Alembic, PostgreSQL 16, pgvector, Redis, Python 3.11

### API Modules

| Module | Endpoints | Key Logic |
|--------|-----------|-----------|
| `lottery.py` | `GET /api/lottery/odds`, `POST /api/lottery/simulate` | Weighted lottery engine, seeded reproducibility |
| `draft_sim.py` | `POST /api/draft/simulate`, `POST /api/draft/summary/stream` | 7-round simulation, SSE draft analysis via Claude |
| `counterfactual.py` | `POST /api/draft/counterfactual/swap` | Dual-run baseline vs. pick-swapped scenario, divergence diff |
| `prospects.py` | `GET /api/prospects` | Filter by position/league/nationality with pagination |
| `teams_ext.py` | `GET /api/teams/{id}/tendency` | GM tendency profile, archetype, positional weights |
| `ml.py` | `POST /api/ml/train`, `GET /api/ml/status`, `GET /api/ml/backtest`, `POST /api/ml/calibrate`, `GET /api/ml/explain/{id}` | Full ML lifecycle management |
| `agent.py` | `POST /api/agent/chat`, `POST /api/agent/chat/stream`, `GET /api/agent/history` | Scout tool-use loop, SSE streaming, session history |
| `standings.py` | `GET /api/standings/live` | Live standings with lottery odds overlay |
| `admin.py` | `POST /api/admin/ingest`, `POST /api/admin/seed-prospects-2026`, `POST /api/admin/compute-tendencies`, `POST /api/admin/feature-store/refresh` | Operational pipeline management |

### Database Schema (17 Alembic Migrations)

| Table | Purpose |
|-------|---------|
| `team` | 32 NHL teams with abbreviation and conference |
| `general_manager` | Current GM per team with start date (hand-maintained) |
| `draft_pick_historical` | ~25,000 picks from 2000–2024 with CSS rank, PPG, position |
| `prospect_2025` | Current draft class (2026 class stored here) with CSS rank, stats |
| `lottery_odds` | Per-team lottery probability and combination counts |
| `gm_tendency_profile` | Computed tendency weights and archetype per GM |
| `prospect_stat_history` | Multi-season pre-draft stats per prospect |
| `prospect_features` | Materialized feature rows for inference speed |
| `scout_embeddings` | 768-dim pgvector embeddings (nomic-embed-text) |
| `scout_conversations` | Persistent session chat history |
| `ingestion_run` | Pipeline checkpointing with step-level metadata |
| `prospect_stat_snapshots` | Point-in-time stat snapshots for trend features |

Schema evolved through 17 migrations including: native `vector(768)` type with IVFFlat cosine index (migration 011/015), prospect feature store (012), missing index coverage (013), stat snapshots for trend features (014), pipeline step tracking (016).

---

## Draft Simulation Engine

### How a Pick Is Made

```
for each pick slot in [1 .. 224]:
    team = lottery_result[(pick_slot - 1) % 32]  # round-robin after round 1
    pool = [p for p in prospects if p.id not in taken]
    scores = score_pool_for_team(team, pool, board_state, gm_profile)
    temperature = _pick_temperature(base_T, pick_slot, pool, scores, total_picks)
    selected = temperature_scaled_sample(pool, scores, temperature)
    taken.add(selected.id)
    emit_pick(pick_slot, round, pick_in_round, team, selected)
```

### Temperature Scaling (Key Design Decision)

Draft variance is driven by **consensus gap**, not pick slot. When the top two available prospects are nearly equal, the pick is uncertain. When one clearly dominates, the pick is near-deterministic.

```python
gap = score(rank_1) - score(rank_2)   # normalized [0, 1]
T   = base_T * (1 - gap)^2            # tight gap → high T (more random)
                                       # clear gap → low T (near-deterministic)
```

`DEFAULT_TEMPERATURE = 0.15` strongly favors high-scored prospects while keeping realistic variation. At 0.4 even CSS #188 had non-trivial selection probability — so the lower value was chosen deliberately.

### What Makes It More Than BPA

The engine blends five signals simultaneously:
1. **ML score** — XGBoost ranker accounting for prospect quality, board context, and GM tendencies
2. **Position scarcity** — `pos_remaining_norm` and `pos_quality_rank_norm` features
3. **Team draft state** — `team_drafted_this_pos` tracks double-dipping
4. **GM tendency weights** — each GM's historical position/league/nationality preferences
5. **Controlled randomness** — temperature-scaled sampling from the full scored distribution

---

## ML System

### Problem Framing

This is a **learning-to-rank** problem (LambdaMART), not classification. Each historical draft slot becomes a query group: one positive row (the player actually selected) and N negative rows (prospects available but passed over). The model directly optimizes "the picked prospect should score above all others" — which matches the real GM decision structure.

This is superior to binary classification (`was_picked=1/0`) because:
- Learns relative ordering, not absolute probabilities
- Sees all alternatives simultaneously per pick slot
- Avoids the false assumption that each row is an independent event

### Model

```
Algorithm:     XGBoost LambdaMART (objective: rank:ndcg)
Features:      34 engineered features
Training data: Historical picks 2008–2024 (CSS era), ~92,000 rows
Train split:   ≤ 2020 (eval mode), all data (production mode)
Val split:     ≥ 2021
Early stop:    25 rounds on NDCG@1
Hyperparams:   learning_rate=0.05, max_depth=5, subsample=0.7, colsample_bytree=0.75
```

### Features (34 Total)

| Group | Features | Engineering Note |
|-------|----------|-----------------|
| **Position one-hot** (5) | C, LW, RW, D, G | — |
| **Nationality group** (5) | CAN, USA, NORDIC, SLAVIC, EUR_OTHER | 15 individual nationalities had zero importance; 5 groups give meaningful signal with less noise |
| **League tier** (5) | tier1_CAN, tier1_USA, tier1_EUR, tier2, tier3 | Replaces ordinal integer; model learns independent weights per pipeline |
| **Physical** (2) | height_cm, weight_kg | — |
| **League-normalized quality** (2) | ppg_league_norm, age_league_norm | PPG ÷ league-tier median. Raw PPG/age dropped: zero XGBoost importance because these are strictly better versions |
| **Consensus rank signals** (3) | pick_slot_norm, rank_vs_slot, rank_gap_norm | css_rank_norm dropped: linearly redundant with rank_vs_slot + pick_slot_norm |
| **GM tendency** (3) | gm_pos_weight, gm_league_weight, gm_nat_weight | gm_avg_deviation removed: pick slot is lottery-determined, not a preference signal |
| **Board state** (4) | pos_taken_before_norm, pos_remaining_norm, team_drafted_this_pos, pos_quality_rank_norm | Dynamic: recomputed pick-by-pick. pos_quality_rank_norm = "where does this prospect rank among same-position players still available?" |
| **Season trends** (4) | gp_pre_draft, ppg_prev_season, ppg_trend, has_prev_season | Season-over-season production trajectory |
| **Draft round** (1) | draft_round | Critical for multi-round: without this the model conflates round-1 BPA dynamics with round-7 developmental gambles |

### Negative Sampling

`NEGATIVE_WINDOW = 31` — draw negatives from within ~one round of the actual pick. This forces the model to learn fine-grained distinctions between realistic alternatives rather than wasting gradient on trivially easy negatives (e.g., "why wasn't a round-7 player taken with pick #1?").

### Evaluation

- **Metric**: NDCG@1 — did the model rank the actual pick first in its group?
- **Temporal split** — never train on future years, always val on held-out future draft classes
- **Multi-year backtests** — repeated temporal evaluation across 2021–2024 draft classes
- **Baselines**: CSS best-available, PPG best-available, uniform random
- **Additional**: top-1 accuracy, top-3 accuracy, top-5 accuracy, MRR

### Conformal Prediction

Split conformal prediction (inductive CP) provides statistically valid prediction sets:

```
A prospect marked in_prediction_set=True at 90% coverage means:
"In at least 90% of historical cases, the actual pick fell within this set."
```

Three coverage levels: α = 0.10 / 0.15 / 0.20. Coverage guarantee verified empirically in `tests/test_calibration.py::TestEmpiricalCoverage`. This produces UI labels:
- **High confidence** (≥0.90 confidence score)
- **In model range** (in prediction set, lower confidence)
- **Volatile pick** (`in_prediction_set = False`)

### SHAP Explainability

`ShapDrawer` in the frontend calls `GET /api/ml/explain/{prospect_id}?team_id=&pick=` which runs `shap.TreeExplainer` and returns per-feature attributions. Users can expand any round-1 pick card to see exactly which features drove the model's selection.

---

## GM Tendency Engine

### What It Computes

For each GM, the engine computes:
- Position preference distribution (e.g., this GM over-indexes on defensemen)
- League preference distribution (e.g., heavy CHL preference vs. European)
- Nationality preference distribution
- GM archetype: `BPA | need-based | safe | boom-bust | system-fit`

### Statistical Methods

**Recency weighting (temporal decay)**
```
weight = RECENCY_DECAY ^ (reference_year - pick_year)
RECENCY_DECAY = 0.85
→ last year = 1.0, 5 years ago = 0.44, 10 years ago = 0.20
```

**Bayesian shrinkage (empirical Bayes)**
```
alpha = n / (n + SHRINKAGE_K)    SHRINKAGE_K = 30
→ 10 picks:  25% GM weight / 75% population prior
→ 30 picks:  50/50
→ 100 picks: 77% GM / 23% prior
```

New GMs with few picks are pulled toward the league-wide average, preventing noisy single-sample estimates from dominating the draft engine. Round weights: `{1: 7, 2: 5, 3: 3, 4+: 1}` — earlier rounds carry more signal about a GM's true preferences.

**Archetype validation**: Computed archetypes are cross-checked against a manually curated ground-truth set of 8 well-documented GMs.

---

## AI Scout Agent

### What Makes It Not a Chatbot

The Scout is a **tool-use agent** built on Claude. Before answering any question, it calls real DB-backed tools and reasons over actual data — not pre-baked context or hallucinated facts.

### Tool-Use Loop

```
1. User message → load session history from scout_conversations
2. Send to Claude with TOOL_DEFINITIONS (Anthropic function-calling)
3. Claude returns tool_use blocks → execute ALL concurrently via asyncio.gather
4. Feed tool results back to Claude
5. Repeat up to MAX_TOOL_ROUNDS = 10
6. Stream final text response via SSE
7. Persist conversation to DB
```

Parallel tool execution (asyncio.gather + asyncio.to_thread) means N tool calls take `max(latencies)` not `sum(latencies)`.

### 10 Database-Backed Tools

| Tool | What It Does |
|------|-------------|
| `get_gm_profile` | GM drafting tendencies, archetype, position/league/nationality weights with percentages |
| `get_top_prospects` | Top prospects at any position ordered by CSS rank |
| `get_team_needs` | Positional needs derived from historical drafting gaps |
| `search_prospects` | Keyword search by name, league, or nationality |
| `semantic_prospect_search` | Finds prospects similar to a player archetype or description via pgvector |
| `get_draft_history` | A team's actual historical picks with player, pick number, CSS rank, PPG |
| `get_ml_ranking` | XGBoost score + top SHAP feature attributions for a prospect at a given slot |
| `compare_prospects` | Side-by-side stat comparison of two prospects |
| `get_prospect_detail` | Full profile + multi-season stat history |
| `get_nhl_comp` | Finds current NHL players most similar in playing style to a given prospect |

All tool inputs are validated through Pydantic models that double as the JSON schema for Claude's tool definitions.

### Reliability Features

- **Circuit breaker** (`circuit_breaker.py`): trips OPEN after 5 failures in 60s, fails fast for 30s, then probes — prevents cascading Anthropic API timeouts
- **Topic guardrail**: fast `claude-haiku-4-5-20251001` pre-flight classifier rejects off-topic queries before the expensive tool-use loop runs
- **Session memory compression**: when a session exceeds 20 messages, Claude summarizes the oldest turns into bullets, keeping the 6 most recent verbatim. Prevents context window overflow on long conversations
- **Prompt caching**: `cache_creation_input_tokens` and `cache_read_input_tokens` are emitted as Prometheus counters to verify Anthropic's ephemeral cache is actually hitting

### Hybrid RAG Retrieval

The Scout builds an index over GM profiles, prospect profiles (one rich prose document per prospect), and team draft histories.

```
Retrieval modes:
  Vector search:   pgvector cosine similarity on 768-dim nomic-embed-text embeddings
  Keyword search:  PostgreSQL BM25 via tsvector/ts_rank with GIN index
  Hybrid:          Reciprocal-rank fusion (RRF) combines both ranked lists
```

Embeddings are generated via Ollama (nomic-embed-text, 768-dim), running as a local service — no external API dependency for the retrieval layer.

---

## Counterfactual Analysis

`POST /api/draft/counterfactual/swap` accepts a completed lottery result and two pick positions to swap. It runs both the original and swapped simulations using the same seed, then returns:
- Full pick-by-pick results for both scenarios
- A **divergence list**: every pick slot where the two scenarios produced different prospects

This answers questions like: "What would Toronto have drafted if they had pick #3 instead of #8?" or "How would Vegas' outcome change if they had moved up from #11 to #5?"

---

## Ingestion Pipeline

### What Gets Refreshed

- Teams and current GMs (from NHL API + `gms.json`)
- Full draft history 2000–2024 (~25,000 picks) with CSS ranks and pre-draft stats
- Live standings with points and regulation overtime wins (tiebreaker for lottery)
- Current 2026 prospect class (seeded from bundled `prospects_2026.json` — 224 verified-eligible players)
- Pre-draft PPG and games played per prospect (from NHL Stats API, temporal integrity enforced)
- Season-over-season trends
- Feature store rows (materialized per-prospect features for inference)
- Scout RAG index (embeddings rebuilt on demand)

### Resumability

`PipelineCheckpointer` writes step metadata to `ingestion_run.pipeline_steps`. A partially-completed run (e.g., network failure mid-stats-fetch) resumes from the last successful step. This is essential for the stats fetch step which makes hundreds of individual NHL API calls.

### 2026 Prospect Sourcing

The NHL Records API only has data for completed drafts (2025 and earlier). The 2026 class is sourced from NHL Central Scouting and EliteProspects, hand-verified for draft eligibility (born Jan 1, 2006 – Sept 15, 2008, previously undrafted), and bundled as `backend/app/ingestion/prospects_2026.json`. This file contains 224 players. Top 5: Gavin McKenna (LW, CAN, CSS #1), Ivar Stenberg (LW, SWE, CSS #2), Chase Reid (D, USA, CSS #3), Keaton Verhoeff (D, CAN, CSS #4), Caleb Malhotra (C, CAN, CSS #5).

---

## Infrastructure (AWS CDK)

**All infrastructure is defined as code in TypeScript CDK.**

| Component | AWS Service | Notes |
|-----------|-------------|-------|
| Frontend delivery | CloudFront + S3 | Hashed assets: `max-age=31536000,immutable`. index.html: `no-cache,no-store` |
| Backend compute | EC2 (single instance) | Nginx reverse proxy + Docker Compose (API + Postgres + Redis + Ollama) |
| Static IP | Elastic IP | Stable address for DNS and API URL configuration |
| Object storage | S3 | ML model artifacts (versioned), frontend build artifacts |
| Scheduled ingestion | EventBridge + Lambda | Daily cron → POST /api/admin/ingest |
| Secrets | AWS Secrets Manager + SSM Parameter Store | DB password, admin key, Anthropic API key |
| Cost monitoring | AWS Budgets | Alert at $20/month threshold |
| CDN config | CloudFront custom error pages | 403/404 → index.html with 200 status for SPA routing |

Previous architecture used ECS + RDS + ALB + VPC (~$80/month). Migrated to single-EC2 + Docker Compose (~$20/month) to eliminate idle costs while maintaining all functionality.

---

## CI/CD (GitHub Actions)

### CI Pipeline (every push + PR)

| Job | What It Runs |
|-----|-------------|
| `backend` | `ruff` lint, unit tests with 70% coverage gate, coverage XML artifact |
| `integration` | Testcontainers PostgreSQL (pgvector/pgvector:pg16), integration tests |
| `migrations` | Clean DB, `alembic upgrade head` — catches migration regressions |
| `cdk-synth` | CDK synth validates CloudFormation output without AWS credentials |
| `cdk-diff` | OIDC-authenticated CDK diff against production stack |
| `docker` | `docker build` validates the Dockerfile |

All jobs use OIDC for AWS authentication — no long-lived keys stored as secrets.

### CD Pipeline (main branch only, after all CI passes)

1. Build and push backend Docker image to ECR (tagged with git SHA + `latest`)
2. Run Alembic migrations as a one-shot task (waits for exit code 0)
3. Force-deploy ECS service and wait for stability
4. Backend smoke tests: `/livez`, `/readyz`, `/api/ml/status`
5. Frontend: `npm run build` → S3 sync (hashed assets immutable, index.html no-cache)
6. CloudFront cache invalidation (`/*`)
7. Frontend smoke test: HTML response check
8. Lambda function code update for ingestion trigger

---

## Observability

| Layer | Tool |
|-------|------|
| Application metrics | Prometheus (FastAPI middleware, per-endpoint latency/count) |
| Agent metrics | Per-tool call count + latency, prompt cache hit/write counters |
| Structured logging | JSON logs with `extra={}` context dict, log level config |
| Health probes | `/livez` (process alive), `/readyz` (DB + Redis + model loaded) |
| Infra monitoring | CloudWatch dashboards, alarms on CPU/memory/request error rate |
| Alerting | SNS → email on CloudWatch alarm breach |
| Error tracking | Sentry (frontend + backend) |

---

## Testing

```
backend/tests/
  test_features.py          — feature engineering correctness, normalization invariants
  test_predict.py           — inference path, pool scoring, board-state computation
  test_calibration.py       — empirical coverage verification (conformal prediction)
  test_lottery_engine.py    — lottery weight math, seed reproducibility
  test_agent_embeddings.py  — embedding dimension, cosine similarity
  test_api_routes.py        — FastAPI route smoke tests with mocked DB
  integration/              — Testcontainers Postgres: full migration path, real queries
```

Coverage gate: 70% minimum enforced in CI. Integration tests use `pgvector/pgvector:pg16` (not `postgres:16-alpine`) because migration 011 requires the pgvector extension.

---

## Key Technical Decisions and Why

**LambdaMART over classification**: The draft problem is inherently a ranking problem. Each pick slot is a query; every available prospect is a candidate. Binary classification ignores that GMs see all alternatives simultaneously. NDCG@1 directly measures whether the model ranked the actual pick first — exactly the behavior being modeled.

**Bayesian shrinkage for GM profiles**: New GMs have few picks. Without shrinkage, a GM with 8 picks would produce very noisy tendency estimates that dominate the draft engine inappropriately. Shrinking toward the population prior with K=30 equivalent-picks makes the engine stable for new GMs while still capturing strong individual preferences for established ones.

**Consensus gap temperature**: Real draft unpredictability is not uniform across pick slots. When McDavid is available, the #1 pick is near-deterministic. When picks 15–25 are a cluster of equivalent prospects, variance spikes. Tying temperature to `(1 - gap)²` captures this correctly; tying it to pick slot would not.

**Grouped nationality features**: Individual nationality one-hots (15 groups) had zero XGBoost importance because rare nationalities almost never appear in leaf nodes. Grouping into 5 scouting-pipeline groups (CAN, USA, NORDIC, SLAVIC, EUR_OTHER) gives the model meaningful signal.

**Split conformal prediction over Platt scaling**: Platt scaling produces calibrated probabilities but no distributional coverage guarantee. Conformal prediction provides a mathematically verified set: "at least 90% coverage empirically" is a checkable, auditable claim rather than a confidence score.

**Reciprocal-rank fusion for hybrid retrieval**: Combining pgvector cosine similarity with PostgreSQL BM25 via RRF avoids the need to normalize scores between fundamentally different ranking functions. RRF only needs ranks, not comparable scores.

**Single-EC2 over ECS+RDS**: ECS with a managed RDS instance costs ~$80/month at minimum. A single t3.medium with Docker Compose costs ~$20/month. For a non-production app, the complexity and cost of ECS+RDS is unjustified — the EC2 approach still provides a repeatable, container-managed deployment.

---

## Current Limitations (Honest)

- **The model predicts GM behavior, not player quality.** High NDCG@1 means the model predicts what GMs do, not what they should do. A "correct" pick in training data might be a historically bad decision.
- **No trade simulation.** Real drafts involve pick trades before and during the event. Pick order is fixed post-lottery.
- **2026 prospect data is hand-curated.** The NHL API has no 2026 data. CSS rankings are bundled from the May 2026 published list and must be manually updated if rankings change.
- **Ollama runs on the same EC2.** For the RAG layer to work in production, Ollama must be running on the instance. Semantic search silently degrades if it's down; keyword search continues working.
- **GMs are hand-maintained.** `gms.json` must be updated when GMs are hired or fired. Stale entries produce tendency profiles attributed to the wrong person.
- **Round 2–7 dynamics are simplified.** The same model that drives round-1 picks drives later rounds, but GM behavior in rounds 6–7 is more opportunistic and harder to model. The `draft_round` feature helps, but late-round picks have more unexplained variance.

---

## Codebase Size

| Area | Files | Lines |
|------|-------|-------|
| Backend API | 11 | ~2,700 |
| ML pipeline | 7 | ~2,130 |
| Agent + Scout | 9 | ~2,800 |
| Engines | 2 | ~570 |
| Ingestion | 5 | ~800 |
| Models + DB | 3 | ~400 |
| Frontend pages | 9 | ~2,500 |
| Frontend components | 6 | ~600 |
| Tests | 12 | ~1,200 |
| Infrastructure (CDK) | 10 | ~800 |
| **Total** | | **~17,000** |

---

## Resume Bullet Points (Technical)

**ML / Modeling**
- Built an XGBoost LambdaMART learning-to-rank model to predict NHL draft picks, framing each draft slot as a query group with 1 positive (actual pick) and ~30 negatives (passed-over prospects) — directly optimizing NDCG@1 on held-out temporal splits
- Engineered 34 features across 10 groups including league-normalized PPG, pick-slot-relative rank, dynamic board-state features (position saturation, positional quality rank), and GM-specific tendency weights
- Implemented split conformal prediction at three coverage levels (90%/85%/80%) to produce statistically valid prediction sets for each pick; empirically verified coverage in automated tests
- Added SHAP TreeExplainer with per-pick feature attributions surfaced inline in the React UI, providing pick-level explainability without leaving the draft board
- Designed GM tendency profiles using Bayesian shrinkage (K=30 empirical Bayes, pulls new GMs toward population prior) and exponential recency decay (λ=0.85/year), making tendency estimates stable for GMs with few picks while capturing strong preferences for established ones

**Backend / Systems**
- Built a 7-round, 224-pick draft simulation engine in FastAPI that scores every available prospect for every team pick-by-pick, using consensus-gap-driven temperature scaling to model realistic pick uncertainty
- Designed a counterfactual analysis API that runs dual baseline/swapped simulations with a shared seed and returns a pick-by-pick divergence diff
- Architected a hybrid RAG retrieval system combining pgvector cosine similarity on 768-dim Ollama embeddings with PostgreSQL BM25 (tsvector/ts_rank), fused via reciprocal-rank fusion — no external vector DB dependency
- Built a Claude tool-use agent (10 DB-backed tools, parallel execution via asyncio.gather, circuit breaker, topic guardrail, session memory compression) that grounds every response in live application data
- Implemented a resumable ingestion pipeline with step-level checkpointing, preventing full restarts after partial failures in multi-hundred-API-call stat fetch operations

**Infrastructure / DevOps**
- Deployed the full stack on AWS using CDK in TypeScript: CloudFront/S3 frontend, EC2 backend with Nginx + Docker Compose, EventBridge + Lambda daily ingestion trigger
- Built a GitHub Actions CI/CD pipeline with OIDC authentication (no long-lived AWS keys): linting, unit tests (70% coverage gate), integration tests via Testcontainers, Alembic migration regression checks, Docker build validation, ECR push, and CloudFront cache invalidation
- Managed 17 Alembic migrations including native pgvector column type migration with IVFFlat cosine index and zero-downtime strategy

**Frontend**
- Built a 9-page React + TypeScript SPA consuming server-sent event streams for live draft simulation and AI Scout chat, with Zustand state persisted across navigation and shareable URLs encoding seed + pick order
- Implemented inline SHAP explainability drawers, pick confidence badges, and drag-and-drop pick order editor in the What-If scenario simulator
