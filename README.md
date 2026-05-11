# NHL Draft Simulator 2026

![CI](https://github.com/your-org/nhl-draft-simulator/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![TypeScript](https://img.shields.io/badge/frontend-react%20%2B%20typescript-3178c6)
![Infra](https://img.shields.io/badge/iac-aws%20cdk-orange)

A full-stack NHL draft product that simulates the 2025 NHL Entry Draft as realistically as possible. It combines a weighted lottery engine, a GM-tendency-aware draft simulator, an ML ranking model, a tool-use AI scout, and production AWS infrastructure.

---

## What This Project Demonstrates

| Area | Details |
|------|---------|
| **ML — ranking** | XGBoost LambdaMART (`rank:ndcg`), 33 features, temporal train/val split, NDCG@1 validation |
| **ML — uncertainty** | Split conformal prediction (inductive CP) with empirically verified 90%/85%/80% coverage |
| **ML — explainability** | SHAP TreeExplainer, per-pick feature attributions surfaced in the UI |
| **ML — evaluation** | Multi-year temporal backtests vs. CSS/PPG/random baselines, MRR, top-1/3/5 accuracy |
| **Statistical modeling** | Bayesian shrinkage (empirical Bayes) + recency decay for GM tendency profiles |
| **AI agent** | Claude tool-use loop, 10 DB-backed tools, prompt caching, circuit breaker, RAG (pgvector + Ollama) |
| **Streaming** | Server-Sent Events (SSE) for live draft summaries and agent chat; Redis Pub/Sub for fan-out |
| **Backend** | FastAPI, SQLAlchemy, Alembic (15 migrations), PostgreSQL 16 + pgvector, Redis |
| **Frontend** | React 18, TypeScript, Tailwind CSS, Vite; 8 pages with streaming SSE consumption |
| **Infrastructure** | AWS CDK (TypeScript): CloudFront/S3, API Gateway → ALB → ECS, RDS, S3 model registry |
| **Observability** | Prometheus metrics, structured JSON logging, CloudWatch dashboards, Sentry |
| **CI/CD** | GitHub Actions with OIDC auth (no long-lived keys), Alembic migration checks, 70% coverage gate |

---

## Architecture

```
Browser
  └── CloudFront → S3 (React SPA)
  └── API Gateway (throttle: 60 req/min)
        └── VPC Link → Private ALB
              └── ECS (FastAPI)
                    ├── RDS PostgreSQL 16 + pgvector
                    ├── Redis (cache + SSE pub/sub)
                    ├── S3 (model artifacts, versioned)
                    └── Anthropic API (Claude)

EventBridge (daily)
  └── Lambda → API Gateway /api/admin/ingest
```

---

## ML System

### GM tendency modeling

For each GM, we compute weighted distributions over position, league, and nationality using:
- **Recency decay** (0.85/year): recent picks count more than decade-old ones
- **Bayesian shrinkage** (K=30): GMs with few picks shrink toward the league average, preventing noisy estimates

Archetypes (BPA, need-based, safe, euro-scout, analytics) are automatically validated against manually curated ground-truth for 8 well-documented GMs.

### Draft simulation temperature

Pick variance is driven by **consensus gap**, not pick slot. When two prospects are ranked very close, the pick is uncertain. When one prospect clearly dominates, the pick is near-deterministic — regardless of where in the round it falls.

```python
gap = score(rank_1) - score(rank_2)
T = base_T * (1 - gap)^2    # tight gap → high uncertainty
```

### Conformal prediction

Calibration provides statistically valid prediction sets, not just raw scores. A prospect marked `in_prediction_set=True` at 90% coverage means:

> "In at least 90% of cases historically, the actual pick fell within this set."

This guarantee is verified empirically in `tests/test_calibration.py::TestEmpiricalCoverage`.

---

## Data Sources

| Data | Source | Notes |
|------|--------|-------|
| Pre-draft PPG, GP, age | NHL Stats API (season before draft year, by player ID) | Temporal integrity enforced — no post-draft stats |
| CSS rankings (2008+) | NHL Records API (`records.nhl.com/site/api/draft`) | Square-root normalised; pre-2008 uses pick-order proxy |
| GM identity + tenure | `backend/app/ingestion/gms.json` (manually maintained) | Must be updated when GMs change |
| 2025 prospect class | Seeded via `POST /api/admin/seed-prospects` | CSS rankings entered manually from published lists |

---

## Known Limitations

- **No trade simulation.** Pick order is fixed post-lottery. Real drafts involve pick trades.
- **GM data is hand-maintained.** `gms.json` must be updated when GMs are hired or fired.

---

## Quick Start

```bash
git clone https://github.com/your-org/nhl-draft-simulator
cd nhl-draft-simulator

# 1. Start dependencies
cp backend/.env.example backend/.env
docker compose up -d db redis ollama

# 2. Pull the embedding model (one-time)
docker compose exec ollama ollama pull nomic-embed-text

# 3. Migrations + ingestion
cd backend
alembic upgrade head
# Then run the data pipeline (see CLAUDE.md for ordered API calls)

# 4. Frontend
cd ../frontend
npm install
npm run dev    # http://localhost:5173
```

---

## Running Tests

```bash
cd backend

# Unit tests (no Docker required)
pytest tests/ -v --tb=short --cov=app --cov-fail-under=70

# Integration tests (requires Docker)
pytest tests/integration/ -v --tb=short

# Key test: conformal calibration coverage guarantee
pytest tests/test_calibration.py::TestEmpiricalCoverage -v
```

---

## Project Structure

```
nhl-draft-simulator/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes (lottery, draft, prospects, ml, agent, admin)
│   │   ├── ml/           # Training, features, predict, calibration, backtest, SHAP
│   │   ├── agent/        # Scout tool-use loop, embeddings, memory, circuit breaker
│   │   ├── engines/      # GM tendency engine, lottery engine
│   │   ├── ingestion/    # NHL API clients, stat ingestion, checkpointer
│   │   ├── models/       # SQLAlchemy ORM models
│   │   └── observability/# Prometheus metrics, structured logging, data quality
│   ├── alembic/versions/ # 15 migrations
│   └── tests/            # Unit + integration tests
├── frontend/
│   └── src/pages/        # 8 React pages
├── infrastructure/
│   └── cdk/              # AWS CDK stacks (TypeScript)
│       └── lib/constructs/  # network, compute, api, frontend, storage, ingestion, observability
└── lambda/               # EventBridge-triggered ingestion Lambda
```

---

## Docs

- [CLAUDE.md](CLAUDE.md) — architecture details, data sources, known limitations, constants reference
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — full ML model documentation
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — runbooks for deployment and incident response
