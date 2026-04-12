# App Summary

## Overview

NHL Draft Simulator is a full-stack sports analytics app that simulates the NHL draft as a realistic product, not just a static mock draft generator. It combines a React frontend, a FastAPI backend, a PostgreSQL data model, an ML ranking pipeline, and a grounded AI Scout assistant to let users:

- simulate the draft lottery
- run a full 32-pick draft
- browse and compare prospects
- inspect team and GM drafting tendencies
- evaluate the ranking model
- ask an AI scout questions backed by live application data
- run ingestion and model operations from an admin surface

At a high level, the app answers two related questions:

1. What is a team likely to do on draft day?
2. Why does that pick make sense given the board, the model, and the team's historical behavior?

## What Users Can Do

### 1. Run the lottery and draft

Users can start from the home page, simulate the lottery, and then run a complete round-one draft. The draft flow supports seeded, reproducible runs, shareable URLs, and AI-generated draft analysis.

### 2. Explore prospects and teams

Users can browse prospects by filters like position and league, inspect team pages, and view how different GMs tend to draft based on historical patterns.

### 3. Use the Scout assistant

Users can ask natural-language questions such as:

- Which defensemen fit Montreal best?
- How does Detroit usually draft?
- Compare two prospects.
- Give me NHL comps for this player.

The Scout is not a generic chatbot. It uses the app's own database tools and retrieval layer before answering.

### 4. Inspect model behavior

Users can view model status, backtest results, baseline comparisons, and SHAP explanations to understand how and why the ranking model behaves the way it does.

### 5. Operate the system

Admin users can trigger ingestion, refresh the feature store, compute GM tendencies, retrain the model, and manage the Scout index.

## Frontend

The frontend is a React + Vite single-page application with route-based product surfaces and lazy-loaded pages. It uses:

- React Router for navigation
- React Query for server-state fetching
- Zustand for draft and lottery state
- Recharts and Framer Motion for analytics and UI polish

### Main Pages

- `frontend/src/pages/HomePage.tsx`
  Guided entry flow that runs the lottery and moves the user into the draft experience.
- `frontend/src/pages/LotteryPage.tsx`
  Live standings, lottery odds, and lottery simulation UI.
- `frontend/src/pages/DraftPage.tsx`
  Pick-by-pick draft board, shareable runs, AI draft analysis, and confidence indicators.
- `frontend/src/pages/ProspectsPage.tsx`
  Prospect browsing and filtering.
- `frontend/src/pages/TeamPage.tsx`
  Team- and GM-specific drafting behavior.
- `frontend/src/pages/ScoutPage.tsx`
  Streaming AI Scout chat experience.
- `frontend/src/pages/ModelPage.tsx`
  Model status, feature importance, baseline comparisons, and multi-year backtests.
- `frontend/src/pages/AdminPage.tsx`
  Operational tasks like ingestion and training.

### Frontend Role in the System

The frontend is more than a dashboard. It acts as the product layer for:

- simulation
- analytics
- explainability
- AI interaction
- operations

## Backend

The backend is a FastAPI application built as a modular API service. The app shell in `backend/app/main.py` wires together:

- router registration
- middleware
- CORS
- metrics
- structured logging
- security headers
- health, liveness, and readiness probes
- model registry startup and reload behavior

### Main API Areas

- `backend/app/api/lottery.py`
  Lottery odds and simulation endpoints.
- `backend/app/api/draft_sim.py`
  Draft simulation and AI draft summary endpoints.
- `backend/app/api/prospects.py`
  Prospect search and filtering APIs.
- `backend/app/api/teams_ext.py`
  Team tendencies and related analytics.
- `backend/app/api/ml.py`
  Training, backtesting, explainability, drift, calibration, and model status.
- `backend/app/api/agent.py`
  Scout chat, streaming, history, and index rebuild APIs.
- `backend/app/api/admin.py`
  Ingestion and operational endpoints.
- `backend/app/api/standings.py`
  Live standings and lottery odds.

The backend owns almost all business logic in the product:

- draft rules
- ML inference
- AI tool execution
- ingestion
- model lifecycle
- observability

## Data Model

The database layer is built around a relational PostgreSQL schema that stores both product data and operational metadata.

### Core Domain Tables

- `team`
- `general_manager`
- `draft_pick_historical`
- `prospect_2025`
- `lottery_odds`
- `gm_tendency_profile`
- `prospect_stat_history`
- `prospect_features`
- `ingestion_run`

This schema matters because the app is not only showing current prospects. It also needs:

- historical draft behavior for training
- stat history for feature freshness
- stored GM profiles for simulation
- ingestion metadata for resumable pipelines
- cached features for inference performance

## Draft Engine

The draft engine is one of the most important parts of the app.

### What It Does

Once a lottery order is known, the engine simulates the draft pick by pick:

1. Identify the team making the pick.
2. Build the pool of still-available prospects.
3. Score each prospect for that team and slot.
4. Sample a selection from the scored distribution.
5. Remove the chosen player from the board.
6. Repeat for the next pick.

### Why It Is More Than "Best Player Available"

The engine does not simply sort players by a single ranking. It blends:

- ML scores from the ranking model
- GM tendency weights
- board-state context
- position scarcity
- pick-slot context
- controlled randomness via temperature-scaled sampling

That design lets the simulator produce believable draft outcomes that are:

- team-specific
- non-deterministic
- still anchored in historical behavior

## GM Tendency Engine

The tendency engine computes how a GM historically drafts.

It looks at:

- position preferences
- league preferences
- nationality preferences
- average deviation from expected ranking

Two notable statistical ideas are built into this layer:

- Recency weighting
  Recent draft behavior counts more than old behavior.
- Bayesian shrinkage
  Small-sample GMs are pulled toward the league-wide average to avoid noisy overfitting.

This gives the draft engine a behavioral layer that makes teams feel distinct rather than interchangeable.

## ML System

The ML system is designed as a ranking problem, not a generic binary classification problem.

### What It Predicts

The model predicts which prospect a team is most likely to select at a given slot. It is a behavioral model of draft decisions, not a pure predictor of long-term NHL value.

### Main ML Components

- `backend/app/ml/features.py`
  Feature engineering and training dataset construction.
- `backend/app/ml/train.py`
  XGBoost LambdaMART training pipeline.
- `backend/app/ml/predict.py`
  Inference and team-specific pool scoring.
- `backend/app/ml/backtest.py`
  Temporal evaluation, baseline comparisons, and multi-year reporting.
- `backend/app/ml/calibration.py`
  Calibration and confidence metadata.
- `backend/app/ml/explain.py`
  SHAP explainability.
- `backend/app/ml/registry.py`
  In-memory model registry with hot-reload support.

### Training Setup

Each historical draft slot is turned into a ranking group:

- 1 positive example: the player actually selected
- N negative examples: players still available but not selected

That setup matches the real problem the app is trying to model: which player gets taken over the alternatives on the board.

### Evaluation

The ML layer includes:

- temporal held-out backtests
- baseline comparisons against simple heuristics
- multi-year evaluation
- calibration metadata
- SHAP explanations for feature-level reasoning

This makes the ML system inspectable instead of opaque.

## AI Scout and Retrieval Layer

The AI Scout is an agent-style system grounded in application data.

### What Makes It Different From a Basic Chatbot

The Scout can call real tools defined in `backend/app/agent/tools.py`, including:

- team/GM profile lookup
- top prospects
- team needs
- keyword prospect search
- semantic prospect search
- draft history lookup
- ML ranking + SHAP explanation
- prospect comparison
- prospect detail
- NHL comp lookup

### Retrieval / RAG

The app builds an index over:

- GM tendency profiles
- prospect profiles
- prospect stat histories
- team draft histories
- NHL player comparison data

It stores embeddings in `scout_embeddings` and retrieves with:

- pgvector cosine similarity
- BM25-style PostgreSQL full-text search
- reciprocal-rank fusion for hybrid retrieval

This gives the agent both semantic and keyword retrieval paths.

### Streaming

The Scout and draft summary flows stream tokens back to the frontend using server-sent events. A Redis-compatible pub/sub layer supports fan-out and reconnection behavior across multiple service instances.

## Ingestion and Data Refresh

The app includes a real ingestion pipeline instead of relying on one-time seed data.

### What It Refreshes

- teams and GMs
- historical draft data
- lottery-related data
- current prospects
- live and historical prospect stats
- feature store rows
- Scout embedding/index content

### Resumability

The pipeline includes step checkpointing through `PipelineCheckpointer`, which means a partially completed run can resume from the last successful step rather than starting from scratch.

### Scheduling

Scheduled refreshes are triggered daily via EventBridge and Lambda. Manual admin endpoints can also run the same workflows on demand.

## Infrastructure

The app is deployed as an AWS-backed full-stack system defined in Terraform.

### Frontend Delivery

- CloudFront
- S3
- optional Route 53 + ACM custom domains

### Backend Delivery

- API Gateway HTTP API
- private ALB
- ECS on EC2
- RDS PostgreSQL

### Supporting Services

- S3 for model artifacts
- Secrets Manager + SSM Parameter Store for configuration and secrets
- Lambda + EventBridge for scheduled ingestion
- CloudWatch + SNS for logs, alarms, and notifications
- AWS Budgets for spend alerting

### Why the Infra Matters

This stack makes the project feel like a production service, not a local-only demo. It covers:

- frontend delivery
- backend orchestration
- managed database infrastructure
- scheduled jobs
- secret management
- monitoring
- cost awareness

## CI/CD and Reliability

Deployment is automated with GitHub Actions.

### CI

On pushes and pull requests, the workflow runs:

- backend linting
- unit tests
- integration tests
- Alembic migration checks
- Terraform validate
- Terraform plan
- Docker build validation

### CD

On pushes to `main`, the workflow:

1. pushes the backend image to ECR
2. runs migrations as an ECS task
3. rolls the ECS service forward
4. smoke-tests the backend
5. builds and deploys the frontend to S3
6. invalidates CloudFront
7. updates the ingestion Lambda

### Observability

The app exposes both infrastructure and application observability through:

- CloudWatch dashboards and alarms
- structured logging
- Prometheus metrics at `/metrics`
- readiness and liveness endpoints

## What Makes This App Strong

This project is strong because it combines several layers that are often separate portfolio projects:

- polished frontend product work
- backend API design
- simulation engine logic
- statistical modeling
- explainable ML
- retrieval-augmented AI
- ingestion pipelines
- AWS deployment and CI/CD

It is not only a user interface, and it is not only a notebook model. It is a full product and service stack.

## Honest Limitations

The most important limitation is that the ranking model predicts draft behavior, not long-term player value. In other words, it models who teams are likely to select, not who will necessarily become the best NHL player.

Other limitations include:

- drafting behavior changes over time
- live data quality depends on ingestion freshness
- late-round picks are noisier than early picks
- the AI layer is only as strong as the underlying retrieval and tools

## Good Short Description

NHL Draft Simulator is a full-stack draft analytics platform that combines lottery simulation, a pick-by-pick draft engine, GM tendency modeling, an explainable ML ranking system, and a grounded AI Scout assistant on top of a FastAPI, React, PostgreSQL, and AWS stack.
