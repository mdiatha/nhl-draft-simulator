"""
Prometheus metrics for the NHL Draft Simulator.

Exposed at GET /metrics (text/plain Prometheus format).
Scraped by Prometheus server or AWS Managed Prometheus.

Metric naming follows Prometheus conventions:
  <namespace>_<subsystem>_<name>_<unit>

Usage — import the metric and record it at the call site:
  from app.observability.metrics import SIMULATIONS_TOTAL
  SIMULATIONS_TOTAL.labels(status="ok").inc()
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, CollectorRegistry

# Use the default registry so /metrics picks them up automatically.

# ── HTTP ──────────────────────────────────────────────────────────────────────

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests by method, path, and status code",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ── Draft simulation ──────────────────────────────────────────────────────────

SIMULATIONS_TOTAL = Counter(
    "draft_simulations_total",
    "Draft simulations completed",
    ["status"],          # ok | error | cache_hit
)

SIMULATION_DURATION = Histogram(
    "draft_simulation_duration_seconds",
    "Time to run a full draft simulation (excluding cache hits)",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

CACHE_HITS_TOTAL = Counter(
    "redis_cache_hits_total",
    "Simulation results served from Redis cache",
)

CACHE_MISSES_TOTAL = Counter(
    "redis_cache_misses_total",
    "Simulation results computed (cache miss)",
)

# ── ML model ──────────────────────────────────────────────────────────────────

MODEL_TRAINING_DURATION = Histogram(
    "model_training_duration_seconds",
    "Time to train and persist the XGBoost model",
    buckets=[10, 30, 60, 120, 300, 600],
)

MODEL_VALIDATION_AUC = Gauge(
    "model_validation_auc",
    "ROC-AUC on the temporal validation split from the last training run",
)

MODEL_TRAINING_SAMPLES = Gauge(
    "model_training_samples_total",
    "Number of rows used to train the last model",
)

MODEL_INFO = Info(
    "model",
    "Metadata about the currently loaded model",
)

# ── Ingestion pipeline ────────────────────────────────────────────────────────

INGESTION_DURATION = Histogram(
    "ingestion_step_duration_seconds",
    "Duration of each ingestion pipeline step",
    ["step"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

INGESTION_ROWS = Gauge(
    "ingestion_rows_total",
    "Rows upserted/updated in the last ingestion run",
    ["entity"],          # prospects | draft_picks | teams | gms
)

# ── Data quality ──────────────────────────────────────────────────────────────

DATA_QUALITY_CHECKS = Counter(
    "data_quality_checks_total",
    "Data quality check results",
    ["check", "result"],   # result: pass | fail | warn
)

PROSPECTS_LOADED = Gauge(
    "prospects_loaded_total",
    "Number of 2025 prospects currently in the database",
)

PROSPECTS_WITH_PREV_SEASON = Gauge(
    "prospects_with_prev_season_total",
    "Prospects where ppg_prev_season is populated",
)

# ── ML prediction ─────────────────────────────────────────────────────────────

MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total calls to score_pool_for_team",
    ["source"],       # draft_sim | scores_api | ensemble
)

MODEL_PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Errors during model prediction scoring",
)

MODEL_PREDICTION_DURATION = Histogram(
    "model_prediction_duration_seconds",
    "Time to score a full prospect pool for one pick slot",
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
)

MODEL_PREDICTION_POOL_SIZE = Histogram(
    "model_prediction_pool_size",
    "Number of prospects scored per prediction call",
    buckets=[10, 25, 50, 100, 150, 200, 250],
)

# ── SHAP explainability ───────────────────────────────────────────────────────

SHAP_CALLS_TOTAL = Counter(
    "model_shap_calls_total",
    "Total SHAP explanation requests",
)

SHAP_ERRORS_TOTAL = Counter(
    "model_shap_errors_total",
    "SHAP explanation failures",
)

SHAP_DURATION = Histogram(
    "model_shap_explanation_duration_seconds",
    "Time to compute a SHAP explanation for one (prospect, team, pick) triple",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# ── Drift monitoring ──────────────────────────────────────────────────────────

DRIFT_JS_DIVERGENCE = Gauge(
    "model_drift_js_divergence",
    "Jensen-Shannon divergence between training and current distributions",
    ["dimension"],    # position | nationality | draft_league_tier | points_per_game | age_at_draft
)

DRIFT_STATUS = Gauge(
    "model_drift_status",
    "Drift status per dimension: 0=pass, 1=warn, 2=fail",
    ["dimension"],
)

# ── Backtest quality ──────────────────────────────────────────────────────────

BACKTEST_TOP1_ACCURACY = Gauge(
    "backtest_top1_accuracy",
    "Top-1 accuracy from the most recent backtest run",
)

BACKTEST_TOP3_ACCURACY = Gauge(
    "backtest_top3_accuracy",
    "Top-3 accuracy from the most recent backtest run",
)

BACKTEST_MRR = Gauge(
    "backtest_mrr",
    "Mean Reciprocal Rank from the most recent backtest run",
)

BACKTEST_WORST_RANK = Gauge(
    "backtest_worst_actual_rank",
    "Actual rank of the worst miss in the most recent backtest run",
)

# ── Model age ─────────────────────────────────────────────────────────────────

MODEL_AGE_DAYS = Gauge(
    "model_age_days",
    "Days since the current model was trained",
)

# ── Scout agent ───────────────────────────────────────────────────────────────

SCOUT_REQUESTS_TOTAL = Counter(
    "scout_requests_total",
    "Total Scout chat requests",
    ["mode"],           # stream | sync
)

SCOUT_TOOL_CALLS_TOTAL = Counter(
    "scout_tool_calls_total",
    "Tool calls executed by the Scout agent",
    ["tool", "status"],  # status: ok | error
)

SCOUT_TOOL_DURATION = Histogram(
    "scout_tool_duration_seconds",
    "Latency of each Scout tool call",
    ["tool"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

SCOUT_TOOL_ROUNDS = Histogram(
    "scout_tool_rounds_total",
    "Number of tool rounds per Scout request",
    buckets=[0, 1, 2, 3, 4, 5],
)

SCOUT_PROMPT_CACHE_READ_TOKENS = Counter(
    "scout_prompt_cache_read_tokens_total",
    "Anthropic prompt-cache tokens read (ephemeral cache hits)",
)

SCOUT_PROMPT_CACHE_WRITE_TOKENS = Counter(
    "scout_prompt_cache_write_tokens_total",
    "Anthropic prompt-cache tokens written (cache misses — new cache entry)",
)

SCOUT_INPUT_TOKENS = Counter(
    "scout_input_tokens_total",
    "Anthropic input tokens consumed by the Scout (cache reads + uncached)",
)

SCOUT_OUTPUT_TOKENS = Counter(
    "scout_output_tokens_total",
    "Anthropic output tokens produced by the Scout",
)

SCOUT_CIRCUIT_BREAKER_OPEN = Counter(
    "scout_circuit_breaker_open_total",
    "Number of times the Anthropic circuit breaker tripped open",
)

SCOUT_CIRCUIT_BREAKER_REJECTED = Counter(
    "scout_circuit_breaker_rejected_total",
    "Requests rejected while the circuit breaker is open",
)
