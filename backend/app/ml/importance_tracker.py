"""
Feature importance drift tracking across training runs.

Problem:
  After a data refresh or feature set change, the model's SHAP global importance
  rankings can shift significantly. If css_rank_norm drops from rank #1 to rank #5,
  that's a signal that something broke in the data pipeline — not a model improvement.
  Currently there's feature *distribution* drift detection (Jensen-Shannon on data),
  but no tracking of model *importance* drift between training runs.

Solution:
  - After each training run, compute SHAP global importances and save to
    model_importance_history.jsonl (one JSON record per training run).
  - `compare_importance_drift()` computes rank correlation (Spearman ρ) between
    the current run and the previous run. Low ρ (< IMPORTANCE_DRIFT_THRESHOLD)
    triggers a warning — feature rankings changed substantially.
  - Individual feature alerts fire if a feature's rank moves by more than
    RANK_SHIFT_ALERT positions (e.g., primary signal drops from top-3 to bottom-10).

Integration:
  Called automatically at the end of train_from_db() in train.py.
  Results surfaced at GET /api/ml/importance-drift.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

IMPORTANCE_HISTORY_PATH = Path(__file__).parent / "model_importance_history.jsonl"
IMPORTANCE_DRIFT_THRESHOLD = 0.70   # Spearman ρ below this → warn
RANK_SHIFT_ALERT = 5                # individual feature rank shift that triggers alert


def compute_and_save_importances(model, feature_cols: list[str], trained_at: str) -> dict:
    """
    Compute SHAP global feature importances for a trained XGBRanker and
    append them to model_importance_history.jsonl.

    Uses XGBoost's built-in feature_importances_ (gain) as a fast proxy.
    For full SHAP computation use app.ml.explain — this is intentionally
    fast so it doesn't slow down the training pipeline.

    Returns the importance record dict (also saved to disk).
    """
    importances = model.feature_importances_   # gain-based, shape (n_features,)
    ranked = sorted(
        zip(feature_cols, importances.tolist()),
        key=lambda x: -x[1],
    )

    record = {
        "trained_at": trained_at,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "importances": {name: round(score, 6) for name, score in ranked},
        "top_10": [name for name, _ in ranked[:10]],
        "bottom_5": [name for name, _ in ranked[-5:]],
    }

    with open(IMPORTANCE_HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        "importance_tracker.saved trained_at=%s top3=%s",
        trained_at, record["top_10"][:3],
    )
    return record


def load_importance_history(n: int = 10) -> list[dict]:
    """Load the last *n* importance records from the JSONL file."""
    if not IMPORTANCE_HISTORY_PATH.exists():
        return []
    records = []
    with open(IMPORTANCE_HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records[-n:]


def compare_importance_drift(feature_cols: list[str]) -> dict:
    """
    Compare current model importances to the previous training run.

    Returns a drift report with:
      - spearman_rho: rank correlation between current and previous (1.0 = identical ranking)
      - drifted: True if ρ < IMPORTANCE_DRIFT_THRESHOLD
      - feature_alerts: features whose rank shifted by > RANK_SHIFT_ALERT positions
      - previous_run: trained_at timestamp of the comparison baseline
    """
    history = load_importance_history(n=2)

    if len(history) < 2:
        return {
            "status": "insufficient_history",
            "message": "Need at least 2 training runs to compare importance drift.",
            "runs_available": len(history),
        }

    current  = history[-1]
    previous = history[-2]

    # Build rank vectors aligned to feature_cols
    def _rank_vector(record: dict) -> np.ndarray:
        importances = record["importances"]
        # Sort features by importance (descending) to get rank
        sorted_features = sorted(importances.keys(), key=lambda k: -importances.get(k, 0))
        rank_map = {f: i + 1 for i, f in enumerate(sorted_features)}
        return np.array([rank_map.get(f, len(feature_cols)) for f in feature_cols], dtype=float)

    curr_ranks = _rank_vector(current)
    prev_ranks = _rank_vector(previous)

    # Spearman rank correlation
    rho = float(_spearman_rho(curr_ranks, prev_ranks))

    # Per-feature rank shifts
    curr_rank_map = {f: r for f, r in zip(feature_cols, curr_ranks.tolist())}
    prev_rank_map = {f: r for f, r in zip(feature_cols, prev_ranks.tolist())}

    alerts = []
    for f in feature_cols:
        shift = abs(curr_rank_map.get(f, 0) - prev_rank_map.get(f, 0))
        if shift >= RANK_SHIFT_ALERT:
            alerts.append({
                "feature": f,
                "prev_rank": int(prev_rank_map.get(f, -1)),
                "curr_rank": int(curr_rank_map.get(f, -1)),
                "rank_shift": int(shift),
            })

    alerts.sort(key=lambda x: -x["rank_shift"])

    drifted = rho < IMPORTANCE_DRIFT_THRESHOLD

    if drifted:
        logger.warning(
            "importance_tracker.drift_detected rho=%.3f threshold=%.2f alerts=%d",
            rho, IMPORTANCE_DRIFT_THRESHOLD, len(alerts),
        )

    return {
        "current_run":    current["trained_at"],
        "previous_run":   previous["trained_at"],
        "spearman_rho":   round(rho, 4),
        "drifted":        drifted,
        "threshold":      IMPORTANCE_DRIFT_THRESHOLD,
        "feature_alerts": alerts[:10],   # top-10 biggest shifts
        "current_top_10": current["top_10"],
        "previous_top_10": previous["top_10"],
    }


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    n = len(a)
    if n == 0:
        return 0.0
    d_sq = np.sum((a - b) ** 2)
    return float(1 - (6 * d_sq) / (n * (n ** 2 - 1)))
