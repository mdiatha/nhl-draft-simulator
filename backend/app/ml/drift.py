"""Drift detection for the NHL draft prediction model.

Compares the distribution of key features in the current 2025 draft class
against the training data distribution. High drift in position/league/nationality
signals the model may need retraining for the current season.

Uses Jensen-Shannon divergence (symmetric, bounded [0,1]) rather than KL divergence.
JS=0 means identical distributions; JS>0.1 is worth monitoring; JS>0.3 suggests drift.
"""
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DRIFT_WARN_THRESHOLD = 0.10   # JS divergence
DRIFT_FAIL_THRESHOLD = 0.30


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two distributions.

    Both arrays are normalized internally so raw counts can be passed in.
    JS=0 means identical distributions; JS=1 means maximally different.
    """
    p = p.astype(float)
    q = q.astype(float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log((a + 1e-10) / (b + 1e-10))))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _status_from_js(js: float) -> str:
    if js >= DRIFT_FAIL_THRESHOLD:
        return "fail"
    if js >= DRIFT_WARN_THRESHOLD:
        return "warn"
    return "pass"


def _categorical_divergence(
    train_vals: list,
    current_vals: list,
    vocab: list[str] | None = None,
) -> dict:
    """Compute JS divergence between two categorical distributions.

    Returns a dict with js_divergence, status, train_distribution, current_distribution.
    """
    from collections import Counter

    train_counts = Counter(v for v in train_vals if v is not None)
    current_counts = Counter(v for v in current_vals if v is not None)

    # Build a shared vocabulary
    if vocab is None:
        vocab = sorted(set(train_counts.keys()) | set(current_counts.keys()))

    if not vocab:
        return {
            "js_divergence": 0.0,
            "status": "pass",
            "train_distribution": {},
            "current_distribution": {},
            "note": "no data",
        }

    train_vec = np.array([train_counts.get(k, 0) for k in vocab], dtype=float)
    current_vec = np.array([current_counts.get(k, 0) for k in vocab], dtype=float)

    js = _js_divergence(train_vec, current_vec)
    status = _status_from_js(js)

    # Normalize for display
    train_total = train_vec.sum()
    current_total = current_vec.sum()

    train_dist = {
        k: round(float(train_vec[i]) / train_total, 4) if train_total > 0 else 0.0
        for i, k in enumerate(vocab)
    }
    current_dist = {
        k: round(float(current_vec[i]) / current_total, 4) if current_total > 0 else 0.0
        for i, k in enumerate(vocab)
    }

    return {
        "js_divergence": round(js, 6),
        "status": status,
        "train_distribution": train_dist,
        "current_distribution": current_dist,
    }


def _numerical_divergence(
    train_vals: list,
    current_vals: list,
    n_bins: int = 5,
) -> dict:
    """Compute JS divergence for a numerical feature by binning into n_bins buckets.

    Returns a dict with js_divergence, status, bin_edges, train_distribution,
    current_distribution.
    """
    train_clean = [v for v in train_vals if v is not None and not np.isnan(v)]
    current_clean = [v for v in current_vals if v is not None and not np.isnan(v)]

    if not train_clean or not current_clean:
        return {
            "js_divergence": 0.0,
            "status": "pass",
            "bin_edges": [],
            "train_distribution": [],
            "current_distribution": [],
            "note": "insufficient data",
        }

    # Determine bin edges from the combined range so both histograms are comparable
    all_vals = train_clean + current_clean
    min_val = float(np.min(all_vals))
    max_val = float(np.max(all_vals))

    if min_val == max_val:
        # No variance — no drift possible
        return {
            "js_divergence": 0.0,
            "status": "pass",
            "bin_edges": [min_val, max_val],
            "train_distribution": [1.0],
            "current_distribution": [1.0],
        }

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    train_hist, _ = np.histogram(train_clean, bins=bin_edges)
    current_hist, _ = np.histogram(current_clean, bins=bin_edges)

    js = _js_divergence(train_hist.astype(float), current_hist.astype(float))
    status = _status_from_js(js)

    train_total = float(train_hist.sum())
    current_total = float(current_hist.sum())

    return {
        "js_divergence": round(js, 6),
        "status": status,
        "bin_edges": [round(float(e), 4) for e in bin_edges],
        "train_distribution": [
            round(float(v) / train_total, 4) if train_total > 0 else 0.0
            for v in train_hist
        ],
        "current_distribution": [
            round(float(v) / current_total, 4) if current_total > 0 else 0.0
            for v in current_hist
        ],
    }


def compute_drift_report(db) -> dict:
    """
    Compare 2025 prospect pool distribution vs training data distribution.

    Checks distributions of: position, nationality, draft_league_tier,
    points_per_game (binned), age_at_draft (binned).

    Returns:
        {
            "overall_status": "pass" | "warn" | "fail",
            "dimensions": {
                "position": {"js_divergence": 0.04, "status": "pass", ...},
                "nationality": {"js_divergence": 0.12, "status": "warn", ...},
                ...
            },
            "recommendation": "No retraining needed." | "Monitor closely." | "Retrain recommended.",
            "checked_at": "2026-03-29T...",
        }
    """
    from app.models import DraftPickHistorical, Prospect
    from app.constants import infer_league_key

    # ── Fetch training data: draft picks from 2015–2024 ───────────────────────
    training_picks = (
        db.query(DraftPickHistorical)
        .filter(
            DraftPickHistorical.year >= 2015,
            DraftPickHistorical.year <= 2024,
        )
        .all()
    )
    logger.info(f"Drift check: {len(training_picks)} training picks (2015-2024)")

    # ── Fetch current 2025 prospects ──────────────────────────────────────────
    current_prospects = db.query(Prospect).all()
    logger.info(f"Drift check: {len(current_prospects)} current 2025 prospects")

    # ── Extract per-dimension values ──────────────────────────────────────────

    # Position
    train_positions = [p.position for p in training_picks if p.position]
    current_positions = [p.position for p in current_prospects if p.position]

    # Nationality
    train_nationalities = [p.nationality for p in training_picks if p.nationality]
    current_nationalities = [p.nationality for p in current_prospects if p.nationality]

    # Draft league tier — map to named key for interpretability
    train_league_tiers = [
        infer_league_key(p.draft_league or "", p.draft_league_tier)
        for p in training_picks
    ]
    current_league_tiers = [
        infer_league_key(p.draft_league or "", p.draft_league_tier)
        for p in current_prospects
    ]

    # PPG (numerical — skip None/0 goalies optionally, keep all for drift purposes)
    train_ppg = [p.points_per_game for p in training_picks if p.points_per_game is not None]
    current_ppg = [p.points_per_game for p in current_prospects if p.points_per_game is not None]

    # Age at draft (numerical)
    train_age = [p.age_at_draft for p in training_picks if p.age_at_draft is not None]
    current_age = [p.age_at_draft for p in current_prospects if p.age_at_draft is not None]

    # ── Compute divergence per dimension ──────────────────────────────────────
    from app.ml.features import POSITIONS, NAT_GROUPS, LEAGUE_KEYS

    position_result = _categorical_divergence(
        train_positions, current_positions, vocab=POSITIONS
    )
    nationality_result = _categorical_divergence(
        train_nationalities, current_nationalities,
        vocab=NAT_GROUPS + ["other"],
    )
    league_tier_result = _categorical_divergence(
        train_league_tiers, current_league_tiers, vocab=LEAGUE_KEYS
    )
    ppg_result = _numerical_divergence(train_ppg, current_ppg, n_bins=5)
    age_result = _numerical_divergence(train_age, current_age, n_bins=5)

    dimensions = {
        "position":         position_result,
        "nationality":      nationality_result,
        "draft_league_tier": league_tier_result,
        "points_per_game":  ppg_result,
        "age_at_draft":     age_result,
    }

    # ── Aggregate to overall status ───────────────────────────────────────────
    statuses = [d["status"] for d in dimensions.values()]
    if "fail" in statuses:
        overall_status = "fail"
        recommendation = "Retrain recommended."
    elif "warn" in statuses:
        overall_status = "warn"
        recommendation = "Monitor closely."
    else:
        overall_status = "pass"
        recommendation = "No retraining needed."

    # ── Emit Prometheus gauges for Grafana alerting ───────────────────────────
    _emit_drift_metrics(dimensions)

    return {
        "overall_status": overall_status,
        "dimensions":     dimensions,
        "recommendation": recommendation,
        "checked_at":     datetime.now(timezone.utc).isoformat(),
        "train_sample_size":   len(training_picks),
        "current_sample_size": len(current_prospects),
    }


_STATUS_TO_INT = {"pass": 0, "warn": 1, "fail": 2}


def _emit_drift_metrics(dimensions: dict) -> None:
    """Publish per-dimension JS divergence and status to Prometheus."""
    try:
        from app.observability.metrics import DRIFT_JS_DIVERGENCE, DRIFT_STATUS
        for dim, result in dimensions.items():
            js  = result.get("js_divergence", 0.0)
            st  = _STATUS_TO_INT.get(result.get("status", "pass"), 0)
            DRIFT_JS_DIVERGENCE.labels(dimension=dim).set(js)
            DRIFT_STATUS.labels(dimension=dim).set(st)
        logger.info(
            "drift.metrics_emitted dimensions=%s",
            {d: result.get("status") for d, result in dimensions.items()},
        )
    except Exception as exc:
        logger.warning("drift.metrics_emit_failed error=%s", exc)
