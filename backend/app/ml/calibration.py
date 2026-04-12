"""
Calibrated uncertainty via split conformal prediction (inductive conformal prediction).

Conformal prediction gives statistically valid prediction sets rather than raw scores.
For each pick, we ask: "which prospects are in the prediction set at confidence level α?"
A prospect is in the 90% prediction set if its nonconformity score is below the
calibration set's 90th percentile — meaning at least 90% of the time, the actual
pick would have been in the set.

Nonconformity score: 1 - softmax(rank_score)[position_of_positive_in_sorted_list]
Equivalently: how "surprising" was the actual pick given the model's ranking?

Split conformal: calibrate on held-out validation years (2020-2024),
then use stored quantiles at inference time — O(1) cost.

Files:
  calibration.json  — quantiles, n_calibration, computed_at (persisted alongside model.pkl)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_PATH = Path(__file__).parent / "calibration.json"

# Coverage levels to pre-compute (1 - alpha)
ALPHA_LEVELS = [0.10, 0.15, 0.20]   # → 90%, 85%, 80% prediction sets


# ── Calibration (run once after eval-mode training) ───────────────────────────

def calibrate(model, val_df, target_col: str = "was_picked") -> dict:
    """
    Compute conformal prediction quantiles on the validation set.

    For each pick group in val_df, compute:
      nc_score = 1 - p_positive
    where p_positive is the softmax probability assigned to the actual pick.

    Then store the empirical quantiles at each alpha level.

    Returns the calibration dict and also persists it to calibration.json.
    """
    from app.ml.features import build_features, FEATURE_COLS
    from xgboost import XGBRanker

    nc_scores = []

    for _, group in val_df.groupby(["year", "overall_pick"], sort=False):
        pos_mask = group[target_col] == 1
        if pos_mask.sum() == 0:
            continue

        X_group = build_features(group)

        if isinstance(model, XGBRanker):
            raw_scores = model.predict(X_group)
        else:
            raw_scores = model.predict_proba(X_group)[:, 1]

        # Softmax-normalize within the group
        raw_scores = np.array(raw_scores, dtype=np.float64)
        shifted = raw_scores - raw_scores.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()

        # Nonconformity score: 1 - probability of the positive (actual pick)
        pos_prob = probs[pos_mask.values].max()
        nc_scores.append(1.0 - float(pos_prob))

    nc_arr = np.array(nc_scores, dtype=np.float64)
    n = len(nc_arr)

    if n == 0:
        logger.warning("calibration.empty_val_set — skipping calibration")
        return {}

    # Compute quantile thresholds: at alpha=0.10, threshold = 90th percentile nc_score.
    # At inference, a prospect is in the 90% prediction set if its nc_score ≤ threshold.
    quantiles: dict[str, float] = {}
    for alpha in ALPHA_LEVELS:
        level = 1.0 - alpha
        # Conformal correction: use ceil((n+1)*(1-alpha))/n to get finite-sample guarantee
        q_idx = int(np.ceil((n + 1) * level)) - 1
        q_idx = max(0, min(q_idx, n - 1))
        quantiles[str(alpha)] = float(np.sort(nc_arr)[q_idx])

    calibration = {
        "quantiles":     quantiles,
        "n_calibration": n,
        "nc_mean":       float(nc_arr.mean()),
        "nc_std":        float(nc_arr.std()),
        "computed_at":   datetime.now(timezone.utc).isoformat(),
    }

    with open(CALIBRATION_PATH, "w") as f:
        json.dump(calibration, f, indent=2)

    logger.info(
        "calibration.saved n=%d quantiles=%s",
        n, {k: round(v, 4) for k, v in quantiles.items()},
    )
    return calibration


def load_calibration() -> Optional[dict]:
    """Load stored calibration data. Returns None if not available."""
    if not CALIBRATION_PATH.exists():
        return None
    try:
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("calibration.load_failed error=%s", exc)
        return None


# ── Inference: apply calibrated intervals ─────────────────────────────────────

def apply_intervals(
    scores: dict[int, float],
    calibration: dict,
    alpha: float = 0.10,
) -> dict[int, dict]:
    """
    Wrap raw model scores with calibrated prediction set membership.

    For each prospect, compute its nonconformity score within the current pool
    and check whether it falls below the calibrated quantile threshold.

    A prospect is in the (1-alpha)*100% prediction set if:
      nc_score ≤ quantile[alpha]

    Returns:
      {
        prospect_id: {
          "score":              float,   # raw model score
          "nc_score":           float,   # nonconformity score (lower = more likely)
          "in_prediction_set":  bool,    # within calibrated confidence set
          "coverage":           float,   # e.g. 0.90 for alpha=0.10
        },
        ...
      }
    """
    if not scores or not calibration:
        return {pid: {"score": s, "nc_score": None, "in_prediction_set": None, "coverage": None}
                for pid, s in scores.items()}

    raw = np.array(list(scores.values()), dtype=np.float64)
    pids = list(scores.keys())

    # Softmax-normalize the pool
    shifted = raw - raw.max()
    probs   = np.exp(shifted) / np.exp(shifted).sum()

    threshold = calibration.get("quantiles", {}).get(str(alpha))
    coverage  = 1.0 - alpha

    result = {}
    for pid, prob in zip(pids, probs):
        nc = 1.0 - float(prob)
        in_set = bool(nc <= threshold) if threshold is not None else None
        result[pid] = {
            "score":             scores[pid],
            "nc_score":          round(nc, 6),
            "in_prediction_set": in_set,
            "coverage":          coverage,
        }

    return result
