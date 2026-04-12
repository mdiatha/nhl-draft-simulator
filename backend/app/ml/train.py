"""Train an XGBoost model to predict which prospect a team will draft.

Architecture: Learning-to-rank (LambdaMART via XGBoost rank:pairwise).

Each draft pick becomes a query group: one positive row (the actual pick) and
N negative rows (prospects available at that slot but passed over). The ranker
directly optimizes "the picked prospect should score above all others" — which
is the actual decision GMs make.

This is superior to binary classification (was_picked=1/0) because:
  - It learns relative ordering, not absolute probabilities
  - The model sees all alternatives simultaneously per pick slot
  - Avoids the false assumption that each row is an independent event

Training data: historical picks (2008-2024, CSS era only) with recency weighting.

Two training modes:

  Evaluation mode (final=False, default):
    Temporal split — train on years ≤ TRAIN_CUTOFF_YEAR, validate on ≥ VAL_START_YEAR.
    Metric: NDCG@1 (did the model rank the actual pick first?).

  Production mode (final=True):
    Train on ALL data. n_estimators fixed from eval mode best_iteration.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBRanker

from app.ml.features import FEATURE_COLS, build_features, build_training_dataset

logger = logging.getLogger(__name__)

MODEL_PATH        = Path(__file__).parent / "model.pkl"
TRAIN_HISTORY_PATH = Path(__file__).parent / "train_history.jsonl"   # one JSON record per line

TRAIN_CUTOFF_YEAR    = 2019   # inclusive upper bound for training split
VAL_START_YEAR       = 2020   # inclusive lower bound for validation split
FINAL_N_ESTIMATORS   = 200    # reset after feature set change (33 features); update after eval run


def train(
    df: pd.DataFrame,
    target_col: str = "was_picked",
    final: bool = False,
) -> tuple[XGBRanker, float]:
    """
    Train on df, persist model.pkl, return (model, val_auc).

    final=False (default — evaluation mode):
      Temporal split: train ≤ TRAIN_CUTOFF_YEAR, val ≥ VAL_START_YEAR.
      Metric: NDCG@1 (did the model rank the actual pick #1 in its group?).

    final=True (production mode):
      Train on ALL rows. n_estimators fixed at FINAL_N_ESTIMATORS.
      Returns NaN as metric. Use when ready to predict 2025.
    """
    # ── Ranker params (tuned for rank:pairwise task) ──────────────────────────
    # learning_rate=0.05 (up from 0.01): with 92k rows the model needs more
    # gradient magnitude to converge before early stopping fires. At 0.01 the
    # first iteration already overshoots the validation optimum because the
    # gradient steps are too small relative to the pairwise loss landscape.
    # max_depth=5 (down from 6): deeper trees overfit with this many negatives;
    # round 1 signal gets diluted by the round 6-7 noise.
    # early_stopping_rounds=25: tighter window avoids false early stops on
    # noisy NDCG@1 swings in the first 20 iterations.
    rank_params = dict(
        objective="rank:ndcg",   # aligns training objective with ndcg@1 eval metric
        n_estimators=FINAL_N_ESTIMATORS if final else 1500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.75,
        min_child_weight=5,
        gamma=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="ndcg@1",
    )

    def _make_groups(split_df: pd.DataFrame) -> np.ndarray:
        """
        Build the XGBoost group array: number of rows per query group.
        Each (year, overall_pick) is one group — 1 positive + N negatives.
        Groups must be contiguous, which they are since df is built in pick order.
        """
        return split_df.groupby(["year", "overall_pick"], sort=False).size().values

    def _group_weights(split_df: pd.DataFrame) -> np.ndarray | None:
        """Per-row sample weights expanded from per-group weights.

        XGBRanker.fit() expects one weight per row, not per group. We derive
        the group weight from the positive row (recency signal lives there), then
        broadcast it to every row in that group so the tensor shapes match.
        """
        if "sample_weight" not in split_df.columns:
            return None
        # Map each (year, overall_pick) group → weight from its positive row
        pos_rows = split_df[split_df[target_col] == 1].copy()
        group_w = (
            pos_rows.groupby(["year", "overall_pick"], sort=False)["sample_weight"]
            .first()
        )
        # Merge back so every row inherits its group's weight
        merged = split_df[["year", "overall_pick"]].merge(
            group_w.rename("_gw"), on=["year", "overall_pick"], how="left"
        )
        return merged["_gw"].fillna(1.0).values

    if final:
        logger.info("Production mode — training ranker on ALL %d rows", len(df))
        X = build_features(df)
        y = df[target_col].astype(int)
        groups = _make_groups(df)
        gw = _group_weights(df)

        model = XGBRanker(**rank_params)
        model.fit(X, y, group=groups, sample_weight=gw, verbose=False)

        model.save_model(MODEL_PATH)
        logger.info("Ranker saved → %s  (n_estimators=%d)", MODEL_PATH, FINAL_N_ESTIMATORS)

        return model, float("nan")

    # ── Evaluation mode: temporal split ──────────────────────────────────────
    if "year" in df.columns:
        train_df = df[df["year"] <= TRAIN_CUTOFF_YEAR]
        val_df   = df[df["year"] >= VAL_START_YEAR]
        if len(val_df) == 0 or val_df[target_col].nunique() < 2:
            logger.warning("Temporal val split empty — falling back to random 80/20 split")
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    else:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])

    X_train  = build_features(train_df)
    X_val    = build_features(val_df)
    y_train  = train_df[target_col].astype(int)
    y_val    = val_df[target_col].astype(int)
    g_train  = _make_groups(train_df)
    g_val    = _make_groups(val_df)
    w_train  = _group_weights(train_df)

    logger.info(
        "Temporal split — train: %d rows (%d groups), val: %d rows (%d groups)",
        len(X_train), len(g_train), len(X_val), len(g_val),
    )

    eval_model = XGBRanker(
        **{**rank_params, "n_estimators": 1500, "early_stopping_rounds": 50,
           "eval_metric": "ndcg@1-"},
    )
    eval_model.fit(
        X_train, y_train,
        group=g_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_group=[g_val],
        verbose=False,
    )

    best_iter = max(eval_model.best_iteration, 1)

    # Compute NDCG@1: fraction of groups where the positive was ranked first
    scores_val = eval_model.predict(X_val)
    ndcg1 = _compute_ndcg1(val_df, scores_val, target_col)
    # Use NDCG@1 as AUC proxy so the quality gate works (gate threshold is 0.60,
    # but NDCG@1 is a harder metric — update threshold in data_quality.py accordingly)
    auc = ndcg1

    logger.info(
        "Validation NDCG@1: %.4f  AUC: %.4f  (n_train=%d, n_val=%d, best_iter=%d)",
        ndcg1, auc, len(X_train), len(X_val), best_iter,
    )
    logger.info("Tip: FINAL_N_ESTIMATORS≈%d", best_iter)

    eval_model.save_model(MODEL_PATH)
    logger.info("Ranker saved → %s", MODEL_PATH)

    return eval_model, auc


def _compute_ndcg1(df: pd.DataFrame, scores: np.ndarray, target_col: str = "was_picked") -> float:
    """
    NDCG@1: fraction of pick groups where the model ranked the actual pick first.
    Each (year, overall_pick) is one group. Returns a value in [0, 1].
    """
    df = df.copy()
    df["_score"] = scores
    hits = 0
    total = 0
    for _, group in df.groupby(["year", "overall_pick"], sort=False):
        if group[target_col].sum() == 0:
            continue
        top_id = group["_score"].idxmax()
        if group.loc[top_id, target_col] == 1:
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def train_from_db(db, final: bool = False) -> dict:
    """Load data, train, persist artifact + metadata. Returns metrics dict."""
    from app.ml.registry import registry
    from app.observability.data_quality import run_training_checks, run_model_quality_checks

    df = build_training_dataset(db)

    # Data quality gate — raises ValueError on FAIL so the API returns 422
    training_report = run_training_checks(df)
    if not training_report.passed:
        raise ValueError(
            f"Training data quality checks failed: "
            + ", ".join(c.name for c in training_report.failed_checks)
        )

    model, auc = train(df, final=final)

    y = df["was_picked"].astype(int)
    feature_importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

    # Post-training quality gate (skipped in final/production mode — no val AUC)
    if not final:
        model_report = run_model_quality_checks(auc, feature_importances)
        if not model_report.passed:
            raise ValueError(
                f"Model quality checks failed: "
                + ", ".join(c.name for c in model_report.failed_checks)
            )

    # Calibrate uncertainty on the validation split (eval mode only)
    calibration_meta = {}
    if not final:
        try:
            from app.ml.calibration import calibrate
            val_df = df[df["year"] >= VAL_START_YEAR]
            if len(val_df) > 0:
                cal = calibrate(model, val_df)
                calibration_meta = {
                    "calibration_n":        cal.get("n_calibration"),
                    "calibration_quantiles": cal.get("quantiles"),
                }
                logger.info(
                    "calibration.done n=%d quantiles=%s",
                    cal.get("n_calibration", 0), cal.get("quantiles", {}),
                )
        except Exception as exc:
            logger.warning("calibration.failed error=%s", exc)

    metrics = {
        "training_samples":    len(df),
        "positive_samples":    int(y.sum()),
        "negative_samples":    int((y == 0).sum()),
        "validation_auc":      None if final else round(auc, 4),
        "mode":                "production" if final else "evaluation",
        "feature_importances": feature_importances,
        "model_path":          str(MODEL_PATH),
        **calibration_meta,
    }

    registry.save_meta(metrics)
    registry.reload()

    # Append to training run history for longitudinal comparison
    _append_train_history(metrics)

    # Track feature importance drift across training runs
    trained_at = metrics.get("trained_at", datetime.now(timezone.utc).isoformat())
    try:
        from app.ml.importance_tracker import compute_and_save_importances
        compute_and_save_importances(model, FEATURE_COLS, trained_at)
        logger.info("importance_tracker.saved trained_at=%s", trained_at)
    except Exception as exc:
        logger.warning("importance_tracker.failed error=%s", exc)

    return metrics


def _append_train_history(metrics: dict) -> None:
    """
    Append a single-line JSON record to train_history.jsonl.

    Each record contains: trained_at, mode, validation_auc, training_samples,
    top_5_features (name + importance). This lets us track model quality over
    time and detect regressions without needing MLflow or a DB table.
    """
    fi = metrics.get("feature_importances", {})
    top5 = sorted(fi.items(), key=lambda x: -x[1])[:5] if fi else []

    record = {
        "trained_at":        datetime.now(timezone.utc).isoformat(),
        "mode":              metrics.get("mode", "evaluation"),
        "validation_auc":    metrics.get("validation_auc"),
        "training_samples":  metrics.get("training_samples"),
        "top_features":      [{"feature": k, "importance": round(v, 6)} for k, v in top5],
        "calibration_n":     metrics.get("calibration_n"),
    }
    try:
        with open(TRAIN_HISTORY_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("train_history.appended trained_at=%s auc=%s",
                    record["trained_at"], record["validation_auc"])
    except Exception as exc:
        logger.warning("train_history.write_failed error=%s", exc)


def model_exists() -> bool:
    return MODEL_PATH.exists()
