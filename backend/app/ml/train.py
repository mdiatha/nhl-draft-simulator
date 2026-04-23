"""Train an XGBoost model to predict which prospect a team will draft.

Architecture: Learning-to-rank (LambdaMART via XGBoost rank:pairwise).

Each draft pick becomes a query group: one positive row (the actual pick) and
N negative rows (prospects available at that slot but passed over). The ranker
directly optimizes "the picked prospect should score above all others" — which
is the actual decision GMs make.

rank:pairwise is used over rank:ndcg because our task is pick-level prediction
(which single player gets chosen), not full-list ranking. Pairwise loss directly
optimizes the comparison between the positive and each negative, which exactly
matches the training data structure (1 positive vs 31 negatives per group).

Training data: historical picks (2008-2024, CSS era only) with recency weighting.

Two phases every run:
  Phase 1 (eval): walk-forward cross-validation across 4 folds to get a reliable
    NDCG@1 estimate, then a final temporal split to find best_iteration.
  Phase 2 (production): retrain on ALL data using best_iteration from phase 1.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from app.ml.features import FEATURE_COLS, NEGATIVE_WINDOW, build_features, build_training_dataset

logger = logging.getLogger(__name__)

from app.ml.registry import MODEL_PATH, MODEL_DIR  # noqa: E402  single source of truth for model location
TRAIN_HISTORY_PATH = Path(__file__).parent / "train_history.jsonl"   # one JSON record per line

TRAIN_CUTOFF_YEAR    = 2022   # inclusive upper bound for final temporal split
VAL_START_YEAR       = 2023   # inclusive lower bound for final temporal split + calibration
# FINAL_N_ESTIMATORS is no longer a hardcoded constant — it is derived
# automatically from eval-mode best_iteration each time train() is called
# with final=False, then used immediately for the production-mode retrain.
# The last derived value is written to best_params.json for reference.


def train(
    df: pd.DataFrame,
    target_col: str = "was_picked",
    final: bool = False,
) -> tuple[XGBRanker, float]:
    """
    Train on df, persist model.pkl, return (model, ndcg1).

    Always runs in two phases:
      Phase 1 (eval): temporal split (train ≤ TRAIN_CUTOFF_YEAR, val ≥
        VAL_START_YEAR), early stopping up to 1500 trees, records
        best_iteration and NDCG@1.
      Phase 2 (production): retrain on ALL data using best_iteration from
        phase 1 — no early stopping, no held-out set.

    final=False (default): return after phase 2 with the eval NDCG@1.
    final=True: same behaviour — the flag is kept for API backwards
      compatibility but no longer skips eval.
    """
    # ── Ranker base params ────────────────────────────────────────────────────
    # rank:pairwise (LambdaMART) directly optimizes pairwise comparisons between
    # the positive and each negative, matching our data construction exactly.
    # scale_pos_weight corrects the 1:31 class imbalance — without it the gradient
    # signal from the single positive is drowned out by 31 negatives per group.
    n_neg = NEGATIVE_WINDOW  # 31 negatives per positive
    base_params = dict(
        objective="rank:pairwise",
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.75,
        min_child_weight=5,
        gamma=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=n_neg,
        random_state=42,
        eval_metric="auc",
    )
    def _make_groups(split_df: pd.DataFrame) -> np.ndarray:
        """
        Build the XGBoost group array: number of rows per query group.
        Each (year, overall_pick) is one group — 1 positive + N negatives.
        Groups must be contiguous, which they are since df is built in pick order.
        """
        return split_df.groupby(["year", "overall_pick"], sort=False).size().values

    def _group_weights(split_df: pd.DataFrame) -> np.ndarray | None:
        """Per-group sample weights for XGBRanker.

        XGBoost ranking API expects one weight per query group (not per row).
        We derive the group weight from the positive row's sample_weight
        (recency + round signal lives there), one value per (year, overall_pick).
        """
        if "sample_weight" not in split_df.columns:
            return None
        pos_rows = split_df[split_df[target_col] == 1].copy()
        group_w = (
            pos_rows.groupby(["year", "overall_pick"], sort=False)["sample_weight"]
            .first()
        )
        return group_w.fillna(1.0).values

    # ── Phase 1: walk-forward cross-validation ───────────────────────────────
    # Each fold trains on all years up to the fold cutoff and validates on the
    # next year. This gives a much more reliable NDCG@1 estimate than a single
    # temporal split because it averages across multiple draft years instead of
    # depending on one potentially unusual cohort (e.g. COVID draft 2020).
    #
    # 9 folds starting from 2014 use the full CSS era (2008-2024) effectively:
    #   2014→2015, 2015→2016, 2016→2017, 2017→2018, 2018→2019,
    #   2019→2020, 2020→2021, 2021→2022, 2022→2023
    #
    # Extending back to 2014 (vs. the old 2017 start) adds 3 extra folds and
    # reduces variance in the CV mean by ~25%. Stopping at 2023 leaves 2024
    # clean for the final temporal split used to find best_iteration.
    _group_keys = [c for c in ["year", "overall_pick"] if c in df.columns]
    CV_FOLDS = [
        (2014, 2015),
        (2015, 2016),
        (2016, 2017),
        (2017, 2018),
        (2018, 2019),
        (2019, 2020),
        (2020, 2021),
        (2021, 2022),
        (2022, 2023),
    ]

    fold_ndcg1s: list[float] = []
    if "year" in df.columns:
        for cutoff, val_year in CV_FOLDS:
            fold_train = df[df["year"] <= cutoff].sort_values(_group_keys).reset_index(drop=True)
            fold_val   = df[df["year"] == val_year].sort_values(_group_keys).reset_index(drop=True)
            if len(fold_val) == 0 or fold_val[target_col].nunique() < 2:
                logger.warning("CV fold val_year=%d empty — skipping", val_year)
                continue

            Xtr = build_features(fold_train)
            Xvl = build_features(fold_val)
            ytr = fold_train[target_col].astype(int)
            gtr = _make_groups(fold_train)
            wtr = _group_weights(fold_train)

            fold_model = XGBRanker(**{**base_params, "n_estimators": 500})
            fold_model.fit(Xtr, ytr, group=gtr, sample_weight=wtr, verbose=False)
            fold_scores = fold_model.predict(Xvl)
            fold_ndcg1  = _compute_ndcg1(fold_val, fold_scores, target_col)
            fold_ndcg1s.append(fold_ndcg1)
            logger.info("CV fold cutoff=%d val=%d  NDCG@1=%.4f", cutoff, val_year, fold_ndcg1)

    if not fold_ndcg1s:
        raise ValueError(
            "Walk-forward CV produced no valid folds — training data may not cover "
            "the expected year range (2018–2021). Check CSS_ERA_START and data ingestion."
        )
    cv_ndcg1 = float(np.mean(fold_ndcg1s))
    cv_std    = float(np.std(fold_ndcg1s))
    logger.info("Walk-forward CV (9-fold) — mean NDCG@1=%.4f  std=%.4f  folds=%d",
                cv_ndcg1, cv_std, len(fold_ndcg1s))

    # ── Final temporal split — find best_iteration for production retrain ────
    # Use the full TRAIN_CUTOFF_YEAR / VAL_START_YEAR split (more training data
    # than any single CV fold) so early stopping has the best signal.
    if "year" in df.columns:
        train_df = df[df["year"] <= TRAIN_CUTOFF_YEAR]
        val_df   = df[df["year"] >= VAL_START_YEAR]
        if len(val_df) == 0 or val_df[target_col].nunique() < 2:
            logger.warning("Temporal val split empty — falling back to random 80/20 split")
            from sklearn.model_selection import train_test_split
            unique_groups = df[_group_keys].drop_duplicates()
            train_groups, val_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)
            train_df = df.merge(train_groups, on=_group_keys)
            val_df   = df.merge(val_groups,   on=_group_keys)
    else:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df = train_df.sort_values(_group_keys).reset_index(drop=True)
    val_df   = val_df.sort_values(_group_keys).reset_index(drop=True)

    X_train = build_features(train_df)
    X_val   = build_features(val_df)
    y_train = train_df[target_col].astype(int)
    y_val   = val_df[target_col].astype(int)
    g_train = _make_groups(train_df)
    g_val   = _make_groups(val_df)
    w_train = _group_weights(train_df)

    logger.info(
        "Final split — train: %d rows (%d groups), val: %d rows (%d groups)",
        len(X_train), len(g_train), len(X_val), len(g_val),
    )

    eval_model = XGBRanker(
        **{**base_params, "n_estimators": 1500, "early_stopping_rounds": 50},
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

    scores_val = eval_model.predict(X_val)
    ndcg1 = _compute_ndcg1(val_df, scores_val, target_col)
    # Report CV mean as the headline metric — more reliable than single-split
    auc = cv_ndcg1 if fold_ndcg1s else ndcg1

    logger.info(
        "Final split NDCG@1: %.4f  CV mean: %.4f  best_iter=%d",
        ndcg1, cv_ndcg1, best_iter,
    )

    # ── Phase 2: production — retrain on ALL data using best_iter ────────────
    logger.info("Phase 2 (production) — retraining on ALL %d rows, n_estimators=%d", len(df), best_iter)

    X_all  = build_features(df)
    y_all  = df[target_col].astype(int)
    g_all  = _make_groups(df)
    gw_all = _group_weights(df)

    prod_model = XGBRanker(**{**base_params, "n_estimators": best_iter})
    prod_model.fit(X_all, y_all, group=g_all, sample_weight=gw_all, verbose=False)

    prod_model.save_model(MODEL_PATH)
    logger.info("Ranker saved → %s  (n_estimators=%d, ndcg1=%.4f)", MODEL_PATH, best_iter, ndcg1)

    # Persist best_iter so HPO and diagnostics can reference it
    _save_best_iter(best_iter, ndcg1)

    return prod_model, auc


def _save_best_iter(best_iter: int, ndcg1: float) -> None:
    """Persist best_iteration to best_params.json for HPO reference."""
    import json
    path = Path(__file__).parent / "best_params.json"
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    existing["best_iteration"] = best_iter
    existing["last_eval_ndcg1"] = round(ndcg1, 4)
    existing["last_eval_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(existing, indent=2))


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
            "Training data quality checks failed: "
            + ", ".join(c.name for c in training_report.failed_checks)
        )

    model, auc = train(df, final=final)

    # Sanity-check: auc must be a real number. A null/NaN here means the eval
    # phase silently failed — catch it before it propagates to model_meta.json.
    if auc is None or (isinstance(auc, float) and (auc != auc)):  # NaN check
        raise ValueError(
            "Training returned null validation_auc — eval phase likely failed. "
            "Check CV fold logs for empty val sets."
        )

    y = df["was_picked"].astype(int)
    feature_importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

    # Post-training quality gate — always runs now since we always have a val AUC
    model_report = run_model_quality_checks(auc, feature_importances)
    if not model_report.passed:
        raise ValueError(
            "Model quality checks failed: "
            + ", ".join(c.name for c in model_report.failed_checks)
        )

    # Calibrate on the validation split — required step, not optional.
    # A retrain without fresh calibration leaves stale quantiles in calibration.json,
    # making conformal prediction set memberships invalid for the new model.
    from app.ml.calibration import calibrate
    val_df = df[df["year"] >= VAL_START_YEAR]
    if len(val_df) == 0:
        raise ValueError(
            f"No validation rows (year >= {VAL_START_YEAR}) available for calibration. "
            "Extend training data or lower VAL_START_YEAR."
        )
    cal = calibrate(model, val_df)
    if not cal:
        raise ValueError(
            "Calibration returned empty result — val set may have no complete pick groups."
        )
    calibration_meta = {
        "calibration_n":         cal.get("n_calibration"),
        "calibration_quantiles": cal.get("quantiles"),
    }
    logger.info(
        "calibration.done n=%d quantiles=%s",
        cal.get("n_calibration", 0), cal.get("quantiles", {}),
    )

    metrics = {
        "training_samples":    len(df),
        "positive_samples":    int(y.sum()),
        "negative_samples":    int((y == 0).sum()),
        "validation_auc":      round(auc, 4),
        "mode":                "production",
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
