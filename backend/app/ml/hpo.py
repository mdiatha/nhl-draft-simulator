"""
Hyperparameter optimization for the XGBoost LambdaMART ranker using Optuna.

Problem:
  The current hyperparameters (learning_rate=0.05, max_depth=5, etc.) were
  manually tuned. Manual tuning misses interactions between parameters and
  is not reproducible. Optuna runs a principled search using Tree-structured
  Parzen Estimator (TPE) — a Bayesian approach that uses previous trial results
  to focus the search toward promising regions of parameter space.

Why NDCG@1 as the objective:
  AUC measures overall discrimination but not ranking quality. NDCG@1
  ("did the model rank the actual pick first?") directly measures what
  matters in production. Optimizing AUC can give a model that correctly
  ranks 70% of cases but consistently misranks round-1 picks, which is
  the highest-value part of the prediction.

Time-series-aware cross-validation:
  Standard k-fold would leak future draft data into training (e.g., fold-3
  trains on 2022 data but fold-1 test includes 2021 data). We use
  TimeSeriesSplit with year-based splits to ensure the model never sees
  future draft history during HPO — same discipline as the main train/val split.

Usage:
  # Tune and print best params (takes 20–60 min depending on n_trials)
  python -m app.ml.hpo

  # Or via API (admin only):
  POST /api/ml/hpo?n_trials=50

Result:
  best_params.json written to ml/ directory.
  Copy the best params into train.py's rank_params dict.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BEST_PARAMS_PATH = Path(__file__).parent / "best_params.json"

# Optuna objective direction
_DIRECTION = "maximize"   # maximise NDCG@1

# Default trial budget (override via n_trials arg)
DEFAULT_N_TRIALS = 50


def _ndcg_at_1(model, X_val: np.ndarray, y_val: np.ndarray, groups_val: np.ndarray) -> float:
    """
    Compute NDCG@1: for each pick group, did the model rank the actual pick first?

    NDCG@1 = fraction of groups where the positive example is ranked #1.
    This is equivalent to top-1 accuracy in a ranking context.
    """
    scores = model.predict(X_val)
    offset = 0
    hits = 0
    total = 0

    for g_size in groups_val:
        g_size = int(g_size)
        g_scores = scores[offset: offset + g_size]
        g_labels = y_val[offset: offset + g_size]
        offset += g_size

        pos_indices = np.where(g_labels == 1)[0]
        if len(pos_indices) == 0:
            continue

        # Rank within group (highest score = rank 0)
        ranked_indices = np.argsort(-g_scores)
        rank_of_positive = int(np.where(ranked_indices == pos_indices[0])[0][0])
        if rank_of_positive == 0:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


def optimize(db, n_trials: int = DEFAULT_N_TRIALS) -> dict:
    """
    Run Optuna HPO on the XGBoost LambdaMART ranker.

    Returns the best hyperparameter dict and NDCG@1 score.
    Also saves results to best_params.json.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError(
            "Optuna not installed. Add it to requirements: pip install optuna"
        )

    from xgboost import XGBRanker
    from app.ml.features import build_features, build_training_dataset, FEATURE_COLS

    logger.info("hpo.start n_trials=%d", n_trials)

    df = build_training_dataset(db)
    X  = build_features(df)
    y  = df["was_picked"].astype(int).values

    # Time-series split: fold boundaries at year boundaries
    if "year" in df.columns:
        years = sorted(df["year"].unique())
        # Use the last 4 years as fold boundaries
        split_years = years[-4:] if len(years) >= 4 else years
    else:
        split_years = []

    def _make_year_split(split_year: int):
        """Return (train_mask, val_mask) for a temporal fold."""
        train_mask = df["year"] < split_year
        val_mask   = df["year"] == split_year
        return train_mask, val_mask

    def _group_sizes(mask: pd.Series) -> np.ndarray:
        """Compute XGBoost group sizes from the year/pick grouping."""
        sub = df[mask]
        sizes = sub.groupby(["year", "overall_pick"], sort=True).size().values
        return sizes.astype(int)

    def objective(trial) -> float:
        params = {
            "objective":    "rank:pairwise",
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":    trial.suggest_int("max_depth", 3, 7),
            "subsample":    trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":    trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":   trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "tree_method":  "hist",
            "verbosity":    0,
        }

        fold_scores = []

        for split_year in split_years:
            train_mask, val_mask = _make_year_split(split_year)
            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue

            X_train = X[train_mask.values]
            X_val   = X[val_mask.values]
            y_train = y[train_mask.values]
            y_val_f = y[val_mask.values]
            g_train = _group_sizes(train_mask)
            g_val   = _group_sizes(val_mask)

            model = XGBRanker(**params)
            model.fit(X_train, y_train, group=g_train, verbose=False)

            score = _ndcg_at_1(model, X_val, y_val_f, g_val)
            fold_scores.append(score)

        if not fold_scores:
            return 0.0

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction=_DIRECTION)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    result = {
        "best_ndcg_at_1":  round(best.value, 4),
        "best_params":     best.params,
        "n_trials":        n_trials,
        "n_folds":         len(split_years),
        "optimized_at":    datetime.now(timezone.utc).isoformat(),
    }

    BEST_PARAMS_PATH.write_text(json.dumps(result, indent=2))

    logger.info(
        "hpo.complete ndcg@1=%.4f top_params=%s",
        best.value, best.params,
    )
    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        result = optimize(db, n_trials=int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_TRIALS)
        print(json.dumps(result, indent=2))
    finally:
        db.close()
