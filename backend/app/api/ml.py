"""ML endpoints — training, status, hot-reload, SHAP explanations."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.database import get_db, get_redis
from app.middleware.auth import require_admin_key
from app.ml.registry import registry

logger = logging.getLogger(__name__)

# Training/admin routes are protected; read-only status routes are public
router = APIRouter(prefix="/ml", tags=["ml"])

# Strict per-endpoint limiter for expensive CPU-bound operations
_limiter = Limiter(key_func=get_remote_address)

_redis, _REDIS_OK = get_redis()

# ── Distributed training lock ─────────────────────────────────────────────────
# threading.Lock() is per-process and does NOT prevent concurrent training on
# multiple ECS tasks or instances. We use Redis SETNX as the cross-instance
# guard, with the thread-local lock as a secondary in-process guard.
_TRAINING_LOCK_KEY = "nhl:training_lock"
_TRAINING_LOCK_TTL = 900  # seconds — generous upper-bound for a 14-min Lambda


def _acquire_training_lock() -> bool:
    """Return True if this process acquired the distributed training lock."""
    if _REDIS_OK and _redis:
        # nx=True → only set if key does not exist (atomic SETNX)
        return bool(_redis.set(_TRAINING_LOCK_KEY, 1, nx=True, ex=_TRAINING_LOCK_TTL))
    # Redis unavailable — fall back to the thread-local lock (single-instance protection)
    return registry.training_lock.acquire(blocking=False)


def _release_training_lock() -> None:
    if _REDIS_OK and _redis:
        _redis.delete(_TRAINING_LOCK_KEY)
    else:
        try:
            registry.training_lock.release()
        except RuntimeError:
            pass  # lock was never acquired (e.g. exception before acquire)


def _flush_draft_cache() -> int:
    """Delete all cached draft simulation results. Returns number of keys removed."""
    if not _REDIS_OK or not _redis:
        return 0
    keys = _redis.keys("draft_v*")
    if keys:
        return _redis.delete(*keys)
    return 0


@router.post("/train", dependencies=[Depends(require_admin_key)])
async def train_model(final: bool = False, db: Session = Depends(get_db)):
    """
    Train the XGBoost draft-pick prediction model on historical data.

    final=false (default — evaluation mode):
      Temporal split: train ≤ 2019, validate 2020–2024.
      Returns honest out-of-sample AUC. Use for tuning and validation.

    final=true (production mode):
      Train on ALL historical data — no held-out val set, no early stopping.
      Use once you're satisfied with the model architecture.
      Update FINAL_N_ESTIMATORS in train.py to match best_iteration from eval runs.

    Saves model.pkl + model_meta.json to disk, then hot-swaps the in-memory registry.
    """
    import time as _time
    from app.ml.train import train_from_db
    from app.observability.metrics import (
        MODEL_TRAINING_DURATION, MODEL_VALIDATION_AUC,
        MODEL_TRAINING_SAMPLES, MODEL_INFO,
    )

    if not _acquire_training_lock():
        raise HTTPException(status_code=409, detail="A training run is already in progress.")

    t0 = _time.perf_counter()
    try:
        metrics = train_from_db(db, final=final)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail="Training failed due to an internal error. Check server logs.")
    finally:
        _release_training_lock()

    duration = _time.perf_counter() - t0
    MODEL_TRAINING_DURATION.observe(duration)
    MODEL_TRAINING_SAMPLES.set(metrics.get("training_samples", 0))
    if metrics.get("validation_auc") is not None:
        MODEL_VALIDATION_AUC.set(metrics["validation_auc"])
    MODEL_INFO.info({
        "mode":           metrics.get("mode", "evaluation"),
        "validation_auc": str(metrics.get("validation_auc", "N/A")),
        "training_samples": str(metrics.get("training_samples", 0)),
    })

    # Notify other instances to reload — they poll nhl:model:version every 30 s
    if _REDIS_OK and _redis:
        _redis.set("nhl:model:version", metrics.get("trained_at", ""))

    flushed = _flush_draft_cache()
    logger.info("Redis draft cache flushed: %d keys removed", flushed)

    return {"status": "ok", "metrics": metrics, "cache_keys_flushed": flushed}


@router.post("/reload", dependencies=[Depends(require_admin_key)])
async def reload_model():
    """
    Hot-swap: reload model.pkl from disk into the in-memory registry.

    Use this after running the standalone training script
    (python -m app.ml.train_model) to inject the new model without
    restarting the server.
    """
    success = registry.reload()
    if not success:
        raise HTTPException(
            status_code=404,
            detail="No model.pkl found. Run POST /api/ml/train or python -m app.ml.train_model first.",
        )
    return {"status": "ok", "meta": registry.meta}



@router.get("/status")
async def ml_status():
    """Return model registry status and training metadata."""
    return {
        "model_loaded":   registry.is_loaded,
        "is_calibrated":  registry.calibration is not None,
        "meta":           registry.meta if registry.is_loaded else None,
    }


@router.get("/versions")
async def list_model_versions():
    """List all versioned models stored in S3, newest first."""
    from app.aws.s3 import list_model_versions
    versions = list_model_versions()
    return {"versions": versions, "count": len(versions)}


@router.post("/rollback/{version:path}", dependencies=[Depends(require_admin_key)])
async def rollback_model(version: str):
    """
    Roll back to a previously trained model version stored in S3.

    version: the trained_at timestamp (e.g. 2026-03-28T12:00:00+00:00).
    Copies the versioned model back to models/latest/ then hot-reloads it.
    Flushes the Redis simulation cache so users immediately get results
    from the rolled-back model.
    """
    from app.aws.s3 import rollback_model as s3_rollback
    import os
    from app.ml.registry import MODEL_PATH, META_PATH

    success = s3_rollback(version)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Version '{version}' not found in S3, or S3 not configured.",
        )

    # Remove local cache so reload() re-downloads the rolled-back version
    for path in (MODEL_PATH, META_PATH):
        if path.exists():
            path.unlink()

    reloaded = registry.reload()
    if not reloaded:
        raise HTTPException(status_code=500, detail="Rollback uploaded but model failed to reload.")

    flushed = _flush_draft_cache()
    return {"status": "ok", "rolled_back_to": version, "cache_keys_flushed": flushed}


@router.post("/backtest")
async def backtest_model(
    test_year: int = 2024,
    train_cutoff: int = 2023,
    db: Session = Depends(get_db),
):
    """
    Train on picks ≤ train_cutoff, evaluate on test_year.

    Returns top-1/3/5 accuracy, MRR, per-round breakdown, worst misses.
    """
    _YEAR_LO, _YEAR_HI = 2000, 2025
    if not (_YEAR_LO <= test_year <= _YEAR_HI):
        raise HTTPException(status_code=422, detail=f"test_year must be between {_YEAR_LO} and {_YEAR_HI}")
    if not (_YEAR_LO <= train_cutoff <= _YEAR_HI):
        raise HTTPException(status_code=422, detail=f"train_cutoff must be between {_YEAR_LO} and {_YEAR_HI}")
    if train_cutoff >= test_year:
        raise HTTPException(status_code=422, detail="train_cutoff must be strictly less than test_year")

    from app.ml.backtest import run_backtest
    try:
        metrics = run_backtest(db, test_year=test_year, train_cutoff=train_cutoff)
        return {"status": "ok", "metrics": metrics}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=500, detail=f"Backtest error: {e}")


@router.post("/backtest/multi-year")
async def backtest_multi_year(
    years: list[int] | None = None,
    db: Session = Depends(get_db),
):
    """
    Run backtests across multiple held-out years and return mean ± std CI.

    Defaults to years 2021–2024. Each year uses all prior years as training data.
    Returns per-year breakdowns plus aggregate statistics with min/max/std so
    point estimates (top1=0.45) become confidence intervals (0.43 ± 0.06).

    This endpoint takes ~4× longer than a single backtest (one train per year).
    """
    from app.ml.backtest import run_multi_year_backtest, BACKTEST_YEARS
    try:
        result = run_multi_year_backtest(db, years=years or BACKTEST_YEARS)
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.exception("Multi-year backtest failed")
        raise HTTPException(status_code=500, detail=f"Multi-year backtest error: {e}")


@router.get("/explain/{prospect_id}")
@_limiter.limit("10/minute")
async def explain_pick(
    request: Request,
    prospect_id: int,
    team_id: int,
    pick_number: int = 1,
    db: Session = Depends(get_db),
):
    """
    Return SHAP feature attributions explaining why the model would (or would not)
    select a given prospect for a given team at a given pick number.

    Returns top-10 features driving the prediction, sorted by absolute SHAP value.
    Useful for understanding model reasoning and debugging unexpected picks.
    """
    from app.ml.explain import explain_prospect_pick
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run POST /api/ml/train first.")
    try:
        result = explain_prospect_pick(db, registry.model, prospect_id, team_id, pick_number)
        return {"status": "ok", "explanation": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("SHAP explanation failed")
        try:
            from app.observability.metrics import SHAP_ERRORS_TOTAL
            SHAP_ERRORS_TOTAL.inc()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Explanation error: {e}")


@router.post("/hpo", dependencies=[Depends(require_admin_key)])
@_limiter.limit("1/hour")
async def run_hpo(
    request: Request,
    n_trials: int = 50,
    db: Session = Depends(get_db),
):
    """
    Run Optuna hyperparameter optimization for the XGBoost ranker.

    Uses time-series-aware cross-validation (fold per year) and optimises
    NDCG@1 — the fraction of picks where the model ranks the actual pick first.

    Takes 20–60 min depending on n_trials and hardware.
    Results saved to ml/best_params.json — copy into train.py's rank_params.

    Rate-limited to 1/hour — this is CPU-intensive.
    """
    import asyncio
    from app.ml.hpo import optimize

    if not _acquire_training_lock():
        raise HTTPException(
            status_code=409,
            detail="Training/HPO already in progress. Wait for it to finish.",
        )
    try:
        result = await asyncio.to_thread(optimize, db, n_trials)
        return {"status": "ok", "result": result}
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        logger.exception("HPO failed")
        raise HTTPException(status_code=500, detail=f"HPO error: {exc}")
    finally:
        _release_training_lock()


@router.get("/importance-drift")
async def get_importance_drift():
    """
    Compare feature importance rankings between the current and previous training run.

    Returns Spearman rank correlation (ρ) plus alerts for individual features
    whose rank shifted by ≥ 5 positions. Low ρ (< 0.70) means the model's
    internal weighting changed substantially — investigate before deploying.

    Useful after:
      - Adding/removing features in features.py
      - Data pipeline changes that alter feature distributions
      - Hyperparameter changes that shift tree structure
    """
    from app.ml.importance_tracker import compare_importance_drift, load_importance_history
    from app.ml.features import FEATURE_COLS

    history = load_importance_history(n=2)
    if not history:
        return {
            "status": "no_history",
            "message": "No importance history found. Train the model at least once.",
        }

    drift_report = compare_importance_drift(FEATURE_COLS)
    return {"status": "ok", "drift": drift_report}


@router.get("/scores")
async def get_prospect_scores(db: Session = Depends(get_db)):
    """
    Return model scores for all 2025 prospects using a neutral team context
    (no GM tendency bias, pick #1, empty draft state).

    Scores reflect pure prospect quality as seen by the model — useful for
    displaying a model-based ranking on the prospects page.
    """
    from app.models import Prospect2025
    from app.ml.predict import score_pool_for_team, compute_pool_stats

    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run POST /api/ml/train first.")

    prospects = (
        db.query(Prospect2025)
        .order_by(Prospect2025.css_ranking.nullslast())
        .all()
    )
    if not prospects:
        raise HTTPException(status_code=404, detail="No prospects loaded. Run ingestion first.")

    pool_stats = compute_pool_stats(prospects)
    draft_state = {"pos_taken": {}, "team_positions": {}, "total_picked": 0}
    scores = score_pool_for_team(prospects, profile=None, pick_slot=1, draft_state=draft_state, pool_stats=pool_stats)

    # Apply conformal calibration if available
    cal_intervals = {}
    if registry.calibration:
        from app.ml.calibration import apply_intervals
        raw = {p.id: scores.get(p.id, 0.0) for p in prospects}
        cal_intervals = apply_intervals(raw, registry.calibration, alpha=0.10)

    result = sorted(
        [
            {
                "prospect_id":      p.id,
                "name":             p.name,
                "position":         p.position,
                "css_rank":         p.css_ranking,
                "ml_score":         round(scores.get(p.id, 0.0), 4),
                "in_prediction_set": cal_intervals.get(p.id, {}).get("in_prediction_set"),
                "nc_score":         cal_intervals.get(p.id, {}).get("nc_score"),
            }
            for p in prospects
        ],
        key=lambda x: x["ml_score"],
        reverse=True,
    )
    return {"scores": result, "count": len(result), "calibrated": bool(cal_intervals)}


@router.get("/train-history")
async def training_history(limit: int = 20):
    """
    Return the last N training run records from train_history.jsonl.

    Each entry includes: trained_at, mode, validation_auc, training_samples,
    and top-5 feature importances. Use this to detect AUC regressions between
    training runs without needing MLflow.
    """
    from app.ml.train import TRAIN_HISTORY_PATH
    import json as _json

    if not TRAIN_HISTORY_PATH.exists():
        return {"history": [], "count": 0}

    try:
        lines = TRAIN_HISTORY_PATH.read_text().strip().splitlines()
        records = []
        for line in lines:
            try:
                records.append(_json.loads(line))
            except Exception:
                pass
        # Return most recent N runs
        recent = records[-limit:] if len(records) > limit else records
        recent.reverse()
        return {"history": recent, "count": len(records)}
    except Exception as exc:
        logger.exception("Training history read failed")
        raise HTTPException(status_code=500, detail=f"History read error: {exc}")


@router.post("/calibrate", dependencies=[Depends(require_admin_key)])
async def calibrate_model(db: Session = Depends(get_db)):
    """
    (Re)calibrate the current model using the validation split (years ≥ 2020).

    Runs split conformal prediction on the held-out years and saves quantile
    thresholds to calibration.json. Safe to call without retraining — useful
    when the model was trained in production mode (final=true) or after the
    training dataset has been updated with new prospects.

    After calibration, prediction set membership (in_prediction_set, nc_score)
    is automatically included in /api/ml/scores and draft simulation picks.
    """
    from app.ml.features import build_training_dataset
    from app.ml.train import VAL_START_YEAR
    from app.ml.calibration import calibrate, load_calibration

    if not registry.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run POST /api/ml/train first.",
        )

    try:
        df = build_training_dataset(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset build failed: {e}")

    val_df = df[df["year"] >= VAL_START_YEAR]
    if len(val_df) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No validation data (year ≥ {VAL_START_YEAR}) found. "
                   "Ensure historical picks are ingested.",
        )

    try:
        cal = calibrate(registry.model, val_df)
    except Exception as e:
        logger.exception("Standalone calibration failed")
        raise HTTPException(status_code=500, detail=f"Calibration error: {e}")

    # Reload into registry without a full model reload
    registry._calibration = load_calibration()

    return {
        "status": "ok",
        "n_calibration": cal.get("n_calibration"),
        "quantiles": cal.get("quantiles"),
        "nc_mean": cal.get("nc_mean"),
        "nc_std": cal.get("nc_std"),
        "computed_at": cal.get("computed_at"),
    }


@router.get("/calibration")
async def get_calibration():
    """
    Return the stored conformal prediction calibration metadata.

    Shows quantile thresholds at 80/85/90% coverage levels. These thresholds
    are used at inference to mark which prospects are inside the prediction set
    (i.e., statistically plausible picks at the configured confidence level).
    """
    cal = registry.calibration
    if cal is None:
        raise HTTPException(
            status_code=404,
            detail="No calibration data found. Run POST /api/ml/train (eval mode) first.",
        )
    return {"status": "ok", "calibration": cal}


@router.get("/drift")
async def drift_report(db: Session = Depends(get_db)):
    """
    Compare the current draft season's pick distribution against the training
    distribution. Returns KL-divergence per feature dimension and a pass/warn/fail
    status. High drift suggests the model may need retraining.
    """
    from app.ml.drift import compute_drift_report
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        report = compute_drift_report(db)
        return {"status": "ok", "drift": report}
    except Exception as e:
        logger.exception("Drift report failed")
        raise HTTPException(status_code=500, detail=f"Drift error: {e}")
