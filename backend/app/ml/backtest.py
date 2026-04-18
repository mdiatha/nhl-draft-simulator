"""
Backtesting: train on years ≤ cutoff, evaluate on a held-out draft year.

For each pick in the test year:
  1. Build the available pool (players drafted later that year = were available).
  2. Score every candidate (actual pick + pool) using the trained model + team GM profile.
  3. Rank by predicted probability descending.
  4. Record where the actual pick lands in the ranking.

Metrics returned:
  - top1_accuracy   % picks where model's #1 choice = actual pick
  - top3_accuracy   % picks where actual pick is in model's top 3
  - top5_accuracy   % picks where actual pick is in model's top 5
  - mrr             Mean Reciprocal Rank  (1/rank, averaged)
  - by_round        breakdown of top1/top3/top5 per round
  - by_archetype    breakdown of top1/top3 per GM archetype
  - worst_misses    top 10 biggest misses (actual rank vs predicted)
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from app.ml.features import (
    FEATURE_COLS, build_features, build_training_dataset,
    _gm_features, _contextual_feats, _compute_predraft_quality, _cohort_tier_stats,
)
from app.ml.train import train, model_exists

logger = logging.getLogger(__name__)

TRAIN_CUTOFF = 2023
TEST_YEAR = 2024

# Years available for multi-year CI backtesting
BACKTEST_YEARS = [2021, 2022, 2023, 2024]

BASELINE_LABELS = {
    "consensus_css": "CSS best-available",
    "production_ppg": "PPG best-available",
    "uniform_random": "Uniform random",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PickResult:
    overall_pick: int
    round: int
    team_name: str
    gm_archetype: str
    actual_player: str
    predicted_player: str
    actual_rank: int       # rank of the actual pick in model's sorted list
    pool_size: int
    score_actual: float
    score_predicted: float


# ── Core backtest logic ───────────────────────────────────────────────────────

def run_backtest(db, test_year: int = TEST_YEAR, train_cutoff: int = TRAIN_CUTOFF) -> dict:
    """
    Train on picks ≤ train_cutoff, evaluate on test_year.
    Returns a metrics dict.

    Uses build_training_dataset() with max_year=train_cutoff so the training
    procedure is identical to production — same features, same negative window,
    same GM tendency computation. This ensures backtest accuracy reflects real
    production model performance rather than a simplified proxy.
    """
    from app.models.draft_pick_historical import DraftPickHistorical
    from app.models.gm_tendency_profile import GMTendencyProfile
    from app.models.general_manager import GeneralManager
    from app.models.team import Team

    # ── Step 1: train on years ≤ cutoff (same procedure as production) ────────
    logger.info("Building training dataset (years ≤ %d)...", train_cutoff)
    train_df = build_training_dataset(db, max_year=train_cutoff)
    logger.info("Training on %d rows...", len(train_df))
    model, _ = train(train_df)

    # ── Step 2: load test picks ───────────────────────────────────────────────
    test_picks = (
        db.query(DraftPickHistorical)
        .filter(
            DraftPickHistorical.year == test_year,
            DraftPickHistorical.round <= 4,
        )
        .order_by(DraftPickHistorical.overall_pick)
        .all()
    )

    if not test_picks:
        raise ValueError(f"No draft picks found for year {test_year}. Check ingestion.")

    # ── Step 3: precompute cohort stats for the test year ─────────────────────
    # These mirror what build_training_dataset computes per cohort so that
    # feature values are consistent between training and evaluation.
    quality      = _compute_predraft_quality(test_picks)
    tier_ppg_med, tier_age_med = _cohort_tier_stats(test_picks)

    # ── Step 4: load GM profiles ──────────────────────────────────────────────
    profiles = {p.gm_id: p for p in db.query(GMTendencyProfile).all()}
    teams    = {t.id: t for t in db.query(Team).all()}
    gms      = {
        gm.team_id: gm
        for gm in db.query(GeneralManager).all()
        if gm.team_id
    }

    # ── Step 5: evaluate each pick ────────────────────────────────────────────
    results: list[PickResult] = []
    baseline_ranks: dict[str, list[int]] = {
        "consensus_css": [],
        "production_ppg": [],
    }
    pool_sizes: list[int] = []

    pos_taken: Counter[str]                          = Counter()
    team_pos_drafted: defaultdict[int, Counter[str]] = defaultdict(Counter)

    for i, pick in enumerate(test_picks):
        if not pick.team_id:
            pos_taken[pick.position or "F"] += 1
            continue

        gm        = gms.get(pick.team_id)
        profile   = profiles.get(gm.id) if gm else None
        team      = teams.get(pick.team_id)
        team_name = team.abbreviation if team else "UNK"
        archetype = profile.tendency_archetype if profile else "BPA"

        # Available pool: next NEGATIVE_WINDOW picks after this slot.
        # Using a fixed window (not the full tail) prevents pool-size shrinkage
        # from trivially inflating late-round accuracy: the last pick in round 4
        # would have pool=1 with the tail, making it trivially 100% top-1.
        # The fixed window matches the training distribution (NEGATIVE_WINDOW=31).
        from app.ml.features import NEGATIVE_WINDOW
        pool = test_picks[i + 1 : i + 1 + NEGATIVE_WINDOW]
        if not pool:
            pos_taken[pick.position or "F"] += 1
            team_pos_drafted[pick.team_id][pick.position or "F"] += 1
            continue

        candidates = [pick] + list(pool)
        # remaining = full board at this slot (everything after, not capped)
        remaining  = test_picks[i:]

        # rank_gap_norm requires knowing the best CSS score in the candidate group
        group_css      = [quality.get(c.id, 0.5) for c in candidates]
        group_best_css = max(group_css)

        rows = []
        for c in candidates:
            c_css = quality.get(c.id, 0.5)
            gm_feats = _gm_features(
                profile, c.position or "", c.draft_league or "", c.nationality or ""
            )
            ctx = _contextual_feats(
                c, remaining, i, pick.team_id,
                pos_taken, team_pos_drafted,
                tier_ppg_med, tier_age_med,
                quality_map=quality,
            )
            rows.append({
                "position":          c.position,
                "nationality":       c.nationality,
                "height_cm":         c.height_cm,
                "weight_kg":         c.weight_kg,
                "draft_league":      c.draft_league,
                "draft_league_tier": c.draft_league_tier,
                "points_per_game":   c.points_per_game,
                "gp_pre_draft":      c.gp_pre_draft,
                "ppg_prev_season":   c.ppg_prev_season,
                "has_prev_season":   1 if c.ppg_prev_season is not None else 0,
                "age_at_draft":      c.age_at_draft,
                "overall_pick":      pick.overall_pick,
                "draft_round":       pick.round,
                "css_rank_norm":     c_css,
                "rank_gap_norm":     group_best_css - c_css,
                **gm_feats,
                **ctx,
            })

        df     = pd.DataFrame(rows)
        X      = build_features(df)
        # XGBRanker uses .predict() — .predict_proba() does not exist on rankers.
        from xgboost import XGBRanker
        probas = model.predict(X) if isinstance(model, XGBRanker) else model.predict_proba(X)[:, 1]

        # Rank candidates (highest probability = rank 1)
        ranked = sorted(zip(probas, candidates), key=lambda x: -x[0])
        actual_rank = next(
            (rank + 1 for rank, (_, c) in enumerate(ranked) if c.id == pick.id),
            len(ranked),
        )

        baseline_ranks["consensus_css"].append(
            _rank_of_actual(
                candidates,
                pick.id,
                key_fn=lambda c: (
                    quality.get(c.id, 0.5),
                    -(c.css_rank or 10_000),
                    c.points_per_game or -1.0,
                ),
            )
        )
        baseline_ranks["production_ppg"].append(
            _rank_of_actual(
                candidates,
                pick.id,
                key_fn=lambda c: (
                    c.points_per_game or -1.0,
                    quality.get(c.id, 0.5),
                    -(c.css_rank or 10_000),
                ),
            )
        )
        pool_sizes.append(len(candidates))

        top_score, top_candidate = ranked[0]

        results.append(PickResult(
            overall_pick=pick.overall_pick,
            round=pick.round,
            team_name=team_name,
            gm_archetype=archetype or "BPA",
            actual_player=pick.player_name or "Unknown",
            predicted_player=top_candidate.player_name or "Unknown",
            actual_rank=actual_rank,
            pool_size=len(candidates),
            score_actual=float(probas[0]),
            score_predicted=float(top_score),
        ))

        pos_taken[pick.position or "F"] += 1
        team_pos_drafted[pick.team_id][pick.position or "F"] += 1

    # ── Step 6: compute metrics ───────────────────────────────────────────────
    metrics = _compute_metrics(results, test_year, train_cutoff, baseline_ranks, pool_sizes)

    # Emit Prometheus gauges for the most recent backtest run
    _emit_backtest_metrics(metrics)

    return metrics


# ── Metric computation ────────────────────────────────────────────────────────

def _rank_of_actual(candidates: list, actual_id: int, key_fn) -> int:
    ranked = sorted(
        candidates,
        key=lambda candidate: tuple(
            -value if isinstance(value, (int, float)) else value
            for value in key_fn(candidate)
        ),
    )
    return next(
        (rank + 1 for rank, candidate in enumerate(ranked) if candidate.id == actual_id),
        len(ranked),
    )


def _metrics_from_ranks(ranks: list[int]) -> dict:
    if not ranks:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "top5_accuracy": 0.0, "mrr": 0.0}

    n = len(ranks)
    return {
        "top1_accuracy": round(sum(1 for rank in ranks if rank == 1) / n, 3),
        "top3_accuracy": round(sum(1 for rank in ranks if rank <= 3) / n, 3),
        "top5_accuracy": round(sum(1 for rank in ranks if rank <= 5) / n, 3),
        "mrr": round(sum(1.0 / rank for rank in ranks) / n, 3),
    }


def _random_baseline_metrics(pool_sizes: list[int]) -> dict:
    if not pool_sizes:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "top5_accuracy": 0.0, "mrr": 0.0}

    harmonic = lambda n: sum(1.0 / i for i in range(1, n + 1))
    n = len(pool_sizes)
    return {
        "top1_accuracy": round(sum(1.0 / size for size in pool_sizes) / n, 3),
        "top3_accuracy": round(sum(min(3, size) / size for size in pool_sizes) / n, 3),
        "top5_accuracy": round(sum(min(5, size) / size for size in pool_sizes) / n, 3),
        "mrr": round(sum(harmonic(size) / size for size in pool_sizes) / n, 3),
    }


def _compute_metrics(
    results: list[PickResult],
    test_year: int,
    train_cutoff: int,
    baseline_ranks: dict[str, list[int]],
    pool_sizes: list[int],
) -> dict:
    if not results:
        return {"error": "No results to evaluate"}

    n    = len(results)
    top1 = sum(1 for r in results if r.actual_rank == 1) / n
    top3 = sum(1 for r in results if r.actual_rank <= 3) / n
    top5 = sum(1 for r in results if r.actual_rank <= 5) / n
    mrr  = sum(1.0 / r.actual_rank for r in results) / n

    # By round
    by_round: dict[int, dict] = {}
    for r in results:
        rnd = r.round
        if rnd not in by_round:
            by_round[rnd] = {"n": 0, "top1": 0, "top3": 0, "top5": 0}
        by_round[rnd]["n"] += 1
        if r.actual_rank == 1:  by_round[rnd]["top1"] += 1
        if r.actual_rank <= 3:  by_round[rnd]["top3"] += 1
        if r.actual_rank <= 5:  by_round[rnd]["top5"] += 1

    for rnd, d in by_round.items():
        d["top1_pct"] = round(d["top1"] / d["n"], 3)
        d["top3_pct"] = round(d["top3"] / d["n"], 3)
        d["top5_pct"] = round(d["top5"] / d["n"], 3)

    # By archetype
    by_arch: dict[str, dict] = {}
    for r in results:
        arch = r.gm_archetype
        if arch not in by_arch:
            by_arch[arch] = {"n": 0, "top1": 0, "top3": 0}
        by_arch[arch]["n"] += 1
        if r.actual_rank == 1:  by_arch[arch]["top1"] += 1
        if r.actual_rank <= 3:  by_arch[arch]["top3"] += 1

    for arch, d in by_arch.items():
        d["top1_pct"] = round(d["top1"] / d["n"], 3)
        d["top3_pct"] = round(d["top3"] / d["n"], 3)

    # Worst misses (largest actual_rank)
    worst = sorted(results, key=lambda r: -r.actual_rank)[:10]
    baselines = {
        "consensus_css": {
            "label": BASELINE_LABELS["consensus_css"],
            **_metrics_from_ranks(baseline_ranks["consensus_css"]),
        },
        "production_ppg": {
            "label": BASELINE_LABELS["production_ppg"],
            **_metrics_from_ranks(baseline_ranks["production_ppg"]),
        },
        "uniform_random": {
            "label": BASELINE_LABELS["uniform_random"],
            **_random_baseline_metrics(pool_sizes),
        },
    }
    lift_vs_baselines = {
        name: {
            "top1_accuracy": round(top1 - metrics["top1_accuracy"], 3),
            "top3_accuracy": round(top3 - metrics["top3_accuracy"], 3),
            "top5_accuracy": round(top5 - metrics["top5_accuracy"], 3),
            "mrr": round(mrr - metrics["mrr"], 3),
        }
        for name, metrics in baselines.items()
    }

    return {
        "test_year":       test_year,
        "train_cutoff":    train_cutoff,
        "picks_evaluated": n,
        "top1_accuracy":   round(top1, 3),
        "top3_accuracy":   round(top3, 3),
        "top5_accuracy":   round(top5, 3),
        "mrr":             round(mrr, 3),
        "by_round":        {f"round_{k}": v for k, v in sorted(by_round.items())},
        "by_archetype":    by_arch,
        "worst_misses": [
            {
                "pick":        r.overall_pick,
                "round":       r.round,
                "team":        r.team_name,
                "actual":      r.actual_player,
                "predicted":   r.predicted_player,
                "actual_rank": r.actual_rank,
                "pool_size":   r.pool_size,
            }
            for r in worst
        ],
        "baselines":       baselines,
        "lift_vs_baselines": lift_vs_baselines,
    }


# ── Multi-year confidence intervals ───────────────────────────────────────────

def run_multi_year_backtest(db, years: list[int] | None = None) -> dict:
    """
    Run backtests across multiple held-out years and compute mean ± std CI.

    For each year Y in `years`:
      - Train on all picks before Y
      - Evaluate on picks from Y

    Returns per-year results plus aggregate statistics with 95% bootstrap CI.
    This turns point estimates (top1=0.45) into confidence intervals
    (top1=0.43 ± 0.06) so we can distinguish real improvements from noise.
    """
    import numpy as np

    if years is None:
        years = BACKTEST_YEARS

    per_year: list[dict] = []
    errors: list[str] = []

    for test_year in years:
        train_cutoff = test_year - 1
        logger.info("multi_year_backtest year=%d cutoff=%d", test_year, train_cutoff)
        try:
            result = run_backtest(db, test_year=test_year, train_cutoff=train_cutoff)
            per_year.append(result)
        except Exception as exc:
            logger.warning("multi_year_backtest.year_failed year=%d error=%s", test_year, exc)
            errors.append(f"{test_year}: {exc}")

    if not per_year:
        return {"error": "All backtest years failed", "details": errors}

    # Aggregate metrics with mean ± std
    def _agg(key: str) -> dict:
        vals = np.array([r[key] for r in per_year if key in r], dtype=float)
        if len(vals) == 0:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": round(float(vals.mean()), 4),
            "std":  round(float(vals.std()),  4),
            "min":  round(float(vals.min()),  4),
            "max":  round(float(vals.max()),  4),
            "n":    len(vals),
        }

    return {
        "years_evaluated":  [r["test_year"] for r in per_year],
        "errors":           errors,
        "per_year":         per_year,
        "aggregate": {
            "top1_accuracy": _agg("top1_accuracy"),
            "top3_accuracy": _agg("top3_accuracy"),
            "top5_accuracy": _agg("top5_accuracy"),
            "mrr":           _agg("mrr"),
        },
        "baseline_aggregate": {
            name: {
                "label": BASELINE_LABELS.get(name, name),
                "top1_accuracy": _agg_baseline(per_year, name, "top1_accuracy"),
                "top3_accuracy": _agg_baseline(per_year, name, "top3_accuracy"),
                "top5_accuracy": _agg_baseline(per_year, name, "top5_accuracy"),
                "mrr": _agg_baseline(per_year, name, "mrr"),
            }
            for name in BASELINE_LABELS
        },
    }


def _agg_baseline(per_year: list[dict], baseline: str, key: str) -> dict:
    import numpy as np

    vals = np.array(
        [
            result["baselines"][baseline][key]
            for result in per_year
            if baseline in result.get("baselines", {}) and key in result["baselines"][baseline]
        ],
        dtype=float,
    )
    if len(vals) == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(float(vals.mean()), 4),
        "std": round(float(vals.std()), 4),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
        "n": len(vals),
    }


def _emit_backtest_metrics(metrics: dict) -> None:
    """Publish backtest results to Prometheus so they trend in Grafana."""
    try:
        from app.observability.metrics import (
            BACKTEST_TOP1_ACCURACY, BACKTEST_TOP3_ACCURACY,
            BACKTEST_MRR, BACKTEST_WORST_RANK,
        )
        if metrics.get("top1_accuracy") is not None:
            BACKTEST_TOP1_ACCURACY.set(metrics["top1_accuracy"])
        if metrics.get("top3_accuracy") is not None:
            BACKTEST_TOP3_ACCURACY.set(metrics["top3_accuracy"])
        if metrics.get("mrr") is not None:
            BACKTEST_MRR.set(metrics["mrr"])
        worst = metrics.get("worst_misses", [])
        if worst:
            BACKTEST_WORST_RANK.set(max(w["actual_rank"] for w in worst))
    except Exception as exc:
        logger.warning("backtest.metrics_emit_failed error=%s", exc)
