"""Score 2025 prospects using the trained XGBoost model."""
from __future__ import annotations

import logging
import statistics
import time as _time
from collections import Counter, defaultdict

import pandas as pd

from app.ml.registry import registry
from app.ml.features import (
    build_features, _gm_features, _contextual_feats,
)

logger = logging.getLogger(__name__)

# Restrict inference to top-N CSS-ranked prospects before scoring.
# This aligns with NEGATIVE_WINDOW=31 from training (model learned to compare
# prospects within a ~32-player window). We use 40 to give extra room for
# positional variation while still preventing low-CSS prospects from being
# over-scored due to rank_gap_norm anchoring against the full pool.
INFERENCE_CANDIDATE_WINDOW = 40


def compute_pool_stats(prospects: list) -> dict:
    """
    Compute league-group median PPG/age and PPG percentiles from the prospect pool.

    ppg_by_tier / age_by_tier: keyed by infer_league_key() so each league group
    gets its own median — OHL prospects are compared to OHL peers, not KHL pros.

    ppg_percentile: {prospect_id: float} — each prospect's PPG rank within the
    full 2025 class, on [0, 1] where 1.0 = highest scorer.  Used as css_rank_norm
    at inference, matching the PPG-percentile proxy used during training.  This
    makes css_rank_norm mean the same thing in both phases.
    """
    from app.constants import infer_league_key
    tier_ppg: dict[str, list] = {}
    tier_age: dict[str, list] = {}
    for p in prospects:
        key = infer_league_key(p.draft_league or "", p.draft_league_tier)
        # Exclude goalies from PPG medians — their G+A is near zero by nature
        if p.points_per_game is not None and (p.position or "F") != "G":
            tier_ppg.setdefault(key, []).append(p.points_per_game)
        if p.age_at_draft is not None:
            tier_age.setdefault(key, []).append(p.age_at_draft)

    # PPG percentile — same formula as _compute_predraft_quality() in features.py.
    # Computed within position group (goalies vs skaters) so goalie near-zero
    # PPG doesn't land them at the bottom of the skater distribution.
    ppg_percentile: dict[int, float] = {}
    skaters = [p for p in prospects if (p.position or "F") != "G"]
    goalies  = [p for p in prospects if (p.position or "F") == "G"]
    for group in (skaters, goalies):
        ppg_vals = sorted(p.points_per_game or 0.0 for p in group)
        n = len(ppg_vals)
        for p in group:
            v = p.points_per_game or 0.0
            ppg_percentile[p.id] = (sum(1 for x in ppg_vals if x <= v) / n) if n > 0 else 0.5

    return {
        "ppg_by_tier":   {k: statistics.median(v) for k, v in tier_ppg.items() if v},
        "age_by_tier":   {k: statistics.median(v) for k, v in tier_age.items() if v},
        "ppg_percentile": ppg_percentile,
    }


def score_pool_for_team(
    prospects: list,           # available Prospect ORM objects
    profile,                   # GMTendencyProfile ORM object (or None)
    pick_slot: int,            # current overall pick number (1-based)
    draft_state: dict | None = None,
    pool_stats: dict | None = None,
    draft_round: int = 1,      # round being simulated (1 = first round, etc.)
    db=None,                   # optional DB session for feature store cache lookup
) -> dict[int, float]:
    """
    Score every prospect in the available pool for one team pick.

    draft_state keys:
      pos_taken       — Counter[str]: positions taken so far this draft
      team_positions  — Counter[str]: positions this team has drafted today
      total_picked    — int: total picks made before this slot

    pool_stats keys:
      ppg_by_tier    — dict[str, float]: league-group -> median PPG
      age_by_tier    — dict[str, float]: league-group -> median age
      ppg_percentile — dict[int, float]: prospect_id -> PPG percentile in pool

    Returns {prospect_id: probability}, empty dict if no model is loaded.
    """
    if not registry.is_loaded:
        return {}

    _t_start = _time.perf_counter()

    try:
        return _score_pool_inner(
            prospects, profile, pick_slot, draft_state, pool_stats, draft_round, db
        )
    except Exception as exc:
        try:
            from app.observability.metrics import MODEL_PREDICTION_ERRORS
            MODEL_PREDICTION_ERRORS.inc()
        except Exception:
            pass
        logger.exception("score_pool_for_team.failed error=%s", exc)
        raise
    finally:
        duration = _time.perf_counter() - _t_start
        try:
            from app.observability.metrics import (
                MODEL_PREDICTIONS_TOTAL, MODEL_PREDICTION_DURATION, MODEL_PREDICTION_POOL_SIZE,
            )
            MODEL_PREDICTIONS_TOTAL.labels(source="draft_sim").inc()
            MODEL_PREDICTION_DURATION.observe(duration)
            MODEL_PREDICTION_POOL_SIZE.observe(len(prospects))
        except Exception:
            pass


def _score_pool_inner(
    prospects: list,
    profile,
    pick_slot: int,
    draft_state: dict | None,
    pool_stats: dict | None,
    draft_round: int,
    db,
) -> dict[int, float]:
    """Inner implementation — separated so the outer function can time it cleanly."""
    # ── Restrict to top-N CSS-ranked prospects ────────────────────────────────
    # Sort prospects by css_ranking (nulls last — unranked go to end), then keep
    # only the top INFERENCE_CANDIDATE_WINDOW. This aligns inference with training:
    # the model learned to compare prospects within a local ~32-player window, so
    # scoring the full pool of 224 produces wildly miscalibrated rank_gap_norm
    # values that cause low-CSS-rank prospects to be over-scored.
    css_ranked   = [p for p in prospects if p.css_ranking is not None]
    css_unranked = [p for p in prospects if p.css_ranking is None]
    css_ranked_sorted = sorted(css_ranked, key=lambda p: p.css_ranking)
    prospects = (css_ranked_sorted + css_unranked)[:INFERENCE_CANDIDATE_WINDOW]

    # Unpack draft state
    pos_taken      = (draft_state or {}).get("pos_taken", Counter())
    team_positions = (draft_state or {}).get("team_positions", Counter())
    total_picked   = (draft_state or {}).get("total_picked", 0)

    # Unpack pool stats
    tier_ppg_med     = (pool_stats or {}).get("ppg_by_tier", {})
    tier_age_med     = (pool_stats or {}).get("age_by_tier", {})
    ppg_percentile   = (pool_stats or {}).get("ppg_percentile", {})

    # Quality map: prospect_id → css_rank_norm, used by _contextual_feats to
    # compute pos_quality_rank_norm (rank within same-position available players).
    # Must match training exactly: sort all CSS-ranked prospects globally by raw
    # css_ranking, assign global ordinal rank, then apply sqrt normalization using
    # the total ranked count as denominator. Training uses _compute_predraft_quality()
    # which does the same global sort — per-list normalization was wrong because it
    # inflated css_rank_norm for lower-ranked prospects who happen to be high within
    # their list (e.g. CSS #152 goalie who is rank #11 of 15 goalies got ~0.5
    # instead of the ~0.18 the model was trained with).
    import math as _math
    quality_map_inf: dict[int, float] = {}
    _css_ranked = [p for p in prospects if p.css_ranking]
    _css_unranked = [p for p in prospects if not p.css_ranking]
    if _css_ranked:
        _sorted_all = sorted(_css_ranked, key=lambda p: p.css_ranking)
        _cohort_size = len(_sorted_all)
        for _ordinal, p in enumerate(_sorted_all, start=1):
            quality_map_inf[p.id] = max(0.0, 1.0 - _math.sqrt((_ordinal - 1) / _cohort_size))
    for p in _css_unranked:
        quality_map_inf[p.id] = ppg_percentile.get(p.id, 0.15)

    # Build a fake team_pos_drafted defaultdict so _contextual_feats works
    # without needing a real team_id — team_positions IS the team's counter
    _team_pos = defaultdict(Counter)
    _team_id  = 0
    _team_pos[_team_id] = team_positions

    # ── Feature store cache lookup ─────────────────────────────────────────────
    # Cached rows store neutral (profile=None, pick_slot=1) base features.
    # We override the pick-slot-dependent and GM-dependent columns below.
    cached_X = None
    if db is not None:
        try:
            from app.ml.feature_store import get_cached_features
            pid_order = [p.id for p in prospects]
            cached_X = get_cached_features(db, pid_order)
        except Exception as fs_exc:
            logger.debug("feature_store.lookup_failed reason=%s", fs_exc)

    if cached_X is not None:
        # Apply pick-slot norm override (pick_slot changes every pick)
        from app.ml.features import MAX_DRAFT_POOL
        cached_X = cached_X.copy()
        slot_norm = (1.0 - (pick_slot - 1) / MAX_DRAFT_POOL)
        cached_X["pick_slot_norm"] = slot_norm
        cached_X["rank_vs_slot"] = cached_X["css_rank_norm"] - slot_norm
        # rank_gap_norm: gap from the best css_rank_norm in the top-INFERENCE_CANDIDATE_WINDOW
        # scoring window. Prospects is already filtered to that window above.
        _sorted_c = sorted(prospects, key=lambda p: quality_map_inf.get(p.id, 0.0), reverse=True)
        _window_best_c = quality_map_inf.get(_sorted_c[0].id, 0.0) if _sorted_c else 0.0
        pid_to_gap = {
            p.id: max(0.0, _window_best_c - quality_map_inf.get(p.id, 0.0))
            for p in prospects
        }
        cached_X["rank_gap_norm"] = [pid_to_gap.get(p.id, 0.0) for p in prospects]
        # Overwrite round one-hots from cached features with the current round
        for r in [1, 2, 3, 4]:
            cached_X[f"round_{r}"] = int(draft_round == r)

        # Apply GM features per-row (profile-dependent — cannot be cached)
        gm_pos    = []
        gm_league = []
        gm_nat    = []
        for p in prospects:
            pf = _gm_features(profile, p.position or "", p.draft_league or "", p.nationality or "")
            gm_pos.append(pf["gm_pos_weight"])
            gm_league.append(pf["gm_league_weight"])
            gm_nat.append(pf["gm_nat_weight"])
        cached_X["gm_pos_weight"]    = gm_pos
        cached_X["gm_league_weight"] = gm_league
        cached_X["gm_nat_weight"]    = gm_nat

        # Apply draft-state features per-row (pick-dependent — cannot be cached)
        pos_taken_list  = []
        pos_remain_list = []
        team_pos_list   = []
        for p in prospects:
            pos = p.position or "F"
            pt_norm = pos_taken[pos] / total_picked if total_picked > 0 else 0.0
            total_rem = len(prospects)
            pos_rem   = sum(1 for x in prospects if (x.position or "F") == pos)
            pr_norm   = pos_rem / total_rem if total_rem > 0 else 0.0
            tc        = team_positions[pos]
            pos_taken_list.append(pt_norm)
            pos_remain_list.append(pr_norm)
            team_pos_list.append(tc)
        cached_X["pos_taken_before_norm"] = pos_taken_list
        cached_X["pos_remaining_norm"]    = pos_remain_list
        cached_X["team_drafted_this_pos"] = team_pos_list

        # pos_quality_rank_norm and css_rank_within_pos: rank within same-position
        # players on the board. Dynamic — depends on who is still available.
        # Both use the same ordinal formula; always overridden here.
        pos_qual_list: list[float] = []
        css_within_pos_list: list[float] = []
        for p in prospects:
            pos   = p.position or "F"
            this_q = quality_map_inf[p.id]
            same_pos = [x for x in prospects if (x.position or "F") == pos]
            n_pos    = len(same_pos)
            if n_pos > 1:
                n_better = sum(1 for x in same_pos if quality_map_inf.get(x.id, 0.0) > this_q)
                pq = 1.0 - (n_better / n_pos)
            else:
                pq = 1.0
            pos_qual_list.append(pq)
            css_within_pos_list.append(pq)
        cached_X["pos_quality_rank_norm"] = pos_qual_list
        cached_X["css_rank_within_pos"]   = css_within_pos_list
        # slot_pressure: fraction of round remaining (same formula as non-cached path)
        _ROUND_SIZES_C = {1: 32, 2: 32, 3: 32, 4: 32}
        _round_size_c   = _ROUND_SIZES_C.get(draft_round, 32)
        _pick_in_round_c = pick_slot - (draft_round - 1) * 32
        cached_X["slot_pressure"] = max(0.0, 1.0 - (_pick_in_round_c - 1) / max(_round_size_c - 1, 1))

        from app.ml.features import FEATURE_COLS
        missing = [c for c in FEATURE_COLS if c not in cached_X.columns]
        if missing:
            # Feature schema has changed since cache was written — fall through
            # to the non-cached path rather than crashing mid-simulation.
            logger.warning(
                "feature_store.stale_schema missing_cols=%s — falling back to full build",
                missing,
            )
            cached_X = None

    if cached_X is not None:
        from app.ml.features import FEATURE_COLS
        X = cached_X[FEATURE_COLS]
        ids = [p.id for p in prospects]

        from xgboost import XGBRanker
        if isinstance(registry.model, XGBRanker):
            scores = registry.model.predict(X)
        else:
            scores = registry.model.predict_proba(X)[:, 1]
        return {pid: float(s) for pid, s in zip(ids, scores)}

    # rank_gap_norm: gap from the best css_rank_norm in the scoring window.
    # prospects is already filtered to top-INFERENCE_CANDIDATE_WINDOW above,
    # so _window_best is the best of the ~40 candidates the model compares.
    _window_best = max(quality_map_inf.values(), default=0.0)
    rank_gap_map = {
        p.id: max(0.0, _window_best - quality_map_inf.get(p.id, 0.0))
        for p in prospects
    }

    # slot_pressure: fraction of current round remaining. At training this is
    # computed from actual per-round pick counts; at inference we use the standard
    # round sizes (R1=32, R2-4=32 each). 1.0 = first pick of round, ~0 = last.
    _ROUND_SIZES = {1: 32, 2: 32, 3: 32, 4: 32}
    _round_size = _ROUND_SIZES.get(draft_round, 32)
    _pick_in_round = pick_slot - (draft_round - 1) * 32  # approximate 1-based index within round
    _slot_pressure = max(0.0, 1.0 - (_pick_in_round - 1) / max(_round_size - 1, 1))

    # css_rank_within_pos: same ordinal formula as training — rank within same-
    # position players still available. Precompute for the full pool so the inner
    # loop doesn't repeat O(n²) work for each prospect.
    css_within_pos_map: dict[int, float] = {}
    for p in prospects:
        pos      = p.position or "F"
        this_q   = quality_map_inf[p.id]
        same_pos = [x for x in prospects if (x.position or "F") == pos]
        n_pos    = len(same_pos)
        if n_pos > 1:
            n_better = sum(1 for x in same_pos if quality_map_inf.get(x.id, 0.0) > this_q)
            css_within_pos_map[p.id] = 1.0 - (n_better / n_pos)
        else:
            css_within_pos_map[p.id] = 1.0

    rows = []
    ids  = []
    for p in prospects:
        pf  = _gm_features(profile, p.position or "", p.draft_league or "", p.nationality or "")
        ctx = _contextual_feats(
            p, prospects, total_picked, _team_id,
            pos_taken, _team_pos,
            tier_ppg_med, tier_age_med,
            quality_map=quality_map_inf,
        )
        css_norm = quality_map_inf[p.id]

        rows.append({
            "position":            p.position,
            "nationality":         p.nationality,
            "height_cm":           p.height_cm,
            "weight_kg":           p.weight_kg,
            "draft_league":        p.draft_league,
            "draft_league_tier":   p.draft_league_tier,
            "points_per_game":     p.points_per_game,
            "gp_pre_draft":        p.games_played,
            "ppg_prev_season":     p.ppg_prev_season,
            "has_prev_season":     1 if p.ppg_prev_season is not None else 0,
            "age_at_draft":        p.age_at_draft,
            "overall_pick":        pick_slot,
            "draft_round":         draft_round,
            "css_rank_norm":       css_norm,
            "rank_gap_norm":       rank_gap_map[p.id],
            "css_rank_within_pos": css_within_pos_map[p.id],
            "slot_pressure":       _slot_pressure,
            **pf,
            **ctx,
        })
        ids.append(p.id)

    df     = pd.DataFrame(rows)
    X      = build_features(df)

    # ── Feature value validation (catch train/inference skew early) ────────────
    _validate_feature_ranges(X)

    # XGBRanker outputs relevance scores via .predict() (higher = more likely to be picked).
    # XGBClassifier (legacy) outputs probabilities via .predict_proba()[:, 1].
    # Support both so old pickled classifiers still work during transition.
    from xgboost import XGBRanker
    if isinstance(registry.model, XGBRanker):
        scores = registry.model.predict(X)
    else:
        scores = registry.model.predict_proba(X)[:, 1]

    return {pid: float(s) for pid, s in zip(ids, scores)}


def _validate_feature_ranges(X: pd.DataFrame) -> None:
    """
    Warn if any feature values are outside their expected training ranges.
    These checks catch encoding bugs and training/inference skew early.
    Does NOT raise — only logs warnings so scoring always completes.
    """
    checks = {
        "css_rank_norm":   (0.0,  1.0),
        "pick_slot_norm":  (0.0,  1.0),
        "ppg_league_norm": (-0.5, 8.0),   # >8 = extreme scorer, suspicious
        "age_league_norm": (-5.0, 5.0),
        "height_norm":     (-20,  20),
        "weight_norm":     (-25,  25),
        "gp_pre_draft":    (0,    100),
    }
    for col, (lo, hi) in checks.items():
        if col not in X.columns:
            continue
        col_min = X[col].min()
        col_max = X[col].max()
        if col_min < lo or col_max > hi:
            logger.warning(
                "feature_validation.out_of_range feature=%s min=%.4f max=%.4f expected=[%.1f, %.1f]",
                col, col_min, col_max, lo, hi,
            )


def score_single_prospect(prospect, profile, pick_slot: int = 15) -> float:
    """Score a single prospect. Returns 0.5 if no model is loaded."""
    scores = score_pool_for_team([prospect], profile, pick_slot)
    return scores.get(prospect.id, 0.5)
