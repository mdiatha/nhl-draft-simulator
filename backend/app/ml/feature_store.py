"""
Feature Store — pre-materializes prospect features for fast inference.

Problem: score_pool_for_team() rebuilds the full feature matrix for all ~224
prospects on every pick (32 picks × 224 prospects = 7,168 row constructions per
simulation). The league normalization, GM feature lookups, and contextual feature
computation all run from scratch each time.

Solution: nightly (or on-demand) refresh of prospect_features table with
pre-computed feature rows. At inference, get_cached_features() loads a
pre-built DataFrame from DB in a single indexed SELECT rather than recomputing.

Note: GM-specific features (gm_pos_weight, gm_league_weight, gm_nat_weight) and
draft-state features (pos_taken_before_norm, etc.) change per pick slot, so they
are NOT cached — only the prospect-level and pool-level features are stored.
The scoring function applies GM/state features on top of cached base features.

Cache staleness: any row older than CACHE_TTL_HOURS is treated as a miss and
features are recomputed live.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

CACHE_TTL_HOURS = 25   # rows older than this are treated as stale

# Bump this string whenever FEATURE_COLS changes (columns added, removed, or renamed).
# Cached rows written with a different version are treated as a full cache miss,
# preventing silent corruption when features change between deployments.
# Format: "v{N}" where N increments on each breaking feature set change.
FEATURE_SCHEMA_VERSION = "v4"   # v4: removed ppg_trend_norm (train/inference mismatch)


# ── Write path ─────────────────────────────────────────────────────────────────

def refresh_feature_store(db: Session) -> int:
    """
    Compute and upsert feature rows for all 2025 prospects.

    Uses neutral GM context (profile=None, pick_slot=1, empty draft state) so
    the stored vector is pure prospect-level signal. GM/state features are
    applied on top at inference time.

    Returns the number of prospects processed.
    """
    from app.models import Prospect
    from app.models.prospect_features import ProspectFeatures
    from app.ml.predict import compute_pool_stats
    from app.ml.features import build_features, _gm_features, _contextual_feats, _css_list, _css_norm_within_list, FEATURE_COLS
    from collections import Counter, defaultdict

    prospects = db.query(Prospect).order_by(Prospect.css_ranking.nullslast()).all()
    if not prospects:
        logger.warning("feature_store.no_prospects")
        return 0

    pool_stats = compute_pool_stats(prospects)
    tier_ppg_med = pool_stats.get("ppg_by_tier", {})
    tier_age_med = pool_stats.get("age_by_tier", {})
    ppg_percentile = pool_stats.get("ppg_percentile", {})

    # Neutral draft state (no picks made, no team history)
    _team_pos = defaultdict(Counter)
    _team_id  = 0
    pos_taken = Counter()

    # Quality map for pos_quality_rank_norm — computed from the full pool
    # (neutral context: no picks made yet, full 2025 class available).
    # List-aware: normalize within na_skater / eur_skater / goalie separately.
    quality_map_fs: dict[int, float] = {}
    _fs_css_groups: dict[str, list] = {}
    for p in prospects:
        if p.css_ranking:
            lst = _css_list(p.position, p.nationality, p.draft_league)
            _fs_css_groups.setdefault(lst, []).append(p)
        else:
            quality_map_fs[p.id] = ppg_percentile.get(p.id, 0.5)
    for lst, grp in _fs_css_groups.items():
        sorted_grp = sorted(grp, key=lambda p: p.css_ranking)
        list_size = len(sorted_grp)
        for within_rank, p in enumerate(sorted_grp, start=1):
            quality_map_fs[p.id] = _css_norm_within_list(within_rank, list_size)

    now = datetime.now(timezone.utc)
    rows_processed = 0

    for p in prospects:
        # Build neutral GM features (profile=None → uniform priors)
        pf = _gm_features(None, p.position or "", p.draft_league or "", p.nationality or "")

        # Build contextual features (including pos_quality_rank_norm)
        ctx = _contextual_feats(
            p, prospects, 0, _team_id,
            pos_taken, _team_pos, tier_ppg_med, tier_age_med,
            quality_map=quality_map_fs,
        )

        css_norm = quality_map_fs[p.id]

        raw_row = {
            "position":          p.position,
            "nationality":       p.nationality,
            "height_cm":         p.height_cm,
            "weight_kg":         p.weight_kg,
            "draft_league":      p.draft_league,
            "draft_league_tier": p.draft_league_tier,
            "points_per_game":   p.points_per_game,
            "gp_pre_draft":      p.games_played,
            "ppg_prev_season":   p.ppg_prev_season,
            "has_prev_season":   1 if p.ppg_prev_season is not None else 0,
            "age_at_draft":      p.age_at_draft,
            "overall_pick":      1,      # neutral pick slot
            "draft_round":       1,
            "css_rank_norm":     css_norm,
            **pf,
            **ctx,
        }

        df_row    = pd.DataFrame([raw_row])
        feat_df   = build_features(df_row)
        feat_dict = feat_df.iloc[0].to_dict()

        # Convert numpy types to native Python for JSON serialization
        feat_json = {k: float(v) if hasattr(v, "item") else v for k, v in feat_dict.items()}

        # Tag each cached row with the feature schema version so get_cached_features()
        # can detect stale caches after a feature set change without a migration.
        feat_json["__schema_version__"] = FEATURE_SCHEMA_VERSION

        # Upsert into prospect_features
        existing = (
            db.query(ProspectFeatures)
            .filter(ProspectFeatures.prospect_id == p.id)
            .first()
        )
        if existing:
            existing.refreshed_at    = now
            existing.features_json   = feat_json
            existing.css_rank_norm   = feat_json.get("css_rank_norm")
            existing.ppg_league_norm = feat_json.get("ppg_league_norm")
            existing.pick_slot_norm  = feat_json.get("pick_slot_norm")
        else:
            db.add(ProspectFeatures(
                prospect_id     = p.id,
                refreshed_at    = now,
                features_json   = feat_json,
                css_rank_norm   = feat_json.get("css_rank_norm"),
                ppg_league_norm = feat_json.get("ppg_league_norm"),
                pick_slot_norm  = feat_json.get("pick_slot_norm"),
            ))

        rows_processed += 1

    db.commit()
    logger.info("feature_store.refreshed count=%d", rows_processed)
    return rows_processed


# ── Read path ──────────────────────────────────────────────────────────────────

def get_cached_features(
    db: Session,
    prospect_ids: list[int],
) -> Optional[pd.DataFrame]:
    """
    Load pre-computed feature rows from the store.

    Returns a DataFrame with FEATURE_COLS columns ordered by prospect_id,
    or None if the cache is stale / incomplete for the requested prospects.

    Staleness: any row older than CACHE_TTL_HOURS → cache miss for the whole batch.
    """
    from app.models.prospect_features import ProspectFeatures
    from app.ml.features import FEATURE_COLS

    if not prospect_ids:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(hours=CACHE_TTL_HOURS)

    rows = (
        db.query(ProspectFeatures)
        .filter(
            ProspectFeatures.prospect_id.in_(prospect_ids),
            ProspectFeatures.refreshed_at >= cutoff,
        )
        .all()
    )

    if len(rows) != len(prospect_ids):
        logger.debug(
            "feature_store.cache_miss requested=%d found=%d",
            len(prospect_ids), len(rows),
        )
        return None

    # Validate schema version — reject the whole batch if any row was written
    # with a different feature schema (catches deploys that changed FEATURE_COLS).
    for row in rows:
        cached_ver = (row.features_json or {}).get("__schema_version__")
        if cached_ver != FEATURE_SCHEMA_VERSION:
            logger.info(
                "feature_store.schema_version_mismatch cached=%s current=%s — forcing full rebuild",
                cached_ver, FEATURE_SCHEMA_VERSION,
            )
            return None

    # Reconstruct DataFrame in the same order as prospect_ids
    id_to_row = {r.prospect_id: r.features_json for r in rows}
    records = []
    for pid in prospect_ids:
        fj = id_to_row.get(pid)
        if fj is None:
            return None
        records.append(fj)

    df = pd.DataFrame(records)

    # Ensure all FEATURE_COLS are present (handle schema evolution gracefully)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df[FEATURE_COLS]
