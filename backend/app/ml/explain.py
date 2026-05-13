"""SHAP feature attribution for individual prospect pick predictions.

Uses TreeExplainer (fast, exact for XGBoost) to compute SHAP values for a
single (prospect, team, pick_number) combination. Returns the top-N features
by absolute impact, making model reasoning transparent.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def explain_prospect_pick(
    db,
    model,  # XGBClassifier
    prospect_id: int,
    team_id: int,
    pick_number: int = 1,
    top_n: int = 10,
) -> dict:
    """
    Compute SHAP values for a single (prospect, team, pick) combination.

    Returns:
        {
            "prospect_name": str,
            "team_abbreviation": str,
            "predicted_probability": float,
            "top_features": [
                {"feature": "position_C", "shap_value": 0.23, "feature_value": 1.0},
                ...
            ],
            "base_value": float,  # expected model output (log-odds)
        }
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "The 'shap' package is required for explainability. "
            "Install it with: pip install shap"
        ) from exc

    from app.ml.features import build_features, FEATURE_COLS
    from app.models import Prospect, Team, GMTendencyProfile
    from app.models.general_manager import GeneralManager

    # ── Fetch prospect ────────────────────────────────────────────────────────
    prospect = db.query(Prospect).filter(Prospect.id == prospect_id).first()
    if prospect is None:
        raise ValueError(f"Prospect {prospect_id} not found")

    # ── Fetch team ────────────────────────────────────────────────────────────
    team = db.query(Team).filter(Team.id == team_id).first()
    if team is None:
        raise ValueError(f"Team {team_id} not found")

    # ── Fetch GM tendency profile for this team ───────────────────────────────
    gm = (
        db.query(GeneralManager)
        .filter(GeneralManager.team_id == team_id, GeneralManager.is_active.is_(True))
        .first()
    )
    profile = None
    if gm is not None:
        profile = (
            db.query(GMTendencyProfile)
            .filter(GMTendencyProfile.gm_id == gm.id)
            .order_by(GMTendencyProfile.computed_at.desc())
            .first()
        )

    # ── Build GM tendency features ─────────────────────────────────────────────
    pos_weights = profile.position_weights or {} if profile else {}
    league_weights = profile.league_weights or {} if profile else {}
    nat_weights = profile.nationality_weights or {} if profile else {}

    from app.constants import infer_league_key, nat_group

    position = prospect.position or "C"
    draft_league = prospect.draft_league or ""
    nationality = prospect.nationality or ""

    gm_pos_weight = pos_weights.get(position, 0.2)
    gm_league_weight = league_weights.get(infer_league_key(draft_league), 0.2)
    gm_nat_weight = nat_weights.get(nat_group(nationality), 0.33)

    # ── Compute css_rank_norm from css_ranking ────────────────────────────────
    from app.ml.features import MAX_DRAFT_POOL
    css_rank_raw = prospect.css_ranking or 100
    # css_ranking is an ordinal rank (1=best); normalize so 1 -> ~1.0, 224 -> ~0.0
    css_rank_norm = 1.0 - (css_rank_raw - 1) / MAX_DRAFT_POOL
    css_rank_norm = float(max(0.0, min(1.0, css_rank_norm)))

    # ── Build the raw row dict ────────────────────────────────────────────────
    ppg = prospect.points_per_game or 0.0
    ppg_prev = prospect.ppg_prev_season  # may be None
    has_prev = 1 if ppg_prev is not None else 0

    row = {
        "position":              position,
        "nationality":           nationality,
        "height_cm":             prospect.height_cm,
        "weight_kg":             prospect.weight_kg,
        "draft_league":          draft_league,
        "draft_league_tier":     prospect.draft_league_tier,
        "points_per_game":       ppg,
        "age_at_draft":          prospect.age_at_draft,
        "css_rank_norm":         css_rank_norm,
        "overall_pick":          pick_number,
        "draft_round":           1,
        "gm_pos_weight":         gm_pos_weight,
        "gm_league_weight":      gm_league_weight,
        "gm_nat_weight":         gm_nat_weight,
        # Contextual features: neutral defaults for single-row explanation
        "ppg_league_norm":       1.0,
        "age_league_norm":       0.0,
        "pos_taken_before_norm": 0.2,
        "pos_remaining_norm":    0.2,
        "team_drafted_this_pos": 0,
        "pos_quality_rank_norm": 0.5,
        "gp_pre_draft":          prospect.games_played or 30,
        "ppg_prev_season":       ppg_prev if ppg_prev is not None else 0.0,
        "has_prev_season":       has_prev,
    }

    raw_df = pd.DataFrame([row])
    feature_df = build_features(raw_df)

    # ── Run SHAP ──────────────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)

    # XGBRanker returns a single output array; shap_values is never a list
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    base_value = float(explainer.expected_value)

    # ── Predicted score (XGBRanker has no predict_proba) ─────────────────────
    prob = float(model.predict(feature_df)[0])

    # ── Top-N features by absolute SHAP value ────────────────────────────────
    feature_names = list(feature_df.columns)
    feature_values = feature_df.values[0]

    indexed = sorted(
        enumerate(sv),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    top_features = [
        {
            "feature":       feature_names[i],
            "shap_value":    round(float(sv[i]), 6),
            "feature_value": round(float(feature_values[i]), 6),
        }
        for i, _ in indexed
    ]

    # ── SHAP consistency check: sum(SHAP) ≈ prediction - base_value ──────────
    shap_sum   = float(sv.sum())
    residual   = prob - base_value
    shap_error = abs(shap_sum - residual)
    if shap_error > 0.05:
        logger.warning(
            "shap.consistency_check_failed prospect=%s shap_sum=%.4f residual=%.4f error=%.4f",
            prospect.name, shap_sum, residual, shap_error,
        )

    return {
        "prospect_name":         prospect.name,
        "team_abbreviation":     team.abbreviation,
        "predicted_probability": round(prob, 6),
        "top_features":          top_features,
        "base_value":            round(base_value, 6),
        "shap_sum":              round(shap_sum, 6),
    }
