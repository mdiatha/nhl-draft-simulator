"""
Counterfactual "What-If" draft analysis.

Endpoint: POST /api/draft/counterfactual/swap

Given a completed lottery result, swap two pick positions (A and B) and
re-simulate both the original and counterfactual scenarios. Returns:
  - baseline picks (original lottery order)
  - counterfactual picks (with positions A and B swapped)
  - divergence: list of picks where the two scenarios chose different prospects

This lets users ask questions like:
  "What would Toronto have drafted if they had pick #3 instead of #8?"
  "How would Vegas' outcome change if they had moved up from #11 to #5?"
"""
from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Team, GeneralManager, GMTendencyProfile, Prospect2025
from app.ml.predict import score_pool_for_team, compute_pool_stats
from app.ml.registry import registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/draft/counterfactual", tags=["draft"])


class CounterfactualRequest(BaseModel):
    lottery_result: list[int]   # ordered team_ids (picks 1-32), same as SimulateDraftRequest
    swap_pick_a: int            # 1-based pick number
    swap_pick_b: int            # 1-based pick number (must differ from swap_pick_a)
    seed: Optional[int] = None
    temperature: float = 0.15

    @field_validator("swap_pick_a", "swap_pick_b")
    @classmethod
    def pick_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Pick numbers must be ≥ 1")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("temperature must be >= 0")
        return v


@router.post("/swap")
async def counterfactual_swap(body: CounterfactualRequest, db: Session = Depends(get_db)):
    """
    Simulate two drafts: baseline (original lottery) and counterfactual
    (picks swap_pick_a and swap_pick_b swapped), then highlight divergence.

    Returns the full pick-by-pick list for both scenarios plus a divergence
    list showing every pick slot where the two scenarios chose different prospects.
    """
    if not registry.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="No trained model loaded. Run POST /api/ml/train first.",
        )

    n = len(body.lottery_result)
    if body.swap_pick_a > n or body.swap_pick_b > n:
        raise HTTPException(
            status_code=400,
            detail=f"swap_pick_a/b must be ≤ number of picks in lottery_result ({n})",
        )
    if body.swap_pick_a == body.swap_pick_b:
        raise HTTPException(status_code=400, detail="swap_pick_a and swap_pick_b must differ")

    seed = body.seed if body.seed is not None else random.randint(0, 2**31)

    # ── Load shared data once ─────────────────────────────────────────────────
    prospects = (
        db.query(Prospect2025)
        .order_by(Prospect2025.css_ranking.nullslast())
        .all()
    )
    if not prospects:
        raise HTTPException(404, "No prospects loaded. Run ingestion first.")

    teams_map: dict[int, Team] = {t.id: t for t in db.query(Team).all()}
    gms_map: dict[int, GeneralManager] = {
        g.team_id: g
        for g in db.query(GeneralManager).filter(GeneralManager.is_active.is_(True)).all()
    }
    profiles_map: dict[int, GMTendencyProfile] = {
        p.gm_id: p for p in db.query(GMTendencyProfile).all()
    }
    pool_stats = compute_pool_stats(prospects)

    # ── Run baseline ──────────────────────────────────────────────────────────
    baseline_picks = _run_simulation(
        body.lottery_result, seed, body.temperature,
        prospects, teams_map, gms_map, profiles_map, pool_stats,
    )

    # ── Build counterfactual lottery: swap positions A and B ──────────────────
    cf_lottery = list(body.lottery_result)
    a_idx = body.swap_pick_a - 1
    b_idx = body.swap_pick_b - 1
    cf_lottery[a_idx], cf_lottery[b_idx] = cf_lottery[b_idx], cf_lottery[a_idx]

    cf_picks = _run_simulation(
        cf_lottery, seed, body.temperature,
        prospects, teams_map, gms_map, profiles_map, pool_stats,
    )

    # ── Compute divergence ────────────────────────────────────────────────────
    divergence = []
    for bl, cf in zip(baseline_picks, cf_picks):
        if bl["prospect_id"] != cf["prospect_id"]:
            divergence.append({
                "pick":               bl["pick"],
                "team_id":            bl["team_id"],
                "team_name":          bl["team_name"],
                "baseline_prospect":  bl["prospect_name"],
                "baseline_css_rank":  bl.get("css_rank"),
                "cf_prospect":        cf["prospect_name"],
                "cf_css_rank":        cf.get("css_rank"),
            })

    return {
        "seed":            seed,
        "swap_pick_a":     body.swap_pick_a,
        "swap_pick_b":     body.swap_pick_b,
        "baseline":        baseline_picks,
        "counterfactual":  cf_picks,
        "divergence":      divergence,
        "divergence_count": len(divergence),
    }


# ── Core simulation ────────────────────────────────────────────────────────────

def _run_simulation(
    lottery_result: list[int],
    seed: int,
    temperature: float,
    prospects: list,
    teams_map: dict,
    gms_map: dict,
    profiles_map: dict,
    pool_stats: dict,
) -> list[dict]:
    """
    Shared simulation loop used by both the baseline and counterfactual runs.
    Pure in-memory — no DB access inside the pick loop.
    """
    from app.api.draft_sim import _pick_temperature, _sample_pick

    rng       = random.Random(seed)
    available = list(prospects)
    picks     = []

    pos_taken: Counter[str]                          = Counter()
    team_positions_drafted: defaultdict[int, Counter] = defaultdict(Counter)

    for pick_num, team_id in enumerate(lottery_result, 1):
        if not available:
            break
        team = teams_map.get(team_id)
        if not team:
            continue

        gm      = gms_map.get(team_id)
        profile = profiles_map.get(gm.id) if gm else None

        draft_state = {
            "pos_taken":      pos_taken,
            "team_positions": team_positions_drafted[team_id],
            "total_picked":   pick_num - 1,
        }

        scores = score_pool_for_team(available, profile, pick_num, draft_state, pool_stats)

        effective_temp = _pick_temperature(temperature, pick_num, len(lottery_result))
        chosen = _sample_pick(available, scores, rng, effective_temp)

        pos_taken[chosen.position or "F"] += 1
        team_positions_drafted[team_id][chosen.position or "F"] += 1
        available = [p for p in available if p.id != chosen.id]

        picks.append({
            "pick":            pick_num,
            "team_id":         team_id,
            "team_name":       getattr(team, "full_name", team.abbreviation),
            "abbreviation":    team.abbreviation,
            "prospect_name":   chosen.name,
            "prospect_id":     chosen.id,
            "position":        chosen.position,
            "nationality":     chosen.nationality,
            "draft_league":    chosen.draft_league,
            "css_rank":        chosen.css_ranking,
            "points_per_game": chosen.points_per_game,
            "ml_score":        round(scores.get(chosen.id, 0.0), 4),
        })

    return picks
