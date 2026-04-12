"""Lottery API endpoints."""
import hashlib
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db, get_redis
from app.models import Team, LotteryOdds, GeneralManager, GMTendencyProfile
from app.engines.lottery_engine import draw_lottery, simulate_lottery_n_times
from app.constants import DRAFT_YEAR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lottery", tags=["lottery"])

_redis, REDIS_OK = get_redis()

CACHE_TTL = 3600


class SimulateLotteryRequest(BaseModel):
    seed: Optional[int] = None
    teams: Optional[list["LotterySimulationTeam"]] = None


class LotterySimulationTeam(BaseModel):
    team_id: int
    team_name: str
    abbreviation: str
    odds_pct: Optional[float] = None
    standing: Optional[int] = None
    overall_rank: Optional[int] = None
    in_playoffs: Optional[bool] = None
    lottery_slot: Optional[int] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    otl: Optional[int] = None
    points: Optional[int] = None


class SimulateManyRequest(BaseModel):
    n: int = Field(default=1000, ge=1, le=10000)
    seed: Optional[int] = None


def _get_lottery_teams(db: Session) -> list[dict]:
    season = DRAFT_YEAR
    records = (
        db.query(LotteryOdds)
        .filter(LotteryOdds.season == season)
        .order_by(LotteryOdds.final_standing)
        .all()
    )
    if not records:
        return []
    team_ids = [o.team_id for o in records]
    teams_by_id = {
        t.id: t for t in db.query(Team).filter(Team.id.in_(team_ids)).all()
    }
    result = []
    for o in records:
        team = teams_by_id.get(o.team_id)
        if team:
            result.append({
                "team_id": team.id,
                "odds_pct": o.lottery_odds_pct or 0,
                "standing": o.final_standing or 0,
                "team_name": getattr(team, "name", None) or getattr(team, "full_name", ""),
                "abbreviation": team.abbreviation,
            })
    return result


def _custom_lottery_teams(teams: list[LotterySimulationTeam]) -> list[dict]:
    lottery_teams = []
    for team in teams:
        is_lottery_team = team.lottery_slot is not None or team.in_playoffs is False
        if not is_lottery_team:
            continue
        standing = team.standing if team.standing is not None else team.lottery_slot
        if standing is None:
            continue
        lottery_teams.append({
            "team_id": team.team_id,
            "odds_pct": team.odds_pct or 0,
            "standing": standing,
            "team_name": team.team_name,
            "abbreviation": team.abbreviation,
            "wins": team.wins,
            "losses": team.losses,
            "otl": team.otl,
            "points": team.points,
            "overall_rank": team.overall_rank,
            "in_playoffs": team.in_playoffs,
        })
    return sorted(lottery_teams, key=lambda t: t["standing"])


def _custom_teams_cache_key(seed: Optional[int], teams: list[LotterySimulationTeam]) -> str:
    payload = [
        {
            "team_id": team.team_id,
            "odds_pct": team.odds_pct,
            "standing": team.standing,
            "overall_rank": team.overall_rank,
            "in_playoffs": team.in_playoffs,
            "lottery_slot": team.lottery_slot,
            "points": team.points,
        }
        for team in sorted(teams, key=lambda t: t.team_id)
    ]
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"lottery:{seed}:custom:{digest}"


@router.post("/simulate")
async def simulate_lottery(body: SimulateLotteryRequest, db: Session = Depends(get_db)):
    cache_key = (
        _custom_teams_cache_key(body.seed, body.teams)
        if body.teams
        else f"lottery:{body.seed}"
    )
    if REDIS_OK and _redis:
        cached = _redis.get(cache_key)
        if cached:
            return json.loads(cached)

    teams = _custom_lottery_teams(body.teams) if body.teams else _get_lottery_teams(db)
    if not teams:
        raise HTTPException(404, "No lottery data. Run ingestion first.")

    pick_order = draw_lottery(teams, seed=body.seed)

    # Append remaining playoff teams (picks 17-32) ordered best→worst standing
    lottery_team_id_set = {t["team_id"] for t in teams}
    if body.teams:
        custom_teams = [team.model_dump() for team in body.teams]
        playoff_teams = [t for t in custom_teams if t["team_id"] not in lottery_team_id_set]
        playoff_team_ids = [
            t["team_id"]
            for t in sorted(
                playoff_teams,
                key=lambda t: (
                    t.get("overall_rank") is None,
                    -(t.get("overall_rank") or 0),
                ),
            )
        ]
        all_team_ids = list(pick_order + playoff_team_ids)
        teams_map = {t["team_id"]: t for t in custom_teams}
    else:
        all_teams = db.query(Team).all()
        playoff_teams = [
            t for t in all_teams if t.id not in lottery_team_id_set
        ]
        # Sort playoff teams: we don't have their exact standings here, so sort by id as proxy
        # (teams were inserted in standings order during ingestion)
        playoff_team_ids = [t.id for t in sorted(playoff_teams, key=lambda t: t.id, reverse=True)]
        all_team_ids = list(pick_order + playoff_team_ids)
        teams_map = {t.id: t for t in db.query(Team).filter(Team.id.in_(all_team_ids)).all()}
    full_order = pick_order + playoff_team_ids

    gms_map = {
        gm.team_id: gm
        for gm in db.query(GeneralManager).filter(
            GeneralManager.team_id.in_(all_team_ids),
            GeneralManager.is_active.is_(True),
        ).all()
    }
    odds_map = {t["team_id"]: t for t in teams}

    picks = []
    for pos, team_id in enumerate(full_order, 1):
        team = teams_map.get(team_id)
        gm = gms_map.get(team_id)
        odds_entry = odds_map.get(team_id, {})
        if isinstance(team, dict):
            team_name = team.get("team_name", "Unknown")
            abbreviation = team.get("abbreviation", "???")
            wins = team.get("wins")
            losses = team.get("losses")
            otl = team.get("otl")
            points = team.get("points")
        else:
            team_name = getattr(team, "name", None) or getattr(team, "full_name", "") if team else "Unknown"
            abbreviation = team.abbreviation if team else "???"
            wins = None
            losses = None
            otl = None
            points = None
        picks.append({
            "pick": pos,
            "team_id": team_id,
            "team_name": team_name,
            "abbreviation": abbreviation,
            "gm_name": gm.name if gm else None,
            "original_standing": odds_entry.get("standing"),
            "lottery_odds_pct": odds_entry.get("odds_pct"),
            "wins": wins,
            "losses": losses,
            "otl": otl,
            "points": points,
        })

    response = {"pick_order": picks, "seed": body.seed}
    if REDIS_OK and _redis:
        _redis.setex(cache_key, CACHE_TTL, json.dumps(response))
    return response


@router.get("/odds")
async def get_lottery_odds(db: Session = Depends(get_db)):
    season = DRAFT_YEAR
    records = (
        db.query(LotteryOdds)
        .filter(LotteryOdds.season == season)
        .order_by(LotteryOdds.final_standing)
        .all()
    )
    if not records:
        return {"teams": [], "season": season, "message": "No lottery data. Run ingestion first."}

    rec_team_ids = [o.team_id for o in records]
    teams_map = {t.id: t for t in db.query(Team).filter(Team.id.in_(rec_team_ids)).all()}
    gms_map = {
        gm.team_id: gm
        for gm in db.query(GeneralManager).filter(
            GeneralManager.team_id.in_(rec_team_ids),
            GeneralManager.is_active.is_(True),
        ).all()
    }
    gm_ids = [gm.id for gm in gms_map.values()]
    profiles_map = {
        p.gm_id: p
        for p in db.query(GMTendencyProfile).filter(GMTendencyProfile.gm_id.in_(gm_ids)).all()
    }

    result = []
    for o in records:
        team = teams_map.get(o.team_id)
        if not team:
            continue
        gm = gms_map.get(o.team_id)
        archetype = profiles_map[gm.id].tendency_archetype if gm and gm.id in profiles_map else None

        result.append({
            "team_id": team.id,
            "team_name": getattr(team, "name", None) or getattr(team, "full_name", ""),
            "abbreviation": team.abbreviation,
            "city": getattr(team, "city", None),
            "conference": team.conference,
            "division": team.division,
            "final_standing": o.final_standing,
            "lottery_odds_pct": o.lottery_odds_pct,
            "combinations_count": len(o.assigned_combinations or []),
            "gm_name": gm.name if gm else getattr(team, "current_gm_name", None),
            "gm_archetype": archetype,
            "logo_url": f"/team-logos/{team.abbreviation.lower()}.svg",
            "wins": o.wins,
            "losses": o.losses,
            "otl": o.otl,
            "points": o.points,
        })
    return {"teams": result, "season": season}


@router.post("/simulate-many")
async def simulate_lottery_many(body: SimulateManyRequest, db: Session = Depends(get_db)):
    teams = _get_lottery_teams(db)
    if not teams:
        raise HTTPException(404, "No lottery data.")

    probs = simulate_lottery_n_times(teams, n=body.n, seed=body.seed)

    sim_team_ids = [t["team_id"] for t in teams]
    teams_map = {t.id: t for t in db.query(Team).filter(Team.id.in_(sim_team_ids)).all()}

    result = []
    for t in teams:
        team = teams_map.get(t["team_id"])
        result.append({
            "team_id": t["team_id"],
            "team_name": getattr(team, "name", None) or getattr(team, "full_name", "") if team else "?",
            "abbreviation": team.abbreviation if team else "???",
            "original_standing": t["standing"],
            "lottery_odds_pct": t["odds_pct"],
            "pick_probabilities": probs.get(t["team_id"], []),
        })
    return {"simulations": body.n, "seed": body.seed, "results": result}
