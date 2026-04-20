"""
Live standings endpoint.

Fetches current NHL standings from the NHL web API, determines which teams
are on track to miss the playoffs, assigns lottery odds based on current
points ranking among non-playoff teams, and returns a combined payload
ready for the Tankathon-style standings table.

Lottery eligibility: bottom 16 teams by points (ties broken by ROW, then GP).
Odds are the official NHL lottery odds table (fixed by the CBA, not points).

Results are cached in Redis for 5 minutes so the frontend can poll freely.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.constants import LOTTERY_ODDS
from app.database import get_db, get_redis
from app.models import Team

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/standings", tags=["standings"])

NHL_STANDINGS_URL = "https://api-web.nhle.com/v1/standings/now"
CACHE_TTL = 300  # 5 minutes

_redis, _REDIS_OK = get_redis()


def _fetch_standings() -> list[dict]:
    """Pull raw standings from the NHL API."""
    with httpx.Client(timeout=10, follow_redirects=True) as client:
        resp = client.get(NHL_STANDINGS_URL)
        resp.raise_for_status()
        data = resp.json()
    return data.get("standings", [])


def _parse_team(entry: dict) -> dict:
    """Extract the fields we care about from one standings entry."""
    abbrev_raw = entry.get("teamAbbrev", {})
    abbrev = abbrev_raw.get("default", "") if isinstance(abbrev_raw, dict) else str(abbrev_raw)

    name_raw = entry.get("teamName", {})
    name = name_raw.get("default", abbrev) if isinstance(name_raw, dict) else str(name_raw)

    place_raw = entry.get("placeName", {})
    city = place_raw.get("default", "") if isinstance(place_raw, dict) else str(place_raw)

    return {
        "nhl_id":        entry.get("teamId"),
        "abbreviation":  abbrev,
        "team_name":     name,
        "city":          city,
        "conference":    entry.get("conferenceName", ""),
        "division":      entry.get("divisionName", ""),
        "wins":          entry.get("wins", 0),
        "losses":        entry.get("losses", 0),
        "otl":           entry.get("otLosses", 0),
        "games_played":  entry.get("gamesPlayed", 0),
        "points":        entry.get("points", 0),
        "row":               entry.get("regulationAndOtWins", 0),  # ROW tiebreaker
        "goal_diff":         entry.get("goalDifferential", 0),
        # Sequence fields for playoff determination
        "division_sequence":  entry.get("divisionSequence", 99),
        "wildcard_sequence":  entry.get("wildcardSequence", 99),
        "conference_sequence": entry.get("conferenceSequence", 99),
        # Playoff clinch/elimination status: "x"=clinched, "y"=div, "z"=pres, "e"=eliminated
        "playoff_status":    entry.get("clinchIndicator", ""),
    }


def _is_playoff_team(entry: dict) -> bool:
    """
    Determine playoff qualification using real NHL wild-card rules:

    Each conference (East / West) has 2 divisions of 8 teams each.
    Playoff spots per conference (8 total):
      - Top 3 in each division  → 6 spots  (divisionSequence 1, 2, 3)
      - Top 2 wild-card teams   → 2 spots  (wildcardSequence 1, 2;
                                             excludes teams already in top-3)

    The API sets wildcardSequence=0 for teams already clinched via division top-3,
    so we only need divisionSequence <= 3 OR wildcardSequence in {1, 2}.
    """
    div_seq = entry.get("division_sequence", 99)
    wc_seq  = entry.get("wildcard_sequence", 99)
    return div_seq <= 3 or wc_seq in (1, 2)


def _build_standings_payload(raw: list[dict]) -> dict:
    """
    Given raw NHL API standings, determine playoff qualification per the real
    NHL wild-card format, assign lottery slots and odds to non-playoff teams,
    and return a combined payload.

    Playoff format (16 teams qualify):
      Eastern Conference: top 3 Atlantic + top 3 Metropolitan + 2 East wild cards
      Western Conference: top 3 Central + top 3 Pacific   + 2 West wild cards

    Lottery: all non-playoff teams, seeded worst-to-best by points
    (ties: fewer ROW → worse seed; more GP played → worse seed).
    Only 2 lottery draws per current NHL rules (changed from 3 in 2021).
    """
    teams = [_parse_team(e) for e in raw]

    # Mark playoff / non-playoff
    for team in teams:
        team["in_playoffs"] = _is_playoff_team(team)

    # Sort all 32 by points (for overall_rank display)
    teams.sort(key=lambda t: (-t["points"], -t["row"], t["games_played"]))
    for i, team in enumerate(teams):
        team["overall_rank"] = i + 1

    # Lottery seeding: non-playoff teams sorted worst → best
    # Worst = fewest points; tiebreak: fewer ROW; tiebreak: more GP played
    non_playoff = [t for t in teams if not t["in_playoffs"]]
    non_playoff.sort(key=lambda t: (t["points"], t["row"], -t["games_played"]))

    for slot, team in enumerate(non_playoff, 1):
        team["lottery_slot"]      = slot
        team["lottery_odds_pct"]  = LOTTERY_ODDS.get(slot, 0.0)

    for team in teams:
        if team["in_playoffs"]:
            team["lottery_slot"]     = None
            team["lottery_odds_pct"] = 0.0

    return {
        "standings":          teams,
        "non_playoff_count":  len(non_playoff),
        "lottery_draws":      2,   # NHL changed from 3 to 2 draws in 2021
        "fetched_at":         int(time.time()),
    }


@router.get("/live")
async def live_standings(db: Session = Depends(get_db)):
    """
    Return current NHL standings with live lottery odds.

    Non-playoff teams are identified by being in the bottom 16 by points.
    Lottery odds follow the official NHL table (worst record = 18.5%).
    Results are cached for 5 minutes.

    Shape of each team object:
      abbreviation, team_name, conference, division,
      wins, losses, otl, games_played, points, row,
      overall_rank, in_playoffs,
      lottery_slot (1-16 or null), lottery_odds_pct,
      team_id (DB primary key, for lottery simulate)
    """
    cache_key = "standings:live"
    if _REDIS_OK and _redis:
        cached = _redis.get(cache_key)
        if cached:
            return JSONResponse(json.loads(cached))

    try:
        raw = _fetch_standings()
    except httpx.HTTPError as exc:
        logger.error("standings.fetch_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=502, detail="Could not reach NHL API. Try again shortly.")

    payload = _build_standings_payload(raw)

    # Enrich each team with the DB primary key (team.id) matched by abbreviation.
    # The frontend needs this to pass team_id to the lottery simulate endpoint.
    db_teams: dict[str, int] = {
        t.abbreviation: t.id
        for t in db.query(Team.abbreviation, Team.id).all()
    }
    for team in payload["standings"]:
        team["team_id"] = db_teams.get(team["abbreviation"])

    if _REDIS_OK and _redis:
        _redis.setex(cache_key, CACHE_TTL, json.dumps(payload))

    return payload
