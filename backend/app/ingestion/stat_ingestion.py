"""
Real-time prospect stat ingestion pipeline.

Fetches live pre-draft stats from the NHL API for active prospects that have an
`nhl_player_id`. Stores a timestamped row in `prospect_stat_history`, refreshes
the denormalized live columns on `prospects`, and persists a normalized season
record in `player_season_stats`.

Intended to run daily via:
  - EventBridge CronJob → Lambda handler (lambda_handler below)
  - Manual trigger: POST /api/admin/ingest/stats (admin-key protected)

The NHL API endpoint used:
  GET https://api-web.nhle.com/v1/player/{nhl_player_id}/game-log/now
  Returns an array of game log entries for the current season.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy.orm import Session

from app.constants import DRAFT_YEAR
from app.ingestion.normalized import upsert_player_season_stat

logger = logging.getLogger(__name__)

NHL_API_BASE    = "https://api-web.nhle.com/v1"
REQUEST_TIMEOUT = 10.0   # seconds per player request
MAX_PROSPECTS   = 500    # safety cap — avoid hammering the NHL API


# ── Core ingestion logic ──────────────────────────────────────────────────────

async def fetch_and_store_stats(db: Session) -> dict:
    """
    Fetch live stats for all active prospects with a known nhl_player_id.

    Returns a summary dict: {updated, skipped, errors, fetched_at}
    """
    from app.models import Prospect
    from app.models.prospect_stat_history import ProspectStatHistory

    prospects_with_id = (
        db.query(Prospect)
        .filter(
            Prospect.nhl_player_id.isnot(None),
            Prospect.is_active.is_(True),
        )
        .limit(MAX_PROSPECTS)
        .all()
    )

    if not prospects_with_id:
        logger.info("stat_ingestion.no_prospects_with_player_id")
        return {"updated": 0, "skipped": 0, "errors": 0, "fetched_at": datetime.now(timezone.utc).isoformat()}

    updated = 0
    skipped = 0
    errors  = 0
    fetched_at = datetime.now(timezone.utc)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for prospect in prospects_with_id:
            try:
                stats = await _fetch_player_stats(client, prospect.nhl_player_id)
                if stats is None:
                    skipped += 1
                    continue

                # Upsert into stat history
                history_row = ProspectStatHistory(
                    prospect_id     = prospect.id,
                    fetched_at      = fetched_at,
                    season_type     = stats.get("season_type", "pre_draft"),
                    league          = stats.get("league"),
                    games_played    = stats.get("games_played"),
                    goals           = stats.get("goals"),
                    assists         = stats.get("assists"),
                    points          = stats.get("points"),
                    points_per_game = stats.get("points_per_game"),
                    source_url      = f"{NHL_API_BASE}/player/{prospect.nhl_player_id}/game-log/now",
                    raw_payload     = stats.get("raw"),
                )
                db.add(history_row)

                # Update live columns on the prospect
                if stats.get("points_per_game") is not None:
                    prospect.points_per_game = stats["points_per_game"]
                if stats.get("games_played") is not None:
                    prospect.games_played = stats["games_played"]
                if stats.get("goals") is not None:
                    prospect.goals = stats["goals"]
                if stats.get("assists") is not None:
                    prospect.assists = stats["assists"]

                if prospect.player is not None:
                    upsert_player_season_stat(
                        db,
                        player=prospect.player,
                        season_year_start=DRAFT_YEAR - 1,
                        season_year_end=DRAFT_YEAR,
                        league=stats.get("league"),
                        team_name=stats.get("team_name"),
                        games_played=stats.get("games_played"),
                        goals=stats.get("goals"),
                        assists=stats.get("assists"),
                        points=stats.get("points"),
                        points_per_game=stats.get("points_per_game"),
                        season_type=stats.get("season_type", "pre_draft"),
                        source="nhl_game_log",
                        as_of_date=fetched_at.date(),
                        raw_payload=stats.get("raw"),
                    )

                updated += 1

            except Exception as exc:
                logger.error(
                    "stat_ingestion.fetch_failed player_id=%s error=%s",
                    prospect.nhl_player_id, exc,
                )
                errors += 1

    db.commit()
    logger.info(
        "stat_ingestion.complete updated=%d skipped=%d errors=%d",
        updated, skipped, errors,
    )
    return {
        "updated":    updated,
        "skipped":    skipped,
        "errors":     errors,
        "total":      len(prospects_with_id),
        "fetched_at": fetched_at.isoformat(),
    }


# ── NHL API client ────────────────────────────────────────────────────────────

async def _fetch_player_stats(client: httpx.AsyncClient, nhl_player_id: int) -> Optional[dict]:
    """
    Fetch current-season game log for a player from the NHL web API.
    Returns a normalized stats dict, or None if no data.
    """
    url = f"{NHL_API_BASE}/player/{nhl_player_id}/game-log/now"
    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            logger.debug("stat_ingestion.player_not_found player_id=%d", nhl_player_id)
            return None
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("stat_ingestion.http_error player_id=%d status=%d", nhl_player_id, exc.response.status_code)
        return None

    # The NHL API returns {"gameLog": [...], "playerStatsSeasons": [...]}
    game_log = data.get("gameLog") or []
    if not game_log:
        return None

    # Aggregate totals across all games in the log
    total_gp = len(game_log)
    total_g  = sum(g.get("goals",   0) or 0 for g in game_log)
    total_a  = sum(g.get("assists", 0) or 0 for g in game_log)
    total_p  = total_g + total_a
    ppg      = round(total_p / total_gp, 4) if total_gp > 0 else 0.0

    # Infer season type from gameType field (2=regular, 3=playoffs, 1=preseason)
    game_types = {g.get("gameTypeId") for g in game_log}
    season_type = "pre_draft"
    if 2 in game_types:
        season_type = "regular"
    elif 3 in game_types:
        season_type = "playoffs"

    return {
        "games_played":    total_gp,
        "goals":           total_g,
        "assists":         total_a,
        "points":          total_p,
        "points_per_game": ppg,
        "season_type":     season_type,
        "league":          game_log[0].get("leagueAbbrev") if game_log else None,
        "team_name":       game_log[0].get("teamAbbrev") if game_log else None,
        "raw":             {"game_count": total_gp, "source": "nhl_game_log"},
    }


# ── Lambda entry point ────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda handler — triggered by EventBridge CronJob (daily at 06:00 UTC).
    Creates a DB session, runs the ingestion, returns a status dict.
    """
    import asyncio
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        result = asyncio.run(fetch_and_store_stats(db))
        return {"statusCode": 200, "body": result}
    except Exception as exc:
        logger.exception("stat_ingestion.lambda_failed error=%s", exc)
        return {"statusCode": 500, "body": {"error": str(exc)}}
    finally:
        db.close()
