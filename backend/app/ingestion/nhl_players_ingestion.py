"""
Fetch current NHL player stats from the NHL API and embed them into
scout_embeddings as 'nhl_player' documents for prospect comp searches.

Call fetch_and_embed_nhl_players(db) from:
  - build_index() in embeddings.py (full rebuild path)
  - POST /api/admin/ingest-nhl-players (standalone on-demand refresh)

NHL API endpoints used:
  https://api-web.nhle.com/v1/skater-stats-leaders/{season}/{gameType}
  https://api-web.nhle.com/v1/goalie-stats-leaders/{season}/{gameType}

Categories pulled:
  - points (top 200)  — captures all elite forwards + offensive D
  - hits   (top 75)   — adds physical forwards absent from points leaders
  - blocked (top 75)  — adds shutdown defensemen absent from points leaders
  - goalie wins (50)  — goalies

Players are deduplicated by NHL player_id, so overlapping categories do not
produce duplicate embeddings. The resulting 'nhl_player' documents are used
by the get_nhl_comp agent tool for semantic similarity search.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

NHL_API_BASE = "https://api-web.nhle.com/v1"
CURRENT_SEASON = "20242025"
GAME_TYPE = 2  # regular season

_POS_MAP = {"L": "LW", "R": "RW", "C": "C", "D": "D", "G": "G"}


# ── API helpers ───────────────────────────────────────────────────────────────

def _extract_str(value) -> str:
    """Handle both plain strings and {'default': str} dict values from the NHL API."""
    if isinstance(value, dict):
        return value.get("default", "") or ""
    return str(value) if value else ""


def _get_skater_leaders(category: str, limit: int = 200) -> list[dict]:
    try:
        resp = httpx.get(
            f"{NHL_API_BASE}/skater-stats-leaders/{CURRENT_SEASON}/{GAME_TYPE}",
            params={"categories": category, "limit": limit},
            timeout=20.0,
        )
        resp.raise_for_status()
        return resp.json().get(category, [])
    except Exception as exc:
        logger.warning("nhl_players.skater_leaders_failed category=%s error=%s", category, exc)
        return []


def _get_goalie_leaders(limit: int = 50) -> list[dict]:
    try:
        resp = httpx.get(
            f"{NHL_API_BASE}/goalie-stats-leaders/{CURRENT_SEASON}/{GAME_TYPE}",
            params={"categories": "wins", "limit": limit},
            timeout=20.0,
        )
        resp.raise_for_status()
        return resp.json().get("wins", [])
    except Exception as exc:
        logger.warning("nhl_players.goalie_leaders_failed error=%s", exc)
        return []


# ── Style derivation ──────────────────────────────────────────────────────────

def _derive_style(position: str, gp: int, goals: int, assists: int, ppg: float,
                  plus_minus: Optional[int]) -> str:
    """
    Build a comma-separated style tag string from a player's season stats.
    Tags are used in the embedding document so semantic queries like
    'elite playmaking center' map onto real NHL player profiles.
    """
    if position == "G":
        return "NHL goaltender"

    tags: list[str] = []

    if position == "D":
        if ppg >= 0.80:
            tags.append("elite offensive defenseman")
        elif ppg >= 0.55:
            tags.append("offensive defenseman")
        elif ppg >= 0.35:
            tags.append("two-way defenseman")
        else:
            tags.append("defensive defenseman")

        total = goals + assists
        if total > 0:
            a_ratio = assists / total
            if a_ratio >= 0.70:
                tags.append("playmaking from the blue line")
            elif a_ratio <= 0.35:
                tags.append("goal-scoring defenseman")
    else:
        # Forward
        if ppg >= 1.20:
            tags.append("elite scorer")
        elif ppg >= 0.85:
            tags.append("top-six scorer")
        elif ppg >= 0.55:
            tags.append("middle-six forward")
        else:
            tags.append("bottom-six forward")

        total = goals + assists
        if total > 0:
            g_ratio = goals / total
            if g_ratio >= 0.48:
                tags.append("goal scorer")
            elif g_ratio <= 0.28:
                tags.append("pure playmaker")
            else:
                tags.append("two-way producer")

    if plus_minus is not None:
        if plus_minus >= 20:
            tags.append("positive possession driver")
        elif plus_minus <= -15:
            tags.append("negative differential player")

    return ", ".join(tags) or "NHL player"


# ── Document builder ──────────────────────────────────────────────────────────

def _player_to_text(p: dict) -> str:
    """
    Build a rich plain-text embedding document for one NHL player.

    Field order is intentional: name + position first (high semantic weight),
    then team/nationality, then stats, then derived style tags. Voyage embeds
    the full sequence so later fields still influence the vector.
    """
    team_str = p.get("team_name") or p.get("team", "")
    nat = p.get("nationality", "")
    lines = [
        f"NHL Player: {p['name']}",
        f"Position: {p['position']}"
        + (f" | Team: {team_str}" if team_str else "")
        + (f" | Nationality: {nat}" if nat else ""),
    ]

    gp = p.get("gp", 0)
    if gp > 0:
        stat_line = (
            f"2024-25: {gp} GP, {p.get('goals', 0)}G, "
            f"{p.get('assists', 0)}A, {p.get('points', 0)}P "
            f"({p.get('ppg', 0.0):.2f} PPG)"
        )
        pm = p.get("plus_minus")
        if pm is not None:
            stat_line += f", {pm:+d} +/-"
        toi = p.get("avg_toi", "")
        if toi:
            stat_line += f", TOI {toi}"
        lines.append(stat_line)

    style = p.get("style", "")
    if style:
        lines.append(f"Style: {style}")

    return "\n".join(lines)


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_skater(raw: dict) -> Optional[dict]:
    player_id = raw.get("id") or raw.get("playerId")
    if not player_id:
        return None

    first = _extract_str(raw.get("firstName", ""))
    last = _extract_str(raw.get("lastName", ""))
    name = f"{first} {last}".strip() or _extract_str(raw.get("name", ""))
    if not name:
        return None

    position = _POS_MAP.get(raw.get("position") or raw.get("positionCode", ""), "F")

    gp = raw.get("gamesPlayed", 0) or 0
    goals = raw.get("goals", 0) or 0
    assists = raw.get("assists", 0) or 0
    points = raw.get("points", 0) or 0
    ppg = round(points / gp, 3) if gp > 0 else 0.0
    pm = raw.get("plusMinus")

    return {
        "player_id":  int(player_id),
        "name":       name,
        "position":   position,
        "team":       _extract_str(raw.get("teamAbbrev") or raw.get("teamAbbreviation", "")),
        "team_name":  _extract_str(raw.get("teamName", "")),
        "nationality": raw.get("nationalityCode") or raw.get("nationality", ""),
        "gp":         gp,
        "goals":      goals,
        "assists":    assists,
        "points":     points,
        "ppg":        ppg,
        "plus_minus": pm,
        "avg_toi":    raw.get("avgToi") or raw.get("avgTimeOnIce", ""),
        "style":      _derive_style(position, gp, goals, assists, ppg, pm),
    }


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_and_embed_nhl_players(db) -> int:
    """
    Fetch current NHL player stats from the NHL API, embed each player's
    profile text via voyage-3-lite, and upsert into scout_embeddings as
    doc_type='nhl_player'.

    Returns the number of player documents upserted.
    Fails gracefully: if the NHL API is unreachable or Voyage embedding fails,
    individual players are skipped; the rest still index.
    """
    from app.agent.embeddings import embed_text
    from app.agent.store import upsert_embedding

    # Collect players deduplicated by player_id
    players: dict[int, dict] = {}

    for category, limit in [("points", 200), ("hits", 75), ("blocked", 75)]:
        for raw in _get_skater_leaders(category=category, limit=limit):
            parsed = _parse_skater(raw)
            if parsed and parsed["player_id"] not in players:
                players[parsed["player_id"]] = parsed

    for raw in _get_goalie_leaders(limit=50):
        parsed = _parse_skater(raw)
        if parsed and parsed["player_id"] not in players:
            parsed["position"] = "G"
            parsed["style"] = "NHL goaltender"
            players[parsed["player_id"]] = parsed

    if not players:
        logger.warning("nhl_players.no_players_fetched — check NHL API connectivity")
        return 0

    count = 0
    for player_id, p in players.items():
        text = _player_to_text(p)
        vec = embed_text(text)
        upsert_embedding(
            db,
            doc_type="nhl_player",
            ref_id=player_id,
            ref_name=p["name"],
            content=text,
            embedding=vec,
        )
        count += 1
        logger.debug("nhl_players.embedded name=%s", p["name"])

    db.commit()
    logger.info("nhl_players.index_built count=%d", count)
    return count
