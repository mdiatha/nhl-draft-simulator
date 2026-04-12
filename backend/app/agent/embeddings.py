"""
Embedding utilities for the Scout RAG agent.

Uses Voyage AI's voyage-3-lite embedding model to embed text chunks.
Falls back gracefully when the API key is not set.

Indexing strategy:
  - One document per GM tendency profile (position/league/nationality weights + archetype)
  - Multiple documents per prospect in the 2025 draft class:
      * profile chunk (facts + size + league context)
      * style chunk (derived role/archetype tags)
      * trend chunk (recent production trajectory + stat history)
  - One document per team (current GM, archetype, top positional needs)

Documents are stored in scout_embeddings and retrieved by cosine similarity
at query time using pgvector's <=> operator.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

EMBED_DIM = 1024  # voyage-3-lite output dimension


def embed_text(text: str) -> Optional[list[float]]:
    """Return a 1024-dim embedding for *text* using voyage-3-lite."""
    from app.config import settings
    if not settings.VOYAGE_API_KEY:
        logger.warning("embed.skipped", extra={"reason": "VOYAGE_API_KEY not set"})
        return None
    return _embed_via_voyage(text)


def _embed_via_voyage(text: str) -> Optional[list[float]]:
    """Call Voyage AI REST API directly for embeddings."""
    from app.config import settings
    try:
        import httpx
        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.VOYAGE_API_KEY}"},
            json={"model": "voyage-3-lite", "input": [text]},
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as exc:
        logger.warning("embed.voyage_failed", extra={"error": str(exc)})
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Document builders ─────────────────────────────────────────────────────────

def gm_profile_to_text(gm_name: str, team_name: str, profile) -> str:
    """Render a GMTendencyProfile ORM object to a plain-text document."""
    pos = profile.position_weights or {}
    leagues = profile.league_weights or {}
    nations = profile.nationality_weights or {}

    top_pos = sorted(pos.items(), key=lambda x: -x[1])[:3]
    top_leagues = sorted(leagues.items(), key=lambda x: -x[1])[:3]
    top_nations = sorted(nations.items(), key=lambda x: -x[1])[:2]

    lines = [
        f"GM: {gm_name} | Team: {team_name}",
        f"Archetype: {profile.tendency_archetype or 'unknown'}",
        f"Top positions drafted: {', '.join(f'{p}({w:.0%})' for p,w in top_pos)}",
        f"Preferred leagues: {', '.join(f'{l}({w:.0%})' for l,w in top_leagues)}",
        f"Nationality tendencies: {', '.join(f'{n}({w:.0%})' for n,w in top_nations)}",
    ]
    if profile.avg_ranking_deviation is not None:
        lines.append(f"Avg CSS rank deviation: {profile.avg_ranking_deviation:+.1f} picks")
    return "\n".join(lines)


def prospect_to_text(prospect) -> str:
    """Render a Prospect2025 ORM object to a plain-text document."""
    return prospect_profile_to_text(prospect)


def _format_css_rank(rank: Optional[int]) -> str:
    return str(rank) if rank is not None else "unranked"


def _league_context_label(tier: Optional[int]) -> str:
    if tier == 1:
        return "top-tier pro or elite junior competition"
    if tier == 2:
        return "strong junior or college competition"
    if tier == 3:
        return "development league competition"
    return "unclear competition level"


def _size_profile(height_cm: Optional[int], weight_kg: Optional[int]) -> str:
    descriptors: list[str] = []
    if height_cm is not None:
        if height_cm >= 191:
            descriptors.append("pro-size frame")
        elif height_cm <= 178:
            descriptors.append("undersized frame")
        else:
            descriptors.append("average frame")
    if weight_kg is not None:
        if weight_kg >= 90:
            descriptors.append("strong build")
        elif weight_kg <= 76:
            descriptors.append("lighter build")
    return ", ".join(descriptors) if descriptors else "size not available"


def _age_context(age_at_draft: Optional[float]) -> str:
    if age_at_draft is None:
        return "age context unavailable"
    if age_at_draft <= 18.2:
        return "young for the class"
    if age_at_draft >= 18.8:
        return "older for the class"
    return "average draft-age profile"


def _goal_assist_split(goals: Optional[int], assists: Optional[int]) -> Optional[str]:
    if goals is None or assists is None:
        return None
    total = goals + assists
    if total <= 0:
        return None
    goal_share = goals / total
    if goal_share >= 0.48:
        return "goal-driven production"
    if goal_share <= 0.28:
        return "assist-heavy playmaking profile"
    return "balanced scoring profile"


def _style_tags_for_prospect(prospect) -> list[str]:
    position = (prospect.position or "").upper()
    ppg = prospect.points_per_game or 0.0
    tags: list[str] = []

    if position == "G":
        tags.append("goaltending prospect")
    elif position == "D":
        if ppg >= 0.75:
            tags.append("offensive defenseman")
        elif ppg >= 0.45:
            tags.append("two-way defenseman")
        else:
            tags.append("defensive defenseman")
    else:
        if ppg >= 1.30:
            tags.append("high-end scoring forward")
        elif ppg >= 0.95:
            tags.append("top-six scoring forward")
        elif ppg >= 0.65:
            tags.append("middle-six forward")
        else:
            tags.append("depth forward profile")

    split = _goal_assist_split(prospect.goals, prospect.assists)
    if split:
        tags.append(split)

    age_note = _age_context(prospect.age_at_draft)
    if age_note != "age context unavailable":
        tags.append(age_note)

    size_note = _size_profile(prospect.height_cm, prospect.weight_kg)
    if size_note != "size not available":
        tags.append(size_note)

    league_note = _league_context_label(prospect.draft_league_tier)
    if league_note != "unclear competition level":
        tags.append(league_note)

    return tags


def _trend_summary(current_ppg: Optional[float], previous_ppg: Optional[float]) -> str:
    if current_ppg is None and previous_ppg is None:
        return "limited recent production context"
    if current_ppg is None:
        return "current season production unavailable"
    if previous_ppg is None:
        return "single-season production snapshot only"

    delta = current_ppg - previous_ppg
    if delta >= 0.20:
        return "clear upward development trend"
    if delta >= 0.08:
        return "modest year-over-year improvement"
    if delta <= -0.20:
        return "meaningful production drop from the prior season"
    if delta <= -0.08:
        return "slight production regression"
    return "stable production year over year"


def prospect_profile_to_text(prospect) -> str:
    """Render a factual prospect profile with richer context for retrieval."""
    lines = [
        f"Prospect: {prospect.name}",
        (
            f"Position: {prospect.position} | Nationality: {prospect.nationality or 'unknown'} "
            f"| League: {prospect.draft_league or 'unknown'}"
        ),
        (
            f"CSS Rank: {_format_css_rank(prospect.css_ranking)}"
            + (f" | CSS Category: {prospect.css_category}" if prospect.css_category else "")
        ),
    ]
    if prospect.points_per_game is not None:
        stat_line = f"Current production: {prospect.points_per_game:.2f} PPG"
        if prospect.games_played is not None:
            stat_line += f" across {prospect.games_played} GP"
        if prospect.goals is not None and prospect.assists is not None:
            stat_line += f" ({prospect.goals}G, {prospect.assists}A)"
        lines.append(stat_line)
    if prospect.age_at_draft is not None:
        lines.append(f"Age at draft: {prospect.age_at_draft:.1f} ({_age_context(prospect.age_at_draft)})")
    lines.append(
        "Context: "
        f"{_league_context_label(prospect.draft_league_tier)}; {_size_profile(prospect.height_cm, prospect.weight_kg)}."
    )
    return "\n".join(lines)


def draft_history_to_text(team_name: str, year: int, picks: list) -> str:
    """Render a team's draft picks for one year to plain text for embedding."""
    lines = [f"Team: {team_name} | Draft Year: {year}"]
    for pick in picks:
        parts = [f"  Pick #{pick.overall_pick} (Rd {pick.round})"]
        if pick.player_name:
            parts.append(f"— {pick.player_name}")
        if pick.position:
            parts.append(f"({pick.position})")
        if pick.css_rank:
            parts.append(f"CSS #{pick.css_rank}")
        if pick.points_per_game is not None:
            parts.append(f"PPG {pick.points_per_game:.2f}")
        if pick.draft_league:
            parts.append(f"[{pick.draft_league}]")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def prospect_stats_to_text(prospect_name: str, prospect, stat_rows: list) -> str:
    """Render a prospect with their season stat history to plain text for embedding."""
    base = prospect_profile_to_text(prospect)
    if not stat_rows:
        return base
    lines = [base, "Season stats:"]
    for row in stat_rows[:5]:  # cap at 5 seasons
        season_line = f"  {row.league or 'unknown'} ({row.season_type or 'pre_draft'}): "
        season_line += f"GP {row.games_played or '?'} G {row.goals or '?'} A {row.assists or '?'} P {row.points or '?'}"
        if row.points_per_game is not None:
            season_line += f" PPG {row.points_per_game:.2f}"
        lines.append(season_line)
    return "\n".join(lines)


def prospect_style_to_text(prospect) -> str:
    """Render a style-focused chunk for semantic archetype queries."""
    tags = _style_tags_for_prospect(prospect)
    lines = [
        f"Prospect style profile: {prospect.name}",
        f"Projected style: {', '.join(tags) if tags else 'style context unavailable'}",
    ]
    if prospect.points_per_game is not None:
        lines.append(
            f"Production signal: {prospect.points_per_game:.2f} PPG in {prospect.draft_league or 'unknown'} "
            f"with CSS rank {_format_css_rank(prospect.css_ranking)}."
        )
    return "\n".join(lines)


def prospect_trend_to_text(prospect, stat_rows: list) -> str:
    """Render a trend-focused chunk for development and trajectory questions."""
    lines = [
        f"Prospect development trend: {prospect.name}",
        (
            "Trajectory: "
            f"{_trend_summary(prospect.points_per_game, prospect.ppg_prev_season)}"
        ),
    ]

    if prospect.points_per_game is not None or prospect.ppg_prev_season is not None:
        current = f"{prospect.points_per_game:.2f}" if prospect.points_per_game is not None else "unknown"
        previous = f"{prospect.ppg_prev_season:.2f}" if prospect.ppg_prev_season is not None else "unknown"
        lines.append(f"Current season vs previous season: {current} PPG vs {previous} PPG.")

    if stat_rows:
        lines.append("Recent stat history:")
        for row in stat_rows[:3]:
            season_line = f"  {row.league or 'unknown'} ({row.season_type or 'pre_draft'}): "
            season_line += f"{row.games_played or '?'} GP"
            if row.points is not None:
                season_line += f", {row.points} points"
            if row.points_per_game is not None:
                season_line += f", {row.points_per_game:.2f} PPG"
            lines.append(season_line)
    else:
        lines.append("Recent stat history: no multi-season rows available.")

    return "\n".join(lines)


# ── Index building ────────────────────────────────────────────────────────────

def build_index(db: Session) -> int:
    """
    Embed all GM profiles, top-200 prospects (with stat history), and recent team
    draft histories, then upsert into scout_embeddings.
    Returns the number of documents indexed.
    """
    from app.models import (
        GMTendencyProfile, GeneralManager, Team, Prospect2025,
        DraftPickHistorical, ProspectStatHistory,
    )
    from app.agent.store import upsert_embedding

    count = 0

    # ── GM profiles ──────────────────────────────────────────────────────────
    profiles = (
        db.query(GMTendencyProfile, GeneralManager, Team)
        .join(GeneralManager, GMTendencyProfile.gm_id == GeneralManager.id)
        .join(Team, GeneralManager.team_id == Team.id)
        .filter(GeneralManager.is_active.is_(True))
        .all()
    )
    for profile, gm, team in profiles:
        text = gm_profile_to_text(gm.name, team.full_name or team.abbreviation, profile)
        vec = embed_text(text)
        upsert_embedding(db, doc_type="gm_profile", ref_id=gm.id, ref_name=gm.name, content=text, embedding=vec)
        count += 1
        logger.info("embed.indexed_gm", extra={"gm": gm.name})

    # ── Top-200 prospects by CSS rank (with stat history enrichment) ─────────
    prospects = (
        db.query(Prospect2025)
        .order_by(Prospect2025.css_ranking.nullslast())
        .limit(200)
        .all()
    )
    prospect_ids = [p.id for p in prospects]
    all_stat_rows = (
        db.query(ProspectStatHistory)
        .filter(ProspectStatHistory.prospect_id.in_(prospect_ids))
        .order_by(ProspectStatHistory.fetched_at.desc())
        .all()
    )
    stats_by_prospect: dict[int, list] = defaultdict(list)
    for row in all_stat_rows:
        if len(stats_by_prospect[row.prospect_id]) < 5:
            stats_by_prospect[row.prospect_id].append(row)

    # Replace legacy single-document prospect entries with chunked prospect docs.
    db.execute(
        text(
            """
            DELETE FROM scout_embeddings
            WHERE doc_type IN (
                'prospect',
                'prospect_with_stats',
                'prospect_profile',
                'prospect_style',
                'prospect_trend'
            )
            """
        )
    )

    for p in prospects:
        stat_rows = stats_by_prospect[p.id]
        docs: list[tuple[str, str]] = [
            ("prospect_profile", prospect_profile_to_text(p)),
            ("prospect_style", prospect_style_to_text(p)),
            ("prospect_trend", prospect_trend_to_text(p, stat_rows)),
        ]
        # Keep one legacy summary chunk for backward-compatible callers and direct factual retrieval.
        if stat_rows:
            docs.append(("prospect_with_stats", prospect_stats_to_text(p.name, p, stat_rows)))
        else:
            docs.append(("prospect", prospect_to_text(p)))

        for doc_type, text_content in docs:
            vec = embed_text(text_content)
            upsert_embedding(
                db,
                doc_type=doc_type,
                ref_id=p.id,
                ref_name=p.name,
                content=text_content,
                embedding=vec,
            )
            count += 1

    # ── Team draft histories for the last 5 years (2020–2024) ────────────────
    recent_years = list(range(2020, 2025))
    teams = db.query(Team).all()
    for team in teams:
        for year in recent_years:
            picks = (
                db.query(DraftPickHistorical)
                .filter(
                    DraftPickHistorical.team_id == team.id,
                    DraftPickHistorical.year == year,
                )
                .order_by(DraftPickHistorical.overall_pick)
                .all()
            )
            if not picks:
                continue
            team_name = team.full_name or team.abbreviation
            text = draft_history_to_text(team_name, year, picks)
            # Use a synthetic ref_id to avoid collisions: team_id * 10000 + year
            ref_id = team.id * 10000 + year
            ref_name = f"{team_name} {year} Draft"
            vec = embed_text(text)
            upsert_embedding(
                db,
                doc_type="draft_history",
                ref_id=ref_id,
                ref_name=ref_name,
                content=text,
                embedding=vec,
            )
            count += 1
            logger.info("embed.indexed_draft_history", extra={"team": team_name, "year": year})

    # ── Current NHL players (for get_nhl_comp prospect comparisons) ─────────────
    # Fetches live stats from the NHL API and embeds each player as 'nhl_player'.
    # Non-fatal: if the NHL API is unreachable, the rest of the index is unaffected.
    from app.ingestion.nhl_players_ingestion import fetch_and_embed_nhl_players
    try:
        nhl_count = fetch_and_embed_nhl_players(db)
        count += nhl_count
        logger.info("embed.nhl_players_indexed", extra={"count": nhl_count})
    except Exception as exc:
        logger.warning(
            "embed.nhl_players_failed error=%s — continuing without NHL player comps", exc
        )

    db.commit()
    logger.info("embed.index_built", extra={"doc_count": count})
    return count
