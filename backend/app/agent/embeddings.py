"""
Embedding utilities for the Scout RAG agent.

Uses Ollama (nomic-embed-text) for local embeddings — no external API key required.
nomic-embed-text outputs 768-dim vectors. Ollama must be running and the model pulled:

    ollama pull nomic-embed-text

Set OLLAMA_BASE_URL in .env (default: http://localhost:11434).
Falls back gracefully when Ollama is unreachable.

Indexing strategy:
  - One document per prospect: rich prose combining facts, style, and production trend.
    Single document embeds better than sparse chunks and is sufficient at 200-prospect scale.
  - One document per GM tendency profile.
  - One document per team draft history per year (2020-2024).
  - One document per current NHL player (for comp queries).
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

EMBED_DIM = 768
OLLAMA_MODEL = "nomic-embed-text"


def embed_text(text: str) -> Optional[list[float]]:
    """Return a 768-dim embedding for *text* using Ollama nomic-embed-text."""
    return _embed_via_ollama(text)


def _embed_via_ollama(text: str) -> Optional[list[float]]:
    from app.config import settings
    base_url = settings.OLLAMA_BASE_URL.rstrip("/")
    try:
        import httpx
        resp = httpx.post(
            f"{base_url}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as exc:
        logger.warning("embed.ollama_failed", extra={"error": str(exc)})
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Document builders ─────────────────────────────────────────────────────────

def prospect_to_text(prospect, stat_rows: list | None = None) -> str:
    """
    Build a single rich prose document for a prospect combining factual profile,
    style description, production trend, and stat history.

    One document per prospect is correct at this scale (~200 prospects). Chunking
    into profile/style/trend splits was unnecessary complexity: the Scout uses
    retrieval only as a routing signal, not as context stuffed into a prompt,
    so fine-grained chunk separation doesn't improve answer quality.
    """
    lines: list[str] = []

    # ── Identity ──────────────────────────────────────────────────────────────
    css = str(prospect.css_ranking) if prospect.css_ranking is not None else "unranked"
    cat = f" ({prospect.css_category})" if getattr(prospect, "css_category", None) else ""
    lines.append(f"{prospect.name} — {prospect.position or 'F'}, CSS #{css}{cat}")
    lines.append(
        f"Nationality: {prospect.nationality or 'unknown'} | "
        f"League: {prospect.draft_league or 'unknown'}"
    )

    # ── Production ────────────────────────────────────────────────────────────
    if prospect.points_per_game is not None:
        stat = f"{prospect.points_per_game:.2f} PPG"
        if getattr(prospect, "games_played", None) is not None:
            stat += f" in {prospect.games_played} GP"
        goals = getattr(prospect, "goals", None)
        assists = getattr(prospect, "assists", None)
        if goals is not None and assists is not None:
            stat += f" ({goals}G, {assists}A)"
        lines.append(f"Production: {stat}")

    # ── Trend ─────────────────────────────────────────────────────────────────
    prev = getattr(prospect, "ppg_prev_season", None)
    cur = prospect.points_per_game
    if cur is not None and prev is not None:
        delta = cur - prev
        if delta >= 0.20:
            trend = "clear upward development trend"
        elif delta >= 0.08:
            trend = "modest year-over-year improvement"
        elif delta <= -0.20:
            trend = "meaningful production drop from prior season"
        elif delta <= -0.08:
            trend = "slight production regression"
        else:
            trend = "stable production year over year"
        lines.append(f"Trend: {trend} ({prev:.2f} PPG prior season → {cur:.2f} PPG this season)")
    elif cur is not None:
        lines.append("Trend: single-season snapshot, no prior season data")

    # ── Physical + age ────────────────────────────────────────────────────────
    height = getattr(prospect, "height_cm", None)
    weight = getattr(prospect, "weight_kg", None)
    age = getattr(prospect, "age_at_draft", None)

    size_parts: list[str] = []
    if height is not None:
        if height >= 191:
            size_parts.append(f"{height}cm (pro-size frame)")
        elif height <= 178:
            size_parts.append(f"{height}cm (undersized)")
        else:
            size_parts.append(f"{height}cm (average frame)")
    if weight is not None:
        size_parts.append(f"{weight}kg")
    if size_parts:
        lines.append(f"Size: {', '.join(size_parts)}")

    if age is not None:
        if age <= 18.2:
            age_note = "young for the draft class"
        elif age >= 18.8:
            age_note = "older for the draft class"
        else:
            age_note = "average draft age"
        lines.append(f"Age at draft: {age:.1f} — {age_note}")

    # ── Style description ─────────────────────────────────────────────────────
    pos = (prospect.position or "F").upper()
    ppg = prospect.points_per_game or 0.0
    style_parts: list[str] = []

    if pos == "G":
        style_parts.append("goaltending prospect")
    elif pos == "D":
        if ppg >= 0.75:
            style_parts.append("offensive defenseman")
        elif ppg >= 0.45:
            style_parts.append("two-way defenseman")
        else:
            style_parts.append("defensive defenseman")
    else:
        if ppg >= 1.30:
            style_parts.append("high-end scoring forward")
        elif ppg >= 0.95:
            style_parts.append("top-six scoring forward")
        elif ppg >= 0.65:
            style_parts.append("middle-six forward")
        else:
            style_parts.append("depth forward")

    goals = getattr(prospect, "goals", None)
    assists = getattr(prospect, "assists", None)
    if goals is not None and assists is not None:
        total = goals + assists
        if total > 0:
            goal_share = goals / total
            if goal_share >= 0.48:
                style_parts.append("goal-driven production")
            elif goal_share <= 0.28:
                style_parts.append("playmaking / assist-heavy")
            else:
                style_parts.append("balanced scorer")

    tier = getattr(prospect, "draft_league_tier", None)
    if tier == 1:
        style_parts.append("elite competition")
    elif tier == 2:
        style_parts.append("strong junior or college competition")

    if style_parts:
        lines.append(f"Style: {', '.join(style_parts)}")

    # ── Stat history ──────────────────────────────────────────────────────────
    if stat_rows:
        lines.append("Season history:")
        for row in stat_rows[:4]:
            s = f"  {row.league or 'unknown'}: {row.games_played or '?'} GP"
            if row.goals is not None and row.assists is not None:
                s += f"  {row.goals}G {row.assists}A"
            if row.points_per_game is not None:
                s += f"  {row.points_per_game:.2f} PPG"
            lines.append(s)

    return "\n".join(lines)


def gm_profile_to_text(gm_name: str, team_name: str, profile) -> str:
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
        f"Preferred leagues: {', '.join(f'{lg}({w:.0%})' for lg,w in top_leagues)}",
        f"Nationality tendencies: {', '.join(f'{n}({w:.0%})' for n,w in top_nations)}",
    ]
    if profile.avg_ranking_deviation is not None:
        lines.append(f"Avg CSS rank deviation: {profile.avg_ranking_deviation:+.1f} picks")
    return "\n".join(lines)


def draft_history_to_text(team_name: str, year: int, picks: list) -> str:
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


# ── Index building ────────────────────────────────────────────────────────────

def build_index(db) -> int:
    """
    Embed all GM profiles, prospects (with stat history), team draft histories,
    and current NHL players, then upsert into scout_embeddings.
    Returns the number of documents indexed.
    """
    from sqlalchemy.orm import Session
    from sqlalchemy import text
    from app.models import (
        GMTendencyProfile, GeneralManager, Team, Prospect,
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
        doc = gm_profile_to_text(gm.name, team.full_name or team.abbreviation, profile)
        vec = embed_text(doc)
        upsert_embedding(db, doc_type="gm_profile", ref_id=gm.id, ref_name=gm.name, content=doc, embedding=vec)
        count += 1

    # ── Prospects: one rich document per prospect ────────────────────────────
    # Clear existing prospect chunk types before reinserting fresh documents.
    db.execute(text("""
        DELETE FROM scout_embeddings
        WHERE doc_type IN (
            'prospect',
            'prospect_with_stats',
            'prospect_profile',
            'prospect_style',
            'prospect_trend'
        )
    """))

    prospects = (
        db.query(Prospect)
        .order_by(Prospect.css_ranking.nullslast())
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
        if len(stats_by_prospect[row.prospect_id]) < 4:
            stats_by_prospect[row.prospect_id].append(row)

    for p in prospects:
        doc = prospect_to_text(p, stats_by_prospect[p.id])
        vec = embed_text(doc)
        upsert_embedding(db, doc_type="prospect", ref_id=p.id, ref_name=p.name, content=doc, embedding=vec)
        count += 1

    # ── Team draft histories (2020-2024) ─────────────────────────────────────
    teams = db.query(Team).all()
    for team in teams:
        for year in range(2020, 2025):
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
            doc = draft_history_to_text(team_name, year, picks)
            ref_id = team.id * 10000 + year
            vec = embed_text(doc)
            upsert_embedding(
                db, doc_type="draft_history",
                ref_id=ref_id, ref_name=f"{team_name} {year} Draft",
                content=doc, embedding=vec,
            )
            count += 1

    # ── NHL players (for get_nhl_comp) ────────────────────────────────────────
    from app.ingestion.nhl_players_ingestion import fetch_and_embed_nhl_players
    try:
        nhl_count = fetch_and_embed_nhl_players(db)
        count += nhl_count
    except Exception as exc:
        logger.warning("embed.nhl_players_failed error=%s — continuing without NHL player comps", exc)

    db.commit()
    logger.info("embed.index_built", extra={"doc_count": count})
    return count
