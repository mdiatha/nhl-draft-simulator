from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from app.models import DraftClass, Player, Prospect, ProspectRanking, PlayerSeasonStat


def _split_name(full_name: str) -> tuple[str, str]:
    full_name = (full_name or "").strip()
    if not full_name:
        return "Unknown", ""
    if " " not in full_name:
        return full_name, ""
    first, rest = full_name.split(" ", 1)
    return first, rest


def ensure_draft_class(db: Session, year: int) -> DraftClass:
    draft_class = db.query(DraftClass).filter(DraftClass.year == year).first()
    if draft_class:
        return draft_class

    draft_class = DraftClass(year=year)
    db.add(draft_class)
    db.flush()
    return draft_class


def upsert_player(
    db: Session,
    *,
    full_name: str,
    nhl_player_id: int | None = None,
    birth_date=None,
    birth_country: str | None = None,
    position: str | None = None,
    height_cm: int | None = None,
    weight_kg: int | None = None,
    draft_year: int | None = None,
    draft_round: int | None = None,
    draft_overall: int | None = None,
    css_rank: int | None = None,
) -> Player:
    player = None
    if nhl_player_id is not None:
        player = db.query(Player).filter(Player.nhl_player_id == nhl_player_id).first()
    if player is None and full_name:
        query = db.query(Player).filter(Player.full_name == full_name)
        if birth_date is not None:
            query = query.filter(Player.birth_date == birth_date)
        player = query.order_by(Player.draft_year.desc().nullslast()).first()

    if player is None:
        first_name, last_name = _split_name(full_name)
        player = Player(
            nhl_player_id=nhl_player_id,
            full_name=full_name,
            first_name=first_name,
            last_name=last_name,
            birth_date=birth_date,
            birth_country=(birth_country or None),
            position=position or "F",
            height_cm=height_cm,
            weight_kg=weight_kg,
            draft_year=draft_year,
            draft_round=draft_round,
            draft_overall=draft_overall,
            css_rank=css_rank,
        )
        db.add(player)
        db.flush()
        return player

    first_name, last_name = _split_name(full_name or player.full_name or "")
    player.full_name = full_name or player.full_name
    player.first_name = player.first_name or first_name
    player.last_name = player.last_name or last_name
    player.nhl_player_id = player.nhl_player_id or nhl_player_id
    player.birth_date = player.birth_date or birth_date
    player.birth_country = player.birth_country or birth_country
    player.position = position or player.position
    player.height_cm = player.height_cm or height_cm
    player.weight_kg = player.weight_kg or weight_kg
    player.draft_year = player.draft_year or draft_year
    player.draft_round = player.draft_round or draft_round
    player.draft_overall = player.draft_overall or draft_overall
    player.css_rank = player.css_rank or css_rank
    db.flush()
    return player


def sync_prospect(
    db: Session,
    *,
    draft_year: int,
    full_name: str,
    position: str,
    nationality: str | None = None,
    height_cm: int | None = None,
    weight_kg: int | None = None,
    draft_league: str | None = None,
    draft_league_tier: int | None = None,
    css_ranking: int | None = None,
    css_category: str | None = None,
    points: int | None = None,
    goals: int | None = None,
    assists: int | None = None,
    games_played: int | None = None,
    points_per_game: float | None = None,
    ppg_prev_season: float | None = None,
    age_at_draft: float | None = None,
    birth_date=None,
    nhl_player_id: int | None = None,
) -> Prospect:
    draft_class = ensure_draft_class(db, draft_year)
    player = upsert_player(
        db,
        full_name=full_name,
        nhl_player_id=nhl_player_id,
        birth_date=birth_date,
        birth_country=nationality,
        position=position,
        height_cm=height_cm,
        weight_kg=weight_kg,
        draft_year=draft_year,
        draft_overall=css_ranking,
        css_rank=css_ranking,
    )

    prospect = (
        db.query(Prospect)
        .filter(
            Prospect.player_id == player.id,
            Prospect.draft_class_id == draft_class.id,
        )
        .first()
    )
    if prospect is None:
        prospect = (
            db.query(Prospect)
            .filter(
                Prospect.name == full_name,
                Prospect.draft_class_id == draft_class.id,
            )
            .first()
        )

    if prospect is None:
        prospect = Prospect(
            player_id=player.id,
            draft_class_id=draft_class.id,
            name=full_name,
            position=position,
            is_active=True,
        )
        db.add(prospect)

    prospect.player_id = player.id
    prospect.draft_class_id = draft_class.id
    prospect.name = full_name
    prospect.position = position
    prospect.nationality = nationality
    prospect.height_cm = height_cm
    prospect.weight_kg = weight_kg
    prospect.draft_league = draft_league
    prospect.draft_league_tier = draft_league_tier
    prospect.css_ranking = css_ranking
    prospect.css_category = css_category
    prospect.points = points
    prospect.goals = goals
    prospect.assists = assists
    prospect.games_played = games_played
    prospect.points_per_game = points_per_game
    prospect.ppg_prev_season = ppg_prev_season
    prospect.age_at_draft = age_at_draft
    prospect.birth_date = birth_date
    prospect.nhl_player_id = nhl_player_id
    prospect.is_active = True
    db.flush()
    return prospect


def upsert_prospect_ranking(
    db: Session,
    *,
    prospect: Prospect,
    source: str,
    ranking_type: str,
    category: str | None,
    rank: int | None,
    ranking_date: date | None = None,
    metadata_json: dict | None = None,
) -> ProspectRanking | None:
    if rank is None:
        return None

    ranking = (
        db.query(ProspectRanking)
        .filter(
            ProspectRanking.prospect_id == prospect.id,
            ProspectRanking.source == source,
            ProspectRanking.ranking_type == ranking_type,
            ProspectRanking.category == category,
        )
        .first()
    )
    if ranking is None:
        ranking = ProspectRanking(
            prospect_id=prospect.id,
            source=source,
            ranking_type=ranking_type,
            category=category,
            rank=rank,
        )
        db.add(ranking)

    ranking.rank = rank
    ranking.ranking_date = ranking_date
    ranking.metadata_json = metadata_json
    db.flush()
    return ranking


def upsert_player_season_stat(
    db: Session,
    *,
    player: Player,
    season_year_start: int,
    season_year_end: int,
    league: str | None,
    games_played: int | None,
    goals: int | None,
    assists: int | None,
    points: int | None,
    points_per_game: float | None,
    season_type: str,
    source: str,
    as_of_date: date | None,
    raw_payload: dict | None = None,
    team_name: str | None = None,
) -> PlayerSeasonStat:
    stat = (
        db.query(PlayerSeasonStat)
        .filter(
            PlayerSeasonStat.player_id == player.id,
            PlayerSeasonStat.season_year_start == season_year_start,
            PlayerSeasonStat.season_year_end == season_year_end,
            PlayerSeasonStat.league == league,
            PlayerSeasonStat.season_type == season_type,
            PlayerSeasonStat.source == source,
            PlayerSeasonStat.as_of_date == as_of_date,
        )
        .first()
    )
    if stat is None:
        stat = PlayerSeasonStat(
            player_id=player.id,
            season_year_start=season_year_start,
            season_year_end=season_year_end,
            league=league,
            season_type=season_type,
            source=source,
            as_of_date=as_of_date,
        )
        db.add(stat)

    stat.team_name = team_name
    stat.games_played = games_played
    stat.goals = goals
    stat.assists = assists
    stat.points = points
    stat.points_per_game = points_per_game
    stat.raw_payload = raw_payload
    db.flush()
    return stat
