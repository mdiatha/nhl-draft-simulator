"""
NHL API data ingestion module.
Fetches teams, draft history, rosters, standings/lottery odds.
Base URL: https://api-web.nhle.com/v1
"""
import json
import logging
import os
import uuid
from datetime import date, datetime, timezone
from typing import Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sqlalchemy import or_
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.database import SessionLocal
from app.models import Team, GeneralManager, DraftPickHistorical, LotteryOdds, IngestionRun, Player, Prospect
from app.constants import DRAFT_YEAR, STANDINGS_DATE, LOTTERY_ODDS, get_league_tier
from app.ingestion.normalized import (
    ensure_draft_class,
    sync_prospect,
    upsert_player,
    upsert_player_season_stat,
    upsert_prospect_ranking,
)

logger = logging.getLogger(__name__)

NHL_API_BASE = "https://api-web.nhle.com/v1"

# GM data lives in gms.json next to this file — edit that file to update GMs.
_GMS_PATH = os.path.join(os.path.dirname(__file__), "gms.json")
with open(_GMS_PATH, encoding="utf-8") as _f:
    _GM_STINTS: list[dict] = json.load(_f)

# Active GMs by team abbreviation — used by fetch_all_teams for current-roster logic
GM_LOOKUP: dict[str, dict] = {s["team"]: s for s in _GM_STINTS if "until" not in s}


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _get(url: str, client: httpx.Client) -> dict:
    """Make a GET request with retry logic."""
    logger.debug(f"GET {url}")
    response = client.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def fetch_all_teams(db: Session) -> list[dict]:
    """
    Fetch all 32 NHL teams from standings endpoint.
    Upserts teams and their GMs to the database.
    Returns list of team dicts.
    """
    logger.info("Fetching all teams...")
    with httpx.Client(base_url=NHL_API_BASE) as client:
        data = _get("/standings/now", client)

    teams_data = []
    standings = data.get("standings", [])

    for entry in standings:
        team_info = entry.get("teamAbbrev", {})
        abbrev = team_info.get("default", "") if isinstance(team_info, dict) else str(team_info)

        team_dict = {
            "nhl_id": entry.get("teamId"),
            "abbreviation": abbrev,
            "full_name": entry.get("teamName", {}).get("default", ""),
            "city": entry.get("placeName", {}).get("default", ""),
            "conference": entry.get("conferenceName", ""),
            "division": entry.get("divisionName", ""),
        }

        gm_data = GM_LOOKUP.get(abbrev, {})
        team_dict["current_gm_name"] = gm_data.get("name")
        team_dict["current_gm_since"] = (
            date.fromisoformat(gm_data["since"]) if gm_data.get("since") else None
        )

        # Upsert team — skip unique-key fields on update
        SKIP_ON_UPDATE = {"nhl_id", "abbreviation"}
        existing = db.query(Team).filter(Team.abbreviation == abbrev).first()
        if existing:
            for k, v in team_dict.items():
                if k not in SKIP_ON_UPDATE:
                    setattr(existing, k, v)
        else:
            db.add(Team(**team_dict))

        teams_data.append(team_dict)

    # Upsert GMs
    for team_dict in teams_data:
        abbrev = team_dict["abbreviation"]
        gm_data = GM_LOOKUP.get(abbrev, {})
        if not gm_data:
            continue
        team = db.query(Team).filter(Team.abbreviation == abbrev).first()
        if not team:
            continue
        existing_gm = db.query(GeneralManager).filter(
            GeneralManager.name == gm_data["name"],
            GeneralManager.team_id == team.id,
        ).first()
        if not existing_gm:
            gm = GeneralManager(
                name=gm_data["name"],
                team_id=team.id,
                start_date=date.fromisoformat(gm_data["since"]) if gm_data.get("since") else date.today(),
                is_active=True,
            )
            db.add(gm)

    db.commit()
    logger.info(f"Upserted {len(teams_data)} teams")
    return teams_data


RECORDS_API_BASE = "https://records.nhl.com/site/api"


def seed_gm_history(db: Session) -> int:
    """
    Create GeneralManager records for every stint in gms.json — both current
    and historical. Idempotent: skips stints that already exist by (name, team).

    Must run after fetch_all_teams so team records exist.
    Must run before fetch_draft_history so picks can be attributed correctly.
    """
    created = 0
    for stint in _GM_STINTS:
        team = db.query(Team).filter(Team.abbreviation == stint["team"]).first()
        if not team:
            logger.warning(f"Team {stint['team']} not found — skipping GM stint for {stint['name']}")
            continue

        existing = db.query(GeneralManager).filter(
            GeneralManager.name == stint["name"],
            GeneralManager.team_id == team.id,
        ).first()
        if existing:
            continue

        db.add(GeneralManager(
            name=stint["name"],
            team_id=team.id,
            start_date=date.fromisoformat(stint["since"]),
            end_date=date.fromisoformat(stint["until"]) if "until" in stint else None,
            is_active="until" not in stint,
        ))
        created += 1

    db.commit()
    logger.info(f"Seeded {created} GM stint record(s)")
    return created


def fetch_draft_history(db: Session, start_year: int = 2000, end_year: int = 2024) -> int:
    """
    Fetch draft history for given year range from NHL Records API.
    Returns total number of picks upserted.
    """
    total = 0
    with httpx.Client() as client:
        for year in range(start_year, end_year + 1):
            logger.info(f"Fetching {year} draft...")
            try:
                # Fetch all rounds for the year (up to 7 rounds × 32 picks)
                url = f"{RECORDS_API_BASE}/draft?cayenneExp=draftYear={year}&limit=500"
                resp = client.get(url, timeout=30.0, follow_redirects=True)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"Draft {year} not available: {e}")
                continue

            picks_data = data.get("data", [])
            for pick in picks_data:
                abbrev = pick.get("triCode", "")
                team = db.query(Team).filter(Team.abbreviation == abbrev).first()

                gm = None
                if team:
                    # Attribute pick to whoever was GM at this team during this draft year
                    draft_date = date(year, 7, 1)
                    gm = (
                        db.query(GeneralManager)
                        .filter(
                            GeneralManager.team_id == team.id,
                            GeneralManager.start_date <= draft_date,
                            or_(
                                GeneralManager.end_date.is_(None),
                                GeneralManager.end_date >= draft_date,
                            ),
                        )
                        .first()
                    )

                league_name = pick.get("amateurLeague") or ""
                tier, _region = get_league_tier(league_name)

                # Height inches → cm, weight lbs → kg
                height_in = pick.get("height")
                height_cm = round(height_in * 2.54) if height_in else None
                weight_lbs = pick.get("weight")
                weight_kg = round(weight_lbs * 0.453592) if weight_lbs else None

                player_name = f"{pick.get('firstName', '')} {pick.get('lastName', '')}".strip()

                overall_pick_num = pick.get("overallPickNumber", 0)

                # Deduplicate: skip if this (year, overall_pick) already exists.
                # Prevents duplicate rows when ingestion is run more than once.
                existing = (
                    db.query(DraftPickHistorical)
                    .filter(
                        DraftPickHistorical.year == year,
                        DraftPickHistorical.overall_pick == overall_pick_num,
                    )
                    .first()
                )
                if existing:
                    continue

                pick_obj = DraftPickHistorical(
                    year=year,
                    round=pick.get("roundNumber", 0),
                    pick_number=pick.get("pickInRound", 0),
                    overall_pick=overall_pick_num,
                    team_id=team.id if team else None,
                    gm_id=gm.id if gm else None,
                    player_name=player_name,
                    position=pick.get("position"),
                    nationality=pick.get("countryCode"),
                    height_cm=height_cm,
                    weight_kg=weight_kg,
                    draft_league=league_name or None,
                    draft_league_tier=tier,
                    nhl_player_id=pick.get("playerId"),
                )
                player = upsert_player(
                    db,
                    full_name=player_name,
                    nhl_player_id=pick.get("playerId"),
                    birth_country=pick.get("countryCode"),
                    position=pick.get("position"),
                    height_cm=height_cm,
                    weight_kg=weight_kg,
                    draft_year=year,
                    draft_round=pick.get("roundNumber", 0),
                    draft_overall=overall_pick_num,
                )
                pick_obj.player_id = player.id
                db.add(pick_obj)
                total += 1

            db.commit()
            logger.info(f"  Committed {year} draft picks ({len(picks_data)} picks)")

    logger.info(f"Total draft picks ingested: {total}")
    return total




def fetch_lottery_standings(db: Session) -> list[dict]:
    """
    Fetch end-of-2024-25 season standings and build lottery odds for the
    16 non-playoff teams (conferenceSequence > 8).
    Returns list of lottery entries.
    """
    logger.info(f"Fetching standings for {STANDINGS_DATE} (end of {DRAFT_YEAR-1}-{DRAFT_YEAR} season)...")
    url = f"{NHL_API_BASE}/standings/{STANDINGS_DATE}"
    with httpx.Client() as client:
        resp = client.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()

    standings = data.get("standings", [])

    # Non-playoff teams: conferenceSequence > 8 (bottom 8 per conference)
    non_playoff = [
        e for e in standings
        if e.get("conferenceSequence", 0) > 8
    ]

    # Sort by points ascending (worst first = highest lottery odds)
    non_playoff.sort(key=lambda x: (x.get("points", 0), x.get("wins", 0)))

    if len(non_playoff) != 16:
        logger.warning(f"Expected 16 non-playoff teams, got {len(non_playoff)} — using anyway")

    # Delete existing lottery odds for draft year
    db.query(LotteryOdds).filter(LotteryOdds.season == DRAFT_YEAR).delete()

    combinations_assigned = 0
    team_combinations = {}

    for standing_pos, entry in enumerate(non_playoff, start=1):
        odds_pct = LOTTERY_ODDS.get(standing_pos, 0)
        num_combos = round(odds_pct * 10)
        team_combinations[standing_pos] = list(
            range(combinations_assigned + 1, combinations_assigned + num_combos + 1)
        )
        combinations_assigned += num_combos

    lottery_entries = []
    for standing_pos, entry in enumerate(non_playoff, start=1):
        team_info = entry.get("teamAbbrev", {})
        abbrev = team_info.get("default", "") if isinstance(team_info, dict) else str(team_info)
        team = db.query(Team).filter(Team.abbreviation == abbrev).first()
        if not team:
            # Log at ERROR — a missing team means lottery odds are incomplete.
            # This usually means the DB is missing a team (e.g. expansion team)
            # and needs a fresh ingest of teams before lottery odds can be built.
            logger.error(
                "lottery.unknown_team abbrev=%s standing=%d — team not in DB, skipping",
                abbrev, standing_pos,
            )
            continue

        odds = LotteryOdds(
            team_id=team.id,
            season=DRAFT_YEAR,
            final_standing=standing_pos,
            lottery_odds_pct=LOTTERY_ODDS.get(standing_pos, 0),
            assigned_combinations=team_combinations.get(standing_pos, []),
            wins=entry.get("wins"),
            losses=entry.get("losses"),
            otl=entry.get("otLosses"),
            points=entry.get("points"),
        )
        db.add(odds)
        lottery_entries.append({
            "team": abbrev,
            "standing": standing_pos,
            "odds_pct": LOTTERY_ODDS.get(standing_pos, 0),
        })

    missing = len(non_playoff) - len(lottery_entries)
    if missing > 0:
        logger.error(
            "lottery.incomplete %d/%d teams matched — %d teams not found in DB. "
            "Run POST /api/admin/ingest first to populate teams.",
            len(lottery_entries), len(non_playoff), missing,
        )

    db.commit()
    logger.info(f"Lottery odds computed for {len(lottery_entries)} teams (season {DRAFT_YEAR})")
    return lottery_entries




def seed_2026_prospects_from_json(db: Session, force: bool = False) -> int:
    """
    Seed the prospects table with the 2026 draft class from the bundled
    prospects_2026.json file (CSS/EliteProspects consensus rankings).
    Clears existing data when force=True.
    """
    if not force and db.query(Prospect).count() > 0:
        logger.info("prospects already has data, skipping 2026 seed (pass force=True to overwrite).")
        return 0

    json_path = os.path.join(os.path.dirname(__file__), "prospects_2026.json")
    try:
        with open(json_path, "r") as f:
            prospects_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load prospects_2026.json: {e}")
        return 0

    if force:
        db.query(Prospect).delete()
        db.flush()

    ensure_draft_class(db, DRAFT_YEAR)
    count = 0
    for p in prospects_data:
        league_name = p.get("draft_league") or ""
        tier, _region = get_league_tier(league_name)
        prospect = sync_prospect(
            db,
            draft_year=DRAFT_YEAR,
            full_name=p["name"],
            position=p.get("position", "F"),
            nationality=p.get("nationality"),
            height_cm=p.get("height_cm"),
            weight_kg=p.get("weight_kg"),
            draft_league=league_name or None,
            draft_league_tier=tier,
            css_ranking=p.get("css_ranking"),
            css_category=p.get("css_category"),
            games_played=p.get("games_played"),
            goals=p.get("goals"),
            assists=p.get("assists"),
            points=p.get("points"),
            points_per_game=p.get("points_per_game"),
            age_at_draft=p.get("age_at_draft"),
        )
        upsert_prospect_ranking(
            db,
            prospect=prospect,
            source="seeded_consensus",
            ranking_type="pre_draft",
            category=p.get("css_category"),
            rank=p.get("css_ranking"),
        )
        count += 1

    db.commit()
    logger.info(f"Seeded {count} 2026 prospects from prospects_2026.json")
    return count


def seed_2025_prospects(db: Session, force: bool = False) -> int:
    """
    Seed the prospects table by fetching the 2025 draft class from the
    NHL Records API (records.nhl.com/site/api/draft?cayenneExp=draftYear=2025).
    Returns the number of prospects inserted.
    Skips if records already exist (unless force=True).
    """
    if not force and db.query(Prospect).count() > 0:
        logger.info("prospects already seeded, skipping.")
        return 0

    logger.info("Fetching 2025 draft class from NHL Records API...")
    url = f"{RECORDS_API_BASE}/draft?cayenneExp=draftYear=2025&limit=500"
    try:
        with httpx.Client() as client:
            resp = client.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch 2025 draft prospects from API: {e}")
        return 0

    picks = data.get("data", [])
    if not picks:
        logger.warning("No 2025 draft data returned from NHL Records API.")
        return 0

    if force:
        db.query(Prospect).delete()
        db.flush()

    ensure_draft_class(db, 2025)
    count = 0
    for pick in picks:
        player_name = f"{pick.get('firstName', '')} {pick.get('lastName', '')}".strip()
        if not player_name:
            continue

        league_name = pick.get("amateurLeague") or ""
        tier, _region = get_league_tier(league_name)

        height_in = pick.get("height")
        height_cm = round(height_in * 2.54) if height_in else None
        weight_lbs = pick.get("weight")
        weight_kg = round(weight_lbs * 0.453592) if weight_lbs else None

        overall = pick.get("overallPickNumber") or 0
        gp = pick.get("gamesPlayed") or 0
        g = pick.get("goals") or 0
        a = pick.get("assists") or 0
        pts = g + a
        ppg = round(pts / gp, 2) if gp > 0 else 0.0

        # Determine CSS category from position
        position = pick.get("position") or "F"
        if position == "G":
            css_cat = "goalie"
        elif pick.get("countryCode") in ("CAN", "USA"):
            css_cat = "NA skater"
        else:
            css_cat = "EUR skater"

        # age_at_draft in years from ageInDays field
        age_in_days = pick.get("ageInDays")
        age_at_draft = round(age_in_days / 365.25, 2) if age_in_days else None

        prospect = sync_prospect(
            db,
            draft_year=2025,
            full_name=player_name,
            position=position,
            nationality=pick.get("countryCode"),
            height_cm=height_cm,
            weight_kg=weight_kg,
            draft_league=league_name or None,
            draft_league_tier=tier,
            css_ranking=overall,
            css_category=css_cat,
            games_played=gp,
            goals=g,
            assists=a,
            points=pts,
            points_per_game=ppg,
            age_at_draft=age_at_draft,
            nhl_player_id=pick.get("playerId"),
        )
        upsert_prospect_ranking(
            db,
            prospect=prospect,
            source="nhl_records",
            ranking_type="draft_board",
            category=css_cat,
            rank=overall,
        )
        count += 1

    db.commit()
    logger.info(f"Seeded {count} prospects into prospects from NHL Records API")
    return count


def fetch_prospect_stats(db: Session) -> int:
    """
    For every Prospect with an nhl_player_id, fetch their most recent
    major-league season stats from the NHL player landing API and populate
    games_played, goals, assists, points, points_per_game.
    Returns the number of prospects updated.
    """
    MAJOR_LEAGUES = {
        "OHL", "WHL", "QMJHL", "SHL", "LIIGA", "KHL", "NLA",
        "NCAA", "USHL", "AHL", "EXTRALIGA", "MESTIS", "ALLSVENSKAN",
    }

    prospects = db.query(Prospect).filter(
        Prospect.nhl_player_id.isnot(None)
    ).all()

    updated = 0
    with httpx.Client(base_url=NHL_API_BASE) as client:
        for prospect in prospects:
            try:
                data = _get(f"/player/{prospect.nhl_player_id}/landing", client)
            except Exception as e:
                logger.debug(f"Could not fetch stats for prospect {prospect.name}: {e}")
                continue

            seasons = data.get("seasonTotals", [])
            # Target the draft-year season and the one before it for y/y trend.
            # Derived from DRAFT_YEAR so no manual update needed each season.
            DRAFT_SEASON_CODE = (DRAFT_YEAR - 1) * 10000 + DRAFT_YEAR
            PREV_SEASON_CODE  = (DRAFT_YEAR - 2) * 10000 + (DRAFT_YEAR - 1)
            best = None
            prev = None
            fallback = None
            for s in seasons:
                if s.get("gameTypeId") != 2:
                    continue
                league = (s.get("leagueAbbrev") or "").upper()
                if not any(ml in league for ml in MAJOR_LEAGUES):
                    continue
                code = s.get("season", 0)
                if code == DRAFT_SEASON_CODE:
                    best = s
                elif code == PREV_SEASON_CODE:
                    prev = s
                elif fallback is None or code > fallback.get("season", 0):
                    fallback = s
            if best is None:
                best = fallback

            if best:
                gp = best.get("gamesPlayed") or 0
                g = best.get("goals") or 0
                a = best.get("assists") or 0
                pts = g + a
                ppg = round(pts / gp, 2) if gp > 0 else 0.0

                prospect.games_played = gp
                prospect.goals = g
                prospect.assists = a
                prospect.points = pts
                prospect.points_per_game = ppg

            if prev:
                gp_prev = prev.get("gamesPlayed") or 0
                pts_prev = (prev.get("goals") or 0) + (prev.get("assists") or 0)
                prospect.ppg_prev_season = round(pts_prev / gp_prev, 2) if gp_prev > 0 else 0.0

            if best or prev:
                if prospect.player_id and best:
                    player = db.query(Player).filter(Player.id == prospect.player_id).first()
                    if player is not None:
                        upsert_player_season_stat(
                            db,
                            player=player,
                            season_year_start=DRAFT_YEAR - 1,
                            season_year_end=DRAFT_YEAR,
                            league=best.get("leagueAbbrev") if best else prospect.draft_league,
                            games_played=prospect.games_played,
                            goals=prospect.goals,
                            assists=prospect.assists,
                            points=prospect.points,
                            points_per_game=prospect.points_per_game,
                            season_type="regular",
                            source="nhl_player_landing",
                            as_of_date=date.today(),
                            raw_payload=best,
                        )
                updated += 1

    db.commit()
    logger.info(f"Updated stats for {updated} prospects")
    return updated


def fetch_historical_stats(db: Session) -> int:
    """
    For each historical draft pick with an nhl_player_id, fetch their
    pre-draft season stats (the season before they were drafted) and
    populate points_per_game and age_at_draft on DraftPickHistorical.

    Uses the NHL player landing API. Only processes picks where
    points_per_game is NULL (safe to call multiple times).
    Returns the number of records updated.
    """
    from datetime import date as date_type
    from app.models import DraftPickHistorical as DPH

    MAJOR_LEAGUES = {
        "OHL", "WHL", "QMJHL", "SHL", "LIIGA", "KHL", "NLA",
        "NCAA", "USHL", "AHL", "EXTRALIGA", "MESTIS", "ALLSVENSKAN",
    }

    picks = (
        db.query(DPH)
        .filter(DPH.nhl_player_id.isnot(None), DPH.points_per_game.is_(None))
        .all()
    )
    logger.info(f"Fetching pre-draft stats for {len(picks)} historical picks...")

    updated = 0
    with httpx.Client(base_url=NHL_API_BASE) as client:
        for pick in picks:
            try:
                data = _get(f"/player/{pick.nhl_player_id}/landing", client)
            except Exception as e:
                logger.debug(f"Could not fetch {pick.player_name}: {e}")
                continue

            # Pre-draft season = {draft_year-1}{draft_year}, prev = {draft_year-2}{draft_year-1}
            pre_draft_season = int(f"{pick.year - 1}{pick.year}")
            prev_season_code  = int(f"{pick.year - 2}{pick.year - 1}")

            seasons = data.get("seasonTotals", [])
            best = None
            prev = None
            fallback = None
            for s in seasons:
                if s.get("gameTypeId") != 2:
                    continue
                league = (s.get("leagueAbbrev") or "").upper()
                if not any(ml in league for ml in MAJOR_LEAGUES):
                    continue
                code = s.get("season", 0)
                if code == pre_draft_season:
                    best = s
                elif code == prev_season_code:
                    prev = s
                else:
                    # Fallback: most recent season that predates the draft
                    season_end_year = code % 10000
                    if season_end_year <= pick.year:
                        if fallback is None or code > fallback.get("season", 0):
                            fallback = s
            if best is None:
                best = fallback

            if best:
                gp = best.get("gamesPlayed") or 0
                g = best.get("goals") or 0
                a = best.get("assists") or 0
                pts = g + a
                pick.points_per_game = round(pts / gp, 2) if gp > 0 else 0.0
                pick.gp_pre_draft = gp
            else:
                pick.points_per_game = 0.0  # mark as processed even if no data found
                pick.gp_pre_draft = 0

            if prev:
                gp_prev = prev.get("gamesPlayed") or 0
                pts_prev = (prev.get("goals") or 0) + (prev.get("assists") or 0)
                pick.ppg_prev_season = round(pts_prev / gp_prev, 2) if gp_prev > 0 else 0.0

            # Age at draft: derive from birthDate in API response
            birth_date_str = data.get("birthDate")
            if birth_date_str:
                try:
                    bd = date_type.fromisoformat(birth_date_str)
                    draft_date = date_type(pick.year, 6, 28)  # approximate NHL draft date
                    pick.age_at_draft = round((draft_date - bd).days / 365.25, 2)
                except Exception:
                    pass

            if pick.player_id and best:
                player = db.query(Player).filter(Player.id == pick.player_id).first()
                if player is not None:
                    upsert_player_season_stat(
                        db,
                        player=player,
                        season_year_start=pick.year - 1,
                        season_year_end=pick.year,
                        league=best.get("leagueAbbrev") if best else pick.draft_league,
                        games_played=pick.gp_pre_draft,
                        goals=best.get("goals"),
                        assists=best.get("assists"),
                        points=(best.get("goals") or 0) + (best.get("assists") or 0) if best else None,
                        points_per_game=pick.points_per_game,
                        season_type="pre_draft",
                        source="nhl_player_landing",
                        as_of_date=date_type(pick.year, 6, 28),
                        raw_payload=best,
                    )

            updated += 1
            if updated % 100 == 0:
                db.commit()
                logger.info(f"  Committed {updated} picks so far...")

    db.commit()
    logger.info(f"Pre-draft stats populated for {updated} historical picks")
    return updated


def fetch_historical_css_rankings(db: Session, start_year: int = 2008, end_year: int = 2024) -> int:
    """
    Fetch NHL Central Scouting final rankings for each draft year and store
    them on DraftPickHistorical.css_rank.

    The NHL API provides 4 categories per year:
      1 = north-american-skater
      2 = international-skater
      3 = north-american-goalie
      4 = international-goalie

    Each category ranks prospects independently (CSS#1 NA skater ≠ CSS#1 overall).
    We convert to a single overall rank using the same mapping the NHL uses:
      - Combine all categories, sort by: NA skaters first (by rank), then
        INT skaters, then goalies. This mirrors the actual draft order tendency.

    Matching: by normalized full name (lowercase, stripped). Handles >95% of
    picks. Unmatched picks retain css_rank=NULL and fall back to the
    round-position proxy in training.

    Returns: number of picks updated.
    """
    from app.models import DraftPickHistorical as DPH

    # Category ordering for combined rank: NA skaters → INT skaters → goalies
    CATEGORY_ORDER = [1, 2, 3, 4]

    updated = 0
    with httpx.Client(timeout=30.0) as client:
        for year in range(start_year, end_year + 1):
            # Build combined ranked list across all 4 categories
            # Each entry: (combined_rank, player_name_normalized)
            combined: list[tuple[int, str]] = []
            rank_counter = 1

            for cat_id in CATEGORY_ORDER:
                try:
                    resp = client.get(
                        f"{NHL_API_BASE}/draft/rankings/{year}/{cat_id}",
                        follow_redirects=True,
                    )
                    resp.raise_for_status()
                    prospects = resp.json().get("rankings", [])
                except Exception as e:
                    logger.warning("CSS rankings year=%d cat=%d unavailable: %s", year, cat_id, e)
                    continue

                # Sort by finalRank within the category (some responses are unordered)
                prospects.sort(key=lambda p: p.get("finalRank") or 9999)

                for p in prospects:
                    name = f"{p.get('firstName', '')} {p.get('lastName', '')}".strip().lower()
                    if name:
                        combined.append((rank_counter, name))
                        rank_counter += 1

            if not combined:
                logger.warning("No CSS rankings found for year=%d", year)
                continue

            # Build lookup: normalized_name → combined_overall_rank
            css_lookup: dict[str, int] = {name: rank for rank, name in combined}

            # Fetch all picks for this year
            picks = (
                db.query(DPH)
                .filter(DPH.year == year, DPH.css_rank.is_(None))
                .all()
            )

            matched = 0
            for pick in picks:
                norm_name = (pick.player_name or "").strip().lower()
                if norm_name in css_lookup:
                    pick.css_rank = css_lookup[norm_name]
                    matched += 1

            db.commit()
            updated += matched
            logger.info(
                "CSS rankings year=%d: %d picks, %d matched (%d in CSS list)",
                year, len(picks), matched, len(css_lookup),
            )

    logger.info("Total picks updated with CSS rank: %d", updated)
    return updated


def run_full_ingestion(triggered_by: str = "api") -> str:
    """Run the complete ingestion pipeline in order. Returns the run_id."""
    run_id = str(uuid.uuid4())
    logger.info(f"Starting full NHL data ingestion (run_id={run_id}, triggered_by={triggered_by})...")

    db = SessionLocal()
    try:
        # Create the IngestionRun record at the start
        ingestion_run = IngestionRun(
            run_id=run_id,
            started_at=datetime.now(timezone.utc),
            status="running",
            triggered_by=triggered_by,
            nhl_api_version="v1",
        )
        db.add(ingestion_run)
        db.commit()

        logger.info("=== Step 1: Teams + GMs ===")
        teams_data = fetch_all_teams(db)
        teams_count = len(teams_data)

        logger.info("=== Step 2: GM history stints ===")
        gms_count = seed_gm_history(db)

        logger.info("=== Step 3: Draft History (2000-2024) ===")
        picks_count = fetch_draft_history(db, start_year=2000, end_year=2024)

        logger.info("=== Step 4: Lottery Standings ===")
        fetch_lottery_standings(db)

        logger.info("=== Step 5: Seed 2025 Prospects ===")
        prospects_count = seed_2025_prospects(db)

        logger.info("=== Step 6: Fetch 2025 Prospect Stats ===")
        stats_count = fetch_prospect_stats(db)

        # Update run record with success
        ingestion_run.finished_at = datetime.now(timezone.utc)
        ingestion_run.status = "success"
        ingestion_run.teams_upserted = teams_count
        ingestion_run.gms_upserted = gms_count
        ingestion_run.picks_upserted = picks_count
        ingestion_run.prospects_upserted = prospects_count
        ingestion_run.prospect_stats_updated = stats_count
        db.commit()

        logger.info(f"Full ingestion complete! (run_id={run_id})")
        return run_id

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        db.rollback()
        # Re-query the run record after rollback and mark it failed
        try:
            failed_run = db.query(IngestionRun).filter(IngestionRun.run_id == run_id).first()
            if failed_run:
                failed_run.finished_at = datetime.now(timezone.utc)
                failed_run.status = "failed"
                failed_run.error_message = str(e)[:500]
                db.commit()
        except Exception as inner_e:
            logger.error(f"Could not update ingestion run status to failed: {inner_e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_full_ingestion()
