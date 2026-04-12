"""Admin endpoints — data ingestion, tendency computation, ML training triggers."""
import logging
import uuid
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.middleware.auth import require_admin_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_key)])


@router.post("/ingest")
async def run_ingestion(background_tasks: BackgroundTasks):
    """
    Kick off the full NHL data ingestion pipeline in the background:
      1. Teams + GMs
      2. Draft history (2000-2024)
      3. Lottery standings
      4. 2025 prospects
      5. Prospect stats (ppg_prev_season)
    Runs data quality checks on completion and logs results.
    Returns a run_id for tracking in logs.
    """
    run_id = str(uuid.uuid4())

    def _run():
        from app.database import SessionLocal
        from app.ingestion.nhl_api import run_full_ingestion
        from app.observability.data_quality import run_ingestion_checks
        logger.info("ingestion.started", extra={"run_id": run_id})
        try:
            run_full_ingestion()
            db = SessionLocal()
            try:
                report = run_ingestion_checks(db)
                if not report.passed:
                    logger.error(
                        "ingestion.quality_failures",
                        extra={"run_id": run_id, "failures": [c.name for c in report.failed_checks]},
                    )
                else:
                    logger.info("ingestion.complete", extra={"run_id": run_id})
            finally:
                db.close()
        except Exception as e:
            logger.error("ingestion.failed", extra={"run_id": run_id, "error": str(e), "type": type(e).__name__})

    background_tasks.add_task(_run)
    return {"status": "started", "run_id": run_id, "message": "Full ingestion running in background. Check logs for run_id progress."}


@router.post("/seed-prospects")
async def seed_prospects(force: bool = False, db: Session = Depends(get_db)):
    """Seed the 2025 draft prospect table from the NHL Records API."""
    from app.ingestion.nhl_api import seed_2025_prospects
    count = seed_2025_prospects(db, force=force)
    return {"status": "ok", "prospects_seeded": count}


@router.post("/compute-tendencies")
async def compute_tendencies(db: Session = Depends(get_db)):
    """Compute (or recompute) GM tendency profiles for all active GMs."""
    from app.engines.tendency_engine import compute_all_gm_tendencies
    profiles = compute_all_gm_tendencies(db)
    return {"status": "ok", "profiles_computed": len(profiles)}


@router.post("/fetch-prospect-stats")
async def fetch_prospect_stats(background_tasks: BackgroundTasks):
    """
    Fetch current-season stats for all 2025 prospects with an nhl_player_id.
    Populates points_per_game, goals, assists used as ML features.
    """
    def _run():
        from app.database import SessionLocal
        from app.ingestion.nhl_api import fetch_prospect_stats as _fetch
        db = SessionLocal()
        try:
            updated = _fetch(db)
            logger.info("prospect_stats.complete", extra={"updated": updated})
        except Exception as e:
            logger.error("prospect_stats.failed", extra={"error": str(e), "type": type(e).__name__})
        finally:
            db.close()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Fetching prospect stats in background."}


@router.post("/fetch-historical-stats")
async def fetch_historical_stats(background_tasks: BackgroundTasks):
    """
    Fetch pre-draft season stats for historical picks (points_per_game, age_at_draft).
    These are the ML training features — run once after ingesting draft history.
    """
    def _run():
        from app.database import SessionLocal
        from app.ingestion.nhl_api import fetch_historical_stats as _fetch
        db = SessionLocal()
        try:
            updated = _fetch(db)
            logger.info("historical_stats.complete", extra={"updated": updated})
        except Exception as e:
            logger.error("historical_stats.failed", extra={"error": str(e), "type": type(e).__name__})
        finally:
            db.close()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Fetching historical pre-draft stats in background (~15 min)."}


@router.post("/ingest/stats")
async def ingest_live_stats(background_tasks: BackgroundTasks):
    """
    Fetch live pre-draft stats from the NHL API for all 2025 prospects with a
    known nhl_player_id and persist to prospect_stat_history + update prospects_2025.

    This is the same action triggered by the daily EventBridge CronJob → Lambda.
    Use this for manual on-demand refreshes.
    """
    def _run():
        import asyncio
        from app.database import SessionLocal
        from app.ingestion.stat_ingestion import fetch_and_store_stats
        db = SessionLocal()
        try:
            result = asyncio.run(fetch_and_store_stats(db))
            logger.info("manual_stat_ingest.complete result=%s", result)
        except Exception as exc:
            logger.exception("manual_stat_ingest.failed error=%s", exc)
        finally:
            db.close()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Live stat ingestion running in background."}


@router.get("/ingest/stats/history")
async def stat_ingestion_history(prospect_id: int, limit: int = 20, db: Session = Depends(get_db)):
    """
    Return recent stat history rows for a given prospect.
    Useful for verifying live stat ingestion is working and tracking PPG trends.
    """
    from app.models.prospect_stat_history import ProspectStatHistory
    from sqlalchemy import desc

    rows = (
        db.query(ProspectStatHistory)
        .filter(ProspectStatHistory.prospect_id == prospect_id)
        .order_by(desc(ProspectStatHistory.fetched_at))
        .limit(min(limit, 100))
        .all()
    )
    return {
        "prospect_id": prospect_id,
        "history": [
            {
                "fetched_at":      r.fetched_at.isoformat() if r.fetched_at else None,
                "season_type":     r.season_type,
                "league":          r.league,
                "games_played":    r.games_played,
                "goals":           r.goals,
                "assists":         r.assists,
                "points":          r.points,
                "points_per_game": r.points_per_game,
            }
            for r in rows
        ],
    }


@router.post("/feature-store/refresh")
async def refresh_feature_store(background_tasks: BackgroundTasks):
    """
    Recompute and cache all prospect features in the prospect_features table.
    Cuts scoring latency by pre-materializing the feature matrix for all ~224 prospects.
    Run after stat ingestion or before a draft simulation to warm the cache.
    """
    def _run():
        from app.database import SessionLocal
        from app.ml.feature_store import refresh_feature_store as _refresh
        db = SessionLocal()
        try:
            count = _refresh(db)
            logger.info("feature_store.refresh_complete count=%d", count)
        except Exception as exc:
            logger.exception("feature_store.refresh_failed error=%s", exc)
        finally:
            db.close()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Feature store refresh running in background."}


@router.post("/ingest-nhl-players")
async def ingest_nhl_players(background_tasks: BackgroundTasks):
    """
    Fetch current NHL player stats from the NHL API, embed each player profile
    via voyage-3-lite, and upsert into scout_embeddings as 'nhl_player' docs.

    Required for the get_nhl_comp agent tool (prospect → NHL player comps).
    Run once after initial setup; re-run any time you want fresh NHL stats.
    After this completes, run POST /api/agent/index to rebuild the full RAG index
    (or let the next scheduled rebuild pick it up automatically).
    """
    def _run():
        from app.database import SessionLocal
        from app.ingestion.nhl_players_ingestion import fetch_and_embed_nhl_players
        db = SessionLocal()
        try:
            count = fetch_and_embed_nhl_players(db)
            logger.info("nhl_players.ingestion_complete", extra={"count": count})
        except Exception as exc:
            logger.exception("nhl_players.ingestion_failed error=%s", exc)
        finally:
            db.close()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "NHL player ingestion running in background."}


@router.post("/fetch-css-rankings")
async def fetch_css_rankings(background_tasks: BackgroundTasks):
    """
    Fetch NHL Central Scouting final rankings for 2008–2024 and store on
    draft_picks_historical.css_rank. Replaces the tautological round-position
    proxy in training with real pre-draft consensus ranks.
    Run once after draft history is ingested, before training the model.
    """
    def _run():
        from app.database import SessionLocal
        from app.ingestion.nhl_api import fetch_historical_css_rankings
        db = SessionLocal()
        try:
            updated = fetch_historical_css_rankings(db)
            logger.info("css_rankings.complete", extra={"updated": updated})
        except Exception as e:
            logger.error("css_rankings.failed", extra={"error": str(e), "type": type(e).__name__})
        finally:
            db.close()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Fetching historical CSS rankings in background (~2 min)."}
