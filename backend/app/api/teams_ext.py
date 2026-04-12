"""Extended team endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Team, GeneralManager, GMTendencyProfile, DraftPickHistorical
from app.engines.tendency_engine import compute_gm_tendency

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("/{team_id}/tendency")
async def get_team_tendency(team_id: int, db: Session = Depends(get_db)):
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(404, "Team not found")

    gm = db.query(GeneralManager).filter(
        GeneralManager.team_id == team_id,
        GeneralManager.is_active.is_(True),
    ).first()

    profile = None
    tendency = {}

    if gm:
        profile = db.query(GMTendencyProfile).filter(GMTendencyProfile.gm_id == gm.id).first()
        if profile:
            tendency = {
                "position_weights": profile.position_weights or {},
                "league_weights": profile.league_weights or {},
                "nationality_weights": profile.nationality_weights or {},
                "avg_ranking_deviation": profile.avg_ranking_deviation,
            }
        else:
            tendency = compute_gm_tendency(gm.id, db)

        picks = (
            db.query(DraftPickHistorical)
            .filter(DraftPickHistorical.gm_id == gm.id)
            .order_by(DraftPickHistorical.year.desc(), DraftPickHistorical.overall_pick)
            .all()
        )
        draft_history = [
            {
                "year": p.year, "round": p.round, "pick_number": p.pick_number,
                "overall_pick": p.overall_pick, "player_name": p.player_name,
                "position": p.position, "draft_league": p.draft_league,
                "nationality": p.nationality,
            }
            for p in picks
        ]
        total_picks = len(picks)
        year_range = f"{min(p.year for p in picks)}-{max(p.year for p in picks)}" if picks else "N/A"
    else:
        draft_history = []
        total_picks = 0
        year_range = "N/A"

    return {
        "team_id": team_id,
        "team_name": getattr(team, "name", None) or getattr(team, "full_name", ""),
        "abbreviation": team.abbreviation,
        "gm": {"id": gm.id, "name": gm.name, "start_date": str(gm.start_date)} if gm else None,
        "tendency": tendency,
        "draft_history": draft_history,
        "total_picks": total_picks,
        "year_range": year_range,
        "computed_at": str(profile.computed_at) if profile else None,
    }
