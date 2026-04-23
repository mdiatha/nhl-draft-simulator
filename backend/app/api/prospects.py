"""Prospects endpoints."""
import re
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Prospect

router = APIRouter(prefix="/draft", tags=["draft"])

_VALID_POSITION = re.compile(r"^[A-Z]{1,3}$")
_VALID_NATIONALITY = re.compile(r"^[A-Z]{2,3}$")


@router.get("/prospects")
async def list_prospects(
    position: Optional[str] = Query(None, max_length=3),
    css_category: Optional[str] = Query(None, max_length=20),
    nationality: Optional[str] = Query(None, max_length=3),
    search: Optional[str] = Query(None, min_length=1, max_length=100),
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(Prospect)
    if position:
        pos = position.upper()
        if not _VALID_POSITION.match(pos):
            raise HTTPException(status_code=422, detail=f"Invalid position: {position!r}")
        q = q.filter(Prospect.position == pos)
    if css_category:
        q = q.filter(Prospect.css_category == css_category)
    if nationality:
        nat = nationality.upper()
        if not _VALID_NATIONALITY.match(nat):
            raise HTTPException(status_code=422, detail=f"Invalid nationality: {nationality!r}")
        q = q.filter(Prospect.nationality == nat)
    if search:
        # Use explicit string concatenation — SQLAlchemy parameterises the bind value,
        # preventing injection even with special characters in the search term.
        q = q.filter(Prospect.name.ilike("%" + search + "%"))

    total = q.count()
    prospects = q.order_by(Prospect.css_ranking.nullslast()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "prospects": [
            {
                "id": p.id, "name": p.name, "position": p.position,
                "nationality": p.nationality, "css_ranking": p.css_ranking,
                "css_category": p.css_category, "draft_league": p.draft_league,
                "draft_league_tier": p.draft_league_tier,
                "points": p.points, "goals": p.goals, "assists": p.assists,
                "games_played": p.games_played, "points_per_game": p.points_per_game,
                "age_at_draft": p.age_at_draft, "height_cm": p.height_cm,
                "weight_kg": p.weight_kg,
            }
            for p in prospects
        ],
    }
