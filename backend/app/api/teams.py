from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.middleware.auth import require_admin_key
from app.models.team import Team
from app.schemas.team import TeamCreate, TeamRead

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("/", response_model=list[TeamRead])
def list_teams(db: Session = Depends(get_db)):
    return db.query(Team).all()


@router.get("/{team_id}", response_model=TeamRead)
def get_team(team_id: int, db: Session = Depends(get_db)):
    team = db.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


@router.post("/", response_model=TeamRead, status_code=201)
def create_team(payload: TeamCreate, db: Session = Depends(get_db), _: None = Depends(require_admin_key)):
    team = Team(**payload.model_dump())
    db.add(team)
    db.commit()
    db.refresh(team)
    return team
