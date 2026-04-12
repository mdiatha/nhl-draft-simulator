from pydantic import BaseModel


class TeamBase(BaseModel):
    abbreviation: str
    full_name: str
    conference: str | None = None
    division: str | None = None
    tendency_profile: dict | None = None
    need_scores: dict | None = None


class TeamCreate(TeamBase):
    nhl_id: int | None = None


class TeamRead(TeamBase):
    id: int
    nhl_id: int | None = None

    model_config = {"from_attributes": True}
