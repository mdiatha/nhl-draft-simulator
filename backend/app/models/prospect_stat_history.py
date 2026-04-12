"""SQLAlchemy model for prospect_stat_history.

Time-series table storing live pre-draft stats fetched from the NHL API.
Each row is a snapshot of a prospect's stats at the time of the fetch.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.database import Base


class ProspectStatHistory(Base):
    __tablename__ = "prospect_stat_history"

    id              = Column(Integer, primary_key=True)
    prospect_id     = Column(Integer, ForeignKey("prospects_2025.id", ondelete="CASCADE"), nullable=False, index=True)
    fetched_at      = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    season_type     = Column(String(20),  nullable=True)   # pre_draft | regular | playoffs
    league          = Column(String(100), nullable=True)
    games_played    = Column(Integer,     nullable=True)
    goals           = Column(Integer,     nullable=True)
    assists         = Column(Integer,     nullable=True)
    points          = Column(Integer,     nullable=True)
    points_per_game = Column(Float,       nullable=True)
    source_url      = Column(String(500), nullable=True)
    raw_payload     = Column(JSONB,       nullable=True)

    prospect = relationship("Prospect2025", back_populates="stat_history")

    def __repr__(self) -> str:
        return f"<ProspectStatHistory prospect_id={self.prospect_id} ppg={self.points_per_game} fetched={self.fetched_at}>"
