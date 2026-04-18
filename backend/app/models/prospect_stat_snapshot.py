"""SQLAlchemy model for prospect_stat_snapshots (migration 014).

Each row is a point-in-time snapshot of a prospect's stats, tagged with a
snapshot_type so training and the RAG index can use the right data:

  pre_draft     — stats available at draft time (used by ML training to avoid leakage)
  end_of_season — final season stats (used for RAG embeddings — most complete data)
  mid_season    — ad-hoc snapshot taken during the season

See alembic/versions/014_prospect_stat_snapshots.py for the full rationale.
"""
from __future__ import annotations

import datetime

from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, Index, func
from sqlalchemy.orm import relationship

from app.database import Base


class ProspectStatSnapshot(Base):
    __tablename__ = "prospect_stat_snapshots"

    id               = Column(Integer, primary_key=True)
    prospect_id      = Column(Integer, ForeignKey("prospects_2025.id", ondelete="SET NULL"),
                               nullable=True, index=True)
    player_name      = Column(String(100), nullable=False)
    draft_year       = Column(Integer, nullable=False)
    draft_league     = Column(String(100), nullable=True)
    snapshot_type    = Column(String(30), nullable=False)  # pre_draft | end_of_season | mid_season
    snapshot_date    = Column(Date, nullable=False)
    points_per_game  = Column(Float, nullable=True)
    goals_per_game   = Column(Float, nullable=True)
    assists_per_game = Column(Float, nullable=True)
    games_played     = Column(Integer, nullable=True)
    ppg_prev_season  = Column(Float, nullable=True)
    css_ranking      = Column(Integer, nullable=True)
    created_at       = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    prospect = relationship("Prospect2025", back_populates="stat_snapshots", foreign_keys=[prospect_id])

    __table_args__ = (
        Index("idx_snapshots_year_type", "draft_year", "snapshot_type"),
        Index("idx_snapshots_player_name", "player_name"),
    )

    def __repr__(self) -> str:
        return (
            f"<ProspectStatSnapshot id={self.id} player={self.player_name!r} "
            f"year={self.draft_year} type={self.snapshot_type!r}>"
        )
