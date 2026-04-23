"""SQLAlchemy model for prospect_features — the materialized feature store."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.database import Base


class ProspectFeatures(Base):
    __tablename__ = "prospect_features"
    __table_args__ = (
        UniqueConstraint("prospect_id", name="uq_prospect_features_prospect_id"),
    )

    id              = Column(Integer, primary_key=True)
    prospect_id     = Column(Integer, ForeignKey("prospects.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    refreshed_at    = Column(DateTime(timezone=True), nullable=False,
                             default=lambda: datetime.now(timezone.utc))
    features_json   = Column(JSONB, nullable=False)
    css_rank_norm   = Column(Float, nullable=True)
    ppg_league_norm = Column(Float, nullable=True)
    pick_slot_norm  = Column(Float, nullable=True)
    prospect        = relationship("Prospect", back_populates="features")

    def __repr__(self) -> str:
        return f"<ProspectFeatures prospect_id={self.prospect_id} refreshed={self.refreshed_at}>"
