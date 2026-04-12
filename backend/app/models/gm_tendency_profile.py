from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class GMTendencyProfile(Base):
    __tablename__ = "gm_tendency_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    gm_id: Mapped[int] = mapped_column(Integer, ForeignKey("general_managers.id"), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # JSON weights (stored as JSONB for efficient querying)
    position_weights: Mapped[dict | None] = mapped_column(JSONB, nullable=True)    # e.g. {"C": 0.3, "D": 0.4, ...}
    league_weights: Mapped[dict | None] = mapped_column(JSONB, nullable=True)      # e.g. {"OHL": 0.4, "SHL": 0.2, ...}
    nationality_weights: Mapped[dict | None] = mapped_column(JSONB, nullable=True) # e.g. {"CAN": 0.5, "USA": 0.2, ...}

    avg_ranking_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)  # avg deviation from CSS rank

    # Archetype classification
    tendency_archetype: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True
    )  # BPA, need-based, system-fit, safe, boom-bust

    # Relationships
    gm: Mapped["GeneralManager"] = relationship("GeneralManager", back_populates="tendency_profile")

    __table_args__ = (
        Index("ix_gm_tendency_profiles_gm_id", "gm_id"),
        Index("ix_gm_tendency_profiles_computed_at", "computed_at"),
        Index("ix_gm_tendency_profiles_tendency_archetype", "tendency_archetype"),
    )
