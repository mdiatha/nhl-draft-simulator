from datetime import date

from sqlalchemy import Boolean, Date, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class GeneralManager(Base):
    __tablename__ = "general_managers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    team_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("teams.id"), nullable=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    team: Mapped["Team"] = relationship(back_populates="general_managers")
    draft_picks_historical: Mapped[list["DraftPickHistorical"]] = relationship(back_populates="gm")
    tendency_profile: Mapped["GMTendencyProfile"] = relationship(back_populates="gm", uselist=False)

    __table_args__ = (
        Index("ix_general_managers_team_id", "team_id"),
        Index("ix_general_managers_is_active", "is_active"),
        Index("ix_general_managers_name", "name"),
    )
