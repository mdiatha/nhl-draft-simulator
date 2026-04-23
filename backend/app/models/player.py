from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    nhl_player_id: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    birth_country: Mapped[str | None] = mapped_column(String(3), nullable=True)
    position: Mapped[str] = mapped_column(String(10), nullable=False)
    shoots_catches: Mapped[str | None] = mapped_column(String(1), nullable=True)
    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weight_kg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    draft_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    draft_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    draft_overall: Mapped[int | None] = mapped_column(Integer, nullable=True)
    css_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    composite_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    prospects: Mapped[list["Prospect"]] = relationship(back_populates="player")
    season_stats: Mapped[list["PlayerSeasonStat"]] = relationship(back_populates="player")
    historical_picks: Mapped[list["DraftPickHistorical"]] = relationship(back_populates="player")

    __table_args__ = (
        Index("ix_players_full_name_birth_date", "full_name", "birth_date"),
    )
