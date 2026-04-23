from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class PlayerSeasonStat(Base):
    __tablename__ = "player_season_stats"
    __table_args__ = (
        Index(
            "ix_player_season_stats_player_window",
            "player_id",
            "season_year_start",
            "season_year_end",
        ),
        Index("ix_player_season_stats_as_of_date", "as_of_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    season_year_start: Mapped[int] = mapped_column(Integer, nullable=False)
    season_year_end: Mapped[int] = mapped_column(Integer, nullable=False)
    league: Mapped[str | None] = mapped_column(String(100), nullable=True)
    team_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    season_type: Mapped[str] = mapped_column(String(20), nullable=False, default="regular")
    games_played: Mapped[int | None] = mapped_column(Integer, nullable=True)
    goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assists: Mapped[int | None] = mapped_column(Integer, nullable=True)
    points: Mapped[int | None] = mapped_column(Integer, nullable=True)
    points_per_game: Mapped[float | None] = mapped_column(Float, nullable=True)
    as_of_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    raw_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    player: Mapped["Player"] = relationship(back_populates="season_stats")
