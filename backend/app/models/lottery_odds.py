from sqlalchemy import Float, ForeignKey, Index, Integer
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class LotteryOdds(Base):
    __tablename__ = "lottery_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)           # e.g. 2025
    final_standing: Mapped[int | None] = mapped_column(Integer, nullable=True)    # 1 = last place
    lottery_odds_pct: Mapped[float | None] = mapped_column(Float, nullable=True)  # e.g. 18.5 for 18.5%
    assigned_combinations: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)  # array of ints 1-1000
    wins: Mapped[int | None] = mapped_column(Integer, nullable=True)
    losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    otl: Mapped[int | None] = mapped_column(Integer, nullable=True)
    points: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    team: Mapped["Team"] = relationship("Team", foreign_keys=[team_id])

    __table_args__ = (
        Index("ix_lottery_odds_team_id", "team_id"),
        Index("ix_lottery_odds_season", "season"),
        Index("ix_lottery_odds_final_standing", "final_standing"),
        Index("ix_lottery_odds_team_season", "team_id", "season", unique=True),
    )
