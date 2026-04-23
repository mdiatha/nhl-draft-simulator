from datetime import date

from sqlalchemy import Boolean, Date, ForeignKey, Index, Integer, String, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Prospect(Base):
    __tablename__ = "prospects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("players.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    draft_class_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("draft_classes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    position: Mapped[str] = mapped_column(String(10), nullable=False)
    nationality: Mapped[str | None] = mapped_column(String(50), nullable=True)
    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weight_kg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    draft_league: Mapped[str | None] = mapped_column(String(100), nullable=True)
    draft_league_tier: Mapped[int | None] = mapped_column(Integer, nullable=True)
    css_ranking: Mapped[int | None] = mapped_column(Integer, nullable=True)
    css_category: Mapped[str | None] = mapped_column(String(20), nullable=True)
    points: Mapped[int | None] = mapped_column(Integer, nullable=True)
    goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assists: Mapped[int | None] = mapped_column(Integer, nullable=True)
    games_played: Mapped[int | None] = mapped_column(Integer, nullable=True)
    points_per_game: Mapped[float | None] = mapped_column(Float, nullable=True)
    ppg_prev_season: Mapped[float | None] = mapped_column(Float, nullable=True)
    age_at_draft: Mapped[float | None] = mapped_column(Float, nullable=True)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    nhl_player_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    player: Mapped["Player | None"] = relationship(back_populates="prospects")
    draft_class: Mapped["DraftClass | None"] = relationship(back_populates="prospects")
    stat_history: Mapped[list["ProspectStatHistory"]] = relationship(
        back_populates="prospect",
        order_by="ProspectStatHistory.fetched_at.desc()",
        lazy="dynamic",
    )
    stat_snapshots: Mapped[list["ProspectStatSnapshot"]] = relationship(
        back_populates="prospect",
        order_by="ProspectStatSnapshot.snapshot_date.desc()",
        lazy="dynamic",
    )
    features: Mapped["ProspectFeatures | None"] = relationship(
        back_populates="prospect",
        uselist=False,
    )
    rankings: Mapped[list["ProspectRanking"]] = relationship(
        back_populates="prospect",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_prospects_css_ranking", "css_ranking"),
        Index("ix_prospects_css_category", "css_category"),
        Index("ix_prospects_position", "position"),
        Index("ix_prospects_nationality", "nationality"),
        Index("ix_prospects_draft_league", "draft_league"),
        Index("ix_prospects_draft_class_css", "draft_class_id", "css_ranking"),
    )
