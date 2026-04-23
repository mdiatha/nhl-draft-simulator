from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ProspectRanking(Base):
    __tablename__ = "prospect_rankings"
    __table_args__ = (
        UniqueConstraint(
            "prospect_id",
            "source",
            "ranking_type",
            "category",
            name="uq_prospect_rankings_source_type_category",
        ),
        Index("ix_prospect_rankings_source_rank", "source", "rank"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prospect_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("prospects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    ranking_type: Mapped[str] = mapped_column(String(30), nullable=False)
    category: Mapped[str | None] = mapped_column(String(30), nullable=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    ranking_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    prospect: Mapped["Prospect"] = relationship(back_populates="rankings")
