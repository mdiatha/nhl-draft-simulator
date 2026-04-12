from sqlalchemy import Column, Integer, String, Float, Date, Index
from sqlalchemy.orm import relationship
from app.database import Base


class Prospect2025(Base):
    __tablename__ = "prospects_2025"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    position = Column(String(10), nullable=False)   # C, LW, RW, D, G
    nationality = Column(String(50), nullable=True)
    height_cm = Column(Integer, nullable=True)
    weight_kg = Column(Integer, nullable=True)

    # League info
    draft_league = Column(String(100), nullable=True)
    draft_league_tier = Column(Integer, nullable=True)  # 1=top, 2=mid, 3=lower

    # CSS ranking info
    css_ranking = Column(Integer, nullable=True)
    css_category = Column(String(20), nullable=True)  # "NA skater", "EUR skater", "goalie"

    # Stats
    points = Column(Integer, nullable=True)
    goals = Column(Integer, nullable=True)
    assists = Column(Integer, nullable=True)
    games_played = Column(Integer, nullable=True)    # pre-draft season GP
    points_per_game = Column(Float, nullable=True)   # pre-draft season PPG
    ppg_prev_season = Column(Float, nullable=True)   # PPG from prior season

    # Age/birth info
    age_at_draft = Column(Float, nullable=True)   # age on draft day (can be decimal)
    birth_date = Column(Date, nullable=True)

    # NHL player ID (from NHL Records API) — used to fetch stats
    nhl_player_id = Column(Integer, nullable=True, index=True)

    stat_history = relationship("ProspectStatHistory", back_populates="prospect",
                               order_by="ProspectStatHistory.fetched_at.desc()",
                               lazy="dynamic")

    __table_args__ = (
        Index("ix_prospects_2025_css_ranking", "css_ranking"),
        Index("ix_prospects_2025_css_category", "css_category"),
        Index("ix_prospects_2025_position", "position"),
        Index("ix_prospects_2025_nationality", "nationality"),
        Index("ix_prospects_2025_draft_league", "draft_league"),
    )
