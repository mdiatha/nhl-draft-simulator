from sqlalchemy import Column, Integer, String, Float, ForeignKey, Index
from sqlalchemy.orm import relationship
from app.database import Base


class DraftPickHistorical(Base):
    __tablename__ = "draft_picks_historical"

    id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    round = Column(Integer, nullable=False)
    pick_number = Column(Integer, nullable=False)   # pick within the round
    overall_pick = Column(Integer, nullable=False)  # overall draft position

    # Foreign Keys
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    gm_id = Column(Integer, ForeignKey("general_managers.id"), nullable=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True, index=True)

    # Player info (denormalized - player may not be in prospects table)
    player_name = Column(String(200), nullable=True)
    position = Column(String(10), nullable=True)    # C, LW, RW, D, G
    nationality = Column(String(50), nullable=True)
    height_cm = Column(Integer, nullable=True)
    weight_kg = Column(Integer, nullable=True)

    # League/tier info
    draft_league = Column(String(100), nullable=True)  # OHL, WHL, QMJHL, SHL, etc.
    draft_league_tier = Column(Integer, nullable=True)  # 1=top, 2=mid, 3=lower

    # NHL player ID (used to fetch pre-draft stats via NHL API)
    nhl_player_id = Column(Integer, nullable=True, index=True)

    # Pre-draft CSS ranking (fetched from NHL CSS rankings API for 2008+)
    # NULL for picks before 2008 or prospects who weren't CSS-ranked.
    css_rank = Column(Integer, nullable=True)

    # Pre-draft season stats (fetched from NHL player API)
    points_per_game = Column(Float, nullable=True)   # PPG in season before draft
    gp_pre_draft = Column(Integer, nullable=True)    # games played in that season
    ppg_prev_season = Column(Float, nullable=True)   # PPG two seasons before draft
    age_at_draft = Column(Float, nullable=True)

    # Relationships
    team = relationship("Team", foreign_keys=[team_id])
    gm = relationship("GeneralManager", back_populates="draft_picks_historical")
    player = relationship("Player", back_populates="historical_picks")

    __table_args__ = (
        Index("ix_draft_picks_historical_year", "year"),
        Index("ix_draft_picks_historical_team_id", "team_id"),
        Index("ix_draft_picks_historical_gm_id", "gm_id"),
        Index("ix_draft_picks_historical_player_id", "player_id"),
        Index("ix_draft_picks_historical_overall_pick", "overall_pick"),
        Index("ix_draft_picks_historical_position", "position"),
        Index("ix_draft_picks_historical_year_round", "year", "round"),
    )
