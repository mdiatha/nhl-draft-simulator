from datetime import date

from sqlalchemy import Date, String, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    nhl_id: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)
    abbreviation: Mapped[str] = mapped_column(String(3), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(100))
    conference: Mapped[str | None] = mapped_column(String(20), nullable=True)
    division: Mapped[str | None] = mapped_column(String(30), nullable=True)
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    current_gm_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    current_gm_since: Mapped[date | None] = mapped_column(Date, nullable=True)

    general_managers: Mapped[list["GeneralManager"]] = relationship(back_populates="team")
