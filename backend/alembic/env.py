from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base
from app.config import settings

# Import ALL models so Alembic can detect them
from app.models.team import Team
from app.models.general_manager import GeneralManager
from app.models.draft_pick_historical import DraftPickHistorical
from app.models.player import Player
from app.models.draft_class import DraftClass
from app.models.prospect import Prospect
from app.models.prospect_ranking import ProspectRanking
from app.models.player_season_stat import PlayerSeasonStat
from app.models.lottery_odds import LotteryOdds
from app.models.gm_tendency_profile import GMTendencyProfile
from app.models.ingestion_run import IngestionRun

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url():
    # Allow callers (e.g. integration test conftest) to override via set_main_option.
    return config.get_main_option("sqlalchemy.url") or settings.DATABASE_URL


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
