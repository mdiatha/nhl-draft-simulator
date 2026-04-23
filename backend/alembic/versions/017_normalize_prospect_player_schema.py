"""Normalize player and prospect schema for point-in-time draft modeling.

Revision ID: 017
Revises: 016
Create Date: 2026-04-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Reuse and modernize the dormant players table ───────────────────────
    op.alter_column("players", "nhl_id", new_column_name="nhl_player_id")
    op.add_column("players", sa.Column("full_name", sa.String(length=200), nullable=True))
    op.add_column(
        "players",
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.add_column(
        "players",
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.alter_column("players", "position", type_=sa.String(length=10), existing_type=sa.String(length=2))
    op.create_index("ix_players_full_name", "players", ["full_name"], unique=False)
    op.create_index("ix_players_full_name_birth_date", "players", ["full_name", "birth_date"], unique=False)

    op.execute(
        """
        UPDATE players
        SET full_name = NULLIF(trim(concat_ws(' ', first_name, last_name)), '')
        WHERE full_name IS NULL
        """
    )

    # ── New normalized draft context tables ─────────────────────────────────
    op.create_table(
        "draft_classes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("year", name="uq_draft_classes_year"),
    )
    op.create_index("ix_draft_classes_year", "draft_classes", ["year"], unique=True)

    op.execute(
        """
        INSERT INTO draft_classes (year)
        SELECT DISTINCT year
        FROM draft_picks_historical
        WHERE year IS NOT NULL
        ORDER BY year
        """
    )
    op.execute("INSERT INTO draft_classes (year) VALUES (2026) ON CONFLICT (year) DO NOTHING")

    # prospects_2025 becomes the canonical prospects table
    op.rename_table("prospects_2025", "prospects")
    op.execute("ALTER INDEX IF EXISTS ix_prospects_2025_css_ranking RENAME TO ix_prospects_css_ranking")
    op.execute("ALTER INDEX IF EXISTS ix_prospects_2025_css_category RENAME TO ix_prospects_css_category")
    op.execute("ALTER INDEX IF EXISTS ix_prospects_2025_position RENAME TO ix_prospects_position")
    op.execute("ALTER INDEX IF EXISTS ix_prospects_2025_nationality RENAME TO ix_prospects_nationality")
    op.execute("ALTER INDEX IF EXISTS ix_prospects_2025_draft_league RENAME TO ix_prospects_draft_league")

    op.add_column("prospects", sa.Column("player_id", sa.Integer(), nullable=True))
    op.add_column("prospects", sa.Column("draft_class_id", sa.Integer(), nullable=True))
    op.add_column(
        "prospects",
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
    )
    op.create_index("ix_prospects_player_id", "prospects", ["player_id"], unique=False)
    op.create_index("ix_prospects_draft_class_id", "prospects", ["draft_class_id"], unique=False)
    op.create_index("ix_prospects_name", "prospects", ["name"], unique=False)
    op.create_index("ix_prospects_draft_class_css", "prospects", ["draft_class_id", "css_ranking"], unique=False)
    op.create_foreign_key("fk_prospects_player_id", "prospects", "players", ["player_id"], ["id"], ondelete="SET NULL")
    op.create_foreign_key("fk_prospects_draft_class_id", "prospects", "draft_classes", ["draft_class_id"], ["id"], ondelete="SET NULL")

    # Backfill current prospects into players
    op.execute(
        """
        INSERT INTO players (
            nhl_player_id,
            full_name,
            first_name,
            last_name,
            birth_date,
            birth_country,
            position,
            height_cm,
            weight_kg,
            draft_year,
            draft_round,
            draft_overall,
            css_rank
        )
        SELECT DISTINCT
            p.nhl_player_id,
            p.name,
            split_part(p.name, ' ', 1),
            CASE
                WHEN strpos(p.name, ' ') > 0 THEN substr(p.name, strpos(p.name, ' ') + 1)
                ELSE ''
            END,
            p.birth_date,
            left(COALESCE(p.nationality, ''), 3),
            p.position,
            p.height_cm,
            p.weight_kg,
            2026,
            NULL::integer,
            p.css_ranking,
            p.css_ranking
        FROM prospects p
        WHERE NOT EXISTS (
            SELECT 1
            FROM players pl
            WHERE
                (p.nhl_player_id IS NOT NULL AND pl.nhl_player_id = p.nhl_player_id)
                OR (p.nhl_player_id IS NULL AND pl.full_name = p.name)
        )
        """
    )

    op.execute(
        """
        UPDATE prospects p
        SET player_id = pl.id,
            draft_class_id = dc.id
        FROM players pl
        JOIN draft_classes dc ON dc.year = 2026
        WHERE
            ((p.nhl_player_id IS NOT NULL AND pl.nhl_player_id = p.nhl_player_id)
             OR (p.nhl_player_id IS NULL AND pl.full_name = p.name))
            AND p.player_id IS NULL
        """
    )

    op.create_table(
        "prospect_rankings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("prospect_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("ranking_type", sa.String(length=30), nullable=False),
        sa.Column("category", sa.String(length=30), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("ranking_date", sa.Date(), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["prospect_id"], ["prospects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "prospect_id",
            "source",
            "ranking_type",
            "category",
            name="uq_prospect_rankings_source_type_category",
        ),
    )
    op.create_index("ix_prospect_rankings_source_rank", "prospect_rankings", ["source", "rank"], unique=False)

    op.execute(
        """
        INSERT INTO prospect_rankings (prospect_id, source, ranking_type, category, rank)
        SELECT id, 'nhl_css', 'final', css_category, css_ranking
        FROM prospects
        WHERE css_ranking IS NOT NULL
        """
    )

    op.create_table(
        "player_season_stats",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("season_year_start", sa.Integer(), nullable=False),
        sa.Column("season_year_end", sa.Integer(), nullable=False),
        sa.Column("league", sa.String(length=100), nullable=True),
        sa.Column("team_name", sa.String(length=100), nullable=True),
        sa.Column("season_type", sa.String(length=20), nullable=False),
        sa.Column("games_played", sa.Integer(), nullable=True),
        sa.Column("goals", sa.Integer(), nullable=True),
        sa.Column("assists", sa.Integer(), nullable=True),
        sa.Column("points", sa.Integer(), nullable=True),
        sa.Column("points_per_game", sa.Float(), nullable=True),
        sa.Column("as_of_date", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["player_id"], ["players.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_player_season_stats_player_window",
        "player_season_stats",
        ["player_id", "season_year_start", "season_year_end"],
        unique=False,
    )
    op.create_index("ix_player_season_stats_as_of_date", "player_season_stats", ["as_of_date"], unique=False)

    # Link historical picks to canonical players
    op.add_column("draft_picks_historical", sa.Column("player_id", sa.Integer(), nullable=True))
    op.create_index("ix_draft_picks_historical_player_id", "draft_picks_historical", ["player_id"], unique=False)
    op.create_foreign_key(
        "fk_draft_picks_historical_player_id",
        "draft_picks_historical",
        "players",
        ["player_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.execute(
        """
        INSERT INTO players (
            nhl_player_id,
            full_name,
            first_name,
            last_name,
            birth_country,
            position,
            height_cm,
            weight_kg,
            draft_year,
            draft_round,
            draft_overall,
            css_rank
        )
        SELECT DISTINCT
            d.nhl_player_id,
            d.player_name,
            split_part(d.player_name, ' ', 1),
            CASE
                WHEN strpos(d.player_name, ' ') > 0 THEN substr(d.player_name, strpos(d.player_name, ' ') + 1)
                ELSE ''
            END,
            left(COALESCE(d.nationality, ''), 3),
            COALESCE(d.position, 'F'),
            d.height_cm,
            d.weight_kg,
            d.year,
            d.round,
            d.overall_pick,
            d.css_rank
        FROM draft_picks_historical d
        WHERE d.player_name IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM players pl
              WHERE
                (d.nhl_player_id IS NOT NULL AND pl.nhl_player_id = d.nhl_player_id)
                OR (d.nhl_player_id IS NULL AND pl.full_name = d.player_name AND pl.draft_year = d.year)
          )
        ON CONFLICT DO NOTHING
        """
    )

    op.execute(
        """
        UPDATE draft_picks_historical d
        SET player_id = pl.id
        FROM players pl
        WHERE
            d.player_name IS NOT NULL
            AND (
                (d.nhl_player_id IS NOT NULL AND pl.nhl_player_id = d.nhl_player_id)
                OR (d.nhl_player_id IS NULL AND pl.full_name = d.player_name AND pl.draft_year = d.year)
            )
            AND d.player_id IS NULL
        """
    )


def downgrade() -> None:
    op.drop_constraint("fk_draft_picks_historical_player_id", "draft_picks_historical", type_="foreignkey")
    op.drop_index("ix_draft_picks_historical_player_id", table_name="draft_picks_historical")
    op.drop_column("draft_picks_historical", "player_id")

    op.drop_index("ix_player_season_stats_as_of_date", table_name="player_season_stats")
    op.drop_index("ix_player_season_stats_player_window", table_name="player_season_stats")
    op.drop_table("player_season_stats")

    op.drop_index("ix_prospect_rankings_source_rank", table_name="prospect_rankings")
    op.drop_table("prospect_rankings")

    op.drop_constraint("fk_prospects_draft_class_id", "prospects", type_="foreignkey")
    op.drop_constraint("fk_prospects_player_id", "prospects", type_="foreignkey")
    op.drop_index("ix_prospects_draft_class_css", table_name="prospects")
    op.drop_index("ix_prospects_name", table_name="prospects")
    op.drop_index("ix_prospects_draft_class_id", table_name="prospects")
    op.drop_index("ix_prospects_player_id", table_name="prospects")
    op.drop_column("prospects", "is_active")
    op.drop_column("prospects", "draft_class_id")
    op.drop_column("prospects", "player_id")
    op.execute("ALTER INDEX IF EXISTS ix_prospects_css_ranking RENAME TO ix_prospects_2025_css_ranking")
    op.execute("ALTER INDEX IF EXISTS ix_prospects_css_category RENAME TO ix_prospects_2025_css_category")
    op.execute("ALTER INDEX IF EXISTS ix_prospects_position RENAME TO ix_prospects_2025_position")
    op.execute("ALTER INDEX IF EXISTS ix_prospects_nationality RENAME TO ix_prospects_2025_nationality")
    op.execute("ALTER INDEX IF EXISTS ix_prospects_draft_league RENAME TO ix_prospects_2025_draft_league")
    op.rename_table("prospects", "prospects_2025")

    op.drop_index("ix_draft_classes_year", table_name="draft_classes")
    op.drop_table("draft_classes")

    op.drop_index("ix_players_full_name_birth_date", table_name="players")
    op.drop_index("ix_players_full_name", table_name="players")
    op.drop_column("players", "updated_at")
    op.drop_column("players", "created_at")
    op.drop_column("players", "full_name")
    op.alter_column("players", "nhl_player_id", new_column_name="nhl_id")
    op.alter_column("players", "position", type_=sa.String(length=2), existing_type=sa.String(length=10))
