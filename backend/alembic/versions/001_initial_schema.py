"""initial schema

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. teams (no FK deps)
    op.create_table(
        'teams',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('nhl_id', sa.Integer(), nullable=True),
        sa.Column('abbreviation', sa.String(3), nullable=False),
        sa.Column('full_name', sa.String(100), nullable=False),
        sa.Column('conference', sa.String(20), nullable=True),
        sa.Column('division', sa.String(30), nullable=True),
        sa.Column('city', sa.String(100), nullable=True),
        sa.Column('current_gm_name', sa.String(200), nullable=True),
        sa.Column('current_gm_since', sa.Date(), nullable=True),
        sa.Column('tendency_profile', sa.JSON(), nullable=True),
        sa.Column('need_scores', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_teams_id', 'teams', ['id'])
    op.create_index('ix_teams_abbreviation', 'teams', ['abbreviation'], unique=True)
    op.create_index('ix_teams_nhl_id', 'teams', ['nhl_id'], unique=True)

    # 2. general_managers (FK: teams)
    op.create_table(
        'general_managers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('start_date', sa.Date(), nullable=False),
        sa.Column('end_date', sa.Date(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_general_managers_team_id', 'general_managers', ['team_id'])
    op.create_index('ix_general_managers_is_active', 'general_managers', ['is_active'])
    op.create_index('ix_general_managers_name', 'general_managers', ['name'])

    # 3. players (no FK deps)
    op.create_table(
        'players',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('nhl_id', sa.Integer(), nullable=True),
        sa.Column('first_name', sa.String(100), nullable=False),
        sa.Column('last_name', sa.String(100), nullable=False),
        sa.Column('birth_date', sa.Date(), nullable=True),
        sa.Column('birth_country', sa.String(3), nullable=True),
        sa.Column('position', sa.String(2), nullable=False),
        sa.Column('shoots_catches', sa.String(1), nullable=True),
        sa.Column('height_cm', sa.Integer(), nullable=True),
        sa.Column('weight_kg', sa.Integer(), nullable=True),
        sa.Column('draft_year', sa.Integer(), nullable=True),
        sa.Column('draft_round', sa.Integer(), nullable=True),
        sa.Column('draft_overall', sa.Integer(), nullable=True),
        sa.Column('css_rank', sa.Integer(), nullable=True),
        sa.Column('final_rank', sa.Integer(), nullable=True),
        sa.Column('composite_score', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_players_id', 'players', ['id'])
    op.create_index('ix_players_nhl_id', 'players', ['nhl_id'], unique=True)

    # 4. draft_picks_historical (FK: teams, general_managers)
    op.create_table(
        'draft_picks_historical',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('round', sa.Integer(), nullable=False),
        sa.Column('pick_number', sa.Integer(), nullable=False),
        sa.Column('overall_pick', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('gm_id', sa.Integer(), nullable=True),
        sa.Column('player_name', sa.String(200), nullable=True),
        sa.Column('position', sa.String(10), nullable=True),
        sa.Column('nationality', sa.String(50), nullable=True),
        sa.Column('height_cm', sa.Integer(), nullable=True),
        sa.Column('weight_kg', sa.Integer(), nullable=True),
        sa.Column('draft_league', sa.String(100), nullable=True),
        sa.Column('draft_league_tier', sa.Integer(), nullable=True),
        sa.Column('games_played_nhl', sa.Integer(), nullable=True, server_default=sa.text('0')),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id']),
        sa.ForeignKeyConstraint(['gm_id'], ['general_managers.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_draft_picks_historical_year', 'draft_picks_historical', ['year'])
    op.create_index('ix_draft_picks_historical_team_id', 'draft_picks_historical', ['team_id'])
    op.create_index('ix_draft_picks_historical_gm_id', 'draft_picks_historical', ['gm_id'])
    op.create_index('ix_draft_picks_historical_overall_pick', 'draft_picks_historical', ['overall_pick'])
    op.create_index('ix_draft_picks_historical_position', 'draft_picks_historical', ['position'])
    op.create_index('ix_draft_picks_historical_year_round', 'draft_picks_historical', ['year', 'round'])

    # 5. prospects_2025 (no FK deps)
    op.create_table(
        'prospects_2025',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('position', sa.String(10), nullable=False),
        sa.Column('nationality', sa.String(50), nullable=True),
        sa.Column('height_cm', sa.Integer(), nullable=True),
        sa.Column('weight_kg', sa.Integer(), nullable=True),
        sa.Column('draft_league', sa.String(100), nullable=True),
        sa.Column('draft_league_tier', sa.Integer(), nullable=True),
        sa.Column('css_ranking', sa.Integer(), nullable=True),
        sa.Column('css_category', sa.String(20), nullable=True),
        sa.Column('points', sa.Integer(), nullable=True),
        sa.Column('goals', sa.Integer(), nullable=True),
        sa.Column('assists', sa.Integer(), nullable=True),
        sa.Column('games_played', sa.Integer(), nullable=True),
        sa.Column('points_per_game', sa.Float(), nullable=True),
        sa.Column('age_at_draft', sa.Float(), nullable=True),
        sa.Column('birth_date', sa.Date(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_prospects_2025_css_ranking', 'prospects_2025', ['css_ranking'])
    op.create_index('ix_prospects_2025_css_category', 'prospects_2025', ['css_category'])
    op.create_index('ix_prospects_2025_position', 'prospects_2025', ['position'])
    op.create_index('ix_prospects_2025_nationality', 'prospects_2025', ['nationality'])
    op.create_index('ix_prospects_2025_draft_league', 'prospects_2025', ['draft_league'])

    # 6. rosters_current (FK: teams)
    op.create_table(
        'rosters_current',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('player_name', sa.String(200), nullable=False),
        sa.Column('position', sa.String(10), nullable=False),
        sa.Column('age', sa.Float(), nullable=True),
        sa.Column('contract_expiry_year', sa.Integer(), nullable=True),
        sa.Column('is_prospect', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('nhl_games_played', sa.Integer(), nullable=True, server_default=sa.text('0')),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_rosters_current_team_id', 'rosters_current', ['team_id'])
    op.create_index('ix_rosters_current_position', 'rosters_current', ['position'])
    op.create_index('ix_rosters_current_contract_expiry_year', 'rosters_current', ['contract_expiry_year'])
    op.create_index('ix_rosters_current_is_prospect', 'rosters_current', ['is_prospect'])

    # 7. lottery_odds (FK: teams)
    op.create_table(
        'lottery_odds',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('season', sa.Integer(), nullable=False),
        sa.Column('final_standing', sa.Integer(), nullable=True),
        sa.Column('lottery_odds_pct', sa.Float(), nullable=True),
        sa.Column('assigned_combinations', sa.ARRAY(sa.Integer()), nullable=True),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_lottery_odds_team_id', 'lottery_odds', ['team_id'])
    op.create_index('ix_lottery_odds_season', 'lottery_odds', ['season'])
    op.create_index('ix_lottery_odds_final_standing', 'lottery_odds', ['final_standing'])
    op.create_index('ix_lottery_odds_team_season', 'lottery_odds', ['team_id', 'season'], unique=True)

    # 8. gm_tendency_profiles (FK: general_managers)
    op.create_table(
        'gm_tendency_profiles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('gm_id', sa.Integer(), nullable=False),
        sa.Column('computed_at', sa.DateTime(), nullable=False),
        sa.Column('position_weights', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('league_weights', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('nationality_weights', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('avg_ranking_deviation', sa.Float(), nullable=True),
        sa.Column('tendency_archetype', sa.String(20), nullable=True),
        sa.ForeignKeyConstraint(['gm_id'], ['general_managers.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_gm_tendency_profiles_gm_id', 'gm_tendency_profiles', ['gm_id'])
    op.create_index('ix_gm_tendency_profiles_computed_at', 'gm_tendency_profiles', ['computed_at'])
    op.create_index('ix_gm_tendency_profiles_tendency_archetype', 'gm_tendency_profiles', ['tendency_archetype'])

    # 9. simulation_runs (no FK deps)
    op.create_table(
        'simulation_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('draft_year', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_simulation_runs_id', 'simulation_runs', ['id'])

    # 10. draft_picks (FK: teams)
    op.create_table(
        'draft_picks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('round', sa.Integer(), nullable=False),
        sa.Column('overall', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('original_team_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id']),
        sa.ForeignKeyConstraint(['original_team_id'], ['teams.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_draft_picks_id', 'draft_picks', ['id'])
    op.create_index('ix_draft_picks_year', 'draft_picks', ['year'])

    # 11. simulation_results (FK: simulation_runs, teams, players)
    op.create_table(
        'simulation_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('simulation_run_id', sa.Integer(), nullable=False),
        sa.Column('overall_pick', sa.Integer(), nullable=False),
        sa.Column('round', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=False),
        sa.Column('pick_score', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['simulation_run_id'], ['simulation_runs.id']),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id']),
        sa.ForeignKeyConstraint(['player_id'], ['players.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_simulation_results_id', 'simulation_results', ['id'])


def downgrade() -> None:
    # Drop in reverse FK dependency order
    op.drop_index('ix_simulation_results_id', table_name='simulation_results')
    op.drop_table('simulation_results')

    op.drop_index('ix_draft_picks_year', table_name='draft_picks')
    op.drop_index('ix_draft_picks_id', table_name='draft_picks')
    op.drop_table('draft_picks')

    op.drop_index('ix_simulation_runs_id', table_name='simulation_runs')
    op.drop_table('simulation_runs')

    op.drop_index('ix_gm_tendency_profiles_tendency_archetype', table_name='gm_tendency_profiles')
    op.drop_index('ix_gm_tendency_profiles_computed_at', table_name='gm_tendency_profiles')
    op.drop_index('ix_gm_tendency_profiles_gm_id', table_name='gm_tendency_profiles')
    op.drop_table('gm_tendency_profiles')

    op.drop_index('ix_lottery_odds_team_season', table_name='lottery_odds')
    op.drop_index('ix_lottery_odds_final_standing', table_name='lottery_odds')
    op.drop_index('ix_lottery_odds_season', table_name='lottery_odds')
    op.drop_index('ix_lottery_odds_team_id', table_name='lottery_odds')
    op.drop_table('lottery_odds')

    op.drop_index('ix_rosters_current_is_prospect', table_name='rosters_current')
    op.drop_index('ix_rosters_current_contract_expiry_year', table_name='rosters_current')
    op.drop_index('ix_rosters_current_position', table_name='rosters_current')
    op.drop_index('ix_rosters_current_team_id', table_name='rosters_current')
    op.drop_table('rosters_current')

    op.drop_index('ix_prospects_2025_draft_league', table_name='prospects_2025')
    op.drop_index('ix_prospects_2025_nationality', table_name='prospects_2025')
    op.drop_index('ix_prospects_2025_position', table_name='prospects_2025')
    op.drop_index('ix_prospects_2025_css_category', table_name='prospects_2025')
    op.drop_index('ix_prospects_2025_css_ranking', table_name='prospects_2025')
    op.drop_table('prospects_2025')

    op.drop_index('ix_draft_picks_historical_year_round', table_name='draft_picks_historical')
    op.drop_index('ix_draft_picks_historical_position', table_name='draft_picks_historical')
    op.drop_index('ix_draft_picks_historical_overall_pick', table_name='draft_picks_historical')
    op.drop_index('ix_draft_picks_historical_gm_id', table_name='draft_picks_historical')
    op.drop_index('ix_draft_picks_historical_team_id', table_name='draft_picks_historical')
    op.drop_index('ix_draft_picks_historical_year', table_name='draft_picks_historical')
    op.drop_table('draft_picks_historical')

    op.drop_index('ix_players_nhl_id', table_name='players')
    op.drop_index('ix_players_id', table_name='players')
    op.drop_table('players')

    op.drop_index('ix_general_managers_name', table_name='general_managers')
    op.drop_index('ix_general_managers_is_active', table_name='general_managers')
    op.drop_index('ix_general_managers_team_id', table_name='general_managers')
    op.drop_table('general_managers')

    op.drop_index('ix_teams_nhl_id', table_name='teams')
    op.drop_index('ix_teams_abbreviation', table_name='teams')
    op.drop_index('ix_teams_id', table_name='teams')
    op.drop_table('teams')
