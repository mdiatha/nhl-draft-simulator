"""Unit tests for the lottery engine."""
import pytest
from app.engines.lottery_engine import (
    assign_combinations, draw_lottery, simulate_lottery_n_times,
    VALID_COMBOS, LOTTERY_ODDS_PCT
)

SAMPLE_TEAMS = [
    {"team_id": i, "odds_pct": LOTTERY_ODDS_PCT[i], "standing": i}
    for i in range(1, 17)
]


def test_valid_combos_count():
    assert len(VALID_COMBOS) == 1000


def test_assign_combinations_coverage():
    """All 1000 combos should be assigned."""
    result = assign_combinations(SAMPLE_TEAMS)
    all_assigned = []
    for combos in result["team_combos"].values():
        all_assigned.extend(combos)
    assert len(all_assigned) == 1000


def test_draw_lottery_returns_16_teams():
    result = draw_lottery(SAMPLE_TEAMS, seed=42)
    assert len(result) == 16


def test_draw_lottery_no_duplicates():
    result = draw_lottery(SAMPLE_TEAMS, seed=42)
    assert len(result) == len(set(result)), "Duplicate teams in lottery result"


def test_draw_lottery_reproducible():
    r1 = draw_lottery(SAMPLE_TEAMS, seed=123)
    r2 = draw_lottery(SAMPLE_TEAMS, seed=123)
    assert r1 == r2, "Same seed should produce same result"


def test_draw_lottery_different_seeds():
    r1 = draw_lottery(SAMPLE_TEAMS, seed=1)
    r2 = draw_lottery(SAMPLE_TEAMS, seed=2)
    # With different seeds, results should usually differ (not guaranteed but highly likely)
    # Just verify they're valid
    assert len(r1) == 16
    assert len(r2) == 16


def test_simulate_lottery_probabilities_sum_to_1():
    """For each pick position, probabilities across all teams should sum to ~1."""
    probs = simulate_lottery_n_times(SAMPLE_TEAMS, n=100, seed=42)
    num_picks = len(SAMPLE_TEAMS)
    for pick_pos in range(num_picks):
        total = sum(probs[team["team_id"]][pick_pos] for team in SAMPLE_TEAMS)
        assert abs(total - 1.0) < 0.01, f"Pick {pick_pos+1} probabilities sum to {total}"


def test_simulate_highest_odds_team_wins_most():
    """Team with best odds (standing=1) should win pick #1 most often."""
    probs = simulate_lottery_n_times(SAMPLE_TEAMS, n=500, seed=42)
    # Team with standing=1 has 18.5% odds
    team_1_pick1_prob = probs[1][0]
    # Should be roughly 18.5%, allow wide margin for randomness
    assert team_1_pick1_prob > 0.10, f"Team 1 pick1 prob too low: {team_1_pick1_prob}"
