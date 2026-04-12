"""
NHL Draft Lottery engine.
Implements the real NHL lottery system with 14 ping-pong balls (C(14,4) = 1001 combos, one excluded).
"""
import random
import logging
from itertools import combinations
from typing import Optional

logger = logging.getLogger(__name__)

# All 1001 combinations of choosing 4 from 14 balls
_ALL_COMBOS = list(combinations(range(1, 15), 4))
# The excluded combination (historically: (11, 12, 13, 14))
_EXCLUDED_COMBO = (11, 12, 13, 14)
VALID_COMBOS = [c for c in _ALL_COMBOS if c != _EXCLUDED_COMBO]
assert len(VALID_COMBOS) == 1000, f"Expected 1000 valid combos, got {len(VALID_COMBOS)}"

# Official lottery odds
LOTTERY_ODDS_PCT = {
    1: 18.5, 2: 13.5, 3: 11.5, 4: 9.5, 5: 8.5,
    6: 7.5, 7: 6.5, 8: 6.0, 9: 5.0, 10: 3.5,
    11: 3.0, 12: 2.5, 13: 2.0, 14: 1.5, 15: 1.0, 16: 0.5
}


def assign_combinations(teams_with_odds: list[dict]) -> dict:
    """
    Assign lottery combinations (1-1000) to teams proportionally.

    Args:
        teams_with_odds: list of {"team_id": int, "odds_pct": float, "standing": int}
                         Should sum to ~100%

    Returns:
        {team_id: [list of combination tuples]}
    """
    # Sort by standing (worst first = highest odds)
    sorted_teams = sorted(teams_with_odds, key=lambda x: x["standing"])

    # Calculate number of combinations per team (each 0.1% = 1 combo)
    team_combos: dict[int, list] = {}
    combo_idx = 0

    for team in sorted_teams:
        num_combos = round(team["odds_pct"] * 10)  # 18.5% -> 185 combos
        team_id = team["team_id"]
        team_combos[team_id] = VALID_COMBOS[combo_idx: combo_idx + num_combos]
        combo_idx += num_combos

    # Build reverse lookup: combo -> team_id
    combo_to_team = {}
    for team_id, combos in team_combos.items():
        for combo in combos:
            combo_to_team[combo] = team_id

    return {
        "team_combos": team_combos,
        "combo_to_team": combo_to_team,
    }


def draw_lottery(
    teams_with_odds: list[dict],
    seed: Optional[int] = None,
) -> list[int]:
    """
    Simulate the NHL draft lottery.
    2 separate draws for picks 1 and 2 (current NHL rules since 2023).
    10-spot rule: a team cannot move up more than 10 positions.
    Remaining picks 3-16 filled by original standings order.

    Args:
        teams_with_odds: list of {"team_id": int, "odds_pct": float, "standing": int}
        seed: optional random seed for reproducibility

    Returns:
        Ordered list of team_ids for picks 1-16
    """
    rng = random.Random(seed)
    assignment = assign_combinations(teams_with_odds)
    combo_to_team = assignment["combo_to_team"]

    # Build standing lookup: team_id -> standing (1 = worst)
    standing_by_team = {t["team_id"]: t["standing"] for t in teams_with_odds}

    # Original standings order (worst to best = picks 3-16 if not lottery winners)
    standings_order = [t["team_id"] for t in sorted(teams_with_odds, key=lambda x: x["standing"])]

    lottery_winners = []
    for pick_num in range(1, 3):  # Only 2 lottery picks
        winner = None
        attempts = 0
        while winner is None and attempts < 200:
            attempts += 1
            # Draw 4 balls from 14 without replacement
            balls = tuple(sorted(rng.sample(range(1, 15), 4)))

            if balls == _EXCLUDED_COMBO:
                continue  # Re-draw

            team_id = combo_to_team.get(balls)
            if team_id is None:
                continue  # Combo not assigned (shouldn't happen)

            if team_id in lottery_winners:
                continue  # Team already won a pick

            # 10-spot rule: team at standing N cannot win a pick better than N-10
            # e.g. standing=12 can win at most pick #2 (12-10=2)
            original_standing = standing_by_team.get(team_id, 1)
            if (original_standing - pick_num) > 10:
                continue  # Would move up more than 10 spots — invalid

            winner = team_id

        if winner is None:
            logger.error(f"Could not draw winner for pick {pick_num} after {attempts} attempts")
            # Fallback: pick highest-odds eligible team not yet drawn
            for t in sorted(teams_with_odds, key=lambda x: -x["odds_pct"]):
                tid = t["team_id"]
                if tid in lottery_winners:
                    continue
                orig = standing_by_team.get(tid, 1)
                if (orig - pick_num) > 10:
                    continue
                winner = tid
                break

        lottery_winners.append(winner)
        logger.debug(f"Pick #{pick_num} goes to team_id={winner}")

    # Fill remaining picks 3-16 in standings order (excluding lottery winners)
    remaining = [t for t in standings_order if t not in lottery_winners]
    final_order = lottery_winners + remaining

    return final_order


def simulate_lottery_n_times(
    teams_with_odds: list[dict],
    n: int = 1000,
    seed: Optional[int] = None,
) -> dict[int, list[float]]:
    """
    Run the lottery n times and compute probability distributions.

    Returns:
        {team_id: [prob_pick_1, prob_pick_2, ..., prob_pick_16]}
        Each value is a float 0-1 representing probability of that pick.
    """
    num_teams = len(teams_with_odds)
    team_ids = [t["team_id"] for t in teams_with_odds]

    # Count how many times each team gets each pick position
    counts: dict[int, list[int]] = {tid: [0] * num_teams for tid in team_ids}

    base_seed = seed or 42
    for i in range(n):
        result = draw_lottery(teams_with_odds, seed=base_seed + i)
        for pick_pos, team_id in enumerate(result):
            if team_id in counts:
                counts[team_id][pick_pos] += 1

    # Convert to probabilities
    return {
        team_id: [count / n for count in pick_counts]
        for team_id, pick_counts in counts.items()
    }
