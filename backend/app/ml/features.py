"""Feature engineering for draft pick prediction.

Training: builds (prospect + team_context) rows labeled was_picked=1/0.
Inference: builds feature rows for scoring 2025 prospects for a specific team.

─── Feature groups ───────────────────────────────────────────────────────────
  Prospect quality
    css_rank_norm       — consensus rank normalized to [0,1]. Primary signal.
    ppg_league_norm     — PPG ÷ median PPG for this league-tier & cohort year.
                          Removes league scoring-environment bias from raw PPG.
    age_league_norm     — league-tier median age minus prospect age. Positive =
                          younger than peers = higher upside signal.

  Draft-state (dynamic, computed pick-by-pick)
    pos_taken_before_norm  — fraction of picks so far at this position.
                             High = position is saturating the draft.
    pos_remaining_norm     — fraction of available pool that is this position.
                             Low = positional scarcity, raises pick probability.
    team_drafted_this_pos  — how many times this team already took this position
                             this draft. GMs rarely double-dip in round 1.

  Rank signal
    rank_vs_slot        — prospect rank minus slot rank. Positive = value pick,
                          negative = reach.

  GM tendency
    gm_*               — GM-specific weights tell the model how each GM deviates
                         from the consensus board by position/league/nationality.

Encoding: all categoricals are one-hot so the model learns independent weights
per position/nationality rather than assuming ordinal relationships.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import random
import statistics
from collections import Counter, defaultdict

import pandas as pd

from app.constants import infer_league_key, nat_group

logger = logging.getLogger(__name__)

# ── Categorical vocabularies ───────────────────────────────────────────────────

POSITIONS = ["C", "LW", "RW", "D", "G"]

# Grouped by scouting pipeline / development path — matches nat_group() in constants.py.
# Individual nationality one-hots (CAN, RUS, CHE...) had 14/15 with zero importance
# because rare nationalities almost never appear in leaf nodes.
# 5 groups give the model meaningful signal with far less noise.
NAT_GROUPS = ["CAN", "USA", "NORDIC", "SLAVIC", "EUR_OTHER"]

NAT_ALIASES: dict[str, str] = {"CA": "CAN", "US": "USA", "SUI": "CHE"}

MAX_DRAFT_POOL = 224  # normalisation denominator for rank/slot

# Negative sample window: how many picks ahead to draw negatives from.
# Restricting to ~one round forces the model to learn fine-grained distinctions
# between realistic alternatives, rather than wasting capacity on trivially easy
# negatives like "why wasn't a 7th-round player taken with the #1 pick?"
NEGATIVE_WINDOW = 31


# ── Feature columns ────────────────────────────────────────────────────────────

LEAGUE_KEYS = ["tier1_CAN", "tier1_USA", "tier1_EUR", "tier2", "tier3"]

DRAFT_ROUNDS = [1, 2, 3, 4]

FEATURE_COLS: list[str] = (
    # Position one-hot (5)
    [f"pos_{p}" for p in POSITIONS]
    # Nationality group one-hot (5) — CAN, USA, NORDIC, SLAVIC, EUR_OTHER
    + [f"nat_{g}" for g in NAT_GROUPS]
    # League one-hot (5) — replaces ordinal league_tier integer
    + [f"league_{k}" for k in LEAGUE_KEYS]
    # Physical — position-relative deviation from positional median
    + ["height_norm", "weight_norm"]
    # League-normalized quality (2)
    + ["ppg_league_norm", "age_league_norm"]
    # Consensus rank signals (3)
    + ["pick_slot_norm", "rank_vs_slot", "rank_gap_norm"]
    # GM tendency (3)
    + ["gm_pos_weight", "gm_league_weight", "gm_nat_weight"]
    # Draft-state / supply signals (4)
    + ["pos_taken_before_norm", "pos_remaining_norm", "team_drafted_this_pos",
       "pos_quality_rank_norm"]
    # Season-over-season production (3)
    + ["gp_pre_draft", "ppg_prev_season", "has_prev_season"]
    # Draft round one-hot (4)
    + [f"round_{r}" for r in DRAFT_ROUNDS]
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _gm_feats_from_picks(
    gm_picks: list,
    reference_year: int,
    priors: dict,
    position: str,
    draft_league: str,
    nationality: str,
) -> dict:
    """
    Compute GM tendency features directly from a list of historical picks.

    Used during training to avoid data leakage: only picks from years BEFORE
    the current draft year are passed in, so the GM's tendency profile never
    sees the picks it's being used to predict.

    Applies the same recency weighting and Bayesian shrinkage as the tendency
    engine, producing identical features to _gm_features() at inference time.
    """
    _RECENCY_DECAY = 0.85
    _SHRINKAGE_K   = 30
    _ROUND_WEIGHTS = {1: 7, 2: 5, 3: 3}

    if not gm_picks:
        return {"gm_pos_weight": 0.2, "gm_league_weight": 0.2, "gm_nat_weight": 0.33}

    pos_w: dict[str, float]    = {}
    league_w: dict[str, float] = {}
    nat_w: dict[str, float]    = {}
    total_w = 0.0

    for p in gm_picks:
        w = _ROUND_WEIGHTS.get(p.round or 4, 1) * (_RECENCY_DECAY ** (reference_year - p.year))
        total_w += w
        if p.position:
            pos_w[p.position] = pos_w.get(p.position, 0.0) + w
        if p.draft_league or p.draft_league_tier:
            key = infer_league_key(p.draft_league or "", p.draft_league_tier)
            league_w[key] = league_w.get(key, 0.0) + w
        if p.nationality:
            ng = nat_group(p.nationality)
            nat_w[ng] = nat_w.get(ng, 0.0) + w

    def _norm(d: dict) -> dict:
        t = sum(d.values())
        return {k: v / t for k, v in d.items()} if t > 0 else {}

    def _shrink(gm_d: dict, prior_d: dict) -> dict:
        alpha = total_w / (total_w + _SHRINKAGE_K)
        return {
            k: alpha * gm_d.get(k, 0.0) + (1 - alpha) * prior_d.get(k, 0.0)
            for k in set(gm_d) | set(prior_d)
        }

    pos_w    = _shrink(_norm(pos_w),    priors.get("position",    {}))
    league_w = _shrink(_norm(league_w), priors.get("league",      {}))
    nat_w    = _shrink(_norm(nat_w),    priors.get("nationality", {}))

    return {
        "gm_pos_weight":    pos_w.get(position or "C", 0.0),
        "gm_league_weight": league_w.get(infer_league_key(draft_league or ""), 0.0),
        "gm_nat_weight":    nat_w.get(nat_group(nationality or ""), 0.0),
    }


def _gm_features(profile, position: str, draft_league: str, nationality: str) -> dict:
    """Extract raw GM tendency weights for a specific (GM, prospect) pair."""
    if profile is None:
        return {
            "gm_pos_weight":    0.2,
            "gm_league_weight": 0.2,
            "gm_nat_weight":    0.33,
        }

    pos_weights    = profile.position_weights or {}
    league_weights = profile.league_weights or {}
    nat_weights    = profile.nationality_weights or {}

    return {
        "gm_pos_weight":    pos_weights.get(position or "C", 0.0),
        "gm_league_weight": league_weights.get(infer_league_key(draft_league or ""), 0.0),
        "gm_nat_weight":    nat_weights.get(nat_group(nationality or ""), 0.0),
    }


def _contextual_feats(
    prospect,
    remaining: list,        # prospects still on the board (including this one)
    picks_made: int,        # total picks made before this slot
    team_id: int,
    pos_taken: Counter,               # position -> total picks at this pos so far
    team_pos_drafted: defaultdict,    # team_id -> position -> count
    tier_ppg_med: dict[str, float],   # league-group -> median PPG in this cohort
    tier_age_med: dict[str, float],   # league-group -> median age in this cohort
    quality_map: dict | None = None,  # prospect_id -> css_rank_norm for board ranking
) -> dict:
    """
    Compute the 6 draft-context features for a single prospect at a given slot.

    These features capture:
      - How the prospect's production compares to peers in the same league group
      - How saturated / scarce their position is on the current board
      - Whether this team already addressed this position earlier in this draft
      - How this prospect ranks among same-position players still available (new)

    quality_map: maps prospect/pick id → css_rank_norm. When provided, enables
    pos_quality_rank_norm computation. Pass None to get a neutral 0.5 default.
    """
    pos = prospect.position or "F"
    league_key = infer_league_key(
        getattr(prospect, "draft_league", None) or "",
        (getattr(prospect, "draft_league_tier", None)
         or getattr(prospect, "league_tier", None)),
    )

    # League-normalized PPG: prospect PPG relative to cohort median for their league group.
    # Goalies are always neutral (1.0) — their G+A stats are near zero by nature and
    # carry no meaningful signal compared to skater production.
    if pos == "G":
        ppg_norm = 1.0
    elif league_key not in tier_ppg_med:
        # Unknown league group — return neutral rather than distorting by dividing
        # raw PPG by an arbitrary fallback denominator (0.5 would double the value).
        ppg_norm = 1.0
    else:
        ppg_med = tier_ppg_med[league_key]
        raw_ppg = (prospect.points_per_game or 0.0)
        ppg_norm = raw_ppg / ppg_med if ppg_med > 0 else 1.0

    # Age relative to same-league peers: positive = younger than average = higher upside
    age_med  = tier_age_med.get(league_key, 18.5)
    raw_age  = (prospect.age_at_draft or age_med)
    age_norm = age_med - raw_age

    # Positional saturation: what fraction of picks so far were this position
    pos_taken_norm = pos_taken[pos] / picks_made if picks_made > 0 else 0.0

    # Positional scarcity: what fraction of the remaining pool is this position
    total_rem = len(remaining)
    pos_rem   = sum(1 for x in remaining if (x.position or "F") == pos)
    pos_rem_norm = pos_rem / total_rem if total_rem > 0 else 0.0

    # Has this team already drafted this position today
    team_count = team_pos_drafted[team_id][pos] if isinstance(team_pos_drafted[team_id], Counter) else 0

    # Positional quality rank: where does this prospect rank among same-position
    # players still on the board? 1.0 = best available at this position.
    #
    # This is the primary heuristic positional GMs use: "who is the best C/D/G
    # remaining?" A positional GM (high gm_pos_weight) will strongly prefer the
    # top-ranked player at their target position over the 3rd-best at that slot.
    # Interacts with gm_pos_weight: high weight + rank 1.0 = very strong signal.
    pid = getattr(prospect, "id", None)
    if quality_map is not None and pid is not None:
        this_q = quality_map.get(pid, 0.5)
        same_pos_pool = [x for x in remaining if (x.position or "F") == pos]
        n_pos = len(same_pos_pool)
        if n_pos > 1:
            n_better = sum(
                1 for x in same_pos_pool
                if quality_map.get(getattr(x, "id", None), 0.0) > this_q
            )
            pos_quality_rank_norm = 1.0 - (n_better / n_pos)
        else:
            pos_quality_rank_norm = 1.0
    else:
        # Neutral fallback when quality map is unavailable
        pos_quality_rank_norm = 0.5

    return {
        "ppg_league_norm":         round(ppg_norm,              4),
        "age_league_norm":         round(age_norm,               4),
        "pos_taken_before_norm":   round(pos_taken_norm,         4),
        "pos_remaining_norm":      round(pos_rem_norm,           4),
        "team_drafted_this_pos":   team_count,
        "pos_quality_rank_norm":   round(pos_quality_rank_norm,  4),
    }


# ── Feature matrix builder ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw DataFrame into the ML feature matrix.

    Required columns in df:
      position, nationality, height_cm, weight_kg,
      draft_league_tier (or league_tier), points_per_game, age_at_draft,
      css_rank_norm, overall_pick (= pick slot),
      gm_pos_weight, gm_league_weight, gm_nat_weight,
      ppg_league_norm, age_league_norm,
      pos_taken_before_norm, pos_remaining_norm, team_drafted_this_pos
    """
    out = pd.DataFrame(index=df.index)

    # ── Position one-hot ──────────────────────────────────────────────────────
    pos_series = df["position"].fillna("").str.split("/").str[0].str.upper()
    for pos in POSITIONS:
        out[f"pos_{pos}"] = (pos_series == pos).astype(int)

    # ── Nationality group one-hot ─────────────────────────────────────────────
    # Map individual nationality codes → 5 scouting-pipeline groups via nat_group().
    # This replaces 15 individual one-hots (14 of which had zero importance) with
    # 5 meaningful groups that the model can actually learn from.
    nat_col = "nationality" if "nationality" in df.columns else "birth_country"
    nat_series = (
        df.get(nat_col, pd.Series("", index=df.index))
        .fillna("")
        .map(lambda n: nat_group(NAT_ALIASES.get(n, n)))
    )
    for grp in NAT_GROUPS:
        out[f"nat_{grp}"] = (nat_series == grp).astype(int)

    # ── Physical — position-relative ─────────────────────────────────────────
    # Normalize height and weight as deviation from position median so the model
    # learns body-type signal relative to positional expectations rather than
    # absolute size. A 6'4" center is expected; a 6'4" winger is notable.
    pos_for_phys = df["position"].fillna("").str.split("/").str[0].str.upper()
    h_raw = df["height_cm"].fillna(182.0)
    w_raw = df["weight_kg"].fillna(85.0)
    pos_h_med = pos_for_phys.map(h_raw.groupby(pos_for_phys).median()).fillna(182.0)
    pos_w_med = pos_for_phys.map(w_raw.groupby(pos_for_phys).median()).fillna(85.0)
    out["height_norm"] = (h_raw - pos_h_med).astype(float)
    out["weight_norm"] = (w_raw - pos_w_med).astype(float)

    # ── League one-hot ────────────────────────────────────────────────────────
    # Use infer_league_key() to map each row to one of 5 named buckets.
    # One-hot avoids the ordinal assumption baked into the old integer tier.
    if "draft_league" in df.columns:
        tier_col = "draft_league_tier" if "draft_league_tier" in df.columns else "league_tier"
        league_key_series = df.apply(
            lambda row: infer_league_key(
                row.get("draft_league") or "",
                row.get(tier_col),
            ),
            axis=1,
        )
    else:
        league_key_series = pd.Series("tier2", index=df.index)
    for lk in LEAGUE_KEYS:
        out[f"league_{lk}"] = (league_key_series == lk).astype(int)

    # ── Raw production ────────────────────────────────────────────────────────
    out["points_per_game"] = df.get("points_per_game", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float)
    if "age_at_draft" in df.columns:
        median_age = df["age_at_draft"].median()
        out["age_at_draft"] = df["age_at_draft"].fillna(median_age if not pd.isna(median_age) else 18.5).astype(float)
    else:
        out["age_at_draft"] = 18.5

    # ── League-normalized quality ─────────────────────────────────────────────
    # ppg_league_norm=1.0 means average production for their league tier
    # age_league_norm=0.0 means average age; positive = younger than peers
    out["ppg_league_norm"] = df.get("ppg_league_norm", pd.Series(1.0, index=df.index)).fillna(1.0).astype(float)
    out["age_league_norm"] = df.get("age_league_norm", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float)

    # ── Rank signals ──────────────────────────────────────────────────────────
    out["css_rank_norm"] = df.get("css_rank_norm", pd.Series(0.5, index=df.index)).fillna(0.5).astype(float)
    slot = df.get("overall_pick", pd.Series(32, index=df.index)).fillna(32).astype(float)
    out["pick_slot_norm"] = (1.0 - (slot - 1) / MAX_DRAFT_POOL).clip(0.0, 1.0)
    out["rank_vs_slot"] = (out["css_rank_norm"] - out["pick_slot_norm"]).astype(float)
    # rank_gap_norm: gap between the best prospect in this group and each prospect.
    # Training: pre-computed per pick-slot group in build_training_dataset (correct).
    # Inference: computed from max of this batch, which IS the full available pool
    #            for one pick slot (also correct).
    if "rank_gap_norm" in df.columns:
        out["rank_gap_norm"] = df["rank_gap_norm"].fillna(0.0).astype(float)
    else:
        best_norm = out["css_rank_norm"].max()
        out["rank_gap_norm"] = (best_norm - out["css_rank_norm"]).astype(float)

    # ── GM tendency ───────────────────────────────────────────────────────────
    out["gm_pos_weight"]    = df.get("gm_pos_weight",    pd.Series(0.2,  index=df.index)).fillna(0.2).astype(float)
    out["gm_league_weight"] = df.get("gm_league_weight", pd.Series(0.2,  index=df.index)).fillna(0.2).astype(float)
    out["gm_nat_weight"]    = df.get("gm_nat_weight",    pd.Series(0.33, index=df.index)).fillna(0.33).astype(float)

    # ── Draft-state / supply signals ──────────────────────────────────────────
    # Defaults represent a neutral mid-draft state with no team history
    out["pos_taken_before_norm"]  = df.get("pos_taken_before_norm",  pd.Series(0.2,  index=df.index)).fillna(0.2).astype(float)
    out["pos_remaining_norm"]     = df.get("pos_remaining_norm",     pd.Series(0.2,  index=df.index)).fillna(0.2).astype(float)
    out["team_drafted_this_pos"]  = df.get("team_drafted_this_pos",  pd.Series(0,    index=df.index)).fillna(0).astype(int)
    # 0.5 = neutral (average quality at position); overridden dynamically at inference
    out["pos_quality_rank_norm"]  = df.get("pos_quality_rank_norm",  pd.Series(0.5,  index=df.index)).fillna(0.5).astype(float)

    # ── Season-over-season production ─────────────────────────────────────────
    # gp_pre_draft: sample size signal — 1.2 PPG over 10 games ≠ 1.2 PPG over 60
    out["gp_pre_draft"]    = df.get("gp_pre_draft",    pd.Series(30,  index=df.index)).fillna(30).astype(float)
    # ppg_prev_season: raw prior-season production (0 when unavailable)
    ppg_prev_raw           = df.get("ppg_prev_season", pd.Series(dtype=float)).reindex(df.index)
    out["ppg_prev_season"] = ppg_prev_raw.fillna(0.0).astype(float)
    # has_prev_season: 1 = second+ major-league year (prior season data exists)
    out["has_prev_season"] = ppg_prev_raw.notna().astype(int)

    # ── Draft round one-hot ───────────────────────────────────────────────────
    # One-hot instead of integer: round 1 and round 4 are categorically different.
    # Integer encoding implies a false ordinal relationship.
    round_series = df.get("draft_round", pd.Series(1, index=df.index)).fillna(1).astype(int)
    for r in DRAFT_ROUNDS:
        out[f"round_{r}"] = (round_series == r).astype(int)

    return out[FEATURE_COLS]


# ── Training dataset builder ───────────────────────────────────────────────────

# CSS publishes three independent ranked lists each year:
#   NA Skaters  (~250 players, ranks 1–~250)
#   EUR Skaters (~100 players, ranks 1–~400)
#   Goalies     (~30-50 players, ranks ~365–~465)
#
# These lists are NOT comparable by raw rank: EUR #5 ≠ NA #5 in draft value.
# We normalize within each list using within-list ordinal rank (1st, 2nd, 3rd...)
# so the best NA skater, best EUR skater, and best goalie all score ~1.0.
# List identity is inferred from position + nationality + league.

# European nationality codes used to classify EUR vs NA skaters.
_EUR_NATIONALITIES = frozenset({
    "SWE", "FIN", "RUS", "CZE", "SVK", "CHE", "DEU", "NOR", "DNK",
    "AUT", "LVA", "UKR", "BLR", "FRA", "SLO", "SVN", "BEL", "GBR",
})

# League name fragments that indicate a European league.
_EUR_LEAGUE_FRAGMENTS = ("SWEDEN", "FINLAND", "RUSSIA", "KHL", "SHL", "LIIGA",
                         "EXTRALIGA", "NLA", "CZECH", "SLOVAK", "NORWAY", "DENMARK",
                         "AUSTRIA", "SWISS", "DEL", "MESTIS", "ALLSVENSKAN")


def _css_list(position: str | None, nationality: str | None, draft_league: str | None) -> str:
    """
    Return which CSS list a prospect belongs to: 'na_skater', 'eur_skater', or 'goalie'.

    CSS publishes separate NA Skater, EUR Skater, and Goalie lists annually.
    Goalies are identified by position. Skaters are split by nationality/league.
    """
    pos = (position or "").upper().split("/")[0]
    if pos == "G":
        return "goalie"
    nat = (nationality or "").upper()
    league = (draft_league or "").upper()
    if nat in _EUR_NATIONALITIES:
        return "eur_skater"
    if any(frag in league for frag in _EUR_LEAGUE_FRAGMENTS):
        return "eur_skater"
    return "na_skater"


def _css_norm_within_list(within_list_rank: int, list_size: int) -> float:
    """
    Convert a within-list ordinal rank (1 = best on that list) to [0, 1].

    Uses sqrt scaling so the gap between #1 and #2 is amplified vs. #50 vs #51.
    list_size is the total number of ranked players on this list in this cohort.
    Falls back to list_size=100 when the list is empty (shouldn't happen in practice).
    """
    import math
    denom = max(list_size, 1)
    return max(0.0, 1.0 - math.sqrt((within_list_rank - 1) / denom))


def _compute_predraft_quality(year_picks: list) -> dict[int, float]:
    """
    Compute css_rank_norm for each pick in a draft year.

    Strategy: group picks by CSS list (na_skater / eur_skater / goalie), sort each
    group by raw css_rank ascending, then assign within-list ordinal rank (1 = best).
    _css_norm_within_list() converts that to [0,1] using the list's own size as
    the denominator, so #1 on any list scores ~1.0 regardless of raw rank number.

    This fixes the cross-list comparison problem: CSS assigns EUR skaters ranks
    like 1–400 and goalies ranks like 365–465. Raw rank comparison would make
    a EUR #5 (great prospect) look worse than a NA #5 just because the number is
    similar but the lists are independent scales.

    Falls back to round-normalized pick position for pre-2008 or unranked picks.
    """
    has_css = [p for p in year_picks if getattr(p, "css_rank", None) is not None]
    total_picks = len(year_picks)
    css_coverage = len(has_css) / total_picks if total_picks > 0 else 0.0

    result: dict[int, float] = {}

    if css_coverage >= 0.5:
        # Group by CSS list, sort by raw rank within each list
        by_list: dict[str, list] = {}
        for p in has_css:
            lst = _css_list(
                getattr(p, "position", None),
                getattr(p, "nationality", None),
                getattr(p, "draft_league", None),
            )
            by_list.setdefault(lst, []).append(p)

        for lst, players in by_list.items():
            sorted_players = sorted(players, key=lambda p: p.css_rank)
            list_size = len(sorted_players)
            for within_rank, p in enumerate(sorted_players, start=1):
                result[p.id] = _css_norm_within_list(within_rank, list_size)

    # Unranked players: CSS deliberately excluded them, so they should score below
    # any ranked player. A fixed penalty (0.15) reflects "not on the board" rather
    # than a neutral 0.5 or a position-within-round proxy that rewarded early picks.
    # Using a small positive value (not 0.0) preserves the ability to distinguish
    # unranked players with strong PPG from those with weak PPG via other features.
    missing = [p for p in year_picks if p.id not in result]
    for p in missing:
        result[p.id] = 0.15

    return result


def _cohort_tier_stats(year_picks: list) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute median PPG and median age per league group for a single draft cohort.

    Keyed by infer_league_key() — e.g. "tier1_CAN", "tier1_EUR", "tier2" — so
    that OHL players are normalized against other OHL players, not lumped with
    KHL professionals or NCAA players in a single "tier 1" bucket.
    """
    tier_ppg: dict[str, list] = {}
    tier_age: dict[str, list] = {}
    for p in year_picks:
        key = infer_league_key(p.draft_league or "", p.draft_league_tier)
        # Exclude goalies from PPG medians — their G+A is near zero by nature
        # and would drag down the skater median for each league group.
        if p.points_per_game is not None and (p.position or "F") != "G":
            tier_ppg.setdefault(key, []).append(p.points_per_game)
        if p.age_at_draft is not None:
            tier_age.setdefault(key, []).append(p.age_at_draft)
    ppg_med = {k: statistics.median(v) for k, v in tier_ppg.items() if v}
    age_med = {k: statistics.median(v) for k, v in tier_age.items() if v}
    return ppg_med, age_med


def build_training_dataset(
    db,
    neg_samples_per_pick: int | None = None,
    random_seed: int = 42,
    max_year: int | None = None,
) -> pd.DataFrame:
    """
    Build a labeled dataset of (team_context, prospect) rows for 2000–2024.

    Positive (was_picked=1): the actual pick.
    Negative (was_picked=0): every other prospect still available at that slot.

    neg_samples_per_pick=None (default) uses NEGATIVE_WINDOW picks ahead,
    matching the real decision: "from realistic alternatives, why this one?"
    Pass an int to cap negatives further (useful for quick iteration / testing).

    max_year: if set, only include picks from years <= max_year.
    Used by backtest.run_backtest to ensure the training set never leaks
    picks from the test year.

    GM tendency features are computed time-aware: when training on a pick from
    year Y, only picks from years < Y are used to compute that GM's tendency
    profile. This eliminates data leakage where the model could learn from
    the very picks it's trying to predict.

    New features computed here:
      - ppg_league_norm / age_league_norm: from within-cohort tier medians
      - pos_taken_before_norm: tracks positional saturation pick-by-pick
      - pos_remaining_norm: scarcity of this position in the remaining pool
      - team_drafted_this_pos: how many times team already took this position
    """
    from app.models.draft_pick_historical import DraftPickHistorical

    rng = random.Random(random_seed)

    year_filter_max = max_year if max_year is not None else 2024

    # 2008 is the first year with real NHL CSS rankings in the DB.
    # Pre-2008 picks use a pick-order proxy for css_rank_norm which is noisier
    # and teaches relationships that don't reflect how GMs draft today.
    # Training on clean CSS-era data gives better signal despite fewer rows.
    CSS_ERA_START = 2008

    # Limit to rounds 1-4: rounds 5-7 are developmental gambles where GMs
    # deviate heavily from any systematic preference (roster needs, personal
    # relationships, late-round fliers). Including them adds noise without
    # adding signal — the model can't learn anything useful from a round-7
    # "reach" for a college senior. The sim engine only simulates rounds 1-4
    # in practice, so this also eliminates train/inference distribution mismatch.
    MAX_TRAINING_ROUND = 4

    picks = (
        db.query(DraftPickHistorical)
        .filter(
            DraftPickHistorical.year >= CSS_ERA_START,
            DraftPickHistorical.year <= year_filter_max,
            DraftPickHistorical.round <= MAX_TRAINING_ROUND,
        )
        .order_by(DraftPickHistorical.year, DraftPickHistorical.overall_pick)
        .all()
    )

    if not picks:
        raise ValueError("No historical draft picks found. Run ingestion first.")

    # ── Population priors ──────────────────────────────────────────────────────
    # Used for Bayesian shrinkage: GMs with few picks shrink toward the league
    # average rather than returning noisy individual estimates.
    all_pos    = Counter(p.position or "F" for p in picks)
    all_league = Counter(infer_league_key(p.draft_league or "", p.draft_league_tier) for p in picks)
    all_nat    = Counter(nat_group(p.nationality or "") for p in picks)

    def _to_dist(c: Counter) -> dict:
        t = sum(c.values())
        return {k: v / t for k, v in c.items()} if t > 0 else {}

    priors = {
        "position":    _to_dist(all_pos),
        "league":      _to_dist(all_league),
        "nationality": _to_dist(all_nat),
    }

    # ── Per-GM pick history (grows one year at a time) ─────────────────────────
    # For a 2015 pick, pre_by_gm[gm_id] contains only picks from 2000–2014.
    # This ensures no future data contaminates the GM tendency features.
    pre_by_gm: defaultdict[int, list] = defaultdict(list)

    by_year: dict[int, list] = {}
    for pick in picks:
        by_year.setdefault(pick.year, []).append(pick)

    rows = []

    for year in sorted(by_year.keys()):
        year_picks = by_year[year]
        year_picks_sorted = sorted(year_picks, key=lambda p: p.overall_pick)
        quality = _compute_predraft_quality(year_picks_sorted)

        # Cohort-level league stats for normalizing PPG and age
        tier_ppg_med, tier_age_med = _cohort_tier_stats(year_picks_sorted)

        # Draft-state counters (reset each year)
        pos_taken: Counter[str]                           = Counter()
        team_pos_drafted: defaultdict[int, Counter[str]] = defaultdict(Counter)

        for i, pick in enumerate(year_picks_sorted):
            if not pick.gm_id or not pick.team_id:
                # Still update state so counts stay accurate
                pos_taken[pick.position or "F"] += 1
                continue

            # Time-aware GM tendency: only pre-year picks, no leakage
            gm_feats = _gm_feats_from_picks(
                pre_by_gm[pick.gm_id],
                year,
                priors,
                pick.position or "",
                pick.draft_league or "",
                pick.nationality or "",
            )

            # Remaining = current pick + everything after (board at this moment)
            remaining = year_picks_sorted[i:]

            ctx = _contextual_feats(
                pick, remaining, i, pick.team_id,
                pos_taken, team_pos_drafted,
                tier_ppg_med, tier_age_med,
                quality_map=quality,
            )

            # rank_gap_norm for the positive row is computed after we know the
            # group (positive + negatives), so we patch it in below.
            pos_row = {
                "year":              year,
                "position":          pick.position,
                "nationality":       pick.nationality,
                "height_cm":         pick.height_cm,
                "weight_kg":         pick.weight_kg,
                "draft_league":      pick.draft_league,
                "draft_league_tier": pick.draft_league_tier,
                "points_per_game":   pick.points_per_game,
                "gp_pre_draft":      pick.gp_pre_draft,
                "ppg_prev_season":   pick.ppg_prev_season,
                "has_prev_season":   1 if pick.ppg_prev_season is not None else 0,
                "age_at_draft":      pick.age_at_draft,
                "overall_pick":      pick.overall_pick,
                "draft_round":       pick.round,
                "css_rank_norm":     quality[pick.id],
                "rank_gap_norm":     None,  # filled in after negatives are known
                **gm_feats,
                **ctx,
                "was_picked": 1,
            }
            rows.append(pos_row)

            # Negatives: prospects still available at this slot — the next
            # NEGATIVE_WINDOW picks (regardless of round) who could have been
            # chosen here but weren't.  Removing the same-round restriction
            # ensures every pick gets ~31 negatives: early picks compare against
            # round-1 peers; late round-1 picks compare against round-2 prospects
            # (which is the actual decision being made at pick #30-32).
            later = year_picks_sorted[i + 1:][:NEGATIVE_WINDOW]
            negs  = (
                rng.sample(later, min(neg_samples_per_pick, len(later)))
                if neg_samples_per_pick is not None
                else later
            )

            # rank_gap_norm: gap between the best prospect in this group and each
            # prospect.  Must be computed per group so training matches inference
            # (where build_features is called on a single pick-slot at a time).
            group_css_norms = [quality[pick.id]] + [quality[neg.id] for neg in negs]
            group_best_css  = max(group_css_norms) if group_css_norms else quality[pick.id]
            # Patch the positive row now that we know the group's best CSS
            pos_row["rank_gap_norm"] = group_best_css - quality[pick.id]

            for neg in negs:
                neg_gm_feats = _gm_feats_from_picks(
                    pre_by_gm[pick.gm_id],
                    year,
                    priors,
                    neg.position or "",
                    neg.draft_league or "",
                    neg.nationality or "",
                )
                neg_ctx = _contextual_feats(
                    neg, remaining, i, pick.team_id,
                    pos_taken, team_pos_drafted,
                    tier_ppg_med, tier_age_med,
                    quality_map=quality,
                )
                rows.append({
                    "year":              year,
                    "position":          neg.position,
                    "nationality":       neg.nationality,
                    "height_cm":         neg.height_cm,
                    "weight_kg":         neg.weight_kg,
                    "draft_league":      neg.draft_league,
                    "draft_league_tier": neg.draft_league_tier,
                    "points_per_game":   neg.points_per_game,
                    "gp_pre_draft":      neg.gp_pre_draft,
                    "ppg_prev_season":   neg.ppg_prev_season,
                    "has_prev_season":   1 if neg.ppg_prev_season is not None else 0,
                    "age_at_draft":      neg.age_at_draft,
                    "overall_pick":      pick.overall_pick,
                    "draft_round":       pick.round,
                    "css_rank_norm":     quality[neg.id],
                    "rank_gap_norm":     group_best_css - quality[neg.id],
                    **neg_gm_feats,
                    **neg_ctx,
                    "was_picked": 0,
                })

            # Update state after this pick is processed
            pos_taken[pick.position or "F"] += 1
            team_pos_drafted[pick.team_id][pick.position or "F"] += 1

        # After all picks in this year are processed, add them to GM histories
        # so they're available as context for subsequent years.
        for pick in year_picks_sorted:
            if pick.gm_id:
                pre_by_gm[pick.gm_id].append(pick)

    df = pd.DataFrame(rows)

    # Sample weighting: two multiplicative factors per row.
    #
    # 1. Recency decay (0.88/year): recent drafts reflect current GM philosophy
    #    and modern player-development paths better than drafts from 15 years ago.
    #    Decay of 0.88/year → 2024 rows count ~2.9× more than 2008 rows.
    #
    # 2. Round weight: round 1 picks are the most predictable (best players, most
    #    scouting coverage, highest-stakes decisions) and the most important for
    #    simulation quality. Later rounds involve more developmental gambles and
    #    idiosyncratic team preferences that are harder to generalize. Upweighting
    #    early rounds ensures the model learns round-1 dynamics most accurately.
    RECENCY_DECAY = 0.88
    ROUND_WEIGHT  = {1: 4.0, 2: 2.5, 3: 1.5, 4: 1.2, 5: 1.0, 6: 0.8, 7: 0.6}
    MAX_YEAR = df["year"].max() if "year" in df.columns else 2024
    df["sample_weight"] = df.apply(
        lambda row: (
            RECENCY_DECAY ** (MAX_YEAR - int(row["year"]))
            * ROUND_WEIGHT.get(int(row.get("draft_round", 1)), 1.0)
        ),
        axis=1,
    )

    logger.info(
        "Training dataset: %d rows (%d positive, %d negative)",
        len(df), int(df["was_picked"].sum()), int((df["was_picked"] == 0).sum()),
    )
    return df
