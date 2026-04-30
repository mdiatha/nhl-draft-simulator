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

import numpy as np
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
    # Consensus rank signals (4) — css_rank_norm added directly; others kept as context
    + ["css_rank_norm", "pick_slot_norm", "rank_vs_slot", "rank_gap_norm"]
    # GM tendency scalars (4) — includes pick-count normalisation
    + ["gm_pos_weight", "gm_league_weight", "gm_nat_weight", "gm_n_picks_norm"]
    # GM × prospect affinity interactions (3)
    # Explicit product of GM preference weight and the matching one-hot avoids
    # relying on XGBoost to discover the interaction via split combinations.
    + ["gm_pos_affinity", "gm_league_affinity", "gm_nat_affinity"]
    # Draft-state / supply signals (5)
    + ["pos_taken_before_norm", "pos_remaining_norm", "team_drafted_this_pos",
       "pos_quality_rank_norm", "css_rank_within_pos"]
    # Season-over-season production (4) — ppg_delta added
    + ["gp_pre_draft", "ppg_prev_season", "has_prev_season", "ppg_delta"]
    # Slot pressure: fraction of round remaining (1.0 = first pick, ~0 = last)
    + ["slot_pressure"]
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
        return {"gm_pos_weight": 0.2, "gm_league_weight": 0.2, "gm_nat_weight": 0.33, "gm_n_picks_norm": 0.0}

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

    # Adaptive shrinkage: GMs with more picks need less regularisation toward
    # the league average — their actual preferences are more reliable.
    n_picks = len(gm_picks)
    if n_picks < 20:
        _SHRINKAGE_K = 30   # heavy shrinkage — little history
    elif n_picks <= 60:
        _SHRINKAGE_K = 15   # moderate shrinkage
    else:
        _SHRINKAGE_K = 5    # light shrinkage — trust their preferences

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
        "gm_n_picks_norm":  min(n_picks / 200.0, 1.0),
    }


def _gm_features(profile, position: str, draft_league: str, nationality: str) -> dict:
    """Extract raw GM tendency weights for a specific (GM, prospect) pair."""
    if profile is None:
        return {
            "gm_pos_weight":    0.2,
            "gm_league_weight": 0.2,
            "gm_nat_weight":    0.33,
            "gm_n_picks_norm":  0.0,
        }

    pos_weights    = profile.position_weights or {}
    league_weights = profile.league_weights or {}
    nat_weights    = profile.nationality_weights or {}

    # n_picks is stored on the profile if available; fall back to 0.
    raw_n = getattr(profile, "n_picks", None)
    n_picks = int(raw_n) if isinstance(raw_n, (int, float)) else 0

    return {
        "gm_pos_weight":    pos_weights.get(position or "C", 0.0),
        "gm_league_weight": league_weights.get(infer_league_key(draft_league or ""), 0.0),
        "gm_nat_weight":    nat_weights.get(nat_group(nationality or ""), 0.0),
        "gm_n_picks_norm":  min(n_picks / 200.0, 1.0),
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
        leagues = df["draft_league"].fillna("").tolist()
        tiers = df[tier_col].tolist() if tier_col in df.columns else [None] * len(df)
        league_key_series = pd.Series(
            [infer_league_key(lg or "", t) for lg, t in zip(leagues, tiers)],
            index=df.index,
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
    # css_rank_norm: the direct consensus rank signal. Previously excluded and
    # only represented via derived features (rank_vs_slot, rank_gap_norm), which
    # are noisy proxies. Including it directly gives the model the clearest
    # possible signal: "how good is this player, absolutely?"
    out["css_rank_norm"] = df.get("css_rank_norm", pd.Series(0.5, index=df.index)).fillna(0.5).astype(float)
    slot = df.get("overall_pick", pd.Series(32, index=df.index)).fillna(32).astype(float)
    out["pick_slot_norm"] = (1.0 - (slot - 1) / MAX_DRAFT_POOL).clip(0.0, 1.0)
    out["rank_vs_slot"] = (out["css_rank_norm"] - out["pick_slot_norm"]).astype(float)
    # rank_gap_norm: gap from the best prospect in the FULL remaining pool.
    # Previously computed from the pick's group (1 positive + 31 negatives),
    # which made it a near-proxy for was_picked=1 (the picked player was often
    # the best, so gap=0 for positives). Now pre-computed from the full board.
    if "rank_gap_norm" in df.columns:
        out["rank_gap_norm"] = df["rank_gap_norm"].fillna(0.0).astype(float)
    else:
        best_norm = out["css_rank_norm"].max()
        out["rank_gap_norm"] = (best_norm - out["css_rank_norm"]).astype(float)

    # ── GM tendency scalars ───────────────────────────────────────────────────
    out["gm_pos_weight"]    = df.get("gm_pos_weight",    pd.Series(0.2,  index=df.index)).fillna(0.2).astype(float)
    out["gm_league_weight"] = df.get("gm_league_weight", pd.Series(0.2,  index=df.index)).fillna(0.2).astype(float)
    out["gm_nat_weight"]    = df.get("gm_nat_weight",    pd.Series(0.33, index=df.index)).fillna(0.33).astype(float)
    # gm_n_picks_norm: number of GM pre-draft picks / 200, capped at 1.0.
    # Tells the model how much to trust the tendency signal: 0.0 = new GM with
    # no history; 1.0 = experienced GM with 200+ picks tracked.
    out["gm_n_picks_norm"]  = df.get("gm_n_picks_norm",  pd.Series(0.0,  index=df.index)).fillna(0.0).astype(float)

    # ── GM × prospect affinity interactions ───────────────────────────────────
    # gm_pos/league/nat_weight are already prospect-specific scalars (e.g.
    # gm_pos_weight is the GM's weight for *this* prospect's position).
    # The affinity features are their product with css_rank_norm: a high-ranked
    # player gets amplified when a GM has strong preference for that player's
    # position/league/nationality. This captures "top center for a center-first GM"
    # as a single feature rather than requiring XGBoost to discover the 3-way split.
    out["gm_pos_affinity"]    = (out["gm_pos_weight"]    * out["css_rank_norm"]).astype(float)
    out["gm_league_affinity"] = (out["gm_league_weight"] * out["css_rank_norm"]).astype(float)
    out["gm_nat_affinity"]    = (out["gm_nat_weight"]    * out["css_rank_norm"]).astype(float)

    # ── Draft-state / supply signals ──────────────────────────────────────────
    out["pos_taken_before_norm"]  = df.get("pos_taken_before_norm",  pd.Series(0.2,  index=df.index)).fillna(0.2).astype(float)
    out["pos_remaining_norm"]     = df.get("pos_remaining_norm",     pd.Series(0.2,  index=df.index)).fillna(0.2).astype(float)
    out["team_drafted_this_pos"]  = df.get("team_drafted_this_pos",  pd.Series(0,    index=df.index)).fillna(0).astype(int)
    out["pos_quality_rank_norm"]  = df.get("pos_quality_rank_norm",  pd.Series(0.5,  index=df.index)).fillna(0.5).astype(float)
    # css_rank_within_pos: this prospect's css_rank_norm rank among same-position
    # players still on the board. 1.0 = best available at this position.
    # More stable than pos_quality_rank_norm (which uses ordinal rank / n_pos).
    out["css_rank_within_pos"] = df.get("css_rank_within_pos", pd.Series(0.5, index=df.index)).fillna(0.5).astype(float)

    # ── Season-over-season production ─────────────────────────────────────────
    out["gp_pre_draft"]    = df.get("gp_pre_draft",    pd.Series(30,  index=df.index)).fillna(30).astype(float)
    ppg_prev_raw           = df.get("ppg_prev_season", pd.Series(dtype=float)).reindex(df.index)
    out["ppg_prev_season"] = ppg_prev_raw.fillna(0.0).astype(float)
    out["has_prev_season"] = ppg_prev_raw.notna().astype(int)
    # ppg_delta: year-over-year production improvement, normalized by the league
    # median PPG so "improved 0.3 PPG in OHL" and "improved 0.3 PPG in KHL" are
    # comparable. Positive = player improved; negative = regressed.
    # Zero when prior season data is unavailable (has_prev_season=0).
    ppg_current = df.get("points_per_game", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float)
    ppg_med_for_delta = df.get("ppg_league_norm", pd.Series(1.0, index=df.index)).fillna(1.0).astype(float)
    # Avoid division by zero; neutral value 0 when no prior season exists
    raw_delta = ppg_current - ppg_prev_raw.fillna(ppg_current)
    out["ppg_delta"] = (
        (raw_delta / ppg_med_for_delta.clip(lower=0.01))
        .clip(-3.0, 3.0)
        .where(ppg_prev_raw.notna(), 0.0)
        .astype(float)
    )

    # ── Slot pressure ─────────────────────────────────────────────────────────
    # Fraction of the current round remaining after this pick (1.0 = first pick
    # of a round; ~0 = last pick). GMs at end-of-round are more likely to deviate
    # from BPA: risk of a positional run exhausting their target is higher.
    out["slot_pressure"] = df.get("slot_pressure", pd.Series(0.5, index=df.index)).fillna(0.5).astype(float)

    # ── Draft round one-hot ───────────────────────────────────────────────────
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

    Sort all ranked picks by raw CSS rank globally, assign within-cohort
    ordinal rank, then sqrt-normalize to [0, 1] using the cohort size as denom.
    Top prospect scores ~1.0; rank #N scores ~1 - sqrt((N-1)/cohort_size).

    Note: inference (predict.py) uses per-CSS-list normalization. The global
    sort here means EUR skater CSS #50 gets a lower score than NA skater CSS #50,
    which reflects real draft value (the CSS numbers are globally comparable).
    Falls back to 0.15 for any pick without a CSS rank.
    """
    import math

    has_css = [p for p in year_picks if getattr(p, "css_rank", None) is not None]
    result: dict[int, float] = {}

    if has_css:
        sorted_players = sorted(has_css, key=lambda p: p.css_rank)
        cohort_size = len(sorted_players)
        for ordinal, p in enumerate(sorted_players, start=1):
            result[p.id] = max(0.0, 1.0 - math.sqrt((ordinal - 1) / max(cohort_size, 1)))

    for p in year_picks:
        if p.id not in result:
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

        # Precompute per-round slot counts for slot_pressure.
        # picks_in_round[r] = total picks in round r for this draft year.
        picks_in_round: dict[int, int] = Counter(p.round for p in year_picks_sorted)
        # Track pick index within each round (reset each round).
        round_pick_index: Counter[int] = Counter()

        # best_remaining is not needed here; rank_gap_norm is recomputed group-wise
        # after all rows are collected (see below the main loop).

        for i, pick in enumerate(year_picks_sorted):
            round_pick_index[pick.round] += 1

            if not pick.gm_id or not pick.team_id:
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

            # rank_gap_norm is recomputed group-wise after all rows are built.
            # Use 0.0 as placeholder; it will be overwritten.
            pick_css_norm = quality[pick.id]

            # css_rank_within_pos: rank among same-position players still available.
            # 1.0 = best at this position on the board; 0.0 = worst.
            same_pos_remaining = [p for p in remaining if (p.position or "F") == (pick.position or "F")]
            n_same_pos = len(same_pos_remaining)
            if n_same_pos > 1:
                n_better_pos = sum(1 for p in same_pos_remaining if quality[p.id] > pick_css_norm)
                pick_css_rank_within_pos = 1.0 - (n_better_pos / n_same_pos)
            else:
                pick_css_rank_within_pos = 1.0

            # slot_pressure: fraction of current round remaining after this pick.
            # 1.0 = first pick of round; ~0.03 = last pick of round.
            total_in_rnd = picks_in_round.get(pick.round, 32)
            idx_in_rnd   = round_pick_index[pick.round]  # 1-based after increment above
            pick_slot_pressure = max(0.0, 1.0 - (idx_in_rnd - 1) / max(total_in_rnd - 1, 1))

            pos_row = {
                "year":                  year,
                "position":              pick.position,
                "nationality":           pick.nationality,
                "height_cm":             pick.height_cm,
                "weight_kg":             pick.weight_kg,
                "draft_league":          pick.draft_league,
                "draft_league_tier":     pick.draft_league_tier,
                "points_per_game":       pick.points_per_game,
                "gp_pre_draft":          pick.gp_pre_draft,
                "ppg_prev_season":       pick.ppg_prev_season,
                "has_prev_season":       1 if pick.ppg_prev_season is not None else 0,
                "age_at_draft":          pick.age_at_draft,
                "overall_pick":          pick.overall_pick,
                "draft_round":           pick.round,
                "css_rank_norm":         pick_css_norm,
                "rank_gap_norm":         0.0,  # recomputed group-wise after loop
                "css_rank_within_pos":   pick_css_rank_within_pos,
                "slot_pressure":         pick_slot_pressure,
                **gm_feats,
                **ctx,
                "was_picked": 1,
            }
            rows.append(pos_row)

            # ── Position-stratified negative sampling ─────────────────────────
            # Goal: at least 40% of negatives are from the same position as the
            # positive pick (drawn from anywhere in the draft year remaining pool),
            # ensuring the model learns fine-grained within-position distinctions.
            # The remaining 60% use the original slot-window approach.
            #
            # n_pos_target  = floor(NEGATIVE_WINDOW * 0.4) = 12
            # n_slot_target = NEGATIVE_WINDOW - n_pos_target = 19
            # Negatives are deduplicated so the total stays at NEGATIVE_WINDOW.
            later_window = year_picks_sorted[i + 1:][:NEGATIVE_WINDOW]

            pos_target  = int(NEGATIVE_WINDOW * 0.4)  # 12 same-position negatives
            slot_target = NEGATIVE_WINDOW - pos_target  # 19 slot-window negatives

            pick_pos = pick.position or "F"

            # Same-position candidates: any pick after this one in the year,
            # NOT restricted to the slot window.
            same_pos_pool = [
                p for p in year_picks_sorted[i + 1:]
                if (p.position or "F") == pick_pos
            ]

            if len(same_pos_pool) >= pos_target:
                # Enough same-position players available: sample pos_target of them.
                pos_negs = rng.sample(same_pos_pool, pos_target)
                # Slot-window negatives: exclude picks already chosen as pos_negs.
                pos_neg_ids = {p.id for p in pos_negs}
                slot_pool = [p for p in later_window if p.id not in pos_neg_ids]
                slot_negs = slot_pool[:slot_target]
                negs = pos_negs + slot_negs
            else:
                # Not enough same-position players — fall back to original behavior.
                negs = later_window

            if neg_samples_per_pick is not None:
                negs = rng.sample(negs, min(neg_samples_per_pick, len(negs)))

            for neg in negs:
                neg_css_norm = quality[neg.id]

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

                same_pos_neg = [p for p in remaining if (p.position or "F") == (neg.position or "F")]
                n_sp_neg = len(same_pos_neg)
                if n_sp_neg > 1:
                    n_better_neg = sum(1 for p in same_pos_neg if quality[p.id] > neg_css_norm)
                    neg_css_rank_within_pos = 1.0 - (n_better_neg / n_sp_neg)
                else:
                    neg_css_rank_within_pos = 1.0

                rows.append({
                    "year":                  year,
                    "position":              neg.position,
                    "nationality":           neg.nationality,
                    "height_cm":             neg.height_cm,
                    "weight_kg":             neg.weight_kg,
                    "draft_league":          neg.draft_league,
                    "draft_league_tier":     neg.draft_league_tier,
                    "points_per_game":       neg.points_per_game,
                    "gp_pre_draft":          neg.gp_pre_draft,
                    "ppg_prev_season":       neg.ppg_prev_season,
                    "has_prev_season":       1 if neg.ppg_prev_season is not None else 0,
                    "age_at_draft":          neg.age_at_draft,
                    "overall_pick":          pick.overall_pick,
                    "draft_round":           pick.round,
                    "css_rank_norm":         neg_css_norm,
                    "rank_gap_norm":         0.0,  # recomputed group-wise after loop
                    "css_rank_within_pos":   neg_css_rank_within_pos,
                    "slot_pressure":         pick_slot_pressure,
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

    # Recompute rank_gap_norm as group-best minus each player's css_rank_norm.
    # Each group = (year, overall_pick) = 1 positive + up to NEGATIVE_WINDOW negatives.
    # This tells the model "how far from the best available in this pick's comparison set?"
    # which is a direct measure of opportunity cost at this slot.
    if "css_rank_norm" in df.columns and "rank_gap_norm" in df.columns:
        group_best = df.groupby(["year", "overall_pick"])["css_rank_norm"].transform("max")
        df["rank_gap_norm"] = (group_best - df["css_rank_norm"]).clip(lower=0.0)

    # Sample weighting: additive combination of recency and round importance.
    #
    # Old scheme: multiplicative (recency × round_weight). Problem: a round-1 pick
    # from 2008 (weight 0.88^16 × 4.0 = 0.48) could outweigh a round-2 pick from
    # 2022 (weight 0.88^2 × 2.5 = 1.94). Recency and round signal are independent;
    # multiplying them creates an unintended joint prior.
    #
    # New scheme: additive with separate caps.
    #   recency_w = 0.92^(MAX_YEAR - year)   — gentler decay; 2008 data keeps ~26%
    #                                           weight vs 12% with 0.88 base
    #   round_w   = 1 + round_bonus[round]   — additive bonus on top of base 1.0
    # Final weight = recency_w * (1 + round_bonus) so recency still scales round
    # importance, but no single factor dominates the product.
    RECENCY_DECAY  = 0.92
    ROUND_BONUS    = {1: 3.0, 2: 1.0, 3: 0.4, 4: 0.1}   # additive on top of 1.0; R1 boosted to 3.0
    MAX_YEAR   = int(df["year"].max()) if "year" in df.columns else 2024
    years      = np.asarray(df["year"].astype(int))
    rounds     = df.get("draft_round", pd.Series(1, index=df.index)).fillna(1).astype(int).values
    recency_w  = RECENCY_DECAY ** (MAX_YEAR - years)
    round_mult = np.array([1.0 + ROUND_BONUS.get(int(r), 0.0) for r in rounds])
    df["sample_weight"] = recency_w * round_mult

    logger.info(
        "Training dataset: %d rows (%d positive, %d negative)",
        len(df), int(df["was_picked"].sum()), int((df["was_picked"] == 0).sum()),
    )
    return df
