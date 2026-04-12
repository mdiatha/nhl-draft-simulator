"""Unit tests for app.ml.features."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from app.api.draft_sim import _sample_pick
from app.ml.features import (
    FEATURE_COLS,
    LEAGUE_KEYS,
    NAT_GROUPS,
    POSITIONS,
    _compute_predraft_quality,
    _contextual_feats,
    _gm_features,
    build_features,
)


def _minimal_row(**overrides) -> dict:
    base = {
        "position": "C",
        "nationality": "CAN",
        "height_cm": 182.0,
        "weight_kg": 85.0,
        "draft_league": "OHL",
        "draft_league_tier": 1,
        "points_per_game": 1.0,
        "gp_pre_draft": 60,
        "ppg_prev_season": 0.9,
        "has_prev_season": 1,
        "age_at_draft": 18.0,
        "overall_pick": 5,
        "css_rank_norm": 0.8,
        "gm_pos_weight": 0.2,
        "gm_league_weight": 0.2,
        "gm_nat_weight": 0.33,
        "ppg_league_norm": 1.1,
        "age_league_norm": 0.3,
        "pos_taken_before_norm": 0.1,
        "pos_remaining_norm": 0.4,
        "team_drafted_this_pos": 0,
        "draft_round": 1,
    }
    base.update(overrides)
    return base


def _make_df(*rows: dict) -> pd.DataFrame:
    if not rows:
        rows = (_minimal_row(),)
    return pd.DataFrame(list(rows))


class TestBuildFeatures:
    def test_output_columns_match_feature_cols(self):
        out = build_features(_make_df())
        assert list(out.columns) == FEATURE_COLS

    def test_no_nans_in_output(self):
        out = build_features(_make_df(_minimal_row(ppg_prev_season=None, height_cm=None)))
        assert not out.isna().any().any()

    def test_position_one_hot(self):
        for pos in POSITIONS:
            out = build_features(_make_df(_minimal_row(position=pos)))
            assert out[f"pos_{pos}"].iloc[0] == 1
            for other in [p for p in POSITIONS if p != pos]:
                assert out[f"pos_{other}"].iloc[0] == 0

    def test_unknown_position_all_zeros(self):
        out = build_features(_make_df(_minimal_row(position="XW")))
        for pos in POSITIONS:
            assert out[f"pos_{pos}"].iloc[0] == 0

    def test_nationality_group_one_hot(self):
        out = build_features(_make_df(_minimal_row(nationality="SWE")))
        assert out["nat_NORDIC"].iloc[0] == 1
        assert out["nat_CAN"].iloc[0] == 0

    def test_unknown_nationality_goes_to_fallback_group(self):
        out = build_features(_make_df(_minimal_row(nationality="ZZZ")))
        assert out["nat_EUR_OTHER"].iloc[0] == 1
        for group in [g for g in NAT_GROUPS if g != "EUR_OTHER"]:
            assert out[f"nat_{group}"].iloc[0] == 0

    def test_league_one_hot_ohl(self):
        out = build_features(_make_df(_minimal_row(draft_league="OHL")))
        assert out["league_tier1_CAN"].iloc[0] == 1
        for key in [k for k in LEAGUE_KEYS if k != "tier1_CAN"]:
            assert out[f"league_{key}"].iloc[0] == 0

    def test_league_one_hot_shl(self):
        out = build_features(_make_df(_minimal_row(draft_league="SHL")))
        assert out["league_tier1_EUR"].iloc[0] == 1

    def test_has_prev_season_flag(self):
        with_prev = build_features(_make_df(_minimal_row(ppg_prev_season=0.8)))
        without_prev = build_features(_make_df(_minimal_row(ppg_prev_season=None)))
        assert with_prev["has_prev_season"].iloc[0] == 1
        assert without_prev["has_prev_season"].iloc[0] == 0

    def test_ppg_trend_zero_when_no_prev_season(self):
        out = build_features(_make_df(_minimal_row(points_per_game=1.2, ppg_prev_season=None)))
        assert out["ppg_trend"].iloc[0] == 0.0

    def test_ppg_trend_computed_when_prev_season_present(self):
        out = build_features(_make_df(_minimal_row(points_per_game=1.2, ppg_prev_season=0.9)))
        assert abs(out["ppg_trend"].iloc[0] - 0.3) < 1e-6

    def test_rank_vs_slot_direction(self):
        out = build_features(_make_df(_minimal_row(css_rank_norm=0.9, overall_pick=30)))
        assert out["rank_vs_slot"].iloc[0] > 0

    def test_multiple_rows(self):
        rows = [_minimal_row(position=pos) for pos in POSITIONS]
        out = build_features(pd.DataFrame(rows))
        assert len(out) == len(POSITIONS)
        assert list(out.columns) == FEATURE_COLS


class TestSamplePick:
    def _make_prospects(self, scores: dict[int, float]):
        return [SimpleNamespace(id=pid) for pid in scores]

    def test_temperature_zero_returns_argmax(self):
        scores = {1: 0.1, 2: 0.9, 3: 0.5}
        result = _sample_pick(self._make_prospects(scores), scores, random.Random(42), temperature=0)
        assert result.id == 2

    def test_temperature_zero_always_deterministic(self):
        scores = {1: 0.3, 2: 0.7, 3: 0.1}
        results = {_sample_pick(self._make_prospects(scores), scores, random.Random(i), 0).id for i in range(20)}
        assert results == {2}

    def test_temperature_positive_introduces_variation(self):
        scores = {1: 0.51, 2: 0.49}
        results = [_sample_pick(self._make_prospects(scores), scores, random.Random(i), 1.0).id for i in range(200)]
        assert 1 in results and 2 in results

    def test_single_candidate_always_chosen(self):
        scores = {42: 1.0}
        result = _sample_pick(self._make_prospects(scores), scores, random.Random(0), 0.4)
        assert result.id == 42


class TestContextualFeats:
    def _make_prospect(self, pos="C", ppg=1.0, age=18.0, league="OHL", tier=1):
        return SimpleNamespace(
            id=1,
            position=pos,
            points_per_game=ppg,
            age_at_draft=age,
            draft_league=league,
            draft_league_tier=tier,
            league_tier=tier,
        )

    def test_goalie_ppg_norm_is_neutral(self):
        p = self._make_prospect(pos="G", ppg=0.05)
        result = _contextual_feats(p, [p], 0, 1, Counter(), defaultdict(Counter), {"tier1_CAN": 1.0}, {})
        assert result["ppg_league_norm"] == 1.0

    def test_unknown_league_ppg_norm_is_neutral(self):
        p = self._make_prospect(pos="C", ppg=0.8, league="UNKNOWN_LEAGUE", tier=None)
        result = _contextual_feats(p, [p], 0, 1, Counter(), defaultdict(Counter), {}, {})
        assert result["ppg_league_norm"] == 1.0

    def test_ppg_league_norm_computed_correctly(self):
        p = self._make_prospect(pos="C", ppg=2.0, league="OHL")
        result = _contextual_feats(p, [p], 0, 1, Counter(), defaultdict(Counter), {"tier1_CAN": 1.0}, {})
        assert abs(result["ppg_league_norm"] - 2.0) < 1e-6

    def test_pos_taken_before_norm_zero_at_start(self):
        p = self._make_prospect()
        result = _contextual_feats(p, [p], picks_made=0, team_id=1, pos_taken=Counter(), team_pos_drafted=defaultdict(Counter), tier_ppg_med={}, tier_age_med={})
        assert result["pos_taken_before_norm"] == 0.0

    def test_team_drafted_this_pos_increments(self):
        p = self._make_prospect(pos="C")
        team_pos = defaultdict(Counter)
        team_pos[1]["C"] = 2
        result = _contextual_feats(p, [p], 10, 1, Counter(), team_pos, {}, {})
        assert result["team_drafted_this_pos"] == 2


class TestComputePredraftQuality:
    def _make_pick(self, pid, pos, ppg, css_rank=None, round_num=1, overall_pick=1):
        return SimpleNamespace(
            id=pid,
            position=pos,
            points_per_game=ppg,
            css_rank=css_rank,
            round=round_num,
            overall_pick=overall_pick,
        )

    def test_higher_css_rank_gets_higher_quality(self):
        picks = [
            self._make_pick(1, "C", 1.0, css_rank=1, overall_pick=1),
            self._make_pick(2, "C", 1.0, css_rank=10, overall_pick=2),
            self._make_pick(3, "C", 1.0, css_rank=30, overall_pick=3),
        ]
        quality = _compute_predraft_quality(picks)
        assert quality[1] > quality[2] > quality[3]

    def test_missing_css_falls_back_to_round_order(self):
        picks = [
            self._make_pick(1, "C", 1.0, css_rank=None, round_num=2, overall_pick=33),
            self._make_pick(2, "C", 1.0, css_rank=None, round_num=2, overall_pick=34),
        ]
        quality = _compute_predraft_quality(picks)
        assert quality[1] > quality[2]

    def test_all_picks_have_quality_score(self):
        picks = [self._make_pick(i, "C", 1.0, css_rank=i + 1, overall_pick=i + 1) for i in range(10)]
        quality = _compute_predraft_quality(picks)
        assert all(pid in quality for pid in range(10))
        assert all(0.0 <= value <= 1.0 for value in quality.values())


class TestGmFeatures:
    def test_none_profile_returns_defaults(self):
        result = _gm_features(None, "C", "OHL", "CAN")
        assert result["gm_pos_weight"] == 0.2
        assert result["gm_league_weight"] == 0.2
        assert result["gm_nat_weight"] == 0.33

    def test_profile_lookup(self):
        profile = MagicMock()
        profile.position_weights = {"C": 0.45}
        profile.league_weights = {"tier1_CAN": 0.60}
        profile.nationality_weights = {"CAN": 0.55}
        result = _gm_features(profile, "C", "OHL", "CAN")
        assert result["gm_pos_weight"] == 0.45
        assert result["gm_league_weight"] == 0.60
        assert result["gm_nat_weight"] == 0.55
