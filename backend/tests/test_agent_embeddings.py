from __future__ import annotations

from types import SimpleNamespace

from app.agent.embeddings import (
    prospect_profile_to_text,
    prospect_style_to_text,
    prospect_trend_to_text,
)


def _prospect(**overrides):
    base = {
        "name": "Test Prospect",
        "position": "D",
        "nationality": "SWE",
        "draft_league": "SHL",
        "draft_league_tier": 1,
        "css_ranking": 7,
        "css_category": "EUR skater",
        "points_per_game": 0.82,
        "games_played": 42,
        "goals": 10,
        "assists": 24,
        "age_at_draft": 18.1,
        "height_cm": 191,
        "weight_kg": 88,
        "ppg_prev_season": 0.56,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _stat_row(**overrides):
    base = {
        "league": "SHL",
        "season_type": "pre_draft",
        "games_played": 42,
        "goals": 10,
        "assists": 24,
        "points": 34,
        "points_per_game": 0.82,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_prospect_profile_to_text_adds_contextual_details():
    text = prospect_profile_to_text(_prospect())

    assert "Prospect: Test Prospect" in text
    assert "CSS Rank: 7 | CSS Category: EUR skater" in text
    assert "Current production: 0.82 PPG across 42 GP (10G, 24A)" in text
    assert "Age at draft: 18.1 (young for the class)" in text
    assert "Context: top-tier pro or elite junior competition; pro-size frame." in text


def test_prospect_style_to_text_derives_role_tags():
    text = prospect_style_to_text(_prospect())

    assert "Projected style:" in text
    assert "offensive defenseman" in text
    assert "balanced scoring profile" in text
    assert "young for the class" in text
    assert "pro-size frame" in text


def test_prospect_trend_to_text_summarizes_growth_and_history():
    stat_rows = [
        _stat_row(points_per_game=0.82, points=34),
        _stat_row(league="J20 Nationell", season_type="previous", games_played=38, goals=7, assists=14, points=21, points_per_game=0.55),
    ]

    text = prospect_trend_to_text(_prospect(), stat_rows)

    assert "Trajectory: clear upward development trend" in text
    assert "Current season vs previous season: 0.82 PPG vs 0.56 PPG." in text
    assert "Recent stat history:" in text
    assert "J20 Nationell (previous): 38 GP, 21 points, 0.55 PPG" in text
