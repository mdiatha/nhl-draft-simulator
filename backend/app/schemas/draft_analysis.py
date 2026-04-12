"""
Pydantic schemas for structured draft analysis output (Feature 10).

Claude is forced to call the `produce_draft_analysis` tool with this schema,
ensuring every analysis response is machine-readable and frontend-renderable
as rich cards rather than raw text.

Usage:
  The /api/draft/analysis endpoint passes DraftAnalysis.model_json_schema()
  as the input_schema for a forced tool call, then validates the returned
  dict against DraftAnalysis to catch any schema violations.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PickHighlight(BaseModel):
    """A notable pick — either a steal or a reach."""
    pick_number:    int             = Field(description="Overall pick number (1-based)")
    prospect_name:  str             = Field(description="Prospect full name")
    team_name:      str             = Field(description="Drafting team full name")
    label:          Literal["steal", "reach", "value", "need_fit"] = Field(
        description="steal = picked below CSS rank; reach = picked above CSS rank; "
                    "value = inline but good fit; need_fit = addressed team need"
    )
    css_rank:       Optional[int]   = Field(None, description="CSS ranking (1 = best)")
    pick_slot:      int             = Field(description="The pick slot number")
    deviation:      int             = Field(
        description="pick_slot minus css_rank. Negative = reach (picked early vs rank). "
                    "Positive = steal (picked late vs rank). 0 = inline."
    )
    commentary:     str             = Field(description="1-2 sentence analyst commentary on this pick")


class PositionTrend(BaseModel):
    """Aggregate positional distribution across the draft."""
    position:     str   = Field(description="Position code: C, LW, RW, D, G")
    count:        int   = Field(description="Number of prospects drafted at this position")
    pct_of_draft: float = Field(description="Fraction of total picks at this position [0,1]")
    trend_label:  str   = Field(description="'heavy' (overrepresented), 'normal', or 'light' (underrepresented)")


class TeamSpotlight(BaseModel):
    """One team with a particularly notable draft outcome."""
    team_name:    str                           = Field(description="Full team name")
    abbreviation: str                           = Field(description="3-letter abbreviation")
    picks:        list[str]                     = Field(description="Prospect names drafted by this team")
    grade:        Literal["A", "B", "C", "D"]  = Field(description="Draft grade")
    rationale:    str                           = Field(description="2-3 sentence explanation of the grade")


class DraftAnalysis(BaseModel):
    """
    Fully structured draft analysis produced by Claude via forced tool use.

    All fields are required — Claude is forced to populate every field via
    tool_choice={\"type\": \"tool\", \"name\": \"produce_draft_analysis\"}.
    """
    headline:           str                     = Field(
        description="One punchy sentence summarizing the draft's defining storyline"
    )
    steals:             list[PickHighlight]      = Field(
        description="Top 2-3 value picks (prospect drafted significantly below CSS rank)",
        max_length=5,
    )
    reaches:            list[PickHighlight]      = Field(
        description="Top 2-3 reaches (prospect drafted significantly above CSS rank)",
        max_length=5,
    )
    position_trends:    list[PositionTrend]      = Field(
        description="Position distribution summary — include at least 3 positions"
    )
    team_spotlight:     TeamSpotlight            = Field(
        description="One team that had a particularly notable draft (good or bad)"
    )
    overall_narrative:  str                      = Field(
        description="2-3 sentence closing paragraph summarizing the draft's overall character"
    )
