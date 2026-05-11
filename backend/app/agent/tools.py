"""
Tool definitions and executors for the Scout agent.

Claude is given these tools and decides which to call based on the user's
question. Each tool is backed by a real DB query — no hard-coded context
injection.

Input validation: each tool's inputs are parsed through a Pydantic model.
This gives type coercion, field-level constraints, and clear ValidationError
messages that Claude can use to retry — all from a single schema that serves
as both the JSON schema for Claude's tool definition AND the runtime validator.

Tools:
  get_gm_profile(team_name)            — GM tendency profile + archetype for a team
  get_top_prospects(position, n)       — top N prospects by CSS rank for a position
  get_team_needs(team_name)            — positional needs based on roster + tendency
  search_prospects(query)              — keyword search across prospect names / leagues
  semantic_prospect_search(query)      — vector + BM25 hybrid search
  get_draft_history(team_name, year?)  — historical draft picks for a team
  get_ml_ranking(prospect, team?, slot?) — XGBoost score + SHAP explanation
  compare_prospects(a, b)              — side-by-side stat comparison of two prospects
  get_prospect_detail(name)            — full profile + multi-season stat history
  get_nhl_comp(prospect_name)          — semantic NHL player comps via RAG
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Pydantic input models (one per tool) ──────────────────────────────────────

class GMProfileInput(BaseModel):
    team_name: str = Field(..., min_length=2, max_length=100,
                           description="Full team name or abbreviation, e.g. 'Toronto Maple Leafs' or 'TOR'")

    @field_validator("team_name")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class TopProspectsInput(BaseModel):
    position: str = Field(..., description="Position code: C, LW, RW, D, G, or 'all'")
    limit: int = Field(default=10, ge=1, le=25, description="Number of prospects to return")

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: str) -> str:
        v = v.strip().upper()
        valid = {"C", "LW", "RW", "D", "G", "ALL"}
        if v not in valid:
            raise ValueError(f"position must be one of {valid}")
        return v if v != "ALL" else "all"


class TeamNameInput(BaseModel):
    team_name: str = Field(..., min_length=2, max_length=100)

    @field_validator("team_name")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class QueryInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)

    @field_validator("query")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class DraftHistoryInput(BaseModel):
    team_name: str = Field(..., min_length=2, max_length=100)
    year: Optional[int] = Field(default=None, ge=2000, le=2025)

    @field_validator("team_name")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class MLRankingInput(BaseModel):
    prospect_name: str = Field(..., min_length=2, max_length=100,
                               description="Prospect name to score")
    team_name: Optional[str] = Field(default=None, max_length=100,
                                     description="Team name for GM-adjusted score and SHAP")
    pick_slot: int = Field(default=15, ge=1, le=224,
                           description="Overall pick number context (default 15)")

    @field_validator("prospect_name", "team_name")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v else v


class CompareProspectsInput(BaseModel):
    prospect_a: str = Field(..., min_length=2, max_length=100)
    prospect_b: str = Field(..., min_length=2, max_length=100)

    @field_validator("prospect_a", "prospect_b")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class ProspectDetailInput(BaseModel):
    prospect_name: str = Field(..., min_length=2, max_length=100)

    @field_validator("prospect_name")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class NHLCompInput(BaseModel):
    prospect_name: str = Field(..., min_length=2, max_length=100,
                               description="Prospect name to find NHL comps for")

    @field_validator("prospect_name")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

# ── Tool schemas (passed to Claude as tools=[...]) ────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_gm_profile",
        "description": (
            "Returns the GM tendency profile for a team: position weights, league weights, "
            "nationality weights, archetype (BPA / need-based / safe / boom-bust / system-fit), "
            "and average CSS rank deviation. Use this when asked about how a GM or team drafts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {
                    "type": "string",
                    "description": "Full team name or abbreviation, e.g. 'Toronto Maple Leafs' or 'TOR'",
                }
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_top_prospects",
        "description": (
            "Returns the top draft prospects for a given position sorted by CSS ranking. "
            "Use when asked about the best available players at a specific position."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "string",
                    "description": "Position code: C, LW, RW, D, G, or 'all' for all positions",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of prospects to return (default 10, max 25)",
                    "default": 10,
                },
            },
            "required": ["position"],
        },
    },
    {
        "name": "get_team_needs",
        "description": (
            "Returns a team's positional needs based on their GM's historical drafting tendencies "
            "and position weights. Use when asked which positions a team should prioritize."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {
                    "type": "string",
                    "description": "Full team name or abbreviation",
                }
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "search_prospects",
        "description": (
            "Search for specific prospects by name, league, or nationality. "
            "Returns matching prospects with their CSS rank, position, and stats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term — prospect name, league (OHL, SHL, NCAA...), or nationality (CAN, USA, SWE...)",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "semantic_prospect_search",
        "description": (
            "Find prospects semantically similar to a described player archetype or a named player. "
            "Use for queries like 'find me a player like Makar', 'who plays like a two-way center', "
            "'find a Swedish offensive defenseman'. Returns top-5 semantically similar prospects "
            "based on embedded player profiles. Use this instead of search_prospects when the "
            "query describes a style of play, archetype, or similarity to another player."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description or player name to find similar prospects for",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_draft_history",
        "description": (
            "Returns a team's actual historical draft picks for a specific year or recent years (2020–2024). "
            "Shows each pick's overall position, player name, position, CSS rank, league, and pre-draft PPG. "
            "Use when asked what a team drafted in a given year, how they used their picks, "
            "or to analyze their past drafting patterns beyond just tendency weights."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {
                    "type": "string",
                    "description": "Full team name or abbreviation, e.g. 'Detroit Red Wings' or 'DET'",
                },
                "year": {
                    "type": "integer",
                    "description": "Draft year (2008–2024). Omit to get the last 3 years.",
                },
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_ml_ranking",
        "description": (
            "Returns the XGBoost model's predicted score for a prospect, optionally in the context of a "
            "specific team at a given pick slot. When a team is provided, also returns the top SHAP factors "
            "explaining why the model rates this prospect highly or poorly for that team. "
            "Use when asked how the model rates a prospect, what's driving their ranking, or for a "
            "data-driven pick recommendation at a specific slot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prospect_name": {
                    "type": "string",
                    "description": "Name of the prospect to score",
                },
                "team_name": {
                    "type": "string",
                    "description": "Team name or abbreviation for GM-adjusted score + SHAP explanation (optional)",
                },
                "pick_slot": {
                    "type": "integer",
                    "description": "Overall pick number context (1–224). Default 15 if not specified.",
                },
            },
            "required": ["prospect_name"],
        },
    },
    {
        "name": "compare_prospects",
        "description": (
            "Side-by-side comparison of two prospects: CSS rank, PPG (current and prior season), "
            "age, size, league, nationality. Use when asked to directly compare two players or "
            "help decide between them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prospect_a": {"type": "string", "description": "First prospect name"},
                "prospect_b": {"type": "string", "description": "Second prospect name"},
            },
            "required": ["prospect_a", "prospect_b"],
        },
    },
    {
        "name": "get_prospect_detail",
        "description": (
            "Full profile for one prospect including multi-season stat history (games played, "
            "goals, assists, PPG per season), physical measurements, age, league, and nationality. "
            "Use when the user wants deep context on a specific player beyond basic stats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prospect_name": {"type": "string", "description": "Prospect name"},
            },
            "required": ["prospect_name"],
        },
    },
    {
        "name": "get_nhl_comp",
        "description": (
            "Find the current NHL players who play most similarly to a given 2025 draft prospect, "
            "based on semantic similarity of their style profiles. Returns the top 3 NHL comps "
            "with similarity scores and player profiles. "
            "Use for questions like 'who does [prospect] play like?', 'what's their NHL comp?', "
            "or 'which NHL player does [prospect] remind you of?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prospect_name": {
                    "type": "string",
                    "description": "Name of the 2025 draft prospect to find NHL comps for",
                },
            },
            "required": ["prospect_name"],
        },
    },
]


# ── Tool executors ────────────────────────────────────────────────────────────

_TOOL_PARSERS = {
    "get_gm_profile":           GMProfileInput,
    "get_team_needs":           TeamNameInput,
    "search_prospects":         QueryInput,
    "semantic_prospect_search": QueryInput,
    "get_top_prospects":        TopProspectsInput,
    "get_draft_history":        DraftHistoryInput,
    "get_ml_ranking":           MLRankingInput,
    "compare_prospects":        CompareProspectsInput,
    "get_prospect_detail":      ProspectDetailInput,
    "get_nhl_comp":             NHLCompInput,
}


def execute_tool(name: str, inputs: dict, db: Session) -> str:
    """
    Dispatch a tool call to its executor. Returns a JSON string result.

    Inputs are validated through Pydantic models before reaching the executor.
    ValidationErrors are returned as structured error JSON so Claude can
    correct its input and retry within the same tool-use loop.
    """
    from pydantic import ValidationError

    parser = _TOOL_PARSERS.get(name)
    if parser is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        parsed = parser.model_validate(inputs)
    except ValidationError as exc:
        errors = [f"{e['loc'][0]}: {e['msg']}" for e in exc.errors()]
        logger.warning("tool.validation_failed tool=%s errors=%s", name, errors)
        return json.dumps({"error": f"Invalid input: {'; '.join(errors)}"})

    try:
        if name == "get_gm_profile":
            return _get_gm_profile(parsed.team_name, db)
        elif name == "get_top_prospects":
            return _get_top_prospects(parsed.position, parsed.limit, db)
        elif name == "get_team_needs":
            return _get_team_needs(parsed.team_name, db)
        elif name == "search_prospects":
            return _search_prospects(parsed.query, db)
        elif name == "semantic_prospect_search":
            return _semantic_prospect_search(parsed.query, db)
        elif name == "get_draft_history":
            return _get_draft_history(parsed.team_name, parsed.year, db)
        elif name == "get_ml_ranking":
            return _get_ml_ranking(parsed.prospect_name, parsed.team_name, parsed.pick_slot, db)
        elif name == "compare_prospects":
            return _compare_prospects(parsed.prospect_a, parsed.prospect_b, db)
        elif name == "get_prospect_detail":
            return _get_prospect_detail(parsed.prospect_name, db)
        elif name == "get_nhl_comp":
            return _get_nhl_comp(parsed.prospect_name, db)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as exc:
        logger.warning("tool.execute_failed", extra={"tool": name, "error": str(exc)})
        return json.dumps({"error": "Tool execution failed. Please try again."})


def _find_team(team_name: str, db: Session):
    from app.models import Team
    q = team_name.strip()
    team = (
        db.query(Team)
        .filter(
            (Team.full_name.ilike(f"%{q}%")) | (Team.abbreviation.ilike(q))
        )
        .first()
    )
    return team


def _find_team_with_gm_and_profile(team_name: str, db: Session):
    """Return (team, gm, profile) in one JOIN query instead of three round trips."""
    from app.models import Team, GeneralManager, GMTendencyProfile

    q = team_name.strip()
    row = (
        db.query(Team, GeneralManager, GMTendencyProfile)
        .outerjoin(
            GeneralManager,
            (GeneralManager.team_id == Team.id) & (GeneralManager.is_active.is_(True)),
        )
        .outerjoin(GMTendencyProfile, GMTendencyProfile.gm_id == GeneralManager.id)
        .filter((Team.full_name.ilike(f"%{q}%")) | (Team.abbreviation.ilike(q)))
        .order_by(GMTendencyProfile.computed_at.desc())
        .first()
    )
    if row is None:
        return None, None, None
    return row[0], row[1], row[2]


def _doc_type_label(doc_type: str) -> str:
    labels = {
        "prospect": "prospect summary",
        "prospect_with_stats": "prospect summary",
        "prospect_profile": "prospect profile",
        "prospect_style": "style profile",
        "prospect_trend": "development trend",
        "nhl_player": "NHL player profile",
    }
    return labels.get(doc_type, doc_type.replace("_", " "))


def _compact_evidence(content: str, max_lines: int = 3) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return " | ".join(lines[:max_lines])[:500]


def _get_gm_profile(team_name: str, db: Session) -> str:
    team, gm, profile = _find_team_with_gm_and_profile(team_name, db)

    if not team:
        return json.dumps({"error": f"Team '{team_name}' not found"})
    if not gm:
        return json.dumps({"team": team.full_name, "error": "No active GM found"})
    if not profile:
        return json.dumps({"team": team.full_name, "gm": gm.name, "error": "No tendency profile computed yet"})

    pos = profile.position_weights or {}
    leagues = profile.league_weights or {}
    nations = profile.nationality_weights or {}

    return json.dumps({
        "team": team.full_name,
        "abbreviation": team.abbreviation,
        "gm": gm.name,
        "archetype": profile.tendency_archetype,
        "avg_css_rank_deviation": profile.avg_ranking_deviation,
        "top_positions": sorted(pos.items(), key=lambda x: -x[1])[:4],
        "top_leagues": sorted(leagues.items(), key=lambda x: -x[1])[:4],
        "top_nationalities": sorted(nations.items(), key=lambda x: -x[1])[:3],
    })


def _get_top_prospects(position: str, limit: int, db: Session) -> str:
    from app.models import Prospect

    limit = min(int(limit), 25)
    q = db.query(Prospect).order_by(Prospect.css_ranking.nullslast())
    if position.lower() != "all":
        q = q.filter(Prospect.position == position.upper())

    prospects = q.limit(limit).all()
    if not prospects:
        return json.dumps({"error": f"No prospects found for position '{position}'"})

    return json.dumps({
        "position": position,
        "prospects": [
            {
                "name": p.name,
                "css_rank": p.css_ranking,
                "position": p.position,
                "nationality": p.nationality,
                "league": p.draft_league,
                "ppg": p.points_per_game,
                "age": p.age_at_draft,
            }
            for p in prospects
        ],
    })


def _get_team_needs(team_name: str, db: Session) -> str:
    from app.models import Prospect
    from sqlalchemy.sql import func

    team, gm, profile = _find_team_with_gm_and_profile(team_name, db)
    if not team:
        return json.dumps({"error": f"Team '{team_name}' not found"})
    if not gm:
        return json.dumps({"team": team.full_name, "error": "No active GM found"})

    pos_weights = profile.position_weights if profile else {}

    # Draft class availability: aggregate in DB instead of fetching all rows
    rows = (
        db.query(
            func.coalesce(Prospect.position, "F").label("pos"),
            func.count().label("cnt"),
        )
        .group_by(func.coalesce(Prospect.position, "F"))
        .all()
    )
    total = sum(r.cnt for r in rows)
    class_rates = {r.pos: r.cnt / total for r in rows} if total else {}

    # Need = how much the GM wants this position vs. how available it is.
    # A high GM weight against a thin position in the class = genuine need.
    # A high GM weight against a deep position = just their style, not a need.
    POSITIONS = ["C", "LW", "RW", "D", "G"]
    needs = []
    for pos in POSITIONS:
        gm_rate  = pos_weights.get(pos, 0.0)
        avail    = class_rates.get(pos, 0.0)
        # Demand/supply ratio: >1 means GM wants more of this position than exists
        ratio = gm_rate / avail if avail > 0 else (1.5 if gm_rate > 0 else 0.0)
        needs.append({
            "position":       pos,
            "gm_draft_rate":  round(gm_rate, 3),
            "class_share":    round(avail, 3),
            "demand_supply":  round(ratio, 2),
            "need":           "high" if ratio >= 1.3 else ("medium" if ratio >= 0.9 else "low"),
        })

    needs.sort(key=lambda x: -x["demand_supply"])

    return json.dumps({
        "team":             team.full_name,
        "gm":               gm.name,
        "archetype":        profile.tendency_archetype if profile else None,
        "positional_needs": needs,
        "note": (
            "Need = GM's historical draft rate vs. position availability in the current class. "
            "demand_supply > 1.3 = team historically drafts this position more than the class supplies it."
        ),
    })


def _semantic_prospect_search(query: str, db: Session) -> str:
    """
    Find prospects semantically similar to a natural-language description.

    Uses Reciprocal Rank Fusion (RRF) of:
      - Vector ANN (pgvector cosine similarity) — for style/archetype queries
      - BM25 full-text search (pg tsvector) — for exact name/league matches

    Example: "find a player like Makar" → vector search wins
             "find Bedard" → BM25 wins, RRF fuses both for correct top result
    """
    from app.agent import store

    try:
        from app.agent.embeddings import embed_text
        query_vec = embed_text(query)
    except Exception as exc:
        logger.warning("semantic_search.embed_failed falling_back_to_keyword error=%s", exc)
        return _search_prospects(query, db)

    doc_types = ["prospect"]

    if query_vec:
        # Hybrid RRF: best of vector similarity + BM25 keyword
        results = store.retrieve_hybrid(
            db,
            query_vec,
            query,
            doc_types=doc_types,
            top_k=5,
        )
        method = "hybrid_rrf"
    else:
        # No embedding available — fall back to BM25 only
        results = store.retrieve_bm25(db, query, doc_types=doc_types, top_k=5)
        method = "bm25_keyword"

    if not results:
        return _search_prospects(query, db)

    return json.dumps({
        "query": query,
        "method": method,
        "results": [
            {
                "prospect": r["ref_name"],
                "similarity_score": round(r["score"], 4),
                "matched_on": _doc_type_label(r["doc_type"]),
                "source_type": r["doc_type"],
                "evidence": _compact_evidence(r["content"]),
            }
            for r in results
        ],
    })


def _get_draft_history(team_name: str, year: int | None, db: Session) -> str:
    from app.models import DraftPickHistorical

    team = _find_team(team_name, db)
    if not team:
        return json.dumps({"error": f"Team '{team_name}' not found"})

    q = db.query(DraftPickHistorical).filter(DraftPickHistorical.team_id == team.id)
    if year:
        q = q.filter(DraftPickHistorical.year == year)
        years_label = str(year)
    else:
        # Default: last 3 completed drafts
        q = q.filter(DraftPickHistorical.year >= 2022)
        years_label = "2022–2024"

    picks = q.order_by(DraftPickHistorical.year, DraftPickHistorical.overall_pick).all()

    if not picks:
        return json.dumps({
            "team": team.full_name,
            "years": years_label,
            "error": "No draft history found for this team and year range.",
        })

    picks_data = [
        {
            "year": p.year,
            "round": p.round,
            "overall_pick": p.overall_pick,
            "player": p.player_name,
            "position": p.position,
            "css_rank": p.css_rank,
            "league": p.draft_league,
            "ppg": p.points_per_game,
            "nationality": p.nationality,
        }
        for p in picks
    ]

    return json.dumps({
        "team": team.full_name,
        "abbreviation": team.abbreviation,
        "years": years_label,
        "picks": picks_data,
    })


def _get_ml_ranking(prospect_name: str, team_name: Optional[str], pick_slot: int, db: Session) -> str:
    from app.models import Prospect, GeneralManager, GMTendencyProfile
    from app.ml.registry import registry
    from app.ml.predict import score_single_prospect

    if not registry.is_loaded:
        return json.dumps({"error": "ML model not loaded — run POST /api/ml/reload first."})

    prospect = (
        db.query(Prospect)
        .filter(Prospect.name.ilike(f"%{prospect_name}%"))
        .order_by(Prospect.css_ranking.nullslast())
        .first()
    )
    if not prospect:
        return json.dumps({"error": f"Prospect '{prospect_name}' not found"})

    profile = None
    team_label = None
    team_obj = None
    if team_name:
        team_obj = _find_team(team_name, db)
        if team_obj:
            team_label = team_obj.full_name
            gm = db.query(GeneralManager).filter(
                GeneralManager.team_id == team_obj.id,
                GeneralManager.is_active.is_(True),
            ).first()
            if gm:
                profile = (
                    db.query(GMTendencyProfile)
                    .filter(GMTendencyProfile.gm_id == gm.id)
                    .order_by(GMTendencyProfile.computed_at.desc())
                    .first()
                )

    ml_score = score_single_prospect(prospect, profile, pick_slot)

    result: dict = {
        "prospect": prospect.name,
        "position": prospect.position,
        "css_rank": prospect.css_ranking,
        "pick_slot": pick_slot,
        "ml_score": round(ml_score, 4),
        "note": "Higher score = model predicts higher likelihood of selection at this pick slot.",
    }
    if team_label:
        result["team_context"] = team_label

    if team_obj:
        try:
            from app.ml.explain import explain_prospect_pick
            explanation = explain_prospect_pick(
                db, registry.model, prospect.id, team_obj.id, pick_slot, top_n=5
            )
            result["top_factors"] = [
                {
                    "feature": f["feature"],
                    "impact": round(f["shap_value"], 4),
                    "direction": "positive" if f["shap_value"] > 0 else "negative",
                }
                for f in explanation.get("top_features", [])
            ]
        except Exception as exc:
            logger.debug("get_ml_ranking.shap_failed error=%s", exc)

    return json.dumps(result)


def _compare_prospects(prospect_a: str, prospect_b: str, db: Session) -> str:
    from app.models import Prospect

    def _find(name: str):
        return (
            db.query(Prospect)
            .filter(Prospect.name.ilike(f"%{name}%"))
            .order_by(Prospect.css_ranking.nullslast())
            .first()
        )

    pa = _find(prospect_a)
    pb = _find(prospect_b)

    if not pa:
        return json.dumps({"error": f"Prospect '{prospect_a}' not found"})
    if not pb:
        return json.dumps({"error": f"Prospect '{prospect_b}' not found"})

    def _dict(p):
        return {
            "name":            p.name,
            "position":        p.position,
            "css_rank":        p.css_ranking,
            "nationality":     p.nationality,
            "league":          p.draft_league,
            "ppg":             p.points_per_game,
            "ppg_prev_season": p.ppg_prev_season,
            "games_played":    p.games_played,
            "age_at_draft":    p.age_at_draft,
            "height_cm":       p.height_cm,
            "weight_kg":       p.weight_kg,
        }

    css_edge = pa.name if (pa.css_ranking or 9999) < (pb.css_ranking or 9999) else pb.name
    ppg_edge = pa.name if (pa.points_per_game or 0) > (pb.points_per_game or 0) else pb.name

    return json.dumps({
        "prospects":  [_dict(pa), _dict(pb)],
        "css_rank_edge": css_edge,
        "ppg_edge":      ppg_edge,
    })


def _get_prospect_detail(prospect_name: str, db: Session) -> str:
    from app.models import Prospect
    from app.models.prospect_stat_history import ProspectStatHistory

    prospect = (
        db.query(Prospect)
        .filter(Prospect.name.ilike(f"%{prospect_name}%"))
        .order_by(Prospect.css_ranking.nullslast())
        .first()
    )
    if not prospect:
        return json.dumps({"error": f"Prospect '{prospect_name}' not found"})

    stat_history = (
        db.query(ProspectStatHistory)
        .filter(ProspectStatHistory.prospect_id == prospect.id)
        .order_by(ProspectStatHistory.fetched_at.desc())
        .limit(10)
        .all()
    )

    return json.dumps({
        "name":            prospect.name,
        "position":        prospect.position,
        "css_rank":        prospect.css_ranking,
        "nationality":     prospect.nationality,
        "league":          prospect.draft_league,
        "age_at_draft":    prospect.age_at_draft,
        "height_cm":       prospect.height_cm,
        "weight_kg":       prospect.weight_kg,
        "current_ppg":     prospect.points_per_game,
        "prev_season_ppg": prospect.ppg_prev_season,
        "games_played":    prospect.games_played,
        "stat_history": [
            {
                "season_type": s.season_type,
                "league":      s.league,
                "gp":          s.games_played,
                "goals":       s.goals,
                "assists":     s.assists,
                "points":      s.points,
                "ppg":         s.points_per_game,
            }
            for s in stat_history
        ],
    })


def _get_nhl_comp(prospect_name: str, db: Session) -> str:
    """
    Find the current NHL players who play most similarly to a draft prospect.

    Approach:
      1. Look up the prospect in prospects
      2. Build a richer prospect profile using factual, style, and trend chunks
      3. Embed the combined description via voyage-3-lite
      4. Vector search against 'nhl_player' docs in scout_embeddings
      5. Return top 3 matches with similarity scores

    Requires: POST /api/admin/ingest-nhl-players (then /api/agent/index) to have been run.
    """
    from app.models import Prospect, ProspectStatHistory
    from app.agent import store
    from app.agent.embeddings import embed_text, prospect_to_text

    prospect = (
        db.query(Prospect)
        .filter(Prospect.name.ilike(f"%{prospect_name}%"))
        .order_by(Prospect.css_ranking.nullslast())
        .first()
    )
    if not prospect:
        return json.dumps({"error": f"Prospect '{prospect_name}' not found"})

    stat_rows = (
        db.query(ProspectStatHistory)
        .filter(ProspectStatHistory.prospect_id == prospect.id)
        .order_by(ProspectStatHistory.fetched_at.desc())
        .limit(4)
        .all()
    )

    query_vec = embed_text(prospect_to_text(prospect, stat_rows))
    if not query_vec:
        return json.dumps({
            "error": (
                "Embedding service unavailable. "
                "Ensure Ollama is running and run POST /api/agent/index."
            )
        })

    results = store.retrieve_similar(
        db,
        query_vec,
        doc_types=["nhl_player"],
        top_k=3,
    )
    if not results:
        return json.dumps({
            "prospect": prospect.name,
            "error": (
                "No NHL player data in index. "
                "Run POST /api/admin/ingest-nhl-players then POST /api/agent/index."
            ),
        })

    return json.dumps({
        "prospect":  prospect.name,
        "position":  prospect.position or "F",
        "css_rank":  prospect.css_ranking,
        "league":    prospect.draft_league or "unknown",
        "retrieval_method": "vector_similarity",
        "nhl_comps": [
            {
                "nhl_player":  r["ref_name"],
                "similarity":  round(r["score"], 4),
                "profile":     _compact_evidence(r["content"]),
            }
            for r in results
        ],
    })


def _search_prospects(query: str, db: Session) -> str:
    from app.models import Prospect
    from sqlalchemy import or_

    q = query.strip()
    prospects = (
        db.query(Prospect)
        .filter(
            or_(
                Prospect.name.ilike(f"%{q}%"),
                Prospect.draft_league.ilike(f"%{q}%"),
                Prospect.nationality.ilike(f"%{q}%"),
            )
        )
        .order_by(Prospect.css_ranking.nullslast())
        .limit(10)
        .all()
    )

    if not prospects:
        return json.dumps({"error": f"No prospects found matching '{query}'"})

    return json.dumps({
        "query": query,
        "results": [
            {
                "name": p.name,
                "css_rank": p.css_ranking,
                "position": p.position,
                "nationality": p.nationality,
                "league": p.draft_league,
                "ppg": p.points_per_game,
            }
            for p in prospects
        ],
    })
