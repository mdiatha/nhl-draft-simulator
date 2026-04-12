"""
LLM-as-judge evaluation harness for the Scout agent.

Why this matters:
  The Scout's correctness can't be fully validated with unit tests because the
  answers are natural language. An LLM-as-judge approach uses Claude to score
  the Scout's responses against ground-truth facts derived from the live DB.

Architecture:
  1. EVAL_CASES defines factual questions with ground-truth expected_facts
     (key phrases that must appear in a correct answer).
  2. run_eval() calls the Scout synchronously for each case.
  3. An LLM judge (Claude) scores each response 0–10 and returns reasoning.
  4. Results are written to eval_results.jsonl for tracking over time.

Running:
  pytest tests/evals/test_scout_evals.py -v -s
  # or as a standalone script:
  python tests/evals/test_scout_evals.py

Environment: requires ANTHROPIC_API_KEY and a live DB with data.
Skip automatically when ANTHROPIC_API_KEY is not set (CI without keys).

Regression detection:
  The test FAILS if average judge score < PASS_THRESHOLD (default: 6.0 / 10).
  Run after any change to the system prompt, tool definitions, or RAG index.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 6.0   # average judge score below this → test fails
EVAL_RESULTS_PATH = Path(__file__).parent / "eval_results.jsonl"
JUDGE_MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for judging


# ── Eval cases ────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    """A single evaluation question with expected factual content."""
    id: str
    question: str
    expected_facts: list[str]   # key phrases a correct answer should mention
    tool_should_be_called: Optional[str] = None   # tool name, if verifiable


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="gm_archetype_leafs",
        question="What is the Toronto Maple Leafs GM's drafting archetype?",
        expected_facts=["archetype", "BPA", "need-based", "safe", "boom-bust", "system-fit"],
        tool_should_be_called="get_gm_profile",
    ),
    EvalCase(
        id="top_centers",
        question="Who are the top 5 center prospects in the 2025 draft class?",
        expected_facts=["C", "CSS", "rank"],
        tool_should_be_called="get_top_prospects",
    ),
    EvalCase(
        id="team_needs_oilers",
        question="What positions does the Edmonton Oilers need most in the draft?",
        expected_facts=["position", "need", "Oilers"],
        tool_should_be_called="get_team_needs",
    ),
    EvalCase(
        id="semantic_search_makar",
        question="Find me a prospect who plays like Cale Makar — offensive defenseman with high skating ability.",
        expected_facts=["D", "defenseman", "prospect"],
        tool_should_be_called="semantic_prospect_search",
    ),
    EvalCase(
        id="draft_history_detroit",
        question="What did Detroit Red Wings draft in 2023?",
        expected_facts=["2023", "Detroit", "pick"],
        tool_should_be_called="get_draft_history",
    ),
    EvalCase(
        id="offtopic_rejection",
        question="What is the capital of France?",
        expected_facts=["NHL", "draft", "hockey", "specialized"],
        # Should be rejected by the topic guardrail — no tool call expected
        tool_should_be_called=None,
    ),
    EvalCase(
        id="ppg_stat_mention",
        question="Which Swedish prospects have the highest points per game this season?",
        expected_facts=["SWE", "ppg", "Sweden", "prospect"],
        tool_should_be_called="search_prospects",
    ),
    EvalCase(
        id="gm_nationality_bias",
        question="Does the Pittsburgh Penguins GM show a preference for Canadian players?",
        expected_facts=["CAN", "nationality", "weight", "Pittsburgh"],
        tool_should_be_called="get_gm_profile",
    ),
]


# ── Judge ─────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    case_id: str
    question: str
    scout_reply: str
    judge_score: float       # 0–10
    judge_reasoning: str
    expected_facts: list[str]
    facts_found: list[str]   # which expected_facts appeared in reply
    passed: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


async def judge_response(
    client,
    question: str,
    reply: str,
    expected_facts: list[str],
) -> tuple[float, str]:
    """
    Ask Claude to score the Scout's response 0–10.

    Scoring rubric:
      9–10: Correct, cites specific data (numbers, names, ranks)
      7–8:  Mostly correct, minor omissions
      5–6:  Partially correct, missing key facts
      3–4:  Vague or generic, not grounded in data
      0–2:  Wrong, off-topic, or refused without reason
    """
    judge_prompt = f"""You are evaluating an NHL draft analyst chatbot's response.

QUESTION: {question}

CHATBOT RESPONSE:
{reply}

EXPECTED FACTS (one or more of these should appear):
{json.dumps(expected_facts, indent=2)}

Score the response 0–10 based on:
1. Factual accuracy and relevance to the question (0–4 pts)
2. Specificity — cites concrete data like CSS ranks, PPG, position weights (0–3 pts)
3. Completeness — addresses all parts of the question (0–2 pts)
4. Appropriate handling of off-topic queries — should redirect, not hallucinate (0–1 pt)

Reply with ONLY valid JSON in this exact format:
{{"score": <number 0-10>, "reasoning": "<one sentence explanation>"}}"""

    try:
        resp = await client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        text = resp.content[0].text.strip()
        # Extract JSON even if model adds preamble
        start = text.find("{")
        end   = text.rfind("}") + 1
        parsed = json.loads(text[start:end])
        return float(parsed["score"]), parsed.get("reasoning", "")
    except Exception as exc:
        logger.warning("judge.parse_failed error=%s raw=%s", exc, text[:200] if "text" in locals() else "")
        return 0.0, f"Judge parse error: {exc}"


async def run_eval(db, client) -> list[EvalResult]:
    """Run all eval cases and return results."""
    from app.agent.scout import chat

    results: list[EvalResult] = []

    for case in EVAL_CASES:
        session_id = f"eval-{case.id}"
        scout_reply = ""
        error = None

        try:
            scout_reply = await chat(db, session_id, case.question)
        except Exception as exc:
            error = str(exc)
            scout_reply = f"[Error: {exc}]"

        # Check which expected facts appear in the reply (case-insensitive)
        reply_lower = scout_reply.lower()
        facts_found = [f for f in case.expected_facts if f.lower() in reply_lower]

        judge_score, judge_reasoning = await judge_response(
            client, case.question, scout_reply, case.expected_facts
        )

        result = EvalResult(
            case_id=case.id,
            question=case.question,
            scout_reply=scout_reply[:500],   # truncate for storage
            judge_score=judge_score,
            judge_reasoning=judge_reasoning,
            expected_facts=case.expected_facts,
            facts_found=facts_found,
            passed=judge_score >= PASS_THRESHOLD,
            error=error,
        )
        results.append(result)
        logger.info(
            "eval.case_done id=%s score=%.1f passed=%s",
            case.id, judge_score, result.passed,
        )

    return results


def _write_results(results: list[EvalResult]) -> None:
    """Append results to JSONL file for longitudinal tracking."""
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "a") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def _print_summary(results: list[EvalResult]) -> None:
    avg_score = sum(r.judge_score for r in results) / len(results) if results else 0
    passed    = sum(1 for r in results if r.passed)
    print(f"\n{'='*60}")
    print(f"Scout Eval Summary — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")
    print(f"Cases: {len(results)} | Passed: {passed} | Avg score: {avg_score:.1f}/10")
    print(f"Threshold: {PASS_THRESHOLD}/10")
    print()
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        facts_pct = len(r.facts_found) / len(r.expected_facts) * 100 if r.expected_facts else 0
        print(f"  [{status}] {r.case_id:<35} score={r.judge_score:.1f} facts={facts_pct:.0f}%")
        if not r.passed:
            print(f"         → {r.judge_reasoning}")
    print(f"{'='*60}\n")


# ── pytest entry point ────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping Scout evals",
)
def test_scout_evals():
    """
    Run all Scout eval cases and assert average judge score >= PASS_THRESHOLD.

    This test is the regression gate for LLM quality. Run it after:
      - Changing the system prompt
      - Adding/removing/modifying tool definitions
      - Rebuilding the RAG index
      - Upgrading the Claude model version
    """
    import anthropic
    from app.database import SessionLocal
    from app.config import settings

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    db = SessionLocal()

    try:
        results = asyncio.run(run_eval(db, client))
    finally:
        db.close()

    _write_results(results)
    _print_summary(results)

    avg_score = sum(r.judge_score for r in results) / len(results) if results else 0
    failed_cases = [r for r in results if not r.passed]

    if failed_cases:
        failure_details = "\n".join(
            f"  {r.case_id}: score={r.judge_score:.1f} — {r.judge_reasoning}"
            for r in failed_cases
        )
        pytest.fail(
            f"Scout eval failed: avg_score={avg_score:.1f} < {PASS_THRESHOLD}\n"
            f"Failed cases:\n{failure_details}"
        )


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import anthropic
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

    from app.database import SessionLocal
    from app.config import settings

    if not settings.ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    db = SessionLocal()

    try:
        results = asyncio.run(run_eval(db, client))
    finally:
        db.close()

    _write_results(results)
    _print_summary(results)

    avg = sum(r.judge_score for r in results) / len(results) if results else 0
    sys.exit(0 if avg >= PASS_THRESHOLD else 1)
