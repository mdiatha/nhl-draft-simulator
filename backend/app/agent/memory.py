"""
Memory-augmented conversation management for the Scout agent.

Strategy: rolling summary updated after every turn.

After each assistant reply, the oldest turns beyond KEEP_RECENT are summarized
into a single persisted summary row. The summary is prepended to every request
so Claude always has session context — not just the last 6 messages.

This replaces the old overflow-only approach (compress at 20 messages) which
meant turns 7-19 were silently dropped from context between compression events.

DB layout:
  scout_conversations rows with role='system' are summary rows.
  There is at most one summary row per session (upserted each turn).
  All other rows are role='user' or role='assistant'.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

KEEP_RECENT        = 6    # verbatim turns always included
SUMMARY_MAX_TOKENS = 300  # Claude budget for rolling summary (kept tight — it's a digest)


def get_history_with_summary(db: Session, session_id: str, limit: int = 10) -> list[dict]:
    """
    Return conversation history for a session.

    Prepends the rolling summary (if one exists) before the recent turns so
    Claude has full session context even after many messages have been rotated out.

    Drop-in replacement for store.get_history() — same return shape.
    """
    from app.agent import store

    summary = _load_summary(db, session_id)
    recent  = store.get_history(db, session_id, limit=KEEP_RECENT)

    if not summary:
        return recent

    # Inject summary as a user/assistant pair at the start of the message list.
    # This pattern is stable across Claude model versions (no system-role in messages[]).
    injected = [
        {"role": "user",      "content": f"[Session summary — earlier conversation]\n{summary}"},
        {"role": "assistant", "content": "Understood — I have the session context from earlier."},
    ]
    return injected + recent


def update_rolling_summary(db: Session, session_id: str) -> None:
    """
    After a turn completes, roll the oldest turns beyond KEEP_RECENT into
    the persisted summary. Called by scout.py after saving each assistant reply.

    No-op when:
      - Total messages ≤ KEEP_RECENT (nothing to summarize yet)
      - ANTHROPIC_API_KEY is not configured
      - Claude summarization call fails (fails open — conversation continues)
    """
    from app.agent import store

    all_messages = store.get_messages_with_ids(db, session_id)
    # Filter out any existing summary rows stored as role='system'
    chat_messages = [m for m in all_messages if m["role"] in ("user", "assistant")]

    if len(chat_messages) <= KEEP_RECENT:
        return

    from app.config import settings
    if not settings.ANTHROPIC_API_KEY:
        return

    # Everything except the most recent KEEP_RECENT turns feeds the summary
    to_summarize = chat_messages[:-KEEP_RECENT]
    existing_summary = _load_summary(db, session_id)

    new_summary = _build_summary(
        settings.ANTHROPIC_API_KEY,
        settings.ANTHROPIC_MODEL,
        to_summarize,
        existing_summary,
    )
    if not new_summary:
        return

    _upsert_summary(db, session_id, new_summary)

    # Delete the turns that are now captured in the summary
    old_ids = [m["id"] for m in to_summarize]
    store.delete_messages_by_ids(db, old_ids)

    logger.info(
        "memory.summary_updated session=%s rolled=%d summary_len=%d",
        session_id, len(to_summarize), len(new_summary),
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_summary(db: Session, session_id: str) -> str:
    """Return the current rolling summary for a session, or empty string."""
    from sqlalchemy import text
    row = db.execute(
        text(
            "SELECT content FROM scout_conversations "
            "WHERE session_id = :sid AND role = 'system' "
            "ORDER BY id DESC LIMIT 1"
        ),
        {"sid": session_id},
    ).fetchone()
    return row.content if row else ""


def _upsert_summary(db: Session, session_id: str, summary: str) -> None:
    """Replace the summary row for this session (delete + insert)."""
    from sqlalchemy import text
    db.execute(
        text("DELETE FROM scout_conversations WHERE session_id = :sid AND role = 'system'"),
        {"sid": session_id},
    )
    db.execute(
        text(
            "INSERT INTO scout_conversations (session_id, role, content) "
            "VALUES (:sid, 'system', :content)"
        ),
        {"sid": session_id, "content": summary},
    )
    db.commit()


def _build_summary(api_key: str, model: str, turns: list[dict], existing: str) -> str:
    """
    Ask Claude to produce a compact rolling summary of the given turns,
    incorporating any existing summary as prior context.
    Returns empty string on failure.
    """
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}"
        for m in turns
    )

    prior_block = (
        f"Prior summary (update this, don't repeat it verbatim):\n{existing}\n\n"
        if existing else ""
    )

    prompt = (
        f"{prior_block}"
        "Summarize the following NHL draft scouting conversation turns into a compact digest. "
        "Preserve: teams and prospects discussed (with positions/CSS ranks), GM tendencies noted, "
        "draft needs identified, conclusions or recommendations reached. "
        "Drop filler. Each bullet ≤ 15 words. Max 10 bullets.\n\n"
        f"TURNS:\n{convo_text}"
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=SUMMARY_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text.strip()
    except Exception as exc:
        logger.error("memory.summarize_failed error=%s", exc)
    return ""
