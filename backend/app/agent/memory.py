"""
Memory-augmented conversation management for the Scout agent.

Problem: long sessions accumulate dozens of turns. Loading all of them into
Claude's context wastes tokens and risks hitting the context window limit.

Solution: when a session exceeds COMPRESS_THRESHOLD messages, compress the
oldest turns into a bullet-point summary (using Claude), replace them in the DB
with a single summary message, and keep only the most recent KEEP_RECENT turns
verbatim. This preserves semantic continuity while keeping token usage constant.

Compression fires at most once per session turn — it's O(1) amortized because
the check is cheap (single COUNT query) and compression only triggers when the
threshold is crossed.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

KEEP_RECENT         = 6    # always keep this many most-recent turns verbatim
COMPRESS_THRESHOLD  = 20   # compress when session exceeds this many total messages
SUMMARY_MAX_TOKENS  = 400  # Claude budget for the compression summary


def maybe_compress_history(db: Session, session_id: str) -> bool:
    """
    Compress old conversation turns into a single summary if the session is long.

    Returns True if compression occurred, False if no action was taken.
    Compression is a no-op if:
      - Total messages ≤ COMPRESS_THRESHOLD
      - ANTHROPIC_API_KEY is not configured
      - Claude call fails (fails open — conversation continues normally)
    """
    from app.agent import store

    total = store.count_messages(db, session_id)
    if total <= COMPRESS_THRESHOLD:
        return False

    from app.config import settings
    if not settings.ANTHROPIC_API_KEY:
        logger.debug("memory.compress_skipped no_api_key session=%s", session_id)
        return False

    # Load all messages with IDs
    all_messages = store.get_messages_with_ids(db, session_id)
    if len(all_messages) <= KEEP_RECENT:
        return False

    # Split: old messages to compress vs recent messages to keep
    to_compress  = all_messages[:-KEEP_RECENT]
    # recent      = all_messages[-KEEP_RECENT:]  # these stay untouched

    # Build conversation text for the summary prompt
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:400]}"   # truncate per turn to save tokens
        for m in to_compress
    )

    summary = _summarize_with_claude(settings.ANTHROPIC_API_KEY, settings.ANTHROPIC_MODEL, convo_text)
    if not summary:
        logger.warning("memory.compress_failed session=%s — keeping full history", session_id)
        return False

    # Delete old messages and replace with summary
    old_ids = [m["id"] for m in to_compress]
    deleted = store.delete_messages_by_ids(db, old_ids)

    summary_content = f"[Conversation summary — {len(to_compress)} earlier turns compressed]\n\n{summary}"
    store.save_message(db, session_id, "assistant", summary_content)

    logger.info(
        "memory.compressed session=%s deleted=%d summary_len=%d",
        session_id, deleted, len(summary_content),
    )
    return True


def get_history_with_summary(db: Session, session_id: str, limit: int = 10) -> list[dict]:
    """
    Return conversation history, compressing first if the session is long.

    Drop-in replacement for store.get_history() in scout.py — same return shape.
    """
    maybe_compress_history(db, session_id)
    from app.agent import store
    return store.get_history(db, session_id, limit=limit)


# ── Private helpers ────────────────────────────────────────────────────────────

def _summarize_with_claude(api_key: str, model: str, convo_text: str) -> str:
    """
    Ask Claude to produce a bullet-point summary of the conversation excerpt.
    Returns empty string on failure.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=SUMMARY_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize the following NHL draft scouting conversation compactly. "
                        "Preserve key facts as bullet points: teams discussed, prospects named "
                        "(with positions/CSS ranks if mentioned), GM tendencies noted, draft needs "
                        "identified, and any conclusions reached. Be concise — each bullet ≤ 15 words.\n\n"
                        f"CONVERSATION:\n{convo_text}"
                    ),
                }
            ],
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text.strip()
    except Exception as exc:
        logger.error("memory.claude_summarize_failed error=%s", exc)
    return ""
