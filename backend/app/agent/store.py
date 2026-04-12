"""
Database helpers for scout_embeddings and scout_conversations.

Keeps all raw SQL in one place so the rest of the agent code stays clean.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Embeddings ────────────────────────────────────────────────────────────────

def upsert_embedding(
    db: Session,
    *,
    doc_type: str,
    ref_id: Optional[int],
    ref_name: Optional[str],
    content: str,
    embedding: Optional[list[float]],
) -> None:
    """Insert or replace a document in scout_embeddings."""
    embedding_json = json.dumps(embedding) if embedding else None

    # Delete existing entry for this doc_type + ref_id before reinserting
    db.execute(
        text("DELETE FROM scout_embeddings WHERE doc_type = :dt AND ref_id = :rid"),
        {"dt": doc_type, "rid": ref_id},
    )
    db.execute(
        text(
            """
            INSERT INTO scout_embeddings (doc_type, ref_id, ref_name, content, embedding)
            VALUES (:dt, :rid, :rname, :content, :emb)
            """
        ),
        {"dt": doc_type, "rid": ref_id, "rname": ref_name, "content": content, "emb": embedding_json},
    )


def retrieve_similar(
    db: Session,
    query_embedding: list[float],
    *,
    doc_types: Optional[list[str]] = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Return the top-k most similar documents to *query_embedding*.

    Primary path: native pgvector cosine ANN query (O(log N) via IVFFlat index).
    Fallback: Python-side cosine similarity for environments without the native
    vector type (e.g. SQLite in unit tests or pre-migration state).
    """
    # Try native pgvector ANN first
    try:
        return _retrieve_similar_pgvector(db, query_embedding, doc_types=doc_types, top_k=top_k)
    except Exception as pgvec_exc:
        logger.debug("pgvector.ann_failed falling_back_to_python reason=%s", pgvec_exc)

    # Python-side fallback (loads all rows — only used if pgvector unavailable)
    rows = db.execute(
        text("SELECT id, doc_type, ref_id, ref_name, content, embedding FROM scout_embeddings")
    ).fetchall()

    if not rows:
        return []

    from app.agent.embeddings import cosine_similarity

    scored = []
    for row in rows:
        if doc_types and row.doc_type not in doc_types:
            continue
        if not row.embedding:
            continue
        try:
            stored_vec = json.loads(row.embedding) if isinstance(row.embedding, str) else list(row.embedding)
        except Exception:
            continue
        sim = cosine_similarity(query_embedding, stored_vec)
        scored.append(
            {
                "doc_type": row.doc_type,
                "ref_id": row.ref_id,
                "ref_name": row.ref_name,
                "content": row.content,
                "score": sim,
            }
        )

    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def _retrieve_similar_pgvector(
    db: Session,
    query_embedding: list[float],
    *,
    doc_types: Optional[list[str]] = None,
    top_k: int = 5,
) -> list[dict]:
    """Native pgvector cosine ANN — requires vector(1024) column + IVFFlat index."""
    # Format vector literal for Postgres: '[0.1, 0.2, ...]'
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    if doc_types:
        # Parameterised IN clause
        placeholders = ", ".join(f":dt{i}" for i in range(len(doc_types)))
        params = {"vec": vec_str, "k": top_k}
        params.update({f"dt{i}": dt for i, dt in enumerate(doc_types)})
        sql = text(f"""
            SELECT doc_type, ref_id, ref_name, content,
                   1 - (embedding <=> :vec::vector) AS score
            FROM scout_embeddings
            WHERE embedding IS NOT NULL
              AND doc_type IN ({placeholders})
            ORDER BY embedding <=> :vec::vector
            LIMIT :k
        """)
    else:
        params = {"vec": vec_str, "k": top_k}
        sql = text("""
            SELECT doc_type, ref_id, ref_name, content,
                   1 - (embedding <=> :vec::vector) AS score
            FROM scout_embeddings
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> :vec::vector
            LIMIT :k
        """)

    rows = db.execute(sql, params).fetchall()
    return [
        {
            "doc_type": r.doc_type,
            "ref_id": r.ref_id,
            "ref_name": r.ref_name,
            "content": r.content,
            "score": float(r.score),
        }
        for r in rows
    ]


def retrieve_keyword(db: Session, query: str, top_k: int = 5) -> list[dict]:
    """Simple ILIKE keyword fallback when embeddings are unavailable."""
    rows = db.execute(
        text(
            "SELECT doc_type, ref_id, ref_name, content FROM scout_embeddings "
            "WHERE content ILIKE :q ORDER BY doc_type LIMIT :k"
        ),
        {"q": f"%{query}%", "k": top_k},
    ).fetchall()
    return [
        {
            "doc_type": r.doc_type,
            "ref_id": r.ref_id,
            "ref_name": r.ref_name,
            "content": r.content,
            "score": 1.0,
        }
        for r in rows
    ]


def retrieve_bm25(
    db: Session,
    query: str,
    *,
    doc_types: Optional[list[str]] = None,
    top_k: int = 10,
) -> list[dict]:
    """
    BM25 full-text search via PostgreSQL tsvector / ts_rank.

    Uses Postgres built-in full-text search (GIN index on to_tsvector).
    Complements vector ANN for exact name lookups and keyword-heavy queries
    where semantic similarity underperforms.

    Requires the GIN index from migration 014:
      CREATE INDEX idx_scout_embeddings_fts ON scout_embeddings
        USING GIN (to_tsvector('english', content));
    """
    # Sanitise query: remove characters that confuse to_tsquery
    safe_q = " & ".join(
        w for w in query.split() if w.isalnum() or w.replace("-", "").isalnum()
    )
    if not safe_q:
        return []

    if doc_types:
        placeholders = ", ".join(f":dt{i}" for i in range(len(doc_types)))
        params: dict = {"q": safe_q, "k": top_k}
        params.update({f"dt{i}": dt for i, dt in enumerate(doc_types)})
        sql = text(f"""
            SELECT doc_type, ref_id, ref_name, content,
                   ts_rank(to_tsvector('english', content), to_tsquery('english', :q)) AS score
            FROM scout_embeddings
            WHERE to_tsvector('english', content) @@ to_tsquery('english', :q)
              AND doc_type IN ({placeholders})
            ORDER BY score DESC
            LIMIT :k
        """)
    else:
        params = {"q": safe_q, "k": top_k}
        sql = text("""
            SELECT doc_type, ref_id, ref_name, content,
                   ts_rank(to_tsvector('english', content), to_tsquery('english', :q)) AS score
            FROM scout_embeddings
            WHERE to_tsvector('english', content) @@ to_tsquery('english', :q)
            ORDER BY score DESC
            LIMIT :k
        """)

    try:
        rows = db.execute(sql, params).fetchall()
        return [
            {
                "doc_type": r.doc_type,
                "ref_id": r.ref_id,
                "ref_name": r.ref_name,
                "content": r.content,
                "score": float(r.score),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.debug("bm25.failed falling_back reason=%s", exc)
        return []


def retrieve_hybrid(
    db: Session,
    query_embedding: list[float],
    query_text: str,
    *,
    doc_types: Optional[list[str]] = None,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Hybrid search: Reciprocal Rank Fusion (RRF) of BM25 + vector ANN results.

    RRF formula: score(d) = Σ 1 / (k + rank(d))
    where k=60 is a constant that dampens the impact of very high ranks.

    Why RRF instead of score normalisation:
      - BM25 and cosine similarity scores are on incomparable scales
      - Score normalisation requires knowing the min/max across both result sets
      - RRF is rank-based so it's robust to score distribution differences
      - Empirically outperforms score fusion for heterogeneous retrievers

    Concrete benefit: "find a player like Connor Bedard" succeeds on the vector
    side; "find prospects named Bedard" succeeds on the BM25 side; RRF fuses both
    so either query returns the right result.
    """
    # Fetch more candidates than top_k from each retriever to give RRF room to re-rank
    fetch_k = max(top_k * 3, 15)

    vec_results = _retrieve_similar_pgvector(db, query_embedding, doc_types=doc_types, top_k=fetch_k)
    bm25_results = retrieve_bm25(db, query_text, doc_types=doc_types, top_k=fetch_k)

    # Build rank maps keyed by entity identity (ref_id when available) so
    # multiple chunks for the same prospect can still collapse into one hit.
    def _identity_key(result: dict) -> tuple[str, str]:
        if result.get("ref_id") is not None:
            return ("id", str(result["ref_id"]))
        if result.get("ref_name"):
            return ("name", str(result["ref_name"]))
        return ("content", str(result.get("content", ""))[:80])

    def _rank_map(results: list[dict]) -> dict[tuple[str, str], int]:
        ranks: dict[tuple[str, str], int] = {}
        for i, result in enumerate(results, start=1):
            key = _identity_key(result)
            if key not in ranks:
                ranks[key] = i
        return ranks

    vec_rank = _rank_map(vec_results)
    bm25_rank = _rank_map(bm25_results)

    # Collect all unique candidates
    all_ids: set[tuple[str, str]] = set(vec_rank) | set(bm25_rank)
    # Build a lookup for content / doc_type (prefer best-ranked result for metadata)
    meta: dict[tuple[str, str], dict] = {}
    for r in vec_results + bm25_results:
        key = _identity_key(r)
        if key not in meta:
            meta[key] = {
                "doc_type": r["doc_type"],
                "ref_id": r.get("ref_id"),
                "ref_name": r.get("ref_name"),
                "content": r["content"],
            }

    # Compute RRF scores
    fused: list[tuple[float, tuple[str, str]]] = []
    for key in all_ids:
        rrf_score = 0.0
        if key in vec_rank:
            rrf_score += 1.0 / (rrf_k + vec_rank[key])
        if key in bm25_rank:
            rrf_score += 1.0 / (rrf_k + bm25_rank[key])
        fused.append((rrf_score, key))

    fused.sort(key=lambda x: -x[0])

    return [
        {
            "doc_type": meta[key]["doc_type"],
            "ref_id": meta[key]["ref_id"],
            "ref_name": meta[key]["ref_name"],
            "content": meta[key]["content"],
            "score": score,
        }
        for score, key in fused[:top_k]
        if key in meta
    ]


# ── Conversation memory ───────────────────────────────────────────────────────

def save_message(db: Session, session_id: str, role: str, content: str) -> None:
    db.execute(
        text(
            "INSERT INTO scout_conversations (session_id, role, content) VALUES (:sid, :role, :content)"
        ),
        {"sid": session_id, "role": role, "content": content},
    )
    db.commit()


def get_history(db: Session, session_id: str, limit: int = 10) -> list[dict]:
    """Return the last *limit* messages for a session, oldest first."""
    rows = db.execute(
        text(
            "SELECT role, content FROM scout_conversations "
            "WHERE session_id = :sid ORDER BY id DESC LIMIT :lim"
        ),
        {"sid": session_id, "lim": limit},
    ).fetchall()
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


def count_messages(db: Session, session_id: str) -> int:
    """Return the total number of stored messages for a session."""
    row = db.execute(
        text("SELECT COUNT(*) FROM scout_conversations WHERE session_id = :sid"),
        {"sid": session_id},
    ).fetchone()
    return int(row[0]) if row else 0


def get_messages_with_ids(db: Session, session_id: str) -> list[dict]:
    """Return all messages for a session ordered oldest-first, including row ids."""
    rows = db.execute(
        text("SELECT id, role, content FROM scout_conversations WHERE session_id = :sid ORDER BY id ASC"),
        {"sid": session_id},
    ).fetchall()
    return [{"id": r.id, "role": r.role, "content": r.content} for r in rows]


def delete_messages_by_ids(db: Session, message_ids: list[int]) -> int:
    """Delete specific messages by row id. Returns number of rows deleted."""
    if not message_ids:
        return 0
    placeholders = ", ".join(f":id{i}" for i in range(len(message_ids)))
    params = {f"id{i}": mid for i, mid in enumerate(message_ids)}
    result = db.execute(
        text(f"DELETE FROM scout_conversations WHERE id IN ({placeholders})"),
        params,
    )
    db.commit()
    return result.rowcount
