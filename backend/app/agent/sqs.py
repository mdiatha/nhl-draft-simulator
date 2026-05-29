"""
SQS integration for async RAG index rebuild.

Why:
  POST /api/agent/index previously called build_index() synchronously — embedding
  200+ prospect documents via Voyage AI takes several seconds and blocks the
  request. A slow admin endpoint that holds a DB connection for 10+ seconds is
  a reliability hazard.

Architecture:
  API → SQS → Lambda → build_index()
  ↓
  Returns 202 immediately

The SQS message body is a simple JSON object. The consumer (lambda/sqs_handler.py)
deserializes it and calls build_index() with a fresh DB session.

Lambda consumer setup (in infrastructure/lambda/):
  - Trigger: SQS queue (SQS_INDEX_QUEUE_URL)
  - Batch size: 1 (index builds are not idempotent-combinable)
  - Visibility timeout: 120 s (build takes ~10–30 s)
  - Dead-letter queue: after 3 failed attempts, message goes to DLQ + SNS alert
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def enqueue_index_rebuild(queue_url: str) -> str:
    """
    Send an index rebuild job to SQS.

    Returns the SQS MessageId on success.
    Raises on boto3 / network errors (caller handles fallback).
    """
    import boto3
    from app.config import settings

    sqs = boto3.client("sqs", region_name=settings.AWS_REGION)

    message = {
        "job": "rebuild_rag_index",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4()),
    }

    response = sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message),
        MessageGroupId="rag-index",           # for FIFO queues — deduplicate concurrent triggers
        MessageDeduplicationId=message["request_id"],
    )
    msg_id = response["MessageId"]
    logger.info("sqs.index_rebuild_enqueued message_id=%s queue=%s", msg_id, queue_url)
    return msg_id


def handle_sqs_event(event: dict, db_factory) -> dict:
    """
    Lambda handler entry point for SQS-triggered index rebuild.

    event: AWS Lambda SQS event dict (Records list)
    db_factory: callable that returns a SQLAlchemy Session

    Usage in lambda/sqs_handler.py:
        from app.agent.sqs import handle_sqs_event
        from app.database import SessionLocal

        def handler(event, context):
            return handle_sqs_event(event, SessionLocal)
    """
    from app.agent.embeddings import build_index

    results = []
    for record in event.get("Records", []):
        body = json.loads(record["body"])
        job = body.get("job")

        if job != "rebuild_rag_index":
            logger.warning("sqs.unknown_job job=%s", job)
            results.append({"status": "skipped", "job": job})
            continue

        logger.info("sqs.index_rebuild_start request_id=%s", body.get("request_id"))
        db = db_factory()
        try:
            count = build_index(db)
            logger.info("sqs.index_rebuild_complete docs=%d", count)
            results.append({"status": "ok", "docs_indexed": count})
        except Exception as exc:
            logger.error("sqs.index_rebuild_failed error=%s", exc)
            raise  # SQS will retry; after max attempts → DLQ
        finally:
            db.close()

    return {"results": results}
