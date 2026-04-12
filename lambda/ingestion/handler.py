"""
NHL Draft Ingestion Lambda — EventBridge trigger.

Invoked daily at 06:00 UTC by EventBridge.
Posts to FastAPI /api/admin/ingest behind API Gateway, which runs the full NHL
data ingestion pipeline where the DB connection and app state live.

No dependencies beyond stdlib — packages to a tiny zip with no pip install.

Environment variables (set by Terraform):
  API_BASE_URL  — FastAPI base URL (e.g. http://api.example.com)
  ADMIN_API_KEY — X-Admin-Key header value for protected endpoints
  LOG_LEVEL     — default INFO
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Trigger the FastAPI ingestion endpoint for the live backend service.

    Accepts any event shape:
      - EventBridge scheduled event  ({"triggered_by": "cron", ...})
      - Direct Lambda invocation      ({})
    """
    api_url = os.environ["API_BASE_URL"].rstrip("/")
    admin_key = os.environ["ADMIN_API_KEY"]

    triggered_by = event.get("triggered_by", "lambda")
    logger.info(json.dumps({"msg": "ingestion.trigger_start", "triggered_by": triggered_by}))

    req = urllib.request.Request(
        f"{api_url}/api/admin/ingest",
        method="POST",
        headers={
            "X-Admin-Key": admin_key,
            "Content-Type": "application/json",
        },
        data=json.dumps({"triggered_by": triggered_by}).encode(),
    )

    try:
        with urllib.request.urlopen(req, timeout=55) as response:
            body = json.loads(response.read())
            logger.info(json.dumps({"msg": "ingestion.trigger_success", "response": body}))
            return {"statusCode": 200, "body": json.dumps(body)}

    except urllib.error.HTTPError as exc:
        error = {
            "msg": "ingestion.trigger_http_error",
            "code": exc.code,
            "reason": exc.reason,
        }
        logger.error(json.dumps(error))
        raise  # re-raise so Lambda marks as error and DLQ receives it

    except Exception as exc:
        logger.error(json.dumps({"msg": "ingestion.trigger_failed", "error": str(exc)}))
        raise
