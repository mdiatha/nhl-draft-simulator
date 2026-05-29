"""
NHL Draft Ingestion Lambda — EventBridge trigger.

Invoked daily at 06:00 UTC by EventBridge.
Posts to FastAPI /api/admin/ingest (via CloudFront), which runs the full NHL
data ingestion pipeline where the DB connection and app state live.

No dependencies beyond stdlib — packages to a tiny zip with no pip install.

Environment variables:
  API_BASE_URL        — FastAPI base URL (e.g. https://d1234.cloudfront.net)
  ADMIN_API_KEY_PARAM — SSM parameter name for the X-Admin-Key secret
  LOG_LEVEL           — default INFO
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

_admin_key_cache: str | None = None


def _get_admin_key() -> str:
    """Fetch the admin API key from SSM, cached for the lifetime of the container."""
    global _admin_key_cache
    if _admin_key_cache is not None:
        return _admin_key_cache
    param_name = os.environ["ADMIN_API_KEY_PARAM"]
    region = os.environ.get("AWS_REGION", "us-east-1")
    req = urllib.request.Request(
        f"https://ssm.{region}.amazonaws.com/",
        method="POST",
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": "AmazonSSM.GetParameter",
        },
        data=json.dumps({"Name": param_name, "WithDecryption": True}).encode(),
    )
    # Use boto3-style signing via the Lambda execution role credentials.
    # urllib alone can't sign AWS requests — use the SSM SDK via the
    # Lambda runtime's bundled boto3 (available in all Python Lambda runtimes).
    import boto3  # noqa: PLC0415 — boto3 is always present in Lambda runtime
    ssm = boto3.client("ssm", region_name=region)
    value = ssm.get_parameter(Name=param_name, WithDecryption=True)["Parameter"]["Value"]
    _admin_key_cache = value
    return value


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Trigger the FastAPI ingestion endpoint for the live backend service.

    Accepts any event shape:
      - EventBridge scheduled event  ({"triggered_by": "cron", ...})
      - Direct Lambda invocation      ({})
    """
    api_url = os.environ["API_BASE_URL"].rstrip("/")
    admin_key = _get_admin_key()

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
