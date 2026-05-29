"""
AWS Secrets Manager integration.

Fetches application secrets (DATABASE_URL, SECRET_KEY) from
Secrets Manager at startup and injects them into the process environment
before Pydantic Settings reads them.

Why Secrets Manager over in-cluster secrets:
  - Secrets are never stored in git or in a cluster control plane by default
  - Rotation: Secrets Manager rotates DB passwords automatically; the app picks up
    new values on the next service restart
  - Audit trail: every secret access is logged in CloudTrail

Usage:
  Set AWS_SECRETS_NAME=nhl-draft/production in your environment.
  The secret in Secrets Manager should be a JSON object:
    {
      "DATABASE_URL": "postgresql+psycopg2://user:pass@host:5432/db",
      "SECRET_KEY": "..."
    }

  Call load_secrets_into_env() BEFORE instantiating Settings() so that
  pydantic-settings picks up the values via os.environ.

Authentication:
  In AWS, attach a role with secretsmanager:GetSecretValue to the runtime
  (for example an ECS task role or EC2 instance profile).
  Locally, use AWS_PROFILE or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def load_secrets_into_env(secret_name: str | None = None) -> bool:
    """
    Fetch JSON secret from Secrets Manager and write each key into os.environ.

    secret_name defaults to the AWS_SECRETS_NAME environment variable.
    Returns True on success, False if not configured or fetch fails.

    Call this once at process start, before settings = Settings().
    """
    name = secret_name or os.environ.get("AWS_SECRETS_NAME")
    if not name:
        logger.debug("AWS_SECRETS_NAME not set — using environment variables directly")
        return False

    region = os.environ.get("AWS_REGION", "us-east-1")

    try:
        import boto3
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=name)
        secret_str = response.get("SecretString") or ""
        secret_dict = json.loads(secret_str)

        injected = []
        for key, value in secret_dict.items():
            # Only inject if not already set — lets env vars override for local dev
            if key not in os.environ:
                os.environ[key] = str(value)
                injected.append(key)

        logger.info(
            "secrets_manager.loaded",
            extra={"secret_name": name, "keys_injected": injected},
        )
        return True

    except ImportError:
        logger.warning("boto3 not installed — cannot load from Secrets Manager")
        return False
    except Exception as exc:
        logger.error(
            "secrets_manager.load_failed",
            extra={"secret_name": name, "error": str(exc)},
        )
        return False


def get_secret(secret_name: str) -> dict:
    """
    Fetch and return a single Secrets Manager secret as a dict.
    Useful for one-off lookups (e.g., NHL API keys, webhook tokens).
    Raises RuntimeError if the fetch fails.
    """
    region = os.environ.get("AWS_REGION", "us-east-1")
    try:
        import boto3
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response.get("SecretString") or "{}")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch secret '{secret_name}': {exc}") from exc
