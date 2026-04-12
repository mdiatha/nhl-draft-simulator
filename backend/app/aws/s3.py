"""
S3 model artifact storage.

Model versioning strategy:
  s3://{bucket}/models/latest/model.pkl        — always the current production model
  s3://{bucket}/models/latest/model_meta.json  — metadata for the current model
  s3://{bucket}/models/{trained_at}/model.pkl  — immutable versioned copy
  s3://{bucket}/models/{trained_at}/model_meta.json

This means:
  - Any service instance loads the same model on startup (no local-disk dependency)
  - Rollback = copy a versioned object back to latest/
  - Full audit trail of every trained model

All functions are no-ops (log a warning and return False/None) when
AWS_S3_BUCKET is not configured, so local development works without AWS creds.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _client():
    """Return a boto3 S3 client. Lazily imported so boto3 is optional locally."""
    import boto3
    from app.config import settings
    return boto3.client("s3", region_name=settings.AWS_REGION)


def _is_configured() -> bool:
    from app.config import settings
    return bool(settings.AWS_S3_BUCKET)


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_model(local_path: Path, trained_at: str) -> bool:
    """
    Upload model.pkl and model_meta.json to S3 after training.

    Writes to two locations:
      - models/latest/  (always current)
      - models/{trained_at}/  (immutable version for rollback)

    Returns True on success, False if S3 is not configured or upload fails.
    """
    if not _is_configured():
        logger.debug("AWS_S3_BUCKET not set — skipping S3 model upload")
        return False

    from app.config import settings
    bucket = settings.AWS_S3_BUCKET
    meta_path = local_path.parent / "model_meta.json"

    files_to_upload = [
        (local_path,  "model.pkl"),
        (meta_path,   "model_meta.json"),
    ]

    try:
        s3 = _client()
        for file_path, filename in files_to_upload:
            if not file_path.exists():
                continue
            for prefix in ("latest", trained_at):
                key = f"models/{prefix}/{filename}"
                s3.upload_file(str(file_path), bucket, key)
                logger.info(
                    "s3.upload",
                    extra={"bucket": bucket, "key": key, "size_bytes": file_path.stat().st_size},
                )
        return True
    except Exception as exc:
        logger.error("s3.upload_failed", extra={"error": str(exc)})
        return False


# ── Download ──────────────────────────────────────────────────────────────────

def download_model(local_path: Path) -> bool:
    """
    Download model.pkl and model_meta.json from s3://{bucket}/models/latest/.

    Called at startup if local model.pkl is missing (e.g. fresh task, new deploy).
    Returns True if both files downloaded successfully, False otherwise.
    """
    if not _is_configured():
        logger.debug("AWS_S3_BUCKET not set — skipping S3 model download")
        return False

    from app.config import settings
    bucket = settings.AWS_S3_BUCKET
    meta_path = local_path.parent / "model_meta.json"

    downloads = [
        ("models/latest/model.pkl",       local_path),
        ("models/latest/model_meta.json", meta_path),
    ]

    try:
        s3 = _client()
        for key, dest in downloads:
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            logger.info("s3.download", extra={"bucket": bucket, "key": key})
        return True
    except s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
        logger.info("s3.no_model_found", extra={"bucket": bucket, "key": "models/latest/model.pkl"})
        return False
    except Exception as exc:
        logger.error("s3.download_failed", extra={"error": str(exc)})
        return False


# ── List versions ─────────────────────────────────────────────────────────────

def list_model_versions() -> list[dict]:
    """
    List all versioned models in S3, newest first.
    Each entry: {"version": "2026-03-29T...", "size_bytes": int, "last_modified": str}
    """
    if not _is_configured():
        return []

    from app.config import settings
    bucket = settings.AWS_S3_BUCKET

    try:
        s3 = _client()
        response = s3.list_objects_v2(Bucket=bucket, Prefix="models/", Delimiter="/")
        prefixes = [p["Prefix"] for p in response.get("CommonPrefixes", [])]

        versions = []
        for prefix in prefixes:
            version = prefix.rstrip("/").split("/")[-1]
            if version == "latest":
                continue
            # Fetch the meta file to get size
            try:
                meta_obj = s3.get_object(Bucket=bucket, Key=f"{prefix}model_meta.json")
                meta = json.loads(meta_obj["Body"].read())
                versions.append({
                    "version":      version,
                    "trained_at":   meta.get("trained_at", version),
                    "validation_auc": meta.get("validation_auc"),
                    "mode":         meta.get("mode", "evaluation"),
                    "last_modified": str(meta_obj["LastModified"]),
                })
            except Exception:
                versions.append({"version": version})

        return sorted(versions, key=lambda v: v.get("trained_at", ""), reverse=True)
    except Exception as exc:
        logger.error("s3.list_versions_failed", extra={"error": str(exc)})
        return []


# ── Rollback ──────────────────────────────────────────────────────────────────

def rollback_model(version: str) -> bool:
    """
    Copy a versioned model back to models/latest/, then signal the registry to reload.

    version: the trained_at timestamp string (e.g. "2026-03-28T12:00:00+00:00")
    """
    if not _is_configured():
        return False

    from app.config import settings
    bucket = settings.AWS_S3_BUCKET

    try:
        s3 = _client()
        for filename in ("model.pkl", "model_meta.json"):
            src_key = f"models/{version}/{filename}"
            dst_key = f"models/latest/{filename}"
            s3.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": src_key},
                Key=dst_key,
            )
            logger.info("s3.rollback", extra={"from": src_key, "to": dst_key})
        return True
    except Exception as exc:
        logger.error("s3.rollback_failed", extra={"error": str(exc), "version": version})
        return False
