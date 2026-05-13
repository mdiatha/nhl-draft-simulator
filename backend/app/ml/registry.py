"""Model registry — loads and holds the trained XGBoost model in memory.

Instead of loading model.pkl from disk on every prediction call, the registry
loads once at startup and caches the model as a module-level singleton.

Hot-swap: call registry.reload() (or POST /api/ml/reload) to swap in a newly
trained model without restarting the server.

S3 integration:
  On load(), if model.pkl is missing locally, the registry tries to download it
  from s3://{AWS_S3_BUCKET}/models/latest/model.pkl before giving up.
  On save_meta(), the new model and metadata are uploaded to S3 (versioned +
  latest/) so all service instances stay in sync across restarts.

Metadata (AUC, feature importances, trained_at) is persisted alongside the
model in model_meta.json so it can be inspected without reloading the model.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Write to /tmp (or MODEL_DIR env var) so the path is writable inside a
# read-only container filesystem.  The directory is created on first use.
# /tmp is ephemeral — models are always downloaded from S3 on container start.
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/nhl_draft_model"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"
META_PATH  = MODEL_DIR / "model_meta.json"


class ModelRegistry:
    """
    Thread-safe singleton that owns the in-memory XGBoost model (+ ensemble artifacts).

    Usage:
        registry = ModelRegistry()
        registry.load()                  # called at app startup
        model = registry.model           # None if not trained
        registry.reload()                # hot-swap after new training
    """

    def __init__(self) -> None:
        self._model = None
        self._meta: dict = {}
        self._lock = threading.RLock()
        self._training_lock = threading.Lock()   # prevents concurrent training runs
        self._calibration: dict | None = None    # conformal prediction calibration data
        self._model_hash_cache: str = ""         # cached MD5 of model.pkl bytes; reset on reload

    @property
    def model(self):
        return self._model

    @property
    def meta(self) -> dict:
        return self._meta.copy()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def calibration(self) -> dict | None:
        return self._calibration

    @property
    def training_lock(self) -> threading.Lock:
        return self._training_lock

    def load(self) -> bool:
        """
        Load model from disk into memory.

        If model.pkl is absent locally, attempts to download from
        s3://{AWS_S3_BUCKET}/models/latest/ first (transparent on cold starts
        and new service instances). Returns True if the model ends up in memory.
        """
        if not MODEL_PATH.exists():
            logger.info("model.pkl not found locally — attempting S3 download")
            from app.aws.s3 import download_model
            downloaded = download_model(MODEL_PATH)
            if not downloaded:
                logger.info(
                    "No model available (local or S3). Train via POST /api/ml/train."
                )
                return False

        from xgboost import XGBRanker
        with self._lock:
            try:
                self._model_hash_cache = ""  # invalidate cached hash before loading new model
                self._model = XGBRanker()
                self._model.load_model(MODEL_PATH)
                self._meta = self._load_meta()

                # Load calibration data (non-fatal if absent)
                self._calibration = None
                try:
                    from app.ml.calibration import load_calibration
                    self._calibration = load_calibration()
                except Exception as cal_exc:
                    logger.debug("calibration.not_loaded reason=%s", cal_exc)

                logger.info(
                    "model.loaded",
                    extra={
                        "trained_at":     self._meta.get("trained_at", "unknown"),
                        "validation_auc": self._meta.get("validation_auc"),
                        "mode":           self._meta.get("mode", "evaluation"),
                        "calibrated":     self._calibration is not None,
                    },
                )
                return True
            except Exception as exc:
                logger.error("model.load_failed", extra={"error": str(exc)})
                self._model = None
                return False

    @property
    def model_hash(self) -> str:
        """
        MD5 of the model.pkl bytes — stable, content-based cache key.

        Why MD5 of file bytes instead of trained_at timestamp:
          - Timestamps are fragile: if the file is copied, touched, or its
            metadata changes, the timestamp changes without the model changing.
          - Two identical models trained at different times would produce
            different cache keys, causing unnecessary Redis cache misses.
          - MD5 of bytes is deterministic: the key changes if and only if the
            model weights change, which is exactly when cached simulations
            should be invalidated.
          - MD5 is fast enough for a ~10 MB model file (< 5 ms).

        The hash is computed once at load() time and cached in _model_hash_cache
        so repeated property access is O(1).

        Falls back to "nomodel" when no model is loaded, or to a timestamp-based
        hash if the model file is unavailable (e.g., downloaded from S3 and then
        the local path was cleaned up).
        """
        if self._model_hash_cache:
            return self._model_hash_cache
        if not MODEL_PATH.exists():
            # Fallback: timestamp-based hash (less stable but better than nothing)
            trained_at = self._meta.get("trained_at", "")
            if not trained_at:
                return "nomodel"
            return "".join(c for c in trained_at if c.isdigit())[:12]
        try:
            import hashlib
            md5 = hashlib.md5(MODEL_PATH.read_bytes(), usedforsecurity=False).hexdigest()
            self._model_hash_cache = md5[:12]
            return self._model_hash_cache
        except Exception:
            return "nomodel"

    def reload(self) -> bool:
        """Hot-swap: reload model from disk. Safe to call while serving traffic."""
        logger.info("model.reload_requested")
        return self.load()

    def save_meta(self, metrics: dict) -> None:
        """
        Persist training metadata to disk and upload both model + meta to S3.

        Uploads to:
          - s3://.../models/latest/          (always the newest)
          - s3://.../models/{trained_at}/    (immutable versioned copy for rollback)
        """
        trained_at = datetime.now(timezone.utc).isoformat()
        meta = {
            "trained_at": trained_at,
            **{k: v for k, v in metrics.items() if k != "model_path"},
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        with self._lock:
            self._meta = meta

        # Upload to S3 (no-op if AWS_S3_BUCKET is not configured)
        from app.aws.s3 import upload_model
        upload_model(MODEL_PATH, trained_at)

    def _load_meta(self) -> dict:
        if META_PATH.exists():
            try:
                with open(META_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}


# Module-level singleton — import this everywhere
registry = ModelRegistry()
