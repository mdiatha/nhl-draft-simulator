"""Standalone training script.

Run from inside the container or with the correct DATABASE_URL set:

    python -m app.ml.train_model
    python -m app.ml.train_model --cutoff 2023
    python -m app.ml.train_model --cutoff 2021  # earlier cutoff for testing

The script:
  1. Connects to the database
  2. Builds the training dataset (historical picks with negative sampling)
  3. Trains the XGBoost model
  4. Saves model.pkl + model_meta.json to app/ml/
  5. Prints a summary

After running, hit POST /api/ml/reload to inject the new model into the
running API without a restart.
"""
from __future__ import annotations

import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NHL draft pick prediction model.")
    parser.add_argument(
        "--cutoff",
        type=int,
        default=2022,
        help="Train on picks up to and including this year (default: 2022)",
    )
    args = parser.parse_args()

    from app.database import SessionLocal
    from app.ml.train import train_from_db
    from app.ml.registry import registry

    logger.info("Connecting to database...")
    db = SessionLocal()

    try:
        logger.info("Building training dataset (picks ≤ %d)...", args.cutoff)
        metrics = train_from_db(db)
    finally:
        db.close()

    # Persist metadata alongside the model artifact
    registry.save_meta(metrics)

    # Print summary
    print("\n── Training complete ──────────────────────────────")
    print(f"  Training samples : {metrics['training_samples']:,}")
    print(f"  Positives        : {metrics['positive_samples']:,}")
    print(f"  Negatives        : {metrics['negative_samples']:,}")
    print(f"  Validation AUC   : {metrics['validation_auc']:.4f}")
    print(f"  Model saved to   : {metrics['model_path']}")
    print("\n  Top feature importances:")
    fi = metrics.get("feature_importances", {})
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * int(imp * 40)
        print(f"    {feat:25s} {bar} {imp:.4f}")
    print("\n  → Run POST /api/ml/reload to inject into the running API.")


if __name__ == "__main__":
    main()
