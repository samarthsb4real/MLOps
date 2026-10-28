"""Data preparation stage: validate raw data and create reproducible train/test splits."""

import json
import logging
from pathlib import Path
from typing import Dict
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RELEVANCE_THRESHOLD,
    RANDOM_STATE,
)
from src.data.loader import load_movies, load_ratings
from src.data.validator import validate_dataset
from src.data.split import split_ratings_train_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(
    test_ratio: float = 0.2,
    min_ratings_per_user: int = 5,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> Dict:
    """Run data ingestion, validation, and reproducible train/test split."""
    logger.info("Ingesting raw MovieLens dataset...")
    movies_df = load_movies()
    ratings_df = load_ratings(include_feedback=True)

    logger.info("Validating dataset schema and integrity...")
    val_report = validate_dataset(movies_df, ratings_df)
    if not val_report["is_valid"]:
        logger.error(f"Data validation failed: {val_report['errors']}")
        raise ValueError(f"Data validation failed with errors: {val_report['errors']}")

    logger.info(
        f"Validation passed: {val_report['movies_stats']['num_movies']} movies, "
        f"{val_report['ratings_stats']['num_ratings']} ratings, "
        f"{val_report['ratings_stats']['num_users']} users."
    )

    logger.info(f"Splitting ratings (test_ratio={test_ratio}, strategy=temporal)...")
    train_df, test_df = split_ratings_train_test(
        ratings_df=ratings_df,
        test_ratio=test_ratio,
        min_ratings_per_user=min_ratings_per_user,
        random_state=RANDOM_STATE,
        strategy="temporal",
    )

    logger.info(f"Split completed: Train={len(train_df)} ratings, Test={len(test_df)} ratings.")

    # Save processed outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    movies_df.to_csv(output_dir / "movies.csv", index=False)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    summary = {
        "num_movies": len(movies_df),
        "num_ratings_total": len(ratings_df),
        "num_ratings_train": len(train_df),
        "num_ratings_test": len(test_df),
        "num_users": int(ratings_df["userId"].nunique()),
        "test_ratio": test_ratio,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "is_valid": True,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(val_report, f, indent=2)

    logger.info(f"Processed artifacts successfully saved to {output_dir}")
    return summary


if __name__ == "__main__":
    prepare_data()
