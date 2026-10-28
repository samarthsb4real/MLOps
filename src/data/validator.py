"""Data validation module ensuring dataset integrity and schema consistency."""

import logging
from typing import Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def validate_movies(df: pd.DataFrame) -> Tuple[bool, List[str], Dict]:
    """Validate movies dataframe schema and integrity."""
    errors = []
    required_cols = {"movieId", "title", "genres"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Movies missing required columns: {missing}")

    if "movieId" in df.columns:
        if df["movieId"].isnull().any():
            errors.append("Movies contains null movieId values.")
        if not df["movieId"].is_unique:
            errors.append("Movies contains duplicate movieId entries.")

    if "title" in df.columns and df["title"].isnull().any():
        errors.append("Movies contains null title values.")

    stats = {
        "num_movies": len(df),
        "unique_movie_ids": df["movieId"].nunique() if "movieId" in df.columns else 0,
        "genres_available": bool("genres" in df.columns and not df["genres"].isnull().all()),
    }
    is_valid = len(errors) == 0
    return is_valid, errors, stats


def validate_ratings(df: pd.DataFrame) -> Tuple[bool, List[str], Dict]:
    """Validate ratings dataframe schema, value bounds, and constraints."""
    errors = []
    required_cols = {"userId", "movieId", "rating", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Ratings missing required columns: {missing}")

    if df.empty:
        errors.append("Ratings dataframe is empty.")
        return False, errors, {"num_ratings": 0}

    # Null checks
    for col in required_cols.intersection(df.columns):
        if df[col].isnull().any():
            errors.append(f"Ratings column '{col}' contains {df[col].isnull().sum()} null values.")

    # Rating range checks (MovieLens standard: 0.5 to 5.0)
    if "rating" in df.columns:
        invalid_ratings = df[(df["rating"] < 0.5) | (df["rating"] > 5.0)]
        if not invalid_ratings.empty:
            errors.append(f"Found {len(invalid_ratings)} ratings outside valid range [0.5, 5.0].")

    # Timestamp checks
    if "timestamp" in df.columns:
        negative_ts = df[df["timestamp"] <= 0]
        if not negative_ts.empty:
            errors.append(f"Found {len(negative_ts)} invalid non-positive timestamps.")

    # Duplicates check
    if {"userId", "movieId"}.issubset(df.columns):
        duplicate_pairs = df.duplicated(subset=["userId", "movieId"]).sum()
        if duplicate_pairs > 0:
            logger.warning(f"Found {duplicate_pairs} duplicate user-movie ratings; deduplication recommended.")

    stats = {
        "num_ratings": len(df),
        "num_users": df["userId"].nunique() if "userId" in df.columns else 0,
        "num_movies_rated": df["movieId"].nunique() if "movieId" in df.columns else 0,
        "mean_rating": float(df["rating"].mean()) if "rating" in df.columns else 0.0,
        "min_rating": float(df["rating"].min()) if "rating" in df.columns else 0.0,
        "max_rating": float(df["rating"].max()) if "rating" in df.columns else 0.0,
    }

    is_valid = len(errors) == 0
    return is_valid, errors, stats


def validate_dataset(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> Dict:
    """Run full validation across movies and ratings datasets."""
    m_valid, m_errors, m_stats = validate_movies(movies_df)
    r_valid, r_errors, r_stats = validate_ratings(ratings_df)

    # Referential integrity check
    overlap_error = []
    if m_valid and r_valid:
        rated_movies = set(ratings_df["movieId"])
        catalog_movies = set(movies_df["movieId"])
        unmatched = rated_movies - catalog_movies
        if unmatched:
            overlap_error.append(f"{len(unmatched)} rated movies do not exist in the movies catalog.")

    all_errors = m_errors + r_errors + overlap_error
    is_valid = len(all_errors) == 0

    return {
        "is_valid": is_valid,
        "errors": all_errors,
        "movies_stats": m_stats,
        "ratings_stats": r_stats,
    }
