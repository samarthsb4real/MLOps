"""Data loader for MovieLens Small and real-time user feedback."""

from pathlib import Path
import re
import shutil
import pandas as pd
from typing import Optional, Tuple
from src.config import (
    SOURCE_DATASET_DIR,
    RAW_DATA_DIR,
    FEEDBACK_DATA_DIR,
)


def extract_year(title: str) -> Optional[int]:
    """Extract release year from movie title (e.g. 'Toy Story (1995)' -> 1995)."""
    match = re.search(r"\((\d{4})\)", str(title))
    if match:
        return int(match.group(1))
    return None


def copy_raw_dataset_if_needed() -> None:
    """Ensure raw dataset files are copied to data/raw/ from source."""
    if not (RAW_DATA_DIR / "movies.csv").exists() and (SOURCE_DATASET_DIR / "movies.csv").exists():
        for fname in ["movies.csv", "ratings.csv", "links.csv", "tags.csv", "README.txt"]:
            src = SOURCE_DATASET_DIR / fname
            if src.exists():
                shutil.copy2(src, RAW_DATA_DIR / fname)


def load_movies(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and preprocess movies dataset."""
    copy_raw_dataset_if_needed()
    base_dir = data_dir or RAW_DATA_DIR
    if not (base_dir / "movies.csv").exists():
        base_dir = SOURCE_DATASET_DIR

    movies_path = base_dir / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(f"movies.csv not found in {base_dir}")

    df = pd.read_csv(movies_path)
    df["movieId"] = df["movieId"].astype(int)
    df["title"] = df["title"].astype(str)
    df["genres"] = df["genres"].fillna("(no genres listed)").astype(str)
    df["year"] = df["title"].apply(extract_year)
    df["genre_list"] = df["genres"].apply(lambda g: g.split("|"))
    return df


def load_feedback_ratings(feedback_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all incoming user feedback ratings collected by the application."""
    fdir = feedback_dir or FEEDBACK_DATA_DIR
    feedback_files = list(fdir.glob("*.csv"))
    if not feedback_files:
        return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

    dfs = []
    for f in feedback_files:
        try:
            df = pd.read_csv(f)
            if {"userId", "movieId", "rating", "timestamp"}.issubset(df.columns):
                dfs.append(df[["userId", "movieId", "rating", "timestamp"]])
        except Exception:
            continue

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])


def load_ratings(
    data_dir: Optional[Path] = None,
    include_feedback: bool = True,
) -> pd.DataFrame:
    """Load ratings dataset, optionally combining with real-time feedback ratings."""
    copy_raw_dataset_if_needed()
    base_dir = data_dir or RAW_DATA_DIR
    if not (base_dir / "ratings.csv").exists():
        base_dir = SOURCE_DATASET_DIR

    ratings_path = base_dir / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.csv not found in {base_dir}")

    df = pd.read_csv(ratings_path)
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)

    if include_feedback:
        feedback_df = load_feedback_ratings()
        if not feedback_df.empty:
            feedback_df["userId"] = feedback_df["userId"].astype(int)
            feedback_df["movieId"] = feedback_df["movieId"].astype(int)
            feedback_df["rating"] = feedback_df["rating"].astype(float)
            feedback_df["timestamp"] = feedback_df["timestamp"].astype(int)
            df = pd.concat([df, feedback_df], ignore_index=True)

    # In case of duplicate user-movie interactions, keep the latest timestamp
    df = df.sort_values("timestamp").drop_duplicates(subset=["userId", "movieId"], keep="last")
    return df.reset_index(drop=True)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience helper to load both movies and ratings."""
    return load_movies(), load_ratings()
