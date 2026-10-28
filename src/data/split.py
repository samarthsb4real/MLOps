"""Reproducible data splitting module preventing data leakage in recommendation evaluation."""

from typing import Tuple
import pandas as pd
import numpy as np


def split_ratings_train_test(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    min_ratings_per_user: int = 5,
    random_state: int = 42,
    strategy: str = "temporal",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train and test sets without data leakage.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain ['userId', 'movieId', 'rating', 'timestamp'].
    test_ratio : float
        Fraction of ratings per user to hold out for testing.
    min_ratings_per_user : int
        Minimum number of ratings a user must have to participate in the test set.
    random_state : int
        Random seed for reproducibility when sampling.
    strategy : str
        'temporal' (sort by timestamp, hold out most recent interactions) or
        'random_per_user' (stratified random hold-out per user).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    """
    if ratings_df.empty:
        raise ValueError("Cannot split empty ratings dataframe.")

    train_indices = []
    test_indices = []

    # Group by userId
    grouped = ratings_df.groupby("userId")

    for user_id, group in grouped:
        n_user_ratings = len(group)
        if n_user_ratings < min_ratings_per_user:
            # Not enough ratings: keep all in train to preserve signal
            train_indices.extend(group.index.tolist())
            continue

        n_test = max(1, int(np.round(n_user_ratings * test_ratio)))

        if strategy == "temporal" and "timestamp" in group.columns:
            sorted_group = group.sort_values("timestamp")
            user_train_idx = sorted_group.index[:-n_test].tolist()
            user_test_idx = sorted_group.index[-n_test:].tolist()
        else:
            # Random stratified sample per user with seed
            user_test_idx = group.sample(
                n=n_test,
                random_state=random_state,
            ).index.tolist()
            user_train_idx = [idx for idx in group.index if idx not in user_test_idx]

        train_indices.extend(user_train_idx)
        test_indices.extend(user_test_idx)

    train_df = ratings_df.loc[train_indices].copy().reset_index(drop=True)
    test_df = ratings_df.loc[test_indices].copy().reset_index(drop=True)

    return train_df, test_df
