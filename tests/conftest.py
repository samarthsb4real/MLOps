"""Pytest fixtures for CineTrack test suite."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_movies_df() -> pd.DataFrame:
    """Fixture providing a sample movies dataframe."""
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6],
            "title": [
                "Toy Story (1995)",
                "Jumanji (1995)",
                "Grumpier Old Men (1995)",
                "Waiting to Exhale (1995)",
                "Heat (1995)",
                "Sabrina (1995)",
            ],
            "genres": [
                "Adventure|Animation|Children|Comedy|Fantasy",
                "Adventure|Children|Fantasy",
                "Comedy|Romance",
                "Comedy|Drama|Romance",
                "Action|Crime|Thriller",
                "Comedy|Romance",
            ],
            "year": [1995, 1995, 1995, 1995, 1995, 1995],
            "genre_list": [
                ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
                ["Adventure", "Children", "Fantasy"],
                ["Comedy", "Romance"],
                ["Comedy", "Drama", "Romance"],
                ["Action", "Crime", "Thriller"],
                ["Comedy", "Romance"],
            ],
        }
    )


@pytest.fixture
def sample_ratings_df() -> pd.DataFrame:
    """Fixture providing a synthetic ratings dataframe across multiple users."""
    np.random.seed(42)
    records = []
    base_ts = 1000000000
    ts_counter = 0

    # User 1 to 5 ratings across movies 1 to 6
    for u in range(1, 6):
        for m in range(1, 7):
            # Give higher ratings to movies 1, 2, 5
            rating = 4.5 if m in [1, 2, 5] else 2.5
            records.append(
                {
                    "userId": u,
                    "movieId": m,
                    "rating": rating,
                    "timestamp": base_ts + ts_counter,
                }
            )
            ts_counter += 100

    return pd.DataFrame(records)
