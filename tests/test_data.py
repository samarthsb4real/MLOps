"""Unit tests for data loading, schema validation, and train/test split."""

import pandas as pd
from src.data.validator import validate_movies, validate_ratings, validate_dataset
from src.data.split import split_ratings_train_test
from src.data.loader import extract_year


def test_extract_year():
    assert extract_year("Toy Story (1995)") == 1995
    assert extract_year("Iron Man (2008)") == 2008
    assert extract_year("Movie Without Year") is None


def test_validate_movies(sample_movies_df):
    is_valid, errors, stats = validate_movies(sample_movies_df)
    assert is_valid is True
    assert len(errors) == 0
    assert stats["num_movies"] == 6

    # Test invalid movie with null title
    bad_df = sample_movies_df.copy()
    bad_df.loc[0, "title"] = None
    is_valid, errors, _ = validate_movies(bad_df)
    assert is_valid is False
    assert any("null title" in e for e in errors)


def test_validate_ratings(sample_ratings_df):
    is_valid, errors, stats = validate_ratings(sample_ratings_df)
    assert is_valid is True
    assert len(errors) == 0
    assert stats["num_ratings"] == 30
    assert stats["min_rating"] == 2.5
    assert stats["max_rating"] == 4.5

    # Test out of bounds rating
    bad_ratings = sample_ratings_df.copy()
    bad_ratings.loc[0, "rating"] = 7.5
    is_valid, errors, _ = validate_ratings(bad_ratings)
    assert is_valid is False
    assert any("outside valid range" in e for e in errors)


def test_validate_dataset(sample_movies_df, sample_ratings_df):
    report = validate_dataset(sample_movies_df, sample_ratings_df)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0

    # Test referential integrity failure
    orphan_ratings = sample_ratings_df.copy()
    orphan_ratings.loc[0, "movieId"] = 99999
    report = validate_dataset(sample_movies_df, orphan_ratings)
    assert report["is_valid"] is False
    assert any("do not exist in the movies catalog" in e for e in report["errors"])


def test_split_ratings_train_test(sample_ratings_df):
    train_df, test_df = split_ratings_train_test(
        sample_ratings_df,
        test_ratio=0.33,
        min_ratings_per_user=3,
        strategy="temporal",
    )

    # Check that sizes sum to original
    assert len(train_df) + len(test_df) == len(sample_ratings_df)

    # Check temporal ordering per user (no future test timestamp is before train timestamp)
    for u in train_df["userId"].unique():
        u_train = train_df[train_df["userId"] == u]
        u_test = test_df[test_df["userId"] == u]
        if not u_test.empty and not u_train.empty:
            assert u_train["timestamp"].max() <= u_test["timestamp"].min()
