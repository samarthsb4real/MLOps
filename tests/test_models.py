"""Unit tests for Popularity baseline and Collaborative Filtering recommenders."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.models.popularity import PopularityRecommender
from src.models.collaborative import CollaborativeFilteringRecommender
from src.models.base import BaseRecommender


def test_popularity_recommender(sample_ratings_df, sample_movies_df):
    model = PopularityRecommender(min_ratings=2)
    assert model.is_fitted is False

    model.fit(sample_ratings_df, sample_movies_df)
    assert model.is_fitted is True

    # Test top-3 recommendations for user 1
    recs = model.recommend(user_id=1, top_k=3, exclude_rated=False)
    assert len(recs) == 3
    for r in recs:
        assert "movieId" in r
        assert "title" in r
        assert "genres" in r
        assert "score" in r
        assert "rank" in r

    # Test item-based recommendation
    similar_recs = model.recommend(item_id=1, top_k=3)
    assert len(similar_recs) <= 3
    # Target item should not be in similar recommendations
    assert all(r["movieId"] != 1 for r in similar_recs)

    # Test save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "pop_model.joblib"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = PopularityRecommender.load(save_path)
        assert loaded_model.is_fitted is True
        assert loaded_model.model_name == "PopularityRecommender"
        loaded_recs = loaded_model.recommend(user_id=1, top_k=3, exclude_rated=False)
        assert len(loaded_recs) == 3


def test_collaborative_filtering_recommender(sample_ratings_df, sample_movies_df):
    model = CollaborativeFilteringRecommender(
        n_factors=8,
        n_epochs=10,
        learning_rate=0.01,
        regularization=0.02,
        random_state=42,
    )
    assert model.is_fitted is False

    model.fit(sample_ratings_df, sample_movies_df)
    assert model.is_fitted is True
    assert model.metadata["n_users"] == 5
    assert model.metadata["n_items"] == 6

    # Test predict score
    score = model.predict_score(user_id=1, item_id=1)
    assert isinstance(score, float)
    assert 0.5 <= score <= 5.0

    # Test personalized recommendations for user 1
    recs = model.recommend(user_id=1, top_k=3, exclude_rated=False)
    assert len(recs) == 3
    assert recs[0]["rank"] == 1
    assert recs[1]["rank"] == 2
    assert recs[2]["rank"] == 3

    # Test item similarity recommendations
    item_recs = model.recommend(item_id=1, top_k=3)
    assert len(item_recs) <= 3
    assert all(r["movieId"] != 1 for r in item_recs)

    # Test save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "cf_model.joblib"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = BaseRecommender.load(save_path)
        assert loaded_model.is_fitted is True
        assert loaded_model.model_name == "CollaborativeFilteringSVD"

        # Check identical predictions
        pred_original = model.predict_score(user_id=2, item_id=3)
        pred_loaded = loaded_model.predict_score(user_id=2, item_id=3)
        assert np.isclose(pred_original, pred_loaded)
