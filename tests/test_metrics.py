"""Unit tests for evaluation metrics: Precision@K, Recall@K, NDCG@K."""

import pytest
import numpy as np
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, evaluate_model
from src.models.popularity import PopularityRecommender


def test_precision_at_k():
    recs = [10, 20, 30, 40, 50]
    relevant = {20, 50, 60}

    # In top 5: items 20 and 50 are relevant -> 2 / 5 = 0.4
    assert precision_at_k(recs, relevant, k=5) == 0.4

    # In top 2: item 20 is relevant -> 1 / 2 = 0.5
    assert precision_at_k(recs, relevant, k=2) == 0.5

    # Edge cases
    assert precision_at_k(recs, set(), k=5) == 0.0
    assert precision_at_k([], relevant, k=5) == 0.0


def test_recall_at_k():
    recs = [10, 20, 30, 40, 50]
    relevant = {20, 50, 60}  # total 3 relevant items

    # In top 5: 2 hits out of 3 relevant -> 2 / 3 = 0.6667
    assert np.isclose(recall_at_k(recs, relevant, k=5), 2 / 3)

    # In top 1: item 10 not relevant -> 0 hits
    assert recall_at_k(recs, relevant, k=1) == 0.0

    # Edge case: empty relevant set
    assert recall_at_k(recs, set(), k=5) == 0.0


def test_ndcg_at_k():
    # Ideal ranking: items at rank 1 and 2 are relevant
    recs_ideal = [20, 50, 10, 30, 40]
    relevant = {20, 50}

    # When all relevant items appear first, NDCG must be 1.0
    assert np.isclose(ndcg_at_k(recs_ideal, relevant, k=5), 1.0)

    # Non-ideal ranking: items at rank 4 and 5
    recs_suboptimal = [10, 30, 40, 20, 50]
    score_suboptimal = ndcg_at_k(recs_suboptimal, relevant, k=5)
    assert 0.0 < score_suboptimal < 1.0

    # No hits in top-k
    recs_miss = [10, 30, 40, 60, 70]
    assert ndcg_at_k(recs_miss, relevant, k=5) == 0.0


def test_evaluate_model(sample_ratings_df, sample_movies_df):
    model = PopularityRecommender(min_ratings=1)
    model.fit(sample_ratings_df, sample_movies_df)

    # Split into train and test
    train_df = sample_ratings_df.iloc[:20]
    test_df = sample_ratings_df.iloc[20:]

    metrics = evaluate_model(
        model=model,
        train_df=train_df,
        test_df=test_df,
        k=5,
        relevance_threshold=3.5,
    )

    assert "precision@5" in metrics
    assert "recall@5" in metrics
    assert "ndcg@5" in metrics
    assert "catalog_coverage" in metrics
    assert "eval_user_count" in metrics
    assert 0.0 <= metrics["precision@5"] <= 1.0
    assert 0.0 <= metrics["recall@5"] <= 1.0
    assert 0.0 <= metrics["ndcg@5"] <= 1.0
