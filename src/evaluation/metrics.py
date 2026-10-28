"""Precision@K, Recall@K, and NDCG@K evaluation metrics for recommender systems."""

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from src.models.base import BaseRecommender


def precision_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int = 5) -> float:
    """Calculate Precision@K for a single user.

    Precision@K = (# of relevant items in top-K) / K
    """
    if k <= 0:
        return 0.0
    top_k_recs = recommended_ids[:k]
    hits = sum(1 for item in top_k_recs if item in relevant_ids)
    return float(hits / k)


def recall_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int = 5) -> float:
    """Calculate Recall@K for a single user.

    Recall@K = (# of relevant items in top-K) / (Total # of relevant items for user)
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k_recs = recommended_ids[:k]
    hits = sum(1 for item in top_k_recs if item in relevant_ids)
    return float(hits / len(relevant_ids))


def ndcg_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K) for a single user.

    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(K, |Rel|)} (1 / log2(i + 1))
    NDCG@K = DCG@K / IDCG@K
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k_recs = recommended_ids[:k]
    dcg = 0.0
    for rank, item_id in enumerate(top_k_recs, start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG: all relevant items ranked in the top positions
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def evaluate_model(
    model: BaseRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 5,
    relevance_threshold: float = 3.5,
    max_eval_users: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a recommendation model across test users using Precision@K, Recall@K, and NDCG@K.

    Parameters
    ----------
    model : BaseRecommender
        Fitted recommendation model instance.
    train_df : pd.DataFrame
        Training ratings used to fit the model.
    test_df : pd.DataFrame
        Ground truth test ratings for evaluation.
    k : int
        Cutoff rank (default 5).
    relevance_threshold : float
        Minimum rating threshold to treat an item as relevant (default 3.5).
    max_eval_users : Optional[int]
        Cap on test users for quick evaluation or testing.

    Returns
    -------
    Dict[str, float]
        Averaged metrics: precision@k, recall@k, ndcg@k, catalog_coverage, eval_user_count.
    """
    if test_df.empty:
        raise ValueError("Cannot evaluate model with empty test dataset.")

    # Filter test interactions by relevance
    relevant_test = test_df[test_df["rating"] >= relevance_threshold]
    user_relevant_items: Dict[int, Set[int]] = (
        relevant_test.groupby("userId")["movieId"].apply(set).to_dict()
    )

    # Only evaluate users with at least 1 relevant item in test set
    eval_users = list(user_relevant_items.keys())
    if max_eval_users is not None and max_eval_users > 0:
        eval_users = eval_users[:max_eval_users]

    if not eval_users:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "catalog_coverage": 0.0,
            "eval_user_count": 0,
        }

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []
    all_recommended_items: Set[int] = set()

    for user_id in eval_users:
        relevant_items = user_relevant_items[user_id]
        recs = model.recommend(user_id=user_id, top_k=k, exclude_rated=True)
        rec_ids = [item["movieId"] for item in recs]
        all_recommended_items.update(rec_ids)

        precisions.append(precision_at_k(rec_ids, relevant_items, k=k))
        recalls.append(recall_at_k(rec_ids, relevant_items, k=k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant_items, k=k))

    total_catalog_items = len(set(train_df["movieId"]).union(set(test_df["movieId"])))
    catalog_coverage = (
        float(len(all_recommended_items) / total_catalog_items) if total_catalog_items > 0 else 0.0
    )

    return {
        f"precision@{k}": round(float(np.mean(precisions)), 4),
        f"recall@{k}": round(float(np.mean(recalls)), 4),
        f"ndcg@{k}": round(float(np.mean(ndcgs)), 4),
        "catalog_coverage": round(catalog_coverage, 4),
        "eval_user_count": len(eval_users),
    }
