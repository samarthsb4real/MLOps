"""Collaborative Filtering Recommender using Biased Matrix Factorisation (SVD).

Implements the Koren et al. regularised matrix factorisation model:
    r_hat(u, i) = mu + b_u + b_i + p_u . q_i
optimised via Stochastic Gradient Descent (SGD) with L2 regularisation.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from src.models.base import BaseRecommender


class CollaborativeFilteringRecommender(BaseRecommender):
    """Matrix Factorization Collaborative Filtering Recommender with User & Item Biases."""

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 20,
        learning_rate: float = 0.005,
        regularization: float = 0.05,
        random_state: int = 42,
        version: str = "1.0.0",
    ) -> None:
        super().__init__(model_name="CollaborativeFilteringSVD", version=version)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state

        # Parameters learned during training
        self.global_mean: float = 0.0
        self.user_biases: np.ndarray = np.array([])
        self.item_biases: np.ndarray = np.array([])
        self.user_factors: np.ndarray = np.array([])
        self.item_factors: np.ndarray = np.array([])

        # Index mappings
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

        # User interaction history {userId: set(movieIds)}
        self.user_history: Dict[int, set] = {}

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> "CollaborativeFilteringRecommender":
        """Fit regularised biased matrix factorisation model on ratings."""
        if ratings_df.empty:
            raise ValueError("Cannot fit CollaborativeFilteringRecommender with empty ratings.")

        self.movies_df = movies_df
        self.global_mean = float(ratings_df["rating"].mean())

        # Map IDs to consecutive integers
        unique_users = sorted(ratings_df["userId"].unique())
        unique_items = sorted(ratings_df["movieId"].unique())

        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for i, u in enumerate(unique_users)}
        self.item2idx = {m: i for i, m in enumerate(unique_items)}
        self.idx2item = {i: m for i, m in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        # Build user interaction set
        user_groups = ratings_df.groupby("userId")["movieId"].apply(set)
        self.user_history = user_groups.to_dict()

        # Initialize parameters
        rng = np.random.default_rng(self.random_state)
        self.user_biases = np.zeros(n_users, dtype=np.float32)
        self.item_biases = np.zeros(n_items, dtype=np.float32)

        # Factors initialized with small Gaussian noise
        self.user_factors = rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)

        # Prepare numpy arrays for rapid iteration
        u_indices = np.array([self.user2idx[u] for u in ratings_df["userId"]], dtype=np.int32)
        i_indices = np.array([self.item2idx[m] for m in ratings_df["movieId"]], dtype=np.int32)
        ratings = ratings_df["rating"].to_numpy(dtype=np.float32)

        lr = float(self.learning_rate)
        reg = float(self.regularization)
        mu = float(self.global_mean)

        n_samples = len(ratings)
        indices = np.arange(n_samples)

        # SGD Optimization loop
        for epoch in range(self.n_epochs):
            rng.shuffle(indices)
            loss = 0.0

            for idx in indices:
                u = u_indices[idx]
                i = i_indices[idx]
                r = ratings[idx]

                # Prediction: mu + b_u + b_i + dot(P_u, Q_i)
                p_u = self.user_factors[u]
                q_i = self.item_factors[i]
                pred = mu + self.user_biases[u] + self.item_biases[i] + np.dot(p_u, q_i)
                err = r - pred
                loss += err * err

                # Gradient updates with L2 regularization
                self.user_biases[u] += lr * (err - reg * self.user_biases[u])
                self.item_biases[i] += lr * (err - reg * self.item_biases[i])

                p_update = lr * (err * q_i - reg * p_u)
                q_update = lr * (err * p_u - reg * q_i)

                self.user_factors[u] += p_update
                self.item_factors[i] += q_update

        self.is_fitted = True
        self.metadata = {
            "n_users": n_users,
            "n_items": n_items,
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "global_mean": self.global_mean,
            "final_epoch_rmse": float(np.sqrt(loss / max(1, n_samples))),
        }
        return self

    def predict_score(self, user_id: int, item_id: int) -> float:
        """Predict rating score for a user and item with fallback for unseen entities."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        u_idx = self.user2idx.get(user_id)
        i_idx = self.item2idx.get(item_id)

        # Base score starts with global mean
        score = self.global_mean

        if u_idx is not None:
            score += self.user_biases[u_idx]
        if i_idx is not None:
            score += self.item_biases[i_idx]

        if u_idx is not None and i_idx is not None:
            score += float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))

        # Clip predicted score to standard rating range [0.5, 5.0]
        return float(np.clip(score, 0.5, 5.0))

    def _get_movie_details(self, movie_id: int) -> Dict[str, str]:
        """Lookup title and genres for movie_id."""
        if self.movies_df is not None:
            row = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not row.empty:
                return {
                    "title": str(row.iloc[0]["title"]),
                    "genres": str(row.iloc[0]["genres"]),
                }
        return {"title": f"Movie {movie_id}", "genres": "Unknown"}

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 5,
        exclude_rated: bool = True,
        genre: Optional[str] = None,
    ) -> List[Dict]:
        """Generate top_k personalized recommendations for a user."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        u_idx = self.user2idx.get(user_id)

        # If user is completely unknown (cold start), return popular movies
        if u_idx is None:
            rated_ids = self.user_history.get(user_id, set()) if exclude_rated else set()
            scores = self.item_biases + self.global_mean
            candidate_item_indices = np.argsort(scores)[::-1]
        else:
            rated_ids = self.user_history.get(user_id, set()) if exclude_rated else set()
            p_u = self.user_factors[u_idx]
            # Vectorized dot product across all items + biases
            scores = self.global_mean + self.user_biases[u_idx] + self.item_biases + np.dot(self.item_factors, p_u)
            candidate_item_indices = np.argsort(scores)[::-1]

        recommendations = []
        rank = 1
        for idx in candidate_item_indices:
            movie_id = self.idx2item[idx]
            if exclude_rated and movie_id in rated_ids:
                continue

            details = self._get_movie_details(movie_id)
            if genre and genre.lower() != "all":
                if genre.lower() not in details["genres"].lower():
                    continue

            recommendations.append(
                {
                    "movieId": int(movie_id),
                    "title": details["title"],
                    "genres": details["genres"],
                    "score": round(float(scores[idx]), 4),
                    "rank": rank,
                }
            )
            rank += 1
            if len(recommendations) >= top_k:
                break

        return recommendations

    def recommend_similar_items(
        self,
        item_id: int,
        top_k: int = 5,
        genre: Optional[str] = None,
    ) -> List[Dict]:
        """Find movies most similar to a selected movie using latent cosine similarity."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        target_idx = self.item2idx.get(item_id)
        if target_idx is None:
            # Fallback if unknown movie
            return []

        target_factor = self.item_factors[target_idx]
        target_norm = np.linalg.norm(target_factor)
        if target_norm == 0:
            target_norm = 1e-8

        # Compute cosine similarity across all item vectors
        all_norms = np.linalg.norm(self.item_factors, axis=1)
        all_norms[all_norms == 0] = 1e-8
        sims = np.dot(self.item_factors, target_factor) / (all_norms * target_norm)

        # Sort descending
        ranked_indices = np.argsort(sims)[::-1]

        recommendations = []
        rank = 1
        for idx in ranked_indices:
            movie_id = self.idx2item[idx]
            if movie_id == item_id:
                continue

            details = self._get_movie_details(movie_id)
            if genre and genre.lower() != "all":
                if genre.lower() not in details["genres"].lower():
                    continue

            # Convert cosine similarity [-1, 1] to a normalized match score [0, 5]
            normalized_score = float(np.clip((sims[idx] + 1.0) * 2.5, 0.0, 5.0))

            recommendations.append(
                {
                    "movieId": int(movie_id),
                    "title": details["title"],
                    "genres": details["genres"],
                    "score": round(normalized_score, 4),
                    "rank": rank,
                }
            )
            rank += 1
            if len(recommendations) >= top_k:
                break

        return recommendations

    def recommend(
        self,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        top_k: int = 5,
        exclude_rated: bool = True,
        genre: Optional[str] = None,
    ) -> List[Dict]:
        """Dispatch recommendation based on provided context."""
        if item_id is not None:
            return self.recommend_similar_items(item_id=item_id, top_k=top_k, genre=genre)
        elif user_id is not None:
            return self.recommend_for_user(user_id=user_id, top_k=top_k, exclude_rated=exclude_rated, genre=genre)
        else:
            # If neither provided, fallback to top popular movies
            return self.recommend_for_user(user_id=1, top_k=top_k, exclude_rated=False, genre=genre)
