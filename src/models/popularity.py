"""Popularity-Based Recommendation Baseline Model."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommendation baseline using IMDB weighted rating formula.

    Score = (v / (v + m)) * R + (m / (v + m)) * C
    where:
      v = number of ratings for the movie
      m = minimum ratings threshold (e.g., 20 or 75th percentile)
      R = average rating for the movie
      C = mean rating across the entire dataset
    """

    def __init__(
        self,
        min_ratings: int = 20,
        version: str = "1.0.0",
    ) -> None:
        super().__init__(model_name="PopularityRecommender", version=version)
        self.min_ratings = min_ratings
        self.item_scores: pd.DataFrame = pd.DataFrame()
        self.user_history: Dict[int, set] = {}
        self.global_mean: float = 0.0

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> "PopularityRecommender":
        """Compute weighted popularity score for each movie in catalog."""
        if ratings_df.empty:
            raise ValueError("Cannot fit PopularityRecommender with empty ratings.")

        self.movies_df = movies_df
        self.global_mean = float(ratings_df["rating"].mean())

        # Build user history
        user_groups = ratings_df.groupby("userId")["movieId"].apply(set)
        self.user_history = user_groups.to_dict()

        # Compute stats per movie
        stats = ratings_df.groupby("movieId").agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean"),
        ).reset_index()

        # Weighted rating calculation
        m = self.min_ratings
        c = self.global_mean

        stats["score"] = (
            (stats["rating_count"] / (stats["rating_count"] + m)) * stats["rating_mean"]
            + (m / (stats["rating_count"] + m)) * c
        )

        if movies_df is not None:
            stats = pd.merge(stats, movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
            stats["title"] = stats["title"].fillna(stats["movieId"].apply(lambda x: f"Movie {x}"))
            stats["genres"] = stats["genres"].fillna("Unknown")
        else:
            stats["title"] = stats["movieId"].apply(lambda x: f"Movie {x}")
            stats["genres"] = "Unknown"

        self.item_scores = stats.sort_values(by="score", ascending=False).reset_index(drop=True)
        self.is_fitted = True

        self.metadata = {
            "num_movies_scored": len(self.item_scores),
            "global_mean_rating": self.global_mean,
            "min_ratings_threshold": self.min_ratings,
            "top_movie_title": str(self.item_scores.iloc[0]["title"]) if not self.item_scores.empty else "",
        }
        return self

    def predict_score(self, user_id: int, item_id: int) -> float:
        """Return the global popularity score for the item."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        match = self.item_scores[self.item_scores["movieId"] == item_id]
        if not match.empty:
            return float(match.iloc[0]["score"])
        return self.global_mean

    def recommend(
        self,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        genre: Optional[str] = None,
        top_k: int = 5,
        exclude_rated: bool = True,
    ) -> List[Dict]:
        """Generate top_k popularity recommendations."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        candidates = self.item_scores.copy()

        # Filter by genre if requested
        if genre and genre.lower() != "all" and "genres" in candidates.columns:
            candidates = candidates[
                candidates["genres"].str.contains(genre, case=False, na=False)
            ]

        # If item_id provided, exclude the item itself and prioritize matching genres
        if item_id is not None:
            candidates = candidates[candidates["movieId"] != item_id]
            if self.movies_df is not None:
                item_row = self.movies_df[self.movies_df["movieId"] == item_id]
                if not item_row.empty:
                    item_genres = set(str(item_row.iloc[0]["genres"]).split("|"))
                    # Boost movies with overlapping genres
                    def genre_overlap(g_str: str) -> int:
                        return len(set(str(g_str).split("|")).intersection(item_genres))

                    candidates["overlap"] = candidates["genres"].apply(genre_overlap)
                    candidates = candidates.sort_values(
                        by=["overlap", "score"],
                        ascending=[False, False],
                    )

        # Exclude already rated movies for this user
        if user_id is not None and exclude_rated and user_id in self.user_history:
            rated_ids = self.user_history[user_id]
            candidates = candidates[~candidates["movieId"].isin(rated_ids)]

        top_candidates = candidates.head(top_k)
        recommendations = []
        for rank, (_, row) in enumerate(top_candidates.iterrows(), start=1):
            recommendations.append(
                {
                    "movieId": int(row["movieId"]),
                    "title": str(row["title"]),
                    "genres": str(row["genres"]),
                    "score": round(float(row["score"]), 4),
                    "rank": rank,
                }
            )
        return recommendations
