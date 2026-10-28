"""Abstract Base Class for CineTrack Recommendation Models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for all recommendation models in CineTrack."""

    def __init__(self, model_name: str, version: str = "1.0.0") -> None:
        self.model_name = model_name
        self.version = version
        self.is_fitted = False
        self.metadata: Dict = {}
        self.movies_df: Optional[pd.DataFrame] = None

    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame, movies_df: Optional[pd.DataFrame] = None) -> "BaseRecommender":
        """Fit the recommendation model on historical ratings."""
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: Optional[int] = None,
        item_id: Optional[int] = None,
        top_k: int = 5,
        exclude_rated: bool = True,
    ) -> List[Dict]:
        """Generate top_k recommendations for a given user or similar to an item.

        Returns a list of dicts with:
        [{'movieId': int, 'title': str, 'genres': str, 'score': float, 'rank': int}, ...]
        """
        pass

    @abstractmethod
    def predict_score(self, user_id: int, item_id: int) -> float:
        """Predict preference/rating score for a specific user and movie."""
        pass

    def get_metadata(self) -> Dict:
        """Return model metadata dictionary."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "is_fitted": self.is_fitted,
            **self.metadata,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model instance to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseRecommender":
        """Load serialized model instance from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        model = joblib.load(path)
        if not isinstance(model, BaseRecommender):
            raise TypeError(f"Loaded object is not a BaseRecommender instance: {type(model)}")
        return model
