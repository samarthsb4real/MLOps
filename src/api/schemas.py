"""Pydantic schemas for request validation and response formatting."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class MovieItem(BaseModel):
    movieId: int
    title: str
    genres: str
    year: Optional[int] = None
    rating_count: Optional[int] = 0
    rating_mean: Optional[float] = 0.0


class MovieListResponse(BaseModel):
    total: int
    skip: int
    limit: int
    movies: List[MovieItem]


class RecommendationRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="Target User ID for personalized recommendations")
    movie_id: Optional[int] = Field(None, description="Selected Movie ID for item-similarity recommendations")
    genre: Optional[str] = Field(None, description="Optional genre filter")
    top_k: int = Field(5, ge=1, le=50, description="Number of recommendations to return")
    model_version: Optional[str] = Field("production", description="Model stage: 'production', 'candidate', or 'baseline'")


class RecommendedMovie(BaseModel):
    rank: int
    movieId: int
    title: str
    genres: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: Optional[int] = None
    movie_id: Optional[int] = None
    model_name: str
    model_version: str
    recommendations: List[RecommendedMovie]


class FeedbackRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="User ID providing feedback")
    movie_id: int = Field(..., ge=1, description="Movie ID being rated")
    rating: float = Field(..., ge=0.5, le=5.0, description="Rating between 0.5 and 5.0")


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    total_new_feedback: int
    retrain_threshold: int
    retrain_triggered: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_name: str
    model_status: str
    timestamp: str


class ModelStatusResponse(BaseModel):
    production_model: Optional[Dict] = None
    candidate_model: Optional[Dict] = None
    latest_quality_gate: Optional[Dict] = None
    promotion_history: List[Dict] = []


class RetrainRequest(BaseModel):
    n_factors: Optional[int] = 32
    n_epochs: Optional[int] = 20
    learning_rate: Optional[float] = 0.005
    regularization: Optional[float] = 0.05
    force_promote: Optional[bool] = False
