"""FastAPI Recommendation and MLOps Service for CineTrack."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional
import uuid
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PRODUCTION_MODEL_DIR,
    CANDIDATE_MODEL_DIR,
    PROCESSED_DATA_DIR,
    FEEDBACK_DATA_DIR,
    REPORTS_DIR,
    DRIFT_REPORTS_DIR,
    FEEDBACK_TRIGGER_THRESHOLD,
    TOP_K,
)
from src.models.base import BaseRecommender
from src.models.collaborative import CollaborativeFilteringRecommender
from src.models.popularity import PopularityRecommender
from src.data.loader import load_movies, load_feedback_ratings
from src.pipeline.train import train_model
from src.pipeline.quality_gate import run_quality_gate
from src.monitoring.drift import run_drift_analysis, get_latest_drift_summary
from src.api.schemas import (
    MovieItem,
    MovieListResponse,
    RecommendationRequest,
    RecommendedMovie,
    RecommendationResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelStatusResponse,
    RetrainRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global in-memory cache for models and catalog
app_state: Dict = {
    "production_model": None,
    "movies_df": None,
    "movies_dict": {},
    "genres_list": [],
}


def load_cached_catalog() -> pd.DataFrame:
    """Load and cache movies catalog with rating count and mean."""
    movies_file = PROCESSED_DATA_DIR / "movies.csv"
    train_file = PROCESSED_DATA_DIR / "train.csv"

    if movies_file.exists():
        movies_df = pd.read_csv(movies_file)
    else:
        movies_df = load_movies()

    # Enrich with rating counts and means if train.csv exists
    if train_file.exists():
        train_df = pd.read_csv(train_file)
        stats = train_df.groupby("movieId").agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean"),
        ).reset_index()
        movies_df = pd.merge(movies_df, stats, on="movieId", how="left")
        movies_df["rating_count"] = movies_df["rating_count"].fillna(0).astype(int)
        movies_df["rating_mean"] = movies_df["rating_mean"].fillna(0.0).round(2)
    else:
        movies_df["rating_count"] = 0
        movies_df["rating_mean"] = 0.0

    return movies_df


def load_active_production_model() -> Optional[BaseRecommender]:
    """Load active production model from disk."""
    prod_path = PRODUCTION_MODEL_DIR / "model.joblib"
    if prod_path.exists():
        try:
            model = BaseRecommender.load(prod_path)
            logger.info(f"Loaded production model: {model.model_name} (v{model.version})")
            return model
        except Exception as e:
            logger.error(f"Error loading production model: {e}")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    logger.info("Initializing CineTrack API application state...")
    movies_df = load_cached_catalog()
    app_state["movies_df"] = movies_df
    app_state["movies_dict"] = movies_df.set_index("movieId").to_dict(orient="index")

    # Extract all unique genres
    genres = set()
    for g_str in movies_df["genres"].dropna():
        for g in str(g_str).split("|"):
            if g and g != "(no genres listed)":
                genres.add(g.strip())
    app_state["genres_list"] = sorted(list(genres))

    # Load active model
    app_state["production_model"] = load_active_production_model()
    if app_state["production_model"] is None:
        logger.warning("No production model found on startup. Initializing fallback popularity recommender...")
        fallback_model = PopularityRecommender(min_ratings=5)
        train_file = PROCESSED_DATA_DIR / "train.csv"
        if train_file.exists():
            train_df = pd.read_csv(train_file)
            fallback_model.fit(train_df, movies_df)
            app_state["production_model"] = fallback_model

    yield
    logger.info("CineTrack API shutting down.")


app = FastAPI(
    title="CineTrack Recommendation API",
    description="Production-grade movie recommendation API with MLOps pipeline, monitoring, and quality gating.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Service health check returning active model name and system status."""
    model = app_state.get("production_model")
    model_name = model.model_name if model else "None"
    model_status = "production" if model else "unavailable"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_name=model_name,
        model_status=model_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/genres", response_model=List[str], tags=["Catalog"])
def get_genres() -> List[str]:
    """Return all unique movie genres available in the catalog."""
    return app_state.get("genres_list", [])


@app.get("/movies", response_model=MovieListResponse, tags=["Catalog"])
def list_movies(
    search: Optional[str] = Query(None, description="Search by movie title"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
) -> MovieListResponse:
    """Browse, search, and filter movies from the catalog with pagination."""
    movies_df: pd.DataFrame = app_state.get("movies_df")
    if movies_df is None or movies_df.empty:
        movies_df = load_cached_catalog()
        app_state["movies_df"] = movies_df

    filtered = movies_df.copy()

    if search:
        filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]

    if genre and genre.lower() != "all":
        filtered = filtered[filtered["genres"].str.contains(genre, case=False, na=False)]

    total = len(filtered)
    page_df = filtered.iloc[skip : skip + limit]

    items = []
    for _, row in page_df.iterrows():
        items.append(
            MovieItem(
                movieId=int(row["movieId"]),
                title=str(row["title"]),
                genres=str(row["genres"]),
                year=int(row["year"]) if pd.notnull(row.get("year")) else None,
                rating_count=int(row.get("rating_count", 0)),
                rating_mean=float(row.get("rating_mean", 0.0)),
            )
        )

    return MovieListResponse(total=total, skip=skip, limit=limit, movies=items)


@app.get("/movies/{movie_id}", response_model=MovieItem, tags=["Catalog"])
def get_movie_detail(movie_id: int) -> MovieItem:
    """Fetch details for a specific movie by ID."""
    movies_dict = app_state.get("movies_dict", {})
    if movie_id not in movies_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with ID {movie_id} not found.",
        )

    row = movies_dict[movie_id]
    return MovieItem(
        movieId=movie_id,
        title=str(row["title"]),
        genres=str(row["genres"]),
        year=int(row["year"]) if pd.notnull(row.get("year")) else None,
        rating_count=int(row.get("rating_count", 0)),
        rating_mean=float(row.get("rating_mean", 0.0)),
    )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(req: RecommendationRequest) -> RecommendationResponse:
    """Generate personalized recommendations for a user or similar movies for an item."""
    model = app_state.get("production_model")

    # If specific model version requested
    if req.model_version == "candidate":
        cand_path = CANDIDATE_MODEL_DIR / "model.joblib"
        if cand_path.exists():
            model = BaseRecommender.load(cand_path)
    elif req.model_version == "baseline":
        baseline_path = Path("models/baseline/model.joblib")
        if baseline_path.exists():
            model = BaseRecommender.load(baseline_path)

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No recommendation model is currently available.",
        )

    try:
        raw_recs = model.recommend(
            user_id=req.user_id,
            item_id=req.movie_id,
            top_k=req.top_k,
            genre=req.genre,
            exclude_rated=True,
        )
    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation computation failed: {str(e)}",
        )

    recs = [
        RecommendedMovie(
            rank=r["rank"],
            movieId=r["movieId"],
            title=r["title"],
            genres=r["genres"],
            score=r["score"],
        )
        for r in raw_recs
    ]

    return RecommendationResponse(
        user_id=req.user_id,
        movie_id=req.movie_id,
        model_name=model.model_name,
        model_version=model.version,
        recommendations=recs,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback & Monitoring"])
def submit_feedback(fb: FeedbackRequest) -> FeedbackResponse:
    """Record user rating feedback and trigger retraining if threshold reached."""
    FEEDBACK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    feedback_file = FEEDBACK_DATA_DIR / "ratings_feedback.csv"

    now_ts = int(datetime.now(timezone.utc).timestamp())
    feedback_id = str(uuid.uuid4())[:8]

    new_row = pd.DataFrame(
        [
            {
                "userId": fb.user_id,
                "movieId": fb.movie_id,
                "rating": fb.rating,
                "timestamp": now_ts,
                "feedback_id": feedback_id,
            }
        ]
    )

    if feedback_file.exists():
        new_row.to_csv(feedback_file, mode="a", header=False, index=False)
    else:
        new_row.to_csv(feedback_file, mode="w", header=True, index=False)

    # Check total feedback count
    total_feedback = len(load_feedback_ratings())
    retrain_triggered = False

    if total_feedback >= FEEDBACK_TRIGGER_THRESHOLD:
        retrain_triggered = True
        logger.info(
            f"Feedback threshold reached ({total_feedback}/{FEEDBACK_TRIGGER_THRESHOLD}). "
            f"Automated retraining triggered."
        )
        try:
            # Trigger retrain and quality gate
            train_model(model_type="collaborative")
            decision = run_quality_gate()
            if decision["passed"]:
                app_state["production_model"] = load_active_production_model()
        except Exception as e:
            logger.error(f"Automated retraining encountered an error: {e}")

    return FeedbackResponse(
        status="success",
        feedback_id=feedback_id,
        total_new_feedback=total_feedback,
        retrain_threshold=FEEDBACK_TRIGGER_THRESHOLD,
        retrain_triggered=retrain_triggered,
        message="Feedback successfully recorded.",
    )


@app.get("/model/status", response_model=ModelStatusResponse, tags=["MLOps"])
def get_model_status() -> ModelStatusResponse:
    """Return status of production model, candidate model, and quality gate decisions."""
    prod_meta = {}
    prod_meta_file = PRODUCTION_MODEL_DIR / "metadata.json"
    if prod_meta_file.exists():
        with open(prod_meta_file, "r") as f:
            prod_meta = json.load(f)

    cand_meta = {}
    cand_meta_file = CANDIDATE_MODEL_DIR / "metadata.json"
    if cand_meta_file.exists():
        with open(cand_meta_file, "r") as f:
            cand_meta = json.load(f)

    latest_gate = {}
    latest_gate_file = CANDIDATE_MODEL_DIR / "latest_quality_gate.json"
    if latest_gate_file.exists():
        with open(latest_gate_file, "r") as f:
            latest_gate = json.load(f)

    history = []
    history_file = PRODUCTION_MODEL_DIR / "promotion_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)

    return ModelStatusResponse(
        production_model=prod_meta,
        candidate_model=cand_meta,
        latest_quality_gate=latest_gate,
        promotion_history=history,
    )


@app.get("/metrics", tags=["MLOps"])
def get_metrics() -> Dict:
    """Return latest evaluation metrics for the production model."""
    eval_file = PRODUCTION_MODEL_DIR / "evaluation_report.json"
    if not eval_file.exists():
        eval_file = CANDIDATE_MODEL_DIR / "evaluation_report.json"

    if eval_file.exists():
        with open(eval_file, "r") as f:
            return json.load(f)
    return {"message": "No evaluation report found yet."}


@app.post("/retrain", tags=["MLOps"])
def trigger_retraining(req: RetrainRequest) -> Dict:
    """Trigger candidate model retraining and execute automated quality gate evaluation."""
    logger.info("Manual retraining requested via API.")
    params = {
        "n_factors": req.n_factors,
        "n_epochs": req.n_epochs,
        "learning_rate": req.learning_rate,
        "regularization": req.regularization,
    }

    try:
        # 1. Train candidate
        train_model(model_type="collaborative", params=params)

        # 2. Run quality gate
        decision = run_quality_gate(force_promote=req.force_promote or False)

        # 3. If passed, reload production model
        if decision["passed"]:
            app_state["production_model"] = load_active_production_model()

        return {
            "status": "success",
            "decision": decision["decision"],
            "passed": decision["passed"],
            "reasons": decision["reasons"],
            "candidate_metrics": decision["candidate_metrics"],
            "incumbent_metrics": decision["incumbent_metrics"],
            "timestamp": decision["timestamp"],
        }
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining process failed: {str(e)}",
        )


@app.get("/monitoring/drift", tags=["Monitoring"])
def get_drift() -> Dict:
    """Return data drift summary comparing incoming feedback vs training baseline."""
    return get_latest_drift_summary()


@app.get("/monitoring/drift/report", response_class=HTMLResponse, tags=["Monitoring"])
def get_drift_report_html():
    """Serve interactive Evidently HTML drift report."""
    report_file = DRIFT_REPORTS_DIR / "drift_report.html"
    if not report_file.exists():
        run_drift_analysis()

    if report_file.exists():
        return HTMLResponse(content=report_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>Drift report not available yet.</h3>")
