"""Centralized configuration for CineTrack MLOps System."""

from pathlib import Path
import os

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEEDBACK_DATA_DIR = DATA_DIR / "feedback"
SOURCE_DATASET_DIR = PROJECT_ROOT / "ml-latest-small"

# Model Directories
MODELS_DIR = PROJECT_ROOT / "models"
PRODUCTION_MODEL_DIR = MODELS_DIR / "production"
CANDIDATE_MODEL_DIR = MODELS_DIR / "candidates"

# Experiment Tracking & Reports
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
REPORTS_DIR = PROJECT_ROOT / "reports"
DRIFT_REPORTS_DIR = REPORTS_DIR / "drift"

# Ensure essential directories exist
for p in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEEDBACK_DATA_DIR,
    PRODUCTION_MODEL_DIR,
    CANDIDATE_MODEL_DIR,
    REPORTS_DIR,
    DRIFT_REPORTS_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

# Recommendation Parameters
TOP_K = 5
RELEVANCE_THRESHOLD = 3.5  # Ratings >= 3.5 are considered relevant items
MIN_RATINGS_FOR_POPULARITY = 20  # Minimum ratings for popularity calculation

# Collaborative Filtering Hyperparameters
CF_FACTORS = 32
CF_EPOCHS = 20
CF_LEARNING_RATE = 0.005
CF_REGULARIZATION = 0.05
RANDOM_STATE = 42

# Quality Gate Promotion Criteria
GATE_MIN_PRECISION = 0.08
GATE_MIN_NDCG = 0.10
GATE_MAX_NDCG_REGRESSION = 0.01  # Candidate cannot regress NDCG@5 by > 1%

# Retraining Triggers
FEEDBACK_TRIGGER_THRESHOLD = 15  # Retrain candidate when >= 15 new user ratings recorded

# MLflow Tracking
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "cinetrack-recsys")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")

