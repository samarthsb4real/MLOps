"""Model training stage supporting baseline and collaborative filtering with MLflow tracking."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import mlflow
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    CANDIDATE_MODEL_DIR,
    PRODUCTION_MODEL_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    CF_FACTORS,
    CF_EPOCHS,
    CF_LEARNING_RATE,
    CF_REGULARIZATION,
    MIN_RATINGS_FOR_POPULARITY,
    RANDOM_STATE,
)
from src.models.base import BaseRecommender
from src.models.popularity import PopularityRecommender
from src.models.collaborative import CollaborativeFilteringRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_model(
    model_type: str = "collaborative",
    params: Optional[Dict] = None,
    data_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = CANDIDATE_MODEL_DIR,
    promote_to_production: bool = False,
) -> BaseRecommender:
    """Train recommendation model, track in MLflow, and save candidate artifact."""
    params = params or {}
    logger.info(f"Loading training data from {data_dir}...")

    train_path = data_dir / "train.csv"
    movies_path = data_dir / "movies.csv"

    if not train_path.exists() or not movies_path.exists():
        from src.pipeline.prepare import prepare_data
        logger.info("Processed data not found; triggering data preparation first...")
        prepare_data()

    train_df = pd.read_csv(train_path)
    movies_df = pd.read_csv(movies_path)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    run_name = f"train_{model_type}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("pipeline_stage", "train")

        if model_type == "popularity":
            min_ratings = params.get("min_ratings", MIN_RATINGS_FOR_POPULARITY)
            mlflow.log_params({"min_ratings": min_ratings, "model_type": "popularity"})
            model = PopularityRecommender(min_ratings=min_ratings)
        elif model_type == "collaborative":
            n_factors = params.get("n_factors", CF_FACTORS)
            n_epochs = params.get("n_epochs", CF_EPOCHS)
            lr = params.get("learning_rate", CF_LEARNING_RATE)
            reg = params.get("regularization", CF_REGULARIZATION)
            seed = params.get("random_state", RANDOM_STATE)

            mlflow.log_params(
                {
                    "model_type": "collaborative",
                    "n_factors": n_factors,
                    "n_epochs": n_epochs,
                    "learning_rate": lr,
                    "regularization": reg,
                    "random_state": seed,
                }
            )
            model = CollaborativeFilteringRecommender(
                n_factors=n_factors,
                n_epochs=n_epochs,
                learning_rate=lr,
                regularization=reg,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'collaborative' or 'popularity'.")

        logger.info(f"Fitting {model_type} model on {len(train_df)} ratings...")
        model.fit(train_df, movies_df)

        # Log training metrics
        meta = model.get_metadata()
        for k, v in meta.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"train_{k}", float(v))

        # Save to candidates directory
        output_dir.mkdir(parents=True, exist_ok=True)
        model_file = output_dir / "model.joblib"
        meta_file = output_dir / "metadata.json"

        model.save(model_file)
        meta["run_id"] = run.info.run_id
        meta["experiment_id"] = run.info.experiment_id
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        mlflow.log_artifact(str(model_file), artifact_path="model")
        mlflow.log_artifact(str(meta_file), artifact_path="model")

        logger.info(f"Trained {model_type} model saved to {model_file} (Run ID: {run.info.run_id})")

        if promote_to_production:
            PRODUCTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save(PRODUCTION_MODEL_DIR / "model.joblib")
            with open(PRODUCTION_MODEL_DIR / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Model directly promoted to production in {PRODUCTION_MODEL_DIR}")

        return model


if __name__ == "__main__":
    train_model(model_type="collaborative")
