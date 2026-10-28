"""Model evaluation stage computing Precision@5, Recall@5, and NDCG@5 with MLflow logging."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import mlflow
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    CANDIDATE_MODEL_DIR,
    REPORTS_DIR,
    TOP_K,
    RELEVANCE_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from src.models.base import BaseRecommender
from src.evaluation.metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_saved_model(
    model_dir: Path = CANDIDATE_MODEL_DIR,
    data_dir: Path = PROCESSED_DATA_DIR,
    k: int = TOP_K,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    max_eval_users: Optional[int] = None,
    output_report_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate a saved model against test ratings and track metrics in MLflow."""
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading model from {model_path}...")
    model = BaseRecommender.load(model_path)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("train.csv or test.csv not found in data_dir.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(
        f"Evaluating {model.model_name} on {len(test_df)} test ratings "
        f"(k={k}, relevance_threshold={relevance_threshold})..."
    )

    metrics = evaluate_model(
        model=model,
        train_df=train_df,
        test_df=test_df,
        k=k,
        relevance_threshold=relevance_threshold,
        max_eval_users=max_eval_users,
    )

    logger.info(
        f"Evaluation Metrics: "
        f"Precision@{k}={metrics.get(f'precision@{k}')}, "
        f"Recall@{k}={metrics.get(f'recall@{k}')}, "
        f"NDCG@{k}={metrics.get(f'ndcg@{k}')}, "
        f"Coverage={metrics.get('catalog_coverage')}"
    )

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"evaluate_{model.model_name}"):
        mlflow.set_tag("model_type", model.model_name)
        mlflow.set_tag("pipeline_stage", "evaluate")
        for metric_name, val in metrics.items():
            safe_name = metric_name.replace("@", "_at_")
            mlflow.log_metric(safe_name, float(val))

    # Save evaluation report JSON
    report = {
        "model_name": model.model_name,
        "version": model.version,
        "k": k,
        "relevance_threshold": relevance_threshold,
        "metrics": metrics,
    }

    report_file = output_report_path or (model_dir / "evaluation_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Also write a copy to reports/
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / f"latest_eval_{model.model_name}.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {report_file}")
    return metrics


if __name__ == "__main__":
    evaluate_saved_model()
