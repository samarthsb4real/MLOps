"""Quality Gate module for candidate model evaluation and automated promotion.

Applies strict criteria:
1. Candidate Precision@5 and NDCG@5 must meet minimum performance floors.
2. Candidate cannot regress NDCG@5 against incumbent by more than tolerance.
3. If candidate fails, the incumbent model remains unchanged in production.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
from typing import Dict, Optional
import mlflow

from src.config import (
    CANDIDATE_MODEL_DIR,
    PRODUCTION_MODEL_DIR,
    GATE_MIN_PRECISION,
    GATE_MIN_NDCG,
    GATE_MAX_NDCG_REGRESSION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from src.pipeline.evaluate import evaluate_saved_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_quality_gate(
    candidate_dir: Path = CANDIDATE_MODEL_DIR,
    production_dir: Path = PRODUCTION_MODEL_DIR,
    force_promote: bool = False,
    max_eval_users: Optional[int] = None,
) -> Dict:
    """Run automated quality gate comparing candidate to incumbent production model."""
    candidate_model_path = candidate_dir / "model.joblib"
    if not candidate_model_path.exists():
        raise FileNotFoundError(f"Candidate model not found at {candidate_model_path}")

    # Evaluate candidate
    logger.info("Evaluating candidate model for quality gate...")
    candidate_metrics = evaluate_saved_model(
        model_dir=candidate_dir,
        max_eval_users=max_eval_users,
    )

    candidate_precision = candidate_metrics.get("precision@5", 0.0)
    candidate_ndcg = candidate_metrics.get("ndcg@5", 0.0)

    # Check if an incumbent model exists in production
    incumbent_model_path = production_dir / "model.joblib"
    has_incumbent = incumbent_model_path.exists()
    incumbent_metrics: Dict[str, float] = {}

    if has_incumbent:
        logger.info("Evaluating incumbent production model for baseline comparison...")
        incumbent_metrics = evaluate_saved_model(
            model_dir=production_dir,
            max_eval_users=max_eval_users,
            output_report_path=production_dir / "evaluation_report.json",
        )
        incumbent_precision = incumbent_metrics.get("precision@5", 0.0)
        incumbent_ndcg = incumbent_metrics.get("ndcg@5", 0.0)
    else:
        logger.info("No incumbent production model found; initial bootstrap run.")
        incumbent_precision = 0.0
        incumbent_ndcg = 0.0

    # Apply quality gate criteria
    reasons = []
    passed = True

    if not force_promote:
        if candidate_precision < GATE_MIN_PRECISION:
            passed = False
            reasons.append(
                f"Candidate Precision@5 ({candidate_precision:.4f}) below required floor ({GATE_MIN_PRECISION:.4f})"
            )

        if candidate_ndcg < GATE_MIN_NDCG:
            passed = False
            reasons.append(
                f"Candidate NDCG@5 ({candidate_ndcg:.4f}) below required floor ({GATE_MIN_NDCG:.4f})"
            )

        if has_incumbent:
            ndcg_delta = candidate_ndcg - incumbent_ndcg
            if ndcg_delta < -GATE_MAX_NDCG_REGRESSION:
                passed = False
                reasons.append(
                    f"Candidate NDCG@5 ({candidate_ndcg:.4f}) regressed against incumbent ({incumbent_ndcg:.4f}) "
                    f"by {abs(ndcg_delta):.4f} > allowed tolerance ({GATE_MAX_NDCG_REGRESSION:.4f})"
                )

    if force_promote:
        passed = True
        reasons.append("Force promotion explicitly requested.")
    elif passed:
        reasons.append("All evaluation quality gate criteria satisfied.")

    decision_status = "PROMOTED" if passed else "REJECTED"
    now_iso = datetime.now(timezone.utc).isoformat()

    decision_record = {
        "timestamp": now_iso,
        "decision": decision_status,
        "passed": passed,
        "reasons": reasons,
        "candidate_metrics": candidate_metrics,
        "incumbent_metrics": incumbent_metrics,
    }

    # Setup MLflow logging for quality gate decision
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"quality_gate_{decision_status.lower()}"):
        mlflow.set_tag("pipeline_stage", "quality_gate")
        mlflow.set_tag("gate_decision", decision_status)
        mlflow.log_param("passed", str(passed))
        mlflow.log_metrics(
            {
                "candidate_precision_at_5": candidate_precision,
                "candidate_ndcg_at_5": candidate_ndcg,
                "incumbent_precision_at_5": incumbent_precision,
                "incumbent_ndcg_at_5": incumbent_ndcg,
            }
        )

    # Perform or withhold promotion
    if passed:
        logger.info(f"Quality gate PASSED. Promoting candidate to production at {production_dir}...")
        production_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_model_path, production_dir / "model.joblib")

        # Copy metadata if available
        candidate_meta = candidate_dir / "metadata.json"
        if candidate_meta.exists():
            shutil.copy2(candidate_meta, production_dir / "metadata.json")

        # Update promotion history
        history_file = production_dir / "promotion_history.json"
        history = []
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(decision_record)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

    else:
        logger.warning(
            f"Quality gate REJECTED. Incumbent model remains in production. Reasons: {reasons}"
        )
        rejection_file = candidate_dir / "rejection_history.json"
        rejections = []
        if rejection_file.exists():
            try:
                with open(rejection_file, "r") as f:
                    rejections = json.load(f)
            except Exception:
                rejections = []
        rejections.append(decision_record)
        with open(rejection_file, "w") as f:
            json.dump(rejections, f, indent=2)

    # Save latest decision summary
    with open(candidate_dir / "latest_quality_gate.json", "w") as f:
        json.dump(decision_record, f, indent=2)

    return decision_record


if __name__ == "__main__":
    run_quality_gate()
