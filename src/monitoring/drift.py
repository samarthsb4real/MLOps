"""Data drift monitoring using Evidently and statistical divergence analysis."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from src.config import (
    PROCESSED_DATA_DIR,
    FEEDBACK_DATA_DIR,
    DRIFT_REPORTS_DIR,
)
from src.data.loader import load_feedback_ratings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_drift_analysis(
    reference_path: Optional[Path] = None,
    current_df: Optional[pd.DataFrame] = None,
    output_dir: Path = DRIFT_REPORTS_DIR,
    drift_p_value_threshold: float = 0.05,
) -> Dict:
    """Run data drift analysis between reference training ratings and current feedback ratings.

    Uses Evidently to generate visual HTML reports and SciPy for exact KS-test and Wasserstein distance.
    """
    ref_file = reference_path or (PROCESSED_DATA_DIR / "train.csv")
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference dataset not found at {ref_file}")

    ref_df = pd.read_csv(ref_file)

    # If current feedback is not provided, load from data/feedback/
    if current_df is None or current_df.empty:
        current_df = load_feedback_ratings()

    # If feedback is empty or very small, create a comparison window using recent test data
    used_fallback_window = False
    if current_df.empty or len(current_df) < 5:
        test_file = PROCESSED_DATA_DIR / "test.csv"
        if test_file.exists():
            test_df = pd.read_csv(test_file)
            current_df = test_df.sample(min(200, len(test_df)), random_state=42)
            used_fallback_window = True
        else:
            current_df = ref_df.sample(min(200, len(ref_df)), random_state=42)
            used_fallback_window = True

    output_dir.mkdir(parents=True, exist_ok=True)
    html_report_path = output_dir / "drift_report.html"
    json_summary_path = output_dir / "drift_summary.json"

    # Compute statistical tests
    ref_ratings = ref_df["rating"].to_numpy(dtype=float)
    curr_ratings = current_df["rating"].to_numpy(dtype=float)

    ks_stat, ks_p_val = ks_2samp(ref_ratings, curr_ratings)
    w_dist = float(wasserstein_distance(ref_ratings, curr_ratings))

    drift_detected = bool(ks_p_val < drift_p_value_threshold)

    # Run Evidently Report
    evidently_success = False
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset

        cols_to_compare = ["rating"]
        if "timestamp" in ref_df.columns and "timestamp" in current_df.columns:
            cols_to_compare.append("timestamp")

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=ref_df[cols_to_compare],
            current_data=current_df[cols_to_compare],
        )
        report.save_html(str(html_report_path))
        evidently_success = True
        logger.info(f"Evidently interactive drift report saved to {html_report_path}")
    except Exception as e:
        logger.warning(f"Evidently visual report could not be generated: {e}")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_detected": drift_detected,
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": round(float(ks_p_val), 6),
        "wasserstein_distance": round(w_dist, 4),
        "p_value_threshold": drift_p_value_threshold,
        "reference_stats": {
            "count": len(ref_ratings),
            "mean": round(float(np.mean(ref_ratings)), 3),
            "std": round(float(np.std(ref_ratings)), 3),
            "min": float(np.min(ref_ratings)),
            "max": float(np.max(ref_ratings)),
        },
        "current_stats": {
            "count": len(curr_ratings),
            "mean": round(float(np.mean(curr_ratings)), 3),
            "std": round(float(np.std(curr_ratings)), 3),
            "min": float(np.min(curr_ratings)),
            "max": float(np.max(curr_ratings)),
        },
        "used_fallback_window": used_fallback_window,
        "evidently_html_available": evidently_success,
        "html_report_path": str(html_report_path),
    }

    with open(json_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Drift analysis complete: drift_detected={drift_detected}, "
        f"p_value={ks_p_val:.6f}, wasserstein={w_dist:.4f}"
    )
    return summary


def get_latest_drift_summary() -> Dict:
    """Read the latest drift summary from disk, or run analysis if missing."""
    summary_path = DRIFT_REPORTS_DIR / "drift_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return run_drift_analysis()


if __name__ == "__main__":
    run_drift_analysis()
