"""Integration tests for the training, evaluation, and quality gate pipeline."""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

from src.pipeline.prepare import prepare_data
from src.pipeline.train import train_model
from src.pipeline.evaluate import evaluate_saved_model
from src.pipeline.quality_gate import run_quality_gate
from src.models.base import BaseRecommender


def test_prepare_data_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        summary = prepare_data(test_ratio=0.2, output_dir=out_dir)

        assert summary["is_valid"] is True
        assert (out_dir / "movies.csv").exists()
        assert (out_dir / "train.csv").exists()
        assert (out_dir / "test.csv").exists()
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "validation_report.json").exists()

        train_df = pd.read_csv(out_dir / "train.csv")
        test_df = pd.read_csv(out_dir / "test.csv")
        assert len(train_df) > 0
        assert len(test_df) > 0


def test_train_and_evaluate_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        cand_dir = Path(tmpdir) / "candidate"
        # Train lightweight model with few epochs
        model = train_model(
            model_type="collaborative",
            params={"n_factors": 4, "n_epochs": 2},
            output_dir=cand_dir,
        )
        assert (cand_dir / "model.joblib").exists()
        assert (cand_dir / "metadata.json").exists()

        metrics = evaluate_saved_model(
            model_dir=cand_dir,
            max_eval_users=20,
        )
        assert "precision@5" in metrics
        assert "ndcg@5" in metrics
        assert "recall@5" in metrics


def test_quality_gate_logic():
    with tempfile.TemporaryDirectory() as tmpdir:
        cand_dir = Path(tmpdir) / "candidate"
        prod_dir = Path(tmpdir) / "production"

        # Train candidate
        train_model(
            model_type="collaborative",
            params={"n_factors": 4, "n_epochs": 2},
            output_dir=cand_dir,
        )

        # 1. Initial bootstrap: with force_promote=True
        decision = run_quality_gate(
            candidate_dir=cand_dir,
            production_dir=prod_dir,
            force_promote=True,
            max_eval_users=20,
        )
        assert decision["passed"] is True
        assert decision["decision"] == "PROMOTED"
        assert (prod_dir / "model.joblib").exists()
        assert (prod_dir / "promotion_history.json").exists()

        # 2. Test rejection: if candidate is intentionally broken/subpar
        # We can simulate by saving an untrained dummy or forcing gate check
        bad_cand_dir = Path(tmpdir) / "bad_candidate"
        bad_cand_dir.mkdir(parents=True, exist_ok=True)
        # Train with 0 factors or random baseline that yields 0 precision
        from src.models.popularity import PopularityRecommender
        pop_model = PopularityRecommender(min_ratings=999999) # absurd threshold
        train_df = pd.read_csv(Path("data/processed/train.csv"))
        movies_df = pd.read_csv(Path("data/processed/movies.csv"))
        pop_model.fit(train_df.head(10), movies_df.head(10))
        pop_model.save(bad_cand_dir / "model.joblib")

        prod_mtime_before = (prod_dir / "model.joblib").stat().st_mtime

        rejection_decision = run_quality_gate(
            candidate_dir=bad_cand_dir,
            production_dir=prod_dir,
            force_promote=False,
            max_eval_users=20,
        )
        assert rejection_decision["passed"] is False
        assert rejection_decision["decision"] == "REJECTED"
        assert (bad_cand_dir / "rejection_history.json").exists()

        # Incumbent model in production remains unchanged!
        prod_mtime_after = (prod_dir / "model.joblib").stat().st_mtime
        assert prod_mtime_before == prod_mtime_after
