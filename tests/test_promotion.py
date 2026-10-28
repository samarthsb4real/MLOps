"""Automated tests for quality-gated model promotion and retention invariants."""

import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd

from src.pipeline.quality_gate import run_quality_gate
from src.models.collaborative import CollaborativeFilteringRecommender
from src.models.popularity import PopularityRecommender


def test_quality_gate_preserves_incumbent_on_failure():
    """Verify invariant: Incumbent model is retained when candidate fails quality gate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cand_dir = Path(tmpdir) / "candidate"
        prod_dir = Path(tmpdir) / "production"
        cand_dir.mkdir(parents=True)
        prod_dir.mkdir(parents=True)

        # 1. Setup an incumbent model in production
        incumbent = CollaborativeFilteringRecommender(n_factors=8, n_epochs=5)
        train_df = pd.read_csv("data/processed/train.csv")
        movies_df = pd.read_csv("data/processed/movies.csv")
        incumbent.fit(train_df.head(200), movies_df.head(200))
        incumbent.save(prod_dir / "model.joblib")

        incumbent_meta = {"model_name": "CollaborativeFilteringSVD", "version": "1.0.0"}
        with open(prod_dir / "metadata.json", "w") as f:
            json.dump(incumbent_meta, f)

        initial_prod_bytes = (prod_dir / "model.joblib").read_bytes()

        # 2. Create a severely underperforming candidate that will fail the quality gate
        failing_candidate = PopularityRecommender(min_ratings=999999)
        # Empty / disconnected fit
        dummy_df = pd.DataFrame({"userId": [1], "movieId": [999999], "rating": [1.0], "timestamp": [1000]})
        failing_candidate.fit(dummy_df, movies_df.head(10))
        failing_candidate.save(cand_dir / "model.joblib")

        # 3. Run quality gate with strict criteria (force_promote=False)
        decision = run_quality_gate(
            candidate_dir=cand_dir,
            production_dir=prod_dir,
            force_promote=False,
            max_eval_users=30,
        )

        # 4. Verify candidate was REJECTED
        assert decision["passed"] is False
        assert decision["decision"] == "REJECTED"
        assert len(decision["reasons"]) > 0

        # 5. Verify incumbent was NOT overwritten
        current_prod_bytes = (prod_dir / "model.joblib").read_bytes()
        assert initial_prod_bytes == current_prod_bytes

        # 6. Verify rejection record was logged
        rejection_file = cand_dir / "rejection_history.json"
        assert rejection_file.exists()
        with open(rejection_file, "r") as f:
            rejections = json.load(f)
        assert len(rejections) == 1
        assert rejections[0]["decision"] == "REJECTED"


def test_quality_gate_promotes_qualified_candidate():
    """Verify promotion of qualified candidate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cand_dir = Path(tmpdir) / "candidate"
        prod_dir = Path(tmpdir) / "production"
        cand_dir.mkdir(parents=True)
        prod_dir.mkdir(parents=True)

        train_df = pd.read_csv("data/processed/train.csv")
        movies_df = pd.read_csv("data/processed/movies.csv")

        # Qualified candidate
        qualified_model = CollaborativeFilteringRecommender(n_factors=16, n_epochs=10)
        qualified_model.fit(train_df, movies_df)
        qualified_model.save(cand_dir / "model.joblib")
        with open(cand_dir / "metadata.json", "w") as f:
            json.dump({"model_name": "CollaborativeFilteringSVD", "version": "1.1.0"}, f)

        # Run quality gate with force_promote=True or satisfied threshold
        decision = run_quality_gate(
            candidate_dir=cand_dir,
            production_dir=prod_dir,
            force_promote=True,
            max_eval_users=30,
        )

        assert decision["passed"] is True
        assert decision["decision"] == "PROMOTED"
        assert (prod_dir / "model.joblib").exists()

        history_file = prod_dir / "promotion_history.json"
        assert history_file.exists()
        with open(history_file, "r") as f:
            history = json.load(f)
        assert len(history) == 1
        assert history[0]["decision"] == "PROMOTED"
