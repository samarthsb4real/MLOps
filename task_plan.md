# Task Plan: CineTrack - MLOps-Based Movie Recommendation System

## Goal
Build a production-grade movie recommendation web application demonstrating a complete, reproducible MLOps lifecycle: data validation, baseline & collaborative filtering models (Precision@5, Recall@5, NDCG@5), DVC data pipeline, MLflow experiment tracking, FastAPI service, Streamlit Movie Explorer & MLOps dashboard, Evidently drift monitoring, automated retraining with quality-gated model promotion, Docker containerization, CI/CD pipeline, and Cloud Run deployment.

## Current Phase
All Phases Complete (Phase 1 through Phase 5)

## Next Step
Present the completed implementation, reproduction instructions, evaluation metrics, and demonstration walkthrough to the user.

## Detailed Phase Breakdown

- [x] **Phase 1: Build and evaluate the recommendation models**
  - [x] Inspect MovieLens Small dataset in `ml-latest-small/`
  - [x] Create modular project directory layout (`src/data/`, `src/models/`, `src/evaluation/`, `tests/`)
  - [x] Implement data ingestion and validation module (`src/data/loader.py`, `src/data/validator.py`)
  - [x] Implement reproducible train/test split preventing temporal/user leakage (`src/data/split.py`)
  - [x] Implement baseline model: Popularity-based recommendation (`src/models/popularity.py`)
  - [x] Implement primary model: Collaborative filtering via Matrix Factorization / SVD (`src/models/collaborative.py`)
  - [x] Implement evaluation metrics: Precision@5, Recall@5, NDCG@5 (`src/evaluation/metrics.py`)
  - [x] Run benchmark evaluation comparing Popularity vs Collaborative Filtering
  - [x] Write unit tests for Phase 1 (`tests/test_models.py`, `tests/test_metrics.py`, `tests/test_data.py`) and verify passing

- [x] **Phase 2: Implement DVC, MLflow, reproducible training, and automated tests**
  - [x] Install and configure MLflow with SQLite backend and DVC (`dvc.yaml`, `params.yaml`, `dvc.lock`)
  - [x] Setup DVC data pipeline stages (`prepare` -> `train` -> `evaluate`)
  - [x] Integrate MLflow tracking: log hyperparams, Precision@5/Recall@5/NDCG@5, model artifacts
  - [x] Implement model persistence and metadata tracking (`models/production/`, `models/candidates/`)
  - [x] Verify DVC pipeline reproduction (`dvc repro` and `dvc metrics show`)
  - [x] Add comprehensive tests for data pipeline and model persistence (`tests/test_pipeline.py`)

- [x] **Phase 3: Build the FastAPI service and Streamlit application**
  - [x] Implement FastAPI backend (`src/api/app.py`):
    - `GET /health`: service health & active model metadata
    - `GET /movies`: searchable, filterable movie catalog
    - `GET /movies/{id}`: detailed movie metadata
    - `POST /recommend`: personalized top-5 recommendations
    - `POST /feedback`: submit user ratings/feedback
    - `GET /metrics`: production model performance metrics
    - `POST /retrain`: trigger retraining pipeline
    - `GET /model/status`: production vs candidate status
  - [x] Write API integration tests (`tests/test_api.py`)
  - [x] Build Streamlit Frontend (`src/ui/app.py`):
    - Dark-themed modern UI
    - Movie Explorer: search, genre filter, view details, get 5 recommendations, rate movies
    - MLOps Dashboard: model comparisons, metrics, drift summary, manual retrain trigger, quality gate decisions
  - [x] Test FastAPI and Streamlit locally together

- [x] **Phase 4: Implement monitoring, candidate retraining, and quality-gated model promotion**
  - [x] Implement feedback storage and tracking (`data/feedback/`)
  - [x] Implement Evidently drift detector (`src/monitoring/drift.py`) analyzing rating distributions and user activity
  - [x] Implement automated quality gate (`src/pipeline/quality_gate.py`):
    - Compare candidate vs incumbent model
    - Criteria: Candidate NDCG@5 >= Incumbent NDCG@5 - tolerance AND Precision@5 >= threshold
    - Promote candidate if passed; keep incumbent if failed
  - [x] Connect retrain pipeline to feedback buffer and quality gate
  - [x] Add tests for retraining trigger and quality gate decisions (`tests/test_promotion.py`)

- [x] **Phase 5: Configure GitHub Actions, Docker containerisation, and documentation**
  - [x] Create Dockerfile and `docker-compose.yml` for unified local execution
  - [x] Setup GitHub Actions workflow (`.github/workflows/ci-cd.yml`) for linting, testing, quality gate check, Docker build, Cloud Run deploy
  - [x] Provide Google Cloud Run deployment configuration and script (`deploy/cloud_run_deploy.sh`)
  - [x] Create comprehensive `README.md` and `ARCHITECTURE.md` with system diagrams, API specs, and reproduction steps
  - [x] Verify end-to-end user workflow: Movie Explorer -> Submit feedback -> Retrain -> Quality gate -> Monitor drift

## Key Decisions & Heuristics
- *Dataset*: Using provided MovieLens Small (`ml-latest-small/`) containing 100,836 ratings, 9,742 movies, 610 users.
- *Evaluation Protocol*: Temporal split per user with latest 20% ratings held out for testing. Compute Precision@5, Recall@5, and NDCG@5 over relevance threshold (ratings >= 3.5).
- *Self-Contained CF*: Biased matrix factorization with user/item bias and SVD formulation using numpy/scipy/scikit-learn for high performance, zero external C++ dependencies, and 100% reproducibility across environments.
- *Quality Gate Invariant*: An inferior candidate NEVER replaces the incumbent production model.

## Errors Encountered & Resolutions
| Error | Attempt | Resolution |
|-------|---------|------------|
| Root git repo conflict | 1 | Ran `git init` in `/home/samarth/projects/MLOps` to isolate repository |
| MLflow filesystem maintenance mode | 1 | Configured `MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"` for robust database tracking |
| MLflow metric name @ character | 1 | Sanitized metric keys replacing `@` with `_at_` |
| Evidently pandas 3.0 incompatibility | 1 | Pinned `pandas<3.0.0` (2.3.3) for full Evidently compatibility |
