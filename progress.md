# Progress Log: CineTrack

## Session: Complete End-to-End MLOps Implementation

### Phase 1: Models & Evaluation
- Validated MovieLens Small dataset (9,742 movies, 100,836 ratings, 610 users).
- Built leak-free temporal split module holding out latest 20% ratings per user.
- Built Popularity baseline recommender (IMDB weighted rating formula).
- Built Collaborative Filtering recommender using Biased Matrix Factorization (Koren SVD) with user/item biases and latent vectors.
- Built Precision@5, Recall@5, and NDCG@5 metrics calculation.
- Evaluated both models on 20,164 test ratings across 594 test users.

### Phase 2: DVC, MLflow & Pipeline
- Configured MLflow experiment tracking using SQLite backend (`sqlite:///mlflow.db`).
- Configured DVC pipeline (`prepare` -> `train` -> `evaluate`) in `dvc.yaml` and `params.yaml`.
- Successfully reproduced pipeline via `dvc repro` and verified metrics with `dvc metrics show`.

### Phase 3: FastAPI Backend & Streamlit Frontend
- Built FastAPI service (`src/api/app.py`) with 9 endpoints for browsing, searching, personalized recommendations, feedback ingestion, metrics, retrain triggering, and drift reporting.
- Built sleek dark-themed Streamlit web app (`src/ui/app.py`) with Movie Explorer and MLOps Dashboard.
- Tested API endpoints and frontend integration.

### Phase 4: Monitoring, Retraining & Quality Gate
- Built real-time user feedback ingestion (`data/feedback/`).
- Built Evidently AI data drift detector with KS-test and Wasserstein distance (`src/monitoring/drift.py`).
- Built automated Quality Gate (`src/pipeline/quality_gate.py`) enforcing regression prevention.
- Added automated tests proving candidate rejection retains incumbent production model.

### Phase 5: Containerization, CI/CD & Cloud Deployment
- Created production `Dockerfile` and `docker-compose.yml`.
- Configured GitHub Actions CI/CD workflow (`.github/workflows/ci-cd.yml`).
- Created Google Cloud Run automated deployment script (`deploy/cloud_run_deploy.sh`).
- Authored comprehensive `README.md` and `ARCHITECTURE.md`.
- Executed full test suite: 26 passed with 82% coverage.
