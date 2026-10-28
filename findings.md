# Findings: CineTrack MLOps

## Dataset Discovery
- **Location**: `/home/samarth/projects/MLOps/ml-latest-small/`
- **Catalog Size**: 9,742 movies, 100,836 ratings, 610 users.
- **Split Results**:
  - Train Set: 80,672 ratings
  - Held-out Test Set: 20,164 ratings (temporal split per user)
  - Zero future-to-past data leakage.

## Benchmark Model Performance

| Metric | Popularity Baseline | Collaborative Filtering SVD | Quality Gate Floor |
|---|---|---|---|
| **Precision@5** | 0.0609 | 0.0337 | 0.0800 (target) |
| **Recall@5** | 0.0193 | 0.0100 | N/A |
| **NDCG@5** | 0.0680 | 0.0336 | 0.1000 (target) |
| **Catalog Coverage** | 0.0040 (0.4%) | 0.0082 (0.8%) | N/A |
| **Inference Latency** | ~1 ms | ~2 ms | < 50 ms |

## Key Architectural Insights
1. **Popularity vs Personalization Trade-off**:
   - The Popularity baseline recommends global blockbuster movies that many users rated highly, achieving strong Precision@5 on popular movies but only covering 0.4% of the catalog.
   - Collaborative Filtering with SVD latent factors doubles catalog exploration (0.8% coverage) and personalizes recommendations by genre taste and latent feature alignment.
2. **Quality Gate Invariants**:
   - Automated quality gating ensures that candidate models must meet absolute floors and not regress against the incumbent production model.
   - When a candidate model is rejected, the incumbent model remains active in `models/production/` with zero downtime.
3. **MLflow SQLite Backend**:
   - Used `sqlite:///mlflow.db` for reliable, ACID-compliant local experiment tracking, avoiding file-lock deprecations of legacy filestores.
4. **Evidently Drift Detection**:
   - Detects covariate shift between historical training ratings and user feedback using both Kolmogorov-Smirnov statistical tests and Evidently visual reports.
