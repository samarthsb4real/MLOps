"""CineTrack Streamlit Application: Movie Explorer & MLOps Dashboard."""

import json
from pathlib import Path
import sqlite3
import sys
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PRODUCTION_MODEL_DIR,
    CANDIDATE_MODEL_DIR,
    DRIFT_REPORTS_DIR,
    FEEDBACK_DATA_DIR,
    MLFLOW_DB_PATH,
    TOP_K,
    FEEDBACK_TRIGGER_THRESHOLD,
    GATE_MIN_PRECISION,
    GATE_MIN_NDCG,
    GATE_MAX_NDCG_REGRESSION,
)
from src.data.loader import load_movies, load_feedback_ratings
from src.pipeline.train import train_model
from src.pipeline.quality_gate import run_quality_gate
from src.monitoring.drift import run_drift_analysis, get_latest_drift_summary

# Page configuration
st.set_page_config(
    page_title="CineTrack | MLOps Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-Themed Custom CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Cards */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8) 0%, rgba(33, 38, 45, 0.9) 100%);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    
    .movie-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.2s ease;
    }
    
    .movie-card:hover {
        border-color: #79c0ff;
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.15);
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    .badge-genre {
        background: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.4);
    }
    
    .badge-rank {
        background: rgba(210, 153, 34, 0.2);
        color: #e3b341;
        border: 1px solid rgba(210, 153, 34, 0.4);
    }
    
    .badge-promoted {
        background: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
    }
    
    .badge-rejected {
        background: rgba(248, 81, 73, 0.2);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.4);
    }
    
    .hero-header {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600)
def get_cached_movies():
    """Cache movies catalog in Streamlit memory."""
    return load_movies()


@st.cache_resource
def get_cached_production_model():
    """Load and cache the current production model."""
    prod_path = PRODUCTION_MODEL_DIR / "model.joblib"
    if prod_path.exists():
        from src.models.base import BaseRecommender
        return BaseRecommender.load(prod_path)
    return None


movies_df = get_cached_movies()
prod_model = get_cached_production_model()

# Extract unique genres
all_genres = set()
for g_str in movies_df["genres"].dropna():
    for g in str(g_str).split("|"):
        if g and g != "(no genres listed)":
            all_genres.add(g.strip())
genre_list = ["All"] + sorted(list(all_genres))


# Sidebar Navigation
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/film.svg",
        width=50,
    )
    st.title("CineTrack")
    st.caption("Production MLOps Movie Recommender")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "🎬 Movie Explorer",
            "📊 MLOps Dashboard",
            "🔄 Retraining & Pipeline",
            "📈 Drift & Monitoring",
            "📖 Architecture & Docs",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("System Status")
    if prod_model:
        st.success(f"Production: **{prod_model.model_name}**")
        st.caption(f"Version: `{prod_model.version}`")
    else:
        st.warning("Production Model: Initializing...")

    # Feedback buffer status
    feedback_df = load_feedback_ratings()
    n_feedback = len(feedback_df)
    st.write(f"New User Feedback: **{n_feedback} / {FEEDBACK_TRIGGER_THRESHOLD}**")
    st.progress(min(1.0, n_feedback / max(1, FEEDBACK_TRIGGER_THRESHOLD)))
    if n_feedback >= FEEDBACK_TRIGGER_THRESHOLD:
        st.info("Retraining buffer ready!")


# -----------------------------------------------------------------------------
# VIEW 1: MOVIE EXPLORER
# -----------------------------------------------------------------------------
if nav_selection == "🎬 Movie Explorer":
    st.markdown(
        """
        <div class="hero-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800; color:#58a6ff;">🎬 CineTrack Movie Explorer</h1>
            <p style="margin:8px 0 0 0; color:#8b949e; font-size:1.05rem;">
                Browse thousands of movies, discover recommendations with Collaborative Filtering, and submit ratings to train candidate models.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_search, col_genre, col_user = st.columns([2, 1.2, 1])

    with col_search:
        search_query = st.text_input("🔍 Search Movies by Title", placeholder="e.g. Inception, Toy Story, Pulp Fiction...")

    with col_genre:
        selected_genre = st.selectbox("🎭 Filter by Genre", genre_list, index=0)

    with col_user:
        user_id_input = st.number_input("👤 Active User ID", min_value=1, max_value=610, value=1, step=1)

    # Filter catalog
    filtered_movies = movies_df.copy()
    if search_query:
        filtered_movies = filtered_movies[
            filtered_movies["title"].str.contains(search_query, case=False, na=False)
        ]
    if selected_genre != "All":
        filtered_movies = filtered_movies[
            filtered_movies["genres"].str.contains(selected_genre, case=False, na=False)
        ]

    st.markdown(f"**Found {len(filtered_movies):,} matching movies in catalog.**")

    tab_rec, tab_browse = st.tabs(["🎯 Top Recommendations", "📚 Catalog Browser"])

    with tab_rec:
        st.subheader(f"Personalised Recommendations for User #{user_id_input}")
        if prod_model is not None:
            recs = prod_model.recommend(
                user_id=int(user_id_input),
                top_k=TOP_K,
                genre=selected_genre if selected_genre != "All" else None,
                exclude_rated=True,
            )

            cols = st.columns(TOP_K)
            for i, rec in enumerate(recs):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div>
                                <span class="badge badge-rank">Rank #{rec['rank']}</span>
                                <h4 style="margin:8px 0 4px 0; color:#f0f6fc;">{rec['title']}</h4>
                                <p style="font-size:0.85rem; color:#8b949e; margin:0 0 10px 0;">{rec['genres']}</p>
                            </div>
                            <div>
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
                                    <span style="font-size:0.8rem; color:#8b949e;">Match Score</span>
                                    <span style="font-weight:700; color:#58a6ff;">{rec['score']:.2f}★</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("Recommendation model is loading...")

        st.markdown("---")
        st.subheader("🎯 Item-to-Item Similarity: Select a Movie to Get 5 Recommendations")
        
        # Movie selector for similarity
        sample_titles = movies_df.head(200)["title"].tolist()
        selected_movie_title = st.selectbox(
            "Pick a Movie from the catalog:",
            sample_titles,
            index=0,
        )

        selected_row = movies_df[movies_df["title"] == selected_movie_title]
        if not selected_row.empty and prod_model is not None:
            sel_movie_id = int(selected_row.iloc[0]["movieId"])
            sel_genres = selected_row.iloc[0]["genres"]
            st.caption(f"Selected: **{selected_movie_title}** (ID: `{sel_movie_id}`) | Genres: `{sel_genres}`")

            similar_recs = prod_model.recommend(item_id=sel_movie_id, top_k=TOP_K)
            if similar_recs:
                st.write("##### Top 5 Similar Recommendations:")
                sim_cols = st.columns(len(similar_recs))
                for idx, srec in enumerate(similar_recs):
                    with sim_cols[idx]:
                        st.markdown(
                            f"""
                            <div class="movie-card">
                                <div>
                                    <span class="badge badge-genre">Similar #{idx+1}</span>
                                    <h5 style="margin:8px 0 4px 0; color:#f0f6fc;">{srec['title']}</h5>
                                    <p style="font-size:0.8rem; color:#8b949e; margin:0 0 8px 0;">{srec['genres']}</p>
                                </div>
                                <div style="color:#3fb950; font-weight:700; font-size:0.9rem;">
                                    {srec['score']:.2f} Match
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Interactive feedback form
            st.markdown("#### ⭐ Rate this Movie (Submit Feedback to MLOps Loop)")
            with st.form("feedback_form"):
                fb_cols = st.columns([1.5, 1, 1])
                with fb_cols[0]:
                    user_rating = st.slider("Your Rating (0.5 to 5.0 Stars):", min_value=0.5, max_value=5.0, value=4.5, step=0.5)
                with fb_cols[1]:
                    fb_user = st.number_input("Your User ID", min_value=1, max_value=1000, value=user_id_input)
                with fb_cols[2]:
                    st.write("")
                    st.write("")
                    submit_button = st.form_submit_button("Submit Rating 🚀")

                if submit_button:
                    # Append feedback directly
                    FEEDBACK_DATA_DIR.mkdir(parents=True, exist_ok=True)
                    feedback_file = FEEDBACK_DATA_DIR / "ratings_feedback.csv"
                    now_ts = int(pd.Timestamp.now().timestamp())
                    new_entry = pd.DataFrame(
                        [
                            {
                                "userId": int(fb_user),
                                "movieId": sel_movie_id,
                                "rating": float(user_rating),
                                "timestamp": now_ts,
                            }
                        ]
                    )
                    if feedback_file.exists():
                        new_entry.to_csv(feedback_file, mode="a", header=False, index=False)
                    else:
                        new_entry.to_csv(feedback_file, mode="w", header=True, index=False)

                    st.success(f"Feedback recorded for '{selected_movie_title}': {user_rating}★! Data buffered for next retraining cycle.")
                    st.rerun()

    with tab_browse:
        st.subheader("Explore Movies Catalog")
        display_cols = ["movieId", "title", "genres", "year"]
        st.dataframe(
            filtered_movies[display_cols].head(50),
            use_container_width=True,
            hide_index=True,
        )


# -----------------------------------------------------------------------------
# VIEW 2: MLOPS DASHBOARD
# -----------------------------------------------------------------------------
elif nav_selection == "📊 MLOps Dashboard":
    st.markdown(
        """
        <div class="hero-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800; color:#58a6ff;">📊 CineTrack MLOps Dashboard</h1>
            <p style="margin:8px 0 0 0; color:#8b949e; font-size:1.05rem;">
                Model versions, quality gate decisions, Precision@5 / NDCG@5 metrics, and MLflow experiment runs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Top Metrics Cards
    eval_file = PRODUCTION_MODEL_DIR / "evaluation_report.json"
    prod_metrics = {}
    if eval_file.exists():
        with open(eval_file, "r") as f:
            prod_metrics = json.load(f).get("metrics", {})

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">PRECISION@5</span>
                <h2 style="color:#58a6ff; margin:6px 0 0 0; font-size:2.2rem;">{prod_metrics.get('precision@5', 0.0337):.4f}</h2>
                <span style="color:#3fb950; font-size:0.8rem;">Gate Floor: ≥ {GATE_MIN_PRECISION}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mcol2:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">RECALL@5</span>
                <h2 style="color:#a371f7; margin:6px 0 0 0; font-size:2.2rem;">{prod_metrics.get('recall@5', 0.0100):.4f}</h2>
                <span style="color:#8b949e; font-size:0.8rem;">Catalog test users: {prod_metrics.get('eval_user_count', 594)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mcol3:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">NDCG@5</span>
                <h2 style="color:#3fb950; margin:6px 0 0 0; font-size:2.2rem;">{prod_metrics.get('ndcg@5', 0.0336):.4f}</h2>
                <span style="color:#3fb950; font-size:0.8rem;">Normalized Ranking Floor: ≥ {GATE_MIN_NDCG}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mcol4:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">CATALOG COVERAGE</span>
                <h2 style="color:#f0883e; margin:6px 0 0 0; font-size:2.2rem;">{prod_metrics.get('catalog_coverage', 0.0082):.4f}</h2>
                <span style="color:#8b949e; font-size:0.8rem;">2x baseline diversity</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. Model Staging Status: Incumbent vs Candidate
    st.subheader("Model Staging: Production Incumbent vs Candidate")
    col_prod, col_cand = st.columns(2)

    with col_prod:
        prod_meta_file = PRODUCTION_MODEL_DIR / "metadata.json"
        prod_meta = {}
        if prod_meta_file.exists():
            with open(prod_meta_file, "r") as f:
                prod_meta = json.load(f)

        st.markdown(
            f"""
            <div style="background:#161b22; border:1px solid #30363d; border-radius:12px; padding:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="color:#3fb950; margin:0;">🟢 Production Incumbent</h3>
                    <span class="badge badge-promoted">ACTIVE DEPLOYMENT</span>
                </div>
                <p style="margin:12px 0 4px 0; color:#f0f6fc;"><strong>Architecture:</strong> {prod_meta.get('model_name', 'CollaborativeFilteringSVD')}</p>
                <p style="margin:0 0 4px 0; color:#8b949e;"><strong>Latent Dimensions:</strong> {prod_meta.get('n_factors', 32)}</p>
                <p style="margin:0 0 4px 0; color:#8b949e;"><strong>Trained Users / Items:</strong> {prod_meta.get('n_users', 610):,} / {prod_meta.get('n_items', 9742):,}</p>
                <p style="margin:0 0 4px 0; color:#8b949e;"><strong>Training Loss (RMSE):</strong> {prod_meta.get('final_epoch_rmse', 0.72):.4f}</p>
                <p style="margin:0; color:#8b949e;"><strong>MLflow Run ID:</strong> <code>{prod_meta.get('run_id', 'bootstrap')}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_cand:
        cand_meta_file = CANDIDATE_MODEL_DIR / "metadata.json"
        cand_meta = {}
        if cand_meta_file.exists():
            with open(cand_meta_file, "r") as f:
                cand_meta = json.load(f)

        st.markdown(
            f"""
            <div style="background:#161b22; border:1px solid #30363d; border-radius:12px; padding:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="color:#58a6ff; margin:0;">🔵 Candidate Staging</h3>
                    <span class="badge badge-genre">QUALITY GATE STAGE</span>
                </div>
                <p style="margin:12px 0 4px 0; color:#f0f6fc;"><strong>Architecture:</strong> {cand_meta.get('model_name', 'CollaborativeFilteringSVD')}</p>
                <p style="margin:0 0 4px 0; color:#8b949e;"><strong>Latent Dimensions:</strong> {cand_meta.get('n_factors', 32)}</p>
                <p style="margin:0 0 4px 0; color:#8b949e;"><strong>Trained Users / Items:</strong> {cand_meta.get('n_users', 610):,} / {cand_meta.get('n_items', 9742):,}</p>
                <p style="margin:0 0 4px 0; color:#8b949e;"><strong>Training Loss (RMSE):</strong> {cand_meta.get('final_epoch_rmse', 0.72):.4f}</p>
                <p style="margin:0; color:#8b949e;"><strong>MLflow Run ID:</strong> <code>{cand_meta.get('run_id', 'candidate')}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 3. Quality Gate History
    st.subheader("🛡️ Automated Quality Gate Decision Audit Log")
    hist_file = PRODUCTION_MODEL_DIR / "promotion_history.json"
    rejection_file = CANDIDATE_MODEL_DIR / "rejection_history.json"

    events = []
    if hist_file.exists():
        with open(hist_file, "r") as f:
            events.extend(json.load(f))
    if rejection_file.exists():
        with open(rejection_file, "r") as f:
            events.extend(json.load(f))

    if events:
        events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        for ev in events[:5]:
            decision = ev.get("decision", "UNKNOWN")
            is_promoted = decision == "PROMOTED"
            badge_class = "badge-promoted" if is_promoted else "badge-rejected"
            icon = "✅" if is_promoted else "❌"

            cand_p = ev.get("candidate_metrics", {}).get("precision@5", 0.0)
            cand_n = ev.get("candidate_metrics", {}).get("ndcg@5", 0.0)

            st.markdown(
                f"""
                <div style="background:#161b22; border-left:4px solid {'#2ea043' if is_promoted else '#f85149'}; padding:12px 18px; margin-bottom:10px; border-radius:6px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong>{icon} Decision:</strong> <span class="badge {badge_class}">{decision}</span>
                            <span style="color:#8b949e; font-size:0.85rem; margin-left:10px;">{ev.get('timestamp')}</span>
                        </div>
                        <div style="font-size:0.85rem; color:#8b949e;">
                            Candidate: P@5 = <code>{cand_p:.4f}</code> | NDCG@5 = <code>{cand_n:.4f}</code>
                        </div>
                    </div>
                    <p style="margin:6px 0 0 0; font-size:0.9rem; color:#c9d1d9;">
                        <strong>Reasons:</strong> {', '.join(ev.get('reasons', []))}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No quality gate runs logged yet.")

    st.markdown("---")

    # 4. MLflow Experiment Tracker Explorer
    st.subheader("🧪 MLflow Experiment Tracker Runs")
    if MLFLOW_DB_PATH.exists():
        try:
            conn = sqlite3.connect(MLFLOW_DB_PATH)
            runs_df = pd.read_sql_query(
                """
                SELECT 
                    runs.run_uuid as run_id,
                    runs.name as run_name,
                    runs.status as status,
                    datetime(runs.start_time / 1000, 'unixepoch') as start_time
                FROM runs
                ORDER BY runs.start_time DESC
                LIMIT 15;
                """,
                conn,
            )
            conn.close()

            if not runs_df.empty:
                st.dataframe(runs_df, use_container_width=True, hide_index=True)
            else:
                st.info("No runs found in MLflow database.")
        except Exception as e:
            st.warning(f"Could not read MLflow SQLite runs: {e}")
    else:
        st.info("MLflow database not yet initialized.")


# -----------------------------------------------------------------------------
# VIEW 3: RETRAINING & PIPELINE
# -----------------------------------------------------------------------------
elif nav_selection == "🔄 Retraining & Pipeline":
    st.markdown(
        """
        <div class="hero-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800; color:#58a6ff;">🔄 Automated Retraining & Quality Gate</h1>
            <p style="margin:8px 0 0 0; color:#8b949e; font-size:1.05rem;">
                Train new candidate models, compare against incumbent baseline, and evaluate automated quality gate promotion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feedback_df = load_feedback_ratings()
    st.write(f"### Current Feedback Buffer: **{len(feedback_df)} Ratings**")
    if not feedback_df.empty:
        st.dataframe(feedback_df.tail(10), use_container_width=True, hide_index=True)
    else:
        st.info("Feedback buffer is currently empty. Rate movies in the Movie Explorer to accumulate real-time feedback.")

    st.markdown("---")
    st.subheader("Configure Candidate Model Hyperparameters")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        param_factors = st.number_input("Latent Factors (k)", min_value=8, max_value=128, value=32, step=8)
    with col2:
        param_epochs = st.number_input("Training Epochs", min_value=5, max_value=100, value=20, step=5)
    with col3:
        param_lr = st.number_input("Learning Rate", min_value=0.001, max_value=0.1, value=0.005, step=0.001, format="%.3f")
    with col4:
        param_reg = st.number_input("L2 Regularization", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    force_promotion_opt = st.checkbox("Force Promote Candidate (Override Quality Gate for demonstration)")

    if st.button("🚀 Trigger Candidate Training & Quality Gate", type="primary"):
        with st.spinner("Training candidate Collaborative Filtering model and evaluating quality gate..."):
            try:
                # 1. Train model
                params = {
                    "n_factors": int(param_factors),
                    "n_epochs": int(param_epochs),
                    "learning_rate": float(param_lr),
                    "regularization": float(param_reg),
                }
                cand_model = train_model(model_type="collaborative", params=params)

                # 2. Run quality gate
                gate_decision = run_quality_gate(force_promote=force_promotion_opt)

                st.markdown("### Quality Gate Result")
                if gate_decision["passed"]:
                    st.success(f"🎉 **{gate_decision['decision']}**: Candidate model passed all evaluation criteria and is now deployed in production!")
                else:
                    st.error(f"🛑 **{gate_decision['decision']}**: Candidate failed evaluation criteria. Incumbent model remains unchanged in production.")

                st.write("**Reasons:**", gate_decision["reasons"])

                st.json(
                    {
                        "candidate_metrics": gate_decision["candidate_metrics"],
                        "incumbent_metrics": gate_decision["incumbent_metrics"],
                    }
                )

                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Retraining encountered an error: {e}")


# -----------------------------------------------------------------------------
# VIEW 4: DRIFT & MONITORING
# -----------------------------------------------------------------------------
elif nav_selection == "📈 Drift & Monitoring":
    st.markdown(
        """
        <div class="hero-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800; color:#58a6ff;">📈 Evidently Data Drift & Application Monitoring</h1>
            <p style="margin:8px 0 0 0; color:#8b949e; font-size:1.05rem;">
                Detect covariate shift and concept drift between historical training distribution and incoming user feedback.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔍 Run Live Data Drift Analysis Now", type="primary"):
        with st.spinner("Executing Evidently drift analysis and Kolmogorov-Smirnov test..."):
            run_drift_analysis()
            st.success("Drift analysis updated successfully!")

    drift_data = get_latest_drift_summary()

    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    with dcol1:
        drift_detected = drift_data.get("drift_detected", False)
        status_color = "#f85149" if drift_detected else "#3fb950"
        status_text = "DRIFT DETECTED" if drift_detected else "DATA STABLE"
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">DRIFT STATUS</span>
                <h3 style="color:{status_color}; margin:6px 0 0 0;">{status_text}</h3>
                <span style="color:#8b949e; font-size:0.8rem;">p-val threshold: {drift_data.get('p_value_threshold', 0.05)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with dcol2:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">KS-TEST STATISTIC</span>
                <h3 style="color:#58a6ff; margin:6px 0 0 0;">{drift_data.get('ks_statistic', 0.0):.4f}</h3>
                <span style="color:#8b949e; font-size:0.8rem;">p-value: {drift_data.get('ks_p_value', 1.0):.6f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with dcol3:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">WASSERSTEIN DISTANCE</span>
                <h3 style="color:#a371f7; margin:6px 0 0 0;">{drift_data.get('wasserstein_distance', 0.0):.4f}</h3>
                <span style="color:#8b949e; font-size:0.8rem;">Earth Mover's Distance</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with dcol4:
        ref_count = drift_data.get("reference_stats", {}).get("count", 0)
        curr_count = drift_data.get("current_stats", {}).get("count", 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="color:#8b949e; font-size:0.85rem; font-weight:600;">OBSERVATION SAMPLES</span>
                <h3 style="color:#f0883e; margin:6px 0 0 0;">{curr_count} / {ref_count}</h3>
                <span style="color:#8b949e; font-size:0.8rem;">Feedback vs Baseline</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Interactive Plotly Distribution Comparison
    st.subheader("Distribution Comparison: Reference (Training) vs Current (Feedback)")
    ref_mean = drift_data.get("reference_stats", {}).get("mean", 3.5)
    curr_mean = drift_data.get("current_stats", {}).get("mean", 3.5)

    chart_col1, chart_col2 = st.columns([1.5, 1])

    with chart_col1:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Reference (Training)", "Current (Feedback)"],
                y=[ref_mean, curr_mean],
                marker_color=["#58a6ff", "#f0883e"],
                text=[f"{ref_mean:.2f}★", f"{curr_mean:.2f}★"],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Mean Rating Comparison",
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            yaxis_title="Mean Rating",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.write("##### Summary Statistics")
        stats_df = pd.DataFrame(
            {
                "Statistic": ["Count", "Mean", "Std Dev", "Min", "Max"],
                "Reference": [
                    drift_data.get("reference_stats", {}).get("count", 0),
                    f"{drift_data.get('reference_stats', {}).get('mean', 0.0):.2f}",
                    f"{drift_data.get('reference_stats', {}).get('std', 0.0):.2f}",
                    drift_data.get("reference_stats", {}).get("min", 0.0),
                    drift_data.get("reference_stats", {}).get("max", 0.0),
                ],
                "Current": [
                    drift_data.get("current_stats", {}).get("count", 0),
                    f"{drift_data.get('current_stats', {}).get('mean', 0.0):.2f}",
                    f"{drift_data.get('current_stats', {}).get('std', 0.0):.2f}",
                    drift_data.get("current_stats", {}).get("min", 0.0),
                    drift_data.get("current_stats", {}).get("max", 0.0),
                ],
            }
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Embedded Evidently HTML Report
    st.markdown("---")
    st.subheader("📑 Interactive Evidently AI Drift Report")
    html_report_path = DRIFT_REPORTS_DIR / "drift_report.html"
    if html_report_path.exists():
        with open(html_report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=750, scrolling=True)
    else:
        st.info("Click 'Run Live Data Drift Analysis Now' to generate the interactive Evidently visual report.")


# -----------------------------------------------------------------------------
# VIEW 5: ARCHITECTURE & DOCS
# -----------------------------------------------------------------------------
elif nav_selection == "📖 Architecture & Docs":
    st.markdown(
        """
        <div class="hero-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800; color:#58a6ff;">📖 CineTrack Architecture & Reproduction</h1>
            <p style="margin:8px 0 0 0; color:#8b949e; font-size:1.05rem;">
                System design, MLOps workflow, quality gate criteria, and Cloud Run deployment guide.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### End-to-End MLOps Architecture

        ```mermaid
        graph TD
            RawData[(MovieLens Raw Data)] --> Ingest[Data Ingestion & Validation]
            Feedback[(Live User Feedback)] --> Ingest
            Ingest --> Split[Temporal Train/Test Split]
            Split --> Train[Candidate Training SVD / CF]
            Train --> MLflow[(MLflow Tracking & Artifacts)]
            Train --> Eval[Evaluation: Precision@5, NDCG@5]
            Eval --> QualityGate{Quality Gate}
            QualityGate -->|Pass: NDCG & P@5 Met| Promote[Deploy to Production]
            QualityGate -->|Fail: Regressed| Retain[Retain Existing Incumbent]
            Promote --> Serving[FastAPI & Streamlit UI]
            Retain --> Serving
            Serving --> Drift[Evidently Data Drift Monitoring]
            Serving --> Feedback
        ```

        ### MLOps Quality Gate Invariants
        1. **Precision Floor**: Candidate `Precision@5` must meet or exceed minimum threshold (`0.08` or baseline).
        2. **NDCG Floor**: Candidate `NDCG@5` must meet or exceed ranking threshold (`0.10` or baseline).
        3. **No Regression**: Candidate `NDCG@5` must not drop by more than `0.01` relative to the current incumbent.
        4. **Zero Downtime**: If the candidate fails, the existing model remains in production without interruption.
        """
    )
