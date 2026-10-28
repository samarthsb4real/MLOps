"""Integration tests for CineTrack FastAPI service."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    """Create test client with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "timestamp" in data


def test_genres_endpoint(client):
    response = client.get("/genres")
    assert response.status_code == 200
    genres = response.json()
    assert isinstance(genres, list)
    assert len(genres) > 0
    assert "Action" in genres or "Comedy" in genres


def test_list_movies_endpoint(client):
    response = client.get("/movies?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] > 0
    assert len(data["movies"]) <= 10
    assert "movieId" in data["movies"][0]
    assert "title" in data["movies"][0]


def test_search_movies_endpoint(client):
    response = client.get("/movies?search=Toy Story")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert any("Toy Story" in m["title"] for m in data["movies"])


def test_get_movie_detail(client):
    response = client.get("/movies/1")
    assert response.status_code == 200
    movie = response.json()
    assert movie["movieId"] == 1
    assert "Toy Story" in movie["title"]

    # 404 for invalid ID
    response_404 = client.get("/movies/9999999")
    assert response_404.status_code == 404


def test_recommend_user(client):
    payload = {
        "user_id": 1,
        "top_k": 5,
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 5
    for idx, r in enumerate(data["recommendations"], start=1):
        assert r["rank"] == idx
        assert "movieId" in r
        assert "score" in r


def test_recommend_movie_similarity(client):
    payload = {
        "movie_id": 1,
        "top_k": 5,
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["movie_id"] == 1
    assert len(data["recommendations"]) <= 5
    # Recommended movies must not include target movie 1
    assert all(r["movieId"] != 1 for r in data["recommendations"])


def test_submit_feedback(client):
    payload = {
        "user_id": 999,
        "movie_id": 1,
        "rating": 4.5,
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "feedback_id" in data
    assert data["total_new_feedback"] >= 1


def test_model_status(client):
    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert "production_model" in data
    assert "candidate_model" in data


def test_monitoring_drift(client):
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    data = response.json()
    assert "drift_detected" in data
    assert "reference_stats" in data
    assert "current_stats" in data
