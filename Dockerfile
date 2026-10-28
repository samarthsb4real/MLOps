# CineTrack Production Dockerfile
FROM python:3.12-slim

# Prevent Python from writing .pyc and enable buffer-less stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and configurations
COPY pyproject.toml pytest.ini params.yaml dvc.yaml ./
COPY src/ ./src/
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Copy dataset and models if present
COPY ml-latest-small/ ./ml-latest-small/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/

# Cloud Run injects $PORT (default 8080)
EXPOSE 8080 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/_stcore/health || curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
