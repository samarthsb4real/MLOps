#!/bin/bash
set -e

# Default port for Cloud Run is 8080
TARGET_PORT=${PORT:-8080}
APP_MODE=${APP_MODE:-all}

echo "Starting CineTrack in mode: $APP_MODE on port: $TARGET_PORT"

if [ "$APP_MODE" = "api" ]; then
    exec uvicorn src.api.app:app --host 0.0.0.0 --port "$TARGET_PORT"
elif [ "$APP_MODE" = "ui" ]; then
    exec streamlit run src/ui/app.py --server.port "$TARGET_PORT" --server.address 0.0.0.0 --server.headless true
else
    # Default 'all' mode: Run FastAPI on internal port 8000 and Streamlit on Cloud Run $PORT
    echo "Starting internal FastAPI backend on port 8000..."
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
    
    echo "Starting Streamlit UI on public port $TARGET_PORT..."
    exec streamlit run src/ui/app.py --server.port "$TARGET_PORT" --server.address 0.0.0.0 --server.headless true
fi
