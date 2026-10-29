@echo off
REM CineTrack Local Launcher for Windows
echo ========================================================
echo Starting CineTrack (Windows Local Runner)
echo ========================================================

IF EXIST .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) ELSE (
    echo [WARNING] .venv not found. Running with global python...
)

echo Starting FastAPI Backend in background on port 8000...
start "CineTrack Backend" cmd /k "python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"

echo Starting Streamlit UI on port 8501...
python -m streamlit run src/ui/app.py --server.port 8501

pause
