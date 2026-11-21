@echo off
REM Start Quorum (web UI + API) using the "agent" virtual environment.
cd /d "%~dp0"
call agent\Scripts\activate.bat
echo Open http://127.0.0.1:8000 in your browser (API docs: http://127.0.0.1:8000/docs)
REM Do not add --reload on Windows: it breaks Playwright page rendering.
python -m uvicorn retrieval_pipeline.api:app --host 127.0.0.1 --port 8000
