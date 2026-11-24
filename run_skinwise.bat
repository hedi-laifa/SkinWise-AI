@echo off
REM ==========================================================================
REM SkinWise AI - start the app locally (development mode, no Docker needed)
REM Double-click this file. Two windows open: the API and the website.
REM Close those two windows to stop the app.
REM ==========================================================================
cd /d "%~dp0"

echo Starting the API (FastAPI + ONNX model) on http://localhost:8000 ...
start "SkinWise API - port 8000" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"

echo Starting the website (React + Vite) on http://localhost:5173 ...
start "SkinWise website - port 5173" cmd /k "cd frontend && npm run dev"

echo.
echo   Website : http://localhost:5173
echo   API docs: http://localhost:8000/docs
echo.
timeout /t 8 /nobreak >nul
start "" http://localhost:5173
