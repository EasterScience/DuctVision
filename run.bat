@echo off
title DuctVision - HVAC Duct Detection
echo ============================================
echo   DuctVision - HVAC Duct Detection Tool
echo ============================================
echo.

:: Create output directory if it doesn't exist
if not exist "output" mkdir output
if not exist "input" mkdir input

:: Check prerequisites
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] uv is not installed. Install from https://docs.astral.sh/uv/
    pause
    exit /b 1
)

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed.
    pause
    exit /b 1
)

:: Install Python dependencies
echo [1/4] Installing Python dependencies...
uv sync

:: Install frontend dependencies
echo [2/4] Installing frontend dependencies...
cd frontend
call npm install --silent
cd ..

:: Start backend
echo [3/4] Starting backend on port 8000...
start "DuctVision Backend" cmd /k "uv run uvicorn api:app --reload --port 8000"

:: Wait for backend to be ready
echo Waiting for backend...
timeout /t 3 /nobreak >nul

:: Start frontend
echo [4/4] Starting frontend on port 3000...
start "DuctVision Frontend" cmd /k "cd frontend && npm start"

echo.
echo ============================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo ============================================
echo.
echo Both servers are running in separate windows.
echo Close those windows to stop the servers.
pause
