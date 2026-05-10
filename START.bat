@echo off
:: BetIQ — one-command startup for Windows
:: Double-click this file to start

cd /d "%~dp0"

echo ================================
echo   BetIQ - Sports Betting AI
echo ================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Install from https://python.org ^(3.8 or newer^)
    pause
    exit /b 1
)

echo Python found.
echo.

:: Create venv on first run
if not exist ".venv" (
    echo Setting up virtual environment ^(one time only^)...
    python -m venv .venv
    echo Done.
    echo.
)

call .venv\Scripts\activate.bat

:: Install dependencies
echo Checking dependencies...
pip install -r requirements.txt -q --disable-pip-version-check
echo Dependencies OK.
echo.

echo Starting BetIQ backend on http://localhost:8000
echo.
echo   ^> Open frontend\index.html in Chrome or Firefox
echo   ^> API docs at http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop.
echo.

cd backend
python -m uvicorn main:app --reload --port 8000
pause
