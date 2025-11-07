@echo off
cls
echo ================================================
echo   VeriSafe Backend Server
echo ================================================
echo.

cd /d "%~dp0"

echo Activating virtual environment...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
    echo Using global Python...
)

echo.
echo Starting server...
echo Server will be available at: http://localhost:8000/docs
echo Press Ctrl+C to stop
echo.
echo ================================================
echo.

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
