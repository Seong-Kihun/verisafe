@echo off
chcp 65001 > nul
cls
echo ================================================
echo   VeriSafe - Starting Backend + Frontend
echo ================================================
echo.

cd /d "%~dp0"

REM Check if using SQLite or PostgreSQL
findstr /C:"DATABASE_TYPE=sqlite" backend\.env > nul
if %errorlevel% equ 0 (
    echo Using SQLite database (Docker not required)
    goto skip_docker_check
)

REM Check if Docker containers are running (for PostgreSQL)
echo Checking Docker containers...
docker ps --filter "name=verisafe" --format "{{.Names}}" | findstr "verisafe" > nul
if errorlevel 1 (
    echo.
    echo WARNING: Docker containers are not running!
    echo Please run STEP1_database.bat first.
    echo.
    pause
    exit /b 1
)
echo Docker containers are running!

:skip_docker_check

REM Check if .env file exists
echo.
echo Checking backend configuration...
if not exist "backend\.env" (
    echo.
    echo WARNING: backend\.env file not found!
    echo Please run STEP3_setup.bat first.
    echo.
    pause
    exit /b 1
)
echo Configuration file found!

echo.
echo ================================================
echo   Starting Services...
echo ================================================
echo.
echo Backend Server: http://localhost:8000/docs
echo Frontend App: http://localhost:8081
echo Database Admin (PgAdmin): http://localhost:5050
echo   Email: admin@verisafe.com
echo   Password: admin2025
echo.
echo Press Ctrl+C in each window to stop servers
echo ================================================
echo.

REM Start Backend in new window
start "VeriSafe Backend" cmd /k "cd backend && start.bat"

REM Wait 5 seconds for backend to initialize
echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

REM Start Frontend in new window
start "VeriSafe Frontend" cmd /k "cd mobile && start.bat"

echo.
echo Both servers are starting in separate windows...
echo.
echo Useful URLs:
echo - API Documentation: http://localhost:8000/docs
echo - Frontend App: http://localhost:8081
echo - PgAdmin: http://localhost:5050
echo.
echo Close this window or press any key to continue
pause > nul
