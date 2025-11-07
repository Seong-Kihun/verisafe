@echo off
chcp 65001 >nul
cls

echo ================================================
echo   STEP 1: Start Database
echo ================================================
echo.

cd /d "%~dp0"

echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker not found!
    echo Please install Docker Desktop from:
    echo https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

echo Docker found! Starting database...
docker-compose up -d

echo.
echo Waiting for database to be ready (30 seconds)...
timeout /t 30 /nobreak >nul

echo.
echo ================================================
echo   Database started successfully!
echo ================================================
echo.
echo Services running:
echo   PostgreSQL: localhost:5432
echo     - User: verisafe_user
echo     - Password: verisafe_pass_2025
echo     - Database: verisafe_db
echo.
echo   Redis: localhost:6379
echo     - Password: verisafe_redis_2025
echo.
echo   PgAdmin (Database Admin): http://localhost:5050
echo     - Email: admin@verisafe.com
echo     - Password: admin2025
echo.
echo ================================================
echo Next: Run STEP2_install.bat
echo.

pause
