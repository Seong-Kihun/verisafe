@echo off
cls

echo ================================================
echo   VeriSafe - Initial Setup (Run Once)
echo ================================================
echo.
echo This script will:
echo   1. Start Docker databases
echo   2. Install Python packages
echo   3. Setup environment and initialize database
echo.
echo Estimated time: 10 minutes
echo.
pause
cls

cd /d "%~dp0"

REM ============================================
REM STEP 1: Start Docker Databases
REM ============================================
echo.
echo ================================================
echo   [1/3] Starting Docker Databases
echo ================================================
echo.

echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed!
    echo Please install Docker Desktop:
    echo https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

echo Docker found! Starting databases...
docker-compose up -d

if errorlevel 1 (
    echo.
    echo [WARNING] There might be issues starting Docker containers.
    echo Continuing anyway...
    echo.
)

echo.
echo Waiting for databases to be ready (30 seconds)...
timeout /t 30 /nobreak >nul

echo.
echo [DONE] Databases started!
echo   - PostgreSQL: localhost:5432
echo   - Redis: localhost:6379
echo   - PgAdmin: http://localhost:5050
echo.
pause
cls

REM ============================================
REM STEP 2: Install Python Packages
REM ============================================
echo.
echo ================================================
echo   [2/3] Installing Python Packages
echo   (5-10 minutes, please wait...)
echo ================================================
echo.

cd backend

echo Checking Python version...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo Please install Python 3.10 or higher:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Checking Python version (3.10+ required)...
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [WARNING] Python 3.10+ is recommended!
    echo Compatibility issues may occur with current version.
    echo.
    pause
)

echo.
echo Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created!
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing packages (5-10 minutes)...
echo Take a coffee break!
echo.
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Package installation failed!
    echo Please check requirements.txt
    pause
    exit /b 1
)

echo.
echo [DONE] Python packages installed!
echo.
pause
cls

REM ============================================
REM STEP 3: Environment Setup and Database Init
REM ============================================
echo.
echo ================================================
echo   [3/3] Environment Setup and Database Init
echo ================================================
echo.

echo Checking .env file...
if not exist .env (
    echo Creating .env file...
    (
        echo # VeriSafe Environment Configuration - Auto-generated
        echo # Generated on: %date% %time%
        echo.
        echo # Application
        echo APP_NAME=VeriSafe API
        echo DEBUG=true
        echo.
        echo # Database ^(PostgreSQL^)
        echo DATABASE_TYPE=postgresql
        echo DATABASE_HOST=localhost
        echo DATABASE_PORT=5432
        echo DATABASE_USER=verisafe_user
        echo DATABASE_PASSWORD=verisafe_pass_2025
        echo DATABASE_NAME=verisafe_db
        echo.
        echo # Redis
        echo REDIS_HOST=localhost
        echo REDIS_PORT=6379
        echo REDIS_PASSWORD=verisafe_redis_2025
        echo REDIS_DB=0
        echo REDIS_CACHE_TTL=300
        echo.
        echo # JWT Authentication
        echo SECRET_KEY=dev_secret_key_change_in_production_2025
        echo ALGORITHM=HS256
        echo ACCESS_TOKEN_EXPIRE_MINUTES=1440
        echo.
        echo # CORS
        echo ALLOWED_ORIGINS=http://localhost:8081,http://192.168.104.30:8081
        echo.
        echo # File Upload
        echo UPLOAD_DIR=./uploads
        echo MAX_UPLOAD_SIZE=10485760
        echo.
        echo # External APIs ^(Optional^)
        echo ACLED_API_KEY=
        echo GDACS_API_URL=https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH
        echo RELIEFWEB_API_URL=https://api.reliefweb.int/v1
        echo TWITTER_BEARER_TOKEN=
        echo NEWS_API_KEY=
        echo SENTINEL_CLIENT_ID=
        echo SENTINEL_CLIENT_SECRET=
    ) > .env
    echo .env file created!
) else (
    echo .env file already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Initializing database tables...
python init_database.py

if errorlevel 1 (
    echo.
    echo [WARNING] Database initialization may have issues.
    echo But continuing anyway...
    echo.
)

echo.
echo Creating hazard scoring rules...
python app\scripts\create_hazard_scoring_rules.py

echo.
echo Generating sample data...
python app\scripts\create_sample_hazards.py

cd ..

echo.
echo ================================================
echo   Setup Complete!
echo ================================================
echo.
echo Next Steps:
echo   1. Run START.bat to start the server
echo.
echo Service Information:
echo   - API Docs: http://localhost:8000/docs
echo   - Frontend: http://localhost:8081
echo   - Database Admin: http://localhost:5050
echo     Email: admin@verisafe.com
echo     Password: admin2025
echo.
echo ================================================
echo.

pause
