@echo off
chcp 65001 >nul
cls

echo ================================================
echo   STEP 3: Setup Database & Environment
echo ================================================
echo.

cd /d "%~dp0backend"

echo Checking .env file...
if not exist .env (
    echo Creating .env file from template...
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
    echo .env file created successfully!
) else (
    echo .env file already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Creating database tables...
python init_db.py

if errorlevel 1 (
    echo.
    echo WARNING: Database setup had issues
    echo But we can continue...
)

echo.
echo ================================================
echo   Database setup complete!
echo ================================================
echo.
echo Next: Run START.bat to start the server
echo.

pause
