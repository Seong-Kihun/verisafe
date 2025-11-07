@echo off
cls
echo ================================================
echo   VeriSafe Mobile App (Expo)
echo ================================================
echo.
echo Starting app...
echo Options when browser opens:
echo   - w: Run in web browser
echo   - a: Run on Android emulator/device
echo   - i: Run on iOS simulator (Mac only)
echo.
echo Press Ctrl+C to stop
echo ================================================
echo.

cd /d "%~dp0"

echo Checking Node.js...
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not installed.
    echo Please install Node.js: https://nodejs.org/
    pause
    exit /b 1
)

echo Checking node_modules...
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)

echo.
echo Starting Expo server (offline mode)...
call npm start -- --offline

pause
