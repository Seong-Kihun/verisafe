@echo off
chcp 65001 >nul
cls

echo ================================================
echo   STEP 2: Install Python Packages
echo   (This may take 5-10 minutes)
echo ================================================
echo.

cd /d "%~dp0backend"

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Checking Python version (requires 3.10 or higher)...
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo WARNING: Python 3.10 or higher is recommended!
    echo Current version may have compatibility issues.
    echo Please consider upgrading Python.
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
echo Installing packages (please wait, this takes 5-10 minutes)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Installation complete!
echo ================================================
echo.
echo Next: Run STEP3_setup.bat
echo.

pause
