@echo off
REM WalkSense Simple Setup Wrapper for Windows

echo =========================================
echo WalkSense Enhanced Setup
echo =========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please install Python from python.org
    pause
    exit /b 1
)

echo Python found!
python --version
echo.

REM Run the main Python setup script
python scripts/setup_project.py

if %errorlevel% neq 0 (
    echo ERROR: Setup script failed.
    pause
    exit /b 1
)

echo.
echo =========================================
echo Setup Complete!
echo =========================================
pause
