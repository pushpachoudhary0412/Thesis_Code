@echo off
REM Windows batch script for easy project execution
REM This script automatically activates the virtual environment and runs commands

if "%1"=="" (
    echo Usage: run_project.bat [command]
    echo.
    echo Commands:
    echo   setup       - Set up environment
    echo   baseline    - Run baseline experiments
    echo   experiments - Run full backdoor experiments
    echo   benchmark   - Run benchmark suite
    echo   visualize   - Create visualizations
    echo   all         - Run complete workflow
    echo   clean       - Clean up generated files
    echo.
    echo Example: run_project.bat all
    goto :eof
)

REM Check if virtual environment exists and has the activation script
if not exist "mimiciv_env\Scripts\activate.bat" (
    echo Virtual environment not found or incomplete. Setting up first...
    python setup_env.py --force
    if errorlevel 1 (
        echo Setup failed!
        goto :eof
    )
)

REM Check again after setup
if not exist "mimiciv_env\Scripts\activate.bat" (
    echo Setup completed but activation script still not found!
    echo Please check the setup and try again.
    goto :eof
)

REM Activate virtual environment
echo Activating virtual environment...
call mimiciv_env\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment!
    echo Trying to continue anyway...
)

REM Set PYTHONPATH for module imports
set PYTHONPATH=%cd%

REM Run the Python script with the provided command
echo Running: python run_project.py %1
python run_project.py %1

REM Exit with the same error code as the Python script
exit /b %errorlevel%
