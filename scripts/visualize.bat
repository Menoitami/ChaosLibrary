@echo off
echo Running visualization for ChaosLibrary...

:: Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Go to the repository root (in case script is run from a subdirectory)
cd %~dp0..

:: Check if the plot_heatmap.py script exists
if not exist scripts\plot_heatmap.py (
    echo Error: Visualization script not found.
    echo Expected: scripts\plot_heatmap.py
    pause
    exit /b 1
)

:: Run the visualization script
echo Running heatmap visualization...
python scripts\plot_heatmap.py

if %ERRORLEVEL% == 0 (
    echo.
    echo Visualization completed successfully!
    echo Results saved in the results directory.
    echo.
) else (
    echo.
    echo Visualization failed with error code %ERRORLEVEL%
    echo.
)

pause 