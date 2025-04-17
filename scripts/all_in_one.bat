@echo off
echo ChaosLibrary All-in-One Script
echo ============================
echo.

:: Set the error level to 0
exit /b 0 > nul 2>&1

:: Save current directory
set CURRENT_DIR=%CD%

:: Go to repository root
cd %~dp0..

echo Step 1: Updating repository...
echo -----------------------------
call scripts\update.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to update repository.
    echo Continuing with the next steps...
    echo.
)

echo Step 2: Building project...
echo -------------------------
call scripts\build.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to build project.
    cd %CURRENT_DIR%
    pause
    exit /b 1
)

echo Step 3: Running application...
echo ---------------------------
call scripts\run.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to run application.
    cd %CURRENT_DIR%
    pause
    exit /b 1
)

echo Step 4: Visualizing results...
echo ---------------------------
call scripts\visualize.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to visualize results.
    cd %CURRENT_DIR%
    pause
    exit /b 1
)

echo.
echo All steps completed successfully!
echo.

:: Return to original directory
cd %CURRENT_DIR%

pause 