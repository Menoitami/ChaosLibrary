@echo off
echo Cleaning ChaosLibrary build files...

:: Go to the repository root (in case script is run from a subdirectory)
cd %~dp0..

:: Ask for confirmation
echo WARNING: This will delete all build files and temporary outputs.
echo Your source code will NOT be deleted.
choice /C YN /M "Do you want to continue?"
if %ERRORLEVEL% EQU 2 (
    echo Cleaning cancelled.
    pause
    exit /b 0
)

:: Remove build directory
if exist build (
    echo Removing build directory...
    rd /s /q build
)

:: Clean workspace directory
if exist workspace (
    echo Cleaning workspace directory...
    del /q /s workspace\bifurcation\*.*
    del /q /s workspace\lle\*.*
    del /q /s workspace\debri\*.*
)

echo.
echo Cleaning completed successfully!
echo.

pause 