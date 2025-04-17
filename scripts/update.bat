@echo off
echo Updating ChaosLibrary from Git repository...

:: Save the current directory
set CURRENT_DIR=%CD%

:: Go to the repository root (in case script is run from a subdirectory)
cd %~dp0..

:: Check if .git directory exists
if not exist .git (
    echo Error: This is not a Git repository.
    echo Please clone the repository first using:
    echo git clone [repository_url] [destination_directory]
    cd %CURRENT_DIR%
    pause
    exit /b 1
)

:: Fetch the latest changes
echo Fetching latest changes...
git fetch

:: Check if there are local changes
git diff --quiet HEAD
if %ERRORLEVEL% NEQ 0 (
    echo Warning: You have local changes that might be overwritten.
    choice /C YN /M "Do you want to continue updating?"
    if %ERRORLEVEL% EQU 2 (
        echo Update cancelled.
        cd %CURRENT_DIR%
        pause
        exit /b 0
    )
)

:: Pull the latest changes
echo Pulling latest changes...
git pull

if %ERRORLEVEL% == 0 (
    echo.
    echo Repository updated successfully!
    echo You may need to rebuild the project using build.bat
    echo.
) else (
    echo.
    echo Failed to update repository. Error code: %ERRORLEVEL%
    echo.
)

:: Return to the original directory
cd %CURRENT_DIR%

pause 