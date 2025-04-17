@echo off
echo Building ChaosLibrary...

:: Go to repository root
cd %~dp0..

:: Create build directory if it doesn't exist
if not exist build mkdir build

:: Navigate to build directory
cd build

:: Run CMake and build
echo Running CMake...
cmake ..
echo Building with CMake...
cmake --build . --config Release

:: Go back to the repository root
cd ..

if %ERRORLEVEL% == 0 (
    echo Build completed successfully!
) else (
    echo Build failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo You can now run the project using run.bat
echo.

pause 