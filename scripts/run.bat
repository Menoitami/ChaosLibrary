@echo off
echo Running ChaosLibrary...

:: Go to repository root
cd %~dp0..

:: Check if build directory exists
if not exist ..\build (
    echo Error: Build directory not found. Please run build.bat first.
    pause
    exit /b 1
)

:: Check if executable exists in Release configuration
if exist ..\build\Release\ChaosLib.exe (
    echo Running Release version...
    ..\build\Release\ChaosLib.exe
) else if exist ..\build\ChaosLib.exe (
    echo Running from build root...
    ..\build\ChaosLib.exe
) else if exist ..\build\Debug\ChaosLib.exe (
    echo Running Debug version...
    ..\build\Debug\ChaosLib.exe
) else (
    echo Error: Executable not found. Please build the project first.
    pause
    exit /b 1
)

echo.
echo Program execution completed.
echo Results are saved in the workspace directory.
echo You can run visualize.bat to visualize the results.
echo.

pause 