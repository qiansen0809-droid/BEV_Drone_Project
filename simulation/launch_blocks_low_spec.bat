@echo off
setlocal
cd /d "%~dp0"

set "BLOCKS_EXE=%~dp0Blocks\Blocks\WindowsNoEditor\Blocks.exe"
set "SETTINGS=%~dp0settings.json"

netstat -ano | findstr ":41451" | findstr "LISTENING" > nul
if not errorlevel 1 (
    echo AirSim Blocks is already running.
    exit /b 0
)

if not exist "%BLOCKS_EXE%" (
    echo ERROR: Blocks.exe was not found.
    echo Expected: %BLOCKS_EXE%
    pause
    exit /b 1
)

echo Starting AirSim Blocks in low-spec mode...
start "AirSim Blocks" "%BLOCKS_EXE%" -ResX=640 -ResY=480 -windowed -NoVSync -settings="%SETTINGS%"
endlocal
