@echo off
chcp 65001 > nul
title AirSim Drone Keyboard Control
cd /d "%~dp0"

set "PYTHON_EXE=C:\ProgramData\anaconda3\envs\bev_drone\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%USERPROFILE%\.conda\envs\bev_drone\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ERROR: bev_drone Python environment was not found.
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%~dp0keyboard_control.py"
pause
