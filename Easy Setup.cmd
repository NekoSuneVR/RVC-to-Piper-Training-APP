@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -NoLaunch
if errorlevel 1 (
  pause
  exit /b 1
)
if exist "tools\python\python.exe" (
  "tools\python\python.exe" studio.py
) else (
  py -3.12 studio.py 2>nul || python studio.py
)
if errorlevel 1 pause
