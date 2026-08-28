@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0tools\python\python.exe"
if exist "%PYTHON%" (
  "%PYTHON%" "%~dp0builder_launcher.py" --action open
) else (
  py -3.12 "%~dp0builder_launcher.py" --action open 2>nul || python "%~dp0builder_launcher.py" --action open
)
if errorlevel 1 pause
