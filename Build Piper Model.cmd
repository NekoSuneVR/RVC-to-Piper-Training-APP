@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0tools\python\python.exe"
if exist "%PYTHON%" (
  "%PYTHON%" "%~dp0piper_builder_gui.py"
) else (
  py -3.12 "%~dp0piper_builder_gui.py"
)
if errorlevel 1 pause
