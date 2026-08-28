@echo off
setlocal
cd /d "%~dp0"
findstr /c:"setup-v5" "tools\.setup-complete" >nul 2>nul
if errorlevel 1 (
  echo First launch: installing Piper and the RVC engine automatically.
  echo This is a large one-time download and may take a while.
  echo.
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -NoLaunch
  if errorlevel 1 (
    echo.
    echo Automatic setup did not finish. Check the message above, then launch again to retry.
    pause
    exit /b 1
  )
)
if exist "tools\python\python.exe" (
  "tools\python\python.exe" app.py
) else (
  py -3.12 app.py 2>nul || python app.py
)
if errorlevel 1 (
  echo.
  echo The app could not start. Run "Easy Setup.cmd" first.
  pause
)
