@echo off
chcp 65001 >nul
title Campus Rules Agent - one-click launcher
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo.
  echo [ERROR] Virtual environment not found:
  echo   %PY%
  echo Please create it first:
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo   .venv\Scripts\python.exe -m pip install -e .
  echo.
  pause
  exit /b 1
)

echo ============================================================
echo   Campus Rules Agent  /  one-click launcher
echo.
echo   Local service : http://127.0.0.1:8000
echo   Auto-open     : http://127.0.0.1:8000/dashboard
echo   Other pages   : /chat  /demo  /docs
echo.
echo   Keep this window open. Close it or press Ctrl+C to stop.
echo   Starting the service and opening the dashboard cost 0 API calls.
echo ============================================================
echo.

"%PY%" -m campus_rules serve --open %*

echo.
echo Service stopped. Press any key to close this window.
pause >nul
