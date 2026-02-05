@echo off
setlocal

cd /d "%~dp0\.."

set "PYTHON_CMD="
where py >nul 2>&1 && set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
  where python >nul 2>&1 && set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
  echo [ERROR] Python 3.10+ not found.
  echo Install Python from https://www.python.org/downloads/ and re-run.
  pause
  exit /b 1
)

if not exist venv (
  echo [INFO] Creating virtual environment...
  %PYTHON_CMD% -m venv venv
  if errorlevel 1 goto :error
)

call venv\Scripts\activate.bat
if errorlevel 1 goto :error

python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist .env (
  if exist .env.example (
    copy .env.example .env >nul
    echo [INFO] Created .env from .env.example
  ) else (
    echo [WARN] .env.example not found. Create .env manually.
  )
)

echo.
echo Setup complete.
echo 1^) Open .env and set GROQ_API_KEY
echo 2^) Run: python app.py
echo.
pause
exit /b 0

:error
echo.
echo [ERROR] Setup failed.
pause
exit /b 1
