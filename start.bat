@echo off
rem Preflop HUD launcher: creates the virtual env on first run, then starts the server.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.11 -m venv .venv || goto :err
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :err
  rem NVIDIA GPU present: run OCR on it (falls back to the CPU build if that fails)
  where nvidia-smi >nul 2>nul && call setup_gpu.bat
)
".venv\Scripts\python.exe" -m preflop_hud %*
goto :eof
:err
echo Setup failed. See README.md (Troubleshooting).
pause
