@echo off
rem Switches OCR to the NVIDIA GPU: replaces the CPU onnxruntime with onnxruntime-gpu plus the
rem CUDA 13 / cuDNN 9 runtime from pip (about 1.5 GB; no separate CUDA install needed).
rem Needs an NVIDIA driver new enough for CUDA 13 (R580 or later). If anything fails, the CPU
rem build is put back, so the HUD keeps working. Run it again after reinstalling requirements.txt.
rem Undo: .venv\Scripts\python -m pip uninstall -y onnxruntime-gpu  then  pip install onnxruntime
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run start.bat once first to create .venv.
  goto :eof
)
where nvidia-smi >nul 2>nul || (
  echo No NVIDIA driver found: OCR stays on the CPU.
  goto :eof
)
set PY=".venv\Scripts\python.exe"
echo Installing onnxruntime-gpu with CUDA and cuDNN...
%PY% -m pip uninstall -y onnxruntime >nul 2>nul
%PY% -m pip install "onnxruntime-gpu[cuda,cudnn]" || goto :fallback
%PY% -c "import onnxruntime as o; o.preload_dlls(); import sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" || goto :fallback
echo GPU OCR ready. The HUD prints "OCR device: cuda" when it starts.
goto :eof
:fallback
echo GPU setup failed: putting the CPU onnxruntime back.
%PY% -m pip uninstall -y onnxruntime-gpu >nul 2>nul
%PY% -m pip install --force-reinstall onnxruntime
