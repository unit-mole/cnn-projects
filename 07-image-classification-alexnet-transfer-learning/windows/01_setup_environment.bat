@echo off
setlocal
cd /d "%~dp0\.."

echo ============================================================
echo Project 07 - Native Windows RTX Setup
echo ============================================================

py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python 3.12 was not found.
  echo Run: py --list
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating project virtual environment...
  py -3.12 -m venv .venv
  if errorlevel 1 exit /b 1
) else (
  echo Existing .venv found. Reusing it.
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

 echo Installing CUDA-enabled PyTorch for the RTX GPU...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 exit /b 1

 echo Installing the remaining project packages...
python -m pip install -r requirements-training.txt
if errorlevel 1 exit /b 1

 echo Verifying CUDA and GPU access...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT DETECTED')"
if errorlevel 1 exit /b 1

python scripts\validate_project.py
if errorlevel 1 exit /b 1
python -m pytest -q
if errorlevel 1 exit /b 1

 echo.
 echo SETUP COMPLETE.
 echo Next run: windows\02_start_notebook.bat
endlocal
