@echo off
cd /d "%~dp0\.."
if not exist ".venv\Scripts\activate.bat" (
  echo Environment missing. Run windows\01_setup_environment.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
jupyter lab notebooks\07_pytorch_cnn_model_comparison.ipynb
