@echo off
cd /d "%~dp0\.."
if not exist ".venv\Scripts\activate.bat" (
  echo Environment missing. Run windows\01_setup_environment.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\export_to_onnx.py
if errorlevel 1 exit /b 1
python scripts\validate_project.py --require-onnx
if errorlevel 1 exit /b 1
start "" http://127.0.0.1:8000
python scripts\run_local_web_server.py --port 8000
