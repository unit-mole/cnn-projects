@echo off
setlocal
cd /d "%~dp0"
echo Starting the TensorFlow.js demo at http://localhost:8000
python -m http.server 8000
endlocal
