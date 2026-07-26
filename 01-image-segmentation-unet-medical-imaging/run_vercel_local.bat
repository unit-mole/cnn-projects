@echo off
setlocal
cd /d "%~dp0"
echo Validating browser files...
node scripts\validate-web.mjs
if errorlevel 1 exit /b 1
echo.
echo Starting local server at http://127.0.0.1:8000
python -m http.server 8000 --directory web
