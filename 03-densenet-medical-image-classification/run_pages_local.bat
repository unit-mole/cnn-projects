@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-pages\Scripts\python.exe" (
    echo Creating the GitHub Pages conversion environment...
    python -m venv .venv-pages || goto :error
)

call .venv-pages\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip || goto :error
pip install -r requirements-pages.txt || goto :error
python scripts\convert_browser_model.py || goto :error

echo.
echo Open http://localhost:8000 in your browser.
echo Press Ctrl+C to stop the local server.
python -m http.server 8000 --directory web
goto :eof

:error
echo.
echo GitHub Pages local setup failed. Review the message above.
exit /b 1
