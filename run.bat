@echo off
:: ClarityOCR Quick Start
:: Starts the web server on http://127.0.0.1:8008

echo Starting ClarityOCR...

:: Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found.
    echo Please run install.bat first.
    pause
    exit /b 1
)

:: Start server
echo.
echo ClarityOCR Web UI starting...
echo Open your browser to: http://127.0.0.1:8008
echo.
echo Press Ctrl+C to stop the server.
echo.

clarityocr serve

pause
