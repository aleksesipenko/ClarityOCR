@echo off
:: ClarityOCR Windows Installer
:: Requires Python 3.10-3.12 and NVIDIA GPU with CUDA support

echo ============================================================
echo ClarityOCR Installer
echo ============================================================
echo.

:: Try to find compatible Python (3.10, 3.11, or 3.12)
:: Python 3.13+ is NOT supported by PyTorch CUDA yet

set PYTHON_CMD=

:: Check for py launcher with specific versions first
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12
    goto :found_python
)

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.11
    goto :found_python
)

py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.10
    goto :found_python
)

:: Check default python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Detected Python version: %PYVER%

:: Extract major.minor version
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

:: Check if version is compatible (3.10, 3.11, or 3.12)
if "%PYMAJOR%"=="3" (
    if "%PYMINOR%"=="10" set PYTHON_CMD=python
    if "%PYMINOR%"=="11" set PYTHON_CMD=python
    if "%PYMINOR%"=="12" set PYTHON_CMD=python
)

if "%PYTHON_CMD%"=="" (
    echo.
    echo ERROR: Compatible Python not found!
    echo        PyTorch CUDA requires Python 3.10, 3.11, or 3.12
    echo        Your version: %PYVER%
    echo.
    echo Please install Python 3.11 from:
    echo   https://www.python.org/downloads/release/python-31110/
    echo.
    echo Make sure to check "Add to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:found_python
echo Using: %PYTHON_CMD%
for /f "tokens=*" %%v in ('%PYTHON_CMD% --version') do echo %%v
echo.

echo [1/5] Creating virtual environment...
%PYTHON_CMD% -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip wheel setuptools

echo [4/5] Installing PyTorch with CUDA support...
echo       This may take several minutes...
echo       Trying CUDA 12.6 first, then 12.4, then 12.1...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo       CUDA 12.6 not available, trying CUDA 12.4...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo       CUDA 12.4 not available, trying CUDA 12.1...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        if errorlevel 1 (
            echo WARNING: PyTorch CUDA installation failed
            echo          Trying CPU-only version...
            echo          NOTE: GPU acceleration will NOT work!
            pip install torch torchvision
        )
    )
)

echo [5/5] Installing ClarityOCR and dependencies...
pip install -e .
if errorlevel 1 (
    echo ERROR: Failed to install ClarityOCR
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo To start ClarityOCR:
echo   1. Run: run.bat
echo   2. Open browser: http://127.0.0.1:8008
echo.
echo Or use command line:
echo   venv\Scripts\activate
echo   clarityocr serve
echo.
pause
