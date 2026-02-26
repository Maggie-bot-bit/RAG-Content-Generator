@echo off
echo ========================================
echo RAG Content Generator - Setup Script
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Trying python3...
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Python3 not found. Trying py launcher...
        py --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo ERROR: Python is not installed or not in PATH.
            echo Please install Python 3.8+ from https://www.python.org/
            pause
            exit /b 1
        ) else (
            set PYTHON_CMD=py
        )
    ) else (
        set PYTHON_CMD=python3
    )
) else (
    set PYTHON_CMD=python
)

echo Python found: %PYTHON_CMD%
echo.

echo Creating directories...
if not exist "docs" mkdir docs
if not exist "rag_store" mkdir rag_store
echo.

echo Installing dependencies...
echo This may take several minutes...
%PYTHON_CMD% -m pip install --upgrade pip
%PYTHON_CMD% -m pip install -r requirements_rag.txt

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Setup completed successfully!
    echo ========================================
    echo.
    echo Next steps:
    echo 1. Add some documents (.txt, .md, or .pdf) to the 'docs' folder
    echo 2. Run: streamlit run app.py
    echo    OR
    echo    Run: %PYTHON_CMD% rag_local.py ingest --docs ./docs --store ./rag_store
    echo.
) else (
    echo.
    echo ERROR: Installation failed. Please check the error messages above.
    echo.
)

pause

