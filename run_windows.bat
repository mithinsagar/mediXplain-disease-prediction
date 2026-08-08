@echo off
REM ---------------------------------------------------------------------------
REM  MediXplain - Windows launcher
REM  Author:  Mithin Sagar S
REM  GitHub:  https://github.com/mithinsagar
REM  Project: MediXplain - Explainable Disease Prediction System
REM ---------------------------------------------------------------------------

setlocal enabledelayedexpansion

echo.
echo ================================================================
echo   MediXplain - Explainable Disease Prediction System
echo   Author: Mithin Sagar S ^| github.com/mithinsagar
echo ================================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if not exist "data\raw\Final_Augmented_dataset_Diseases_and_Symptoms.csv" (
    echo.
    echo Dataset CSV not found at data\raw\.
    echo Attempting to download via scripts\download_dataset.py...
    python scripts\download_dataset.py
)

echo.
echo Preparing dataset...
python -m ml.prepare_dataset

echo.
echo Training classical models...
python -m ml.train_models

echo.
echo Building FAISS RAG index...
python -m rag.build_faiss_index

echo.
echo Starting Flask app on http://127.0.0.1:5000/
python -m app.app

endlocal
