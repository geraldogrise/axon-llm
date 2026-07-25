@echo off
REM ============================================================
REM  axon-lang - live training (LinearRouter = TF-IDF + LogReg)
REM  Run this file and watch on screen. Ctrl+C to stop.
REM ============================================================
cd /d "%~dp0"
set "PYTHONPATH=%~dp0python"
set "PATH=C:\msys64\mingw64\bin;%PATH%"

echo.
echo === axon-lang: collect + train (live) ===
echo Downloading from several sources, training the router and measuring accuracy.
echo Two CSVs: downloaded.csv (dedup) and trained.csv (what was trained).
echo Ctrl+C to stop. The model is saved periodically.
echo.

REM Optional: adjust the time (seconds) and the training frequency:
REM   set AXON_BUDGET=10800   (3 hours)
REM   set AXON_FIT_EVERY=600  (train every 600 sources)

python examples\train_linear.py

echo.
echo === Finished ===
pause
