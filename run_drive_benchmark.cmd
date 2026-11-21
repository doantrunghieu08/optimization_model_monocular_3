@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "PYTHONPATH=%CD%\.runtime-deps"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
"D:\miniconda\envs\gvhmr\python.exe" -u run_drive_benchmark.py --local-only --reverse %*
set "benchmark_exit=%ERRORLEVEL%"
popd
exit /b %benchmark_exit%
