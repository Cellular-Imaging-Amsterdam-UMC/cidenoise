@echo off
setlocal
set "CI_CONDA=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
set "CI_PYTHON=%LOCALAPPDATA%\miniconda3\envs\cidenoise\python.exe"
if not exist "%CI_PYTHON%" (
  "%CI_CONDA%" env create -f "%~dp0environment.yml"
  if errorlevel 1 exit /b 1
)
"%CI_PYTHON%" -m pip install -r "%~dp0requirements.txt" -r "%~dp0requirements_launcher.txt" -r "%~dp0requirements_dev.txt" -c "%~dp0requirements-lock-windows.txt"
if errorlevel 1 exit /b 1
if not exist "%~dp0inputfolder" mkdir "%~dp0inputfolder"
if not exist "%~dp0outputfolder" mkdir "%~dp0outputfolder"
"%CI_PYTHON%" "%~dp0tools\download_models.py"
if errorlevel 1 exit /b 1
"%CI_PYTHON%" "%~dp0tools\cuda_smoke.py"
