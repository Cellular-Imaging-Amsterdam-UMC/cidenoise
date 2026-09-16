@echo off
setlocal
set "CI_PYTHON=%LOCALAPPDATA%\miniconda3\envs\cidenoise\python.exe"
if not exist "%CI_PYTHON%" (
  echo Run create_env.cmd first.
  exit /b 1
)
"%CI_PYTHON%" "%~dp0launcher.py" %*
