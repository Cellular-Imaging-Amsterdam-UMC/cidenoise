@echo off
setlocal
set "CI_PYTHON=%LOCALAPPDATA%\miniconda3\envs\cidenoise\python.exe"
pushd "%~dp0"
"%CI_PYTHON%" -m training.download_confocal %*
if errorlevel 1 goto failed
"%CI_PYTHON%" -m training.train_noise2noise --prepare-only %*
if errorlevel 1 goto failed
popd
exit /b 0
:failed
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
