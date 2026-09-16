@echo off
setlocal
set "CI_PYTHON=%LOCALAPPDATA%\miniconda3\envs\cidenoise\python.exe"
pushd "%~dp0"
"%CI_PYTHON%" -m training.train_noise2noise %*
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
