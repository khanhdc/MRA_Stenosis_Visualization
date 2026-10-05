@echo off
setlocal
set "PROJ=E:\MRA_Data\MRA_Annotation_Ver2\stenosis labeling"
set "PY=C:\ProgramData\MRA_Labeling_Env\Scripts\python.exe"

if not exist "%PY%" (
  echo ERROR: shared Python not found at
  echo   %PY%
  echo Ask Ulcer to restore the shared environment.
  pause & exit /b 1
)

cd /d "%PROJ%" || (echo ERROR: cannot reach "%PROJ%" & pause & exit /b 1)

title MRA Labeling
"%PY%" interactive_viewer.py
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo The viewer exited with code %RC%. Message above.
  pause
)
exit /b %RC%
