@echo off
cd /d "%~dp0"
set "RUN_DIR=outputs\my_experiment"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pet_classifier prepare
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pet_classifier train --output "%RUN_DIR%"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pet_classifier evaluate --checkpoint "%RUN_DIR%\best.pt" --output "%RUN_DIR%\evaluation"
if errorlevel 1 goto failed
echo Done. Results are in %RUN_DIR%.
pause
exit /b 0
:failed
echo The command failed. See the error above. Existing experiments are never overwritten.
pause
exit /b 1
