@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
rem Streamlit may show a one-time optional email prompt on first launch.
rem Submit a blank line automatically; no email is collected or stored.
(echo.) | ".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
pause
