@echo off
REM SkinWise AI - stop the app started by run_skinwise.bat
REM Ends the processes listening on port 8000 (API) and 5173 (website).
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do taskkill /PID %%p /T /F
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5173 " ^| findstr "LISTENING"') do taskkill /PID %%p /T /F
echo SkinWise stopped. You can close the remaining command windows.
timeout /t 5 /nobreak >nul
