@echo off
REM ============================================================
REM  install_service.bat - Register the bridge as a hidden
REM  Windows startup task. Admin installs at boot; non-admin
REM  falls back to current-user logon auto-start.
REM
REM  This uses Windows Task Scheduler instead of sc.exe because
REM  main.py is a normal Python app, not a native SCM service.
REM ============================================================
setlocal
cd /d "%~dp0"

set "TASK_NAME=MultiProviderOAuthBridge"
set "INSTALL_PS=%~dp0install_service_task.ps1"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if not exist "%POWERSHELL_EXE%" (
    echo [bridge] LOI: khong tim thay Windows PowerShell.
    exit /b 1
)

"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%INSTALL_PS%" -TaskName "%TASK_NAME%"
if errorlevel 1 (
    echo [bridge] LOI: khong tao duoc Windows startup task.
    exit /b 1
)

echo [bridge] Khoi dong bridge ngay bay gio...
call "%~dp0start.bat"

echo.
echo [bridge] Da cai startup task: %TASK_NAME%
echo [bridge] Neu chay voi quyen Administrator: bridge se tu bat cung luc Windows khoi dong.
echo [bridge] Neu khong co quyen Administrator: bridge se tu bat khi user hien tai dang nhap.
echo [bridge] Dung tam thoi: stop.bat
echo [bridge] Tat auto-start va dung bridge: remove_service.bat
endlocal
