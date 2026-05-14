@echo off
REM ============================================================
REM  remove_service.bat - Stop the bridge and remove the Windows
REM  startup task created by install_service.bat.
REM ============================================================
setlocal
cd /d "%~dp0"

set "TASK_NAME=MultiProviderOAuthBridge"
set "REMOVE_PS=%~dp0remove_service_task.ps1"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

call "%~dp0stop.bat"

if not exist "%POWERSHELL_EXE%" (
    echo [bridge] LOI: khong tim thay Windows PowerShell.
    exit /b 1
)

"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%REMOVE_PS%" -TaskName "%TASK_NAME%"
if errorlevel 1 (
    echo [bridge] LOI: khong xoa duoc Windows startup task.
    exit /b 1
)

endlocal
