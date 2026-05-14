@echo off
setlocal
cd /d "%~dp0"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if not defined BRIDGE_PORT set BRIDGE_PORT=12345
if not defined BRIDGE_HOST set BRIDGE_HOST=127.0.0.1

if not exist "%POWERSHELL_EXE%" (
    echo [bridge] Windows PowerShell khong ton tai o %POWERSHELL_EXE%
    exit /b 1
)

"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0bridge_runtime.ps1" -Mode run
