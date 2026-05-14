@echo off
REM ============================================================
REM  restart.bat - Restart the bridge if it is running, or start
REM  it if it is currently stopped.
REM ============================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "NETSTAT_EXE=%SystemRoot%\System32\netstat.exe"
set "FINDSTR_EXE=%SystemRoot%\System32\findstr.exe"
set "PING_EXE=%SystemRoot%\System32\ping.exe"

if not defined BRIDGE_PORT (
    if defined OPENAI_BRIDGE_PORT (
        set "BRIDGE_PORT=%OPENAI_BRIDGE_PORT%"
    ) else if defined GOOGLE_BRIDGE_PORT (
        set "BRIDGE_PORT=%GOOGLE_BRIDGE_PORT%"
    ) else (
        set "BRIDGE_PORT=12345"
    )
)

set "RUNNING=0"

if exist data\bridge.pid (
    set /p PID=<data\bridge.pid
    if defined PID (
        "%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -Command "try { Get-Process -Id !PID! -ErrorAction Stop | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
        if not errorlevel 1 set "RUNNING=1"
    )
)

if "!RUNNING!"=="0" (
    for /f "tokens=5" %%a in ('%NETSTAT_EXE% -ano ^| %FINDSTR_EXE% ":%BRIDGE_PORT% " ^| %FINDSTR_EXE% LISTENING') do (
        set "RUNNING=1"
    )
)

if "!RUNNING!"=="1" (
    echo [bridge] Bridge dang chay, se dung roi khoi dong lai...
    call "%~dp0stop.bat"
    REM Give Windows a moment to release the port after taskkill.
    "%PING_EXE%" -n 3 127.0.0.1 >nul
) else (
    echo [bridge] Bridge dang tat, se bat lai...
)

call "%~dp0start.bat"
endlocal
