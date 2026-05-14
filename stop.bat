@echo off
REM ============================================================
REM  stop.bat - Dung bridge dang chay an
REM ============================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "TASK_NAME=MultiProviderOAuthBridge"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "SCHTASKS_EXE=%SystemRoot%\System32\schtasks.exe"
set "NETSTAT_EXE=%SystemRoot%\System32\netstat.exe"
set "FINDSTR_EXE=%SystemRoot%\System32\findstr.exe"

if not defined BRIDGE_PORT (
    if defined OPENAI_BRIDGE_PORT (
        set "BRIDGE_PORT=%OPENAI_BRIDGE_PORT%"
    ) else if defined GOOGLE_BRIDGE_PORT (
        set "BRIDGE_PORT=%GOOGLE_BRIDGE_PORT%"
    ) else (
        set "BRIDGE_PORT=12345"
    )
)

set "KILLED=0"

REM If a startup task action is still running, ask Task Scheduler to end it.
if exist "%SCHTASKS_EXE%" "%SCHTASKS_EXE%" /Query /TN "%TASK_NAME%" >nul 2>nul
if not errorlevel 1 (
    "%SCHTASKS_EXE%" /End /TN "%TASK_NAME%" >nul 2>nul
)

REM 1) Kill theo PID file
if exist data\bridge.pid (
    set /p PID=<data\bridge.pid
    if defined PID (
        "%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -Command "try { Stop-Process -Id !PID! -Force -ErrorAction Stop; exit 0 } catch { exit 1 }" >nul 2>nul
        if not errorlevel 1 (
            echo [bridge] Da kill PID !PID!.
            set "KILLED=1"
        )
    )
    del data\bridge.pid >nul 2>nul
)

REM 2) Fallback: kill bat ky process nao dang LISTEN port
for /f "tokens=5" %%a in ('%NETSTAT_EXE% -ano ^| %FINDSTR_EXE% ":%BRIDGE_PORT% " ^| %FINDSTR_EXE% LISTENING') do (
    "%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -Command "try { Stop-Process -Id %%a -Force -ErrorAction Stop; exit 0 } catch { exit 1 }" >nul 2>nul
    if not errorlevel 1 (
        echo [bridge] Da kill PID %%a tren port %BRIDGE_PORT%.
        set "KILLED=1"
    )
)

if "!KILLED!"=="0" (
    echo [bridge] Khong tim thay bridge dang chay tren port %BRIDGE_PORT%.
) else (
    echo [bridge] Da dung bridge.
)

endlocal
