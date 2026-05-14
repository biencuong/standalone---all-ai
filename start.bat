@echo off
REM ============================================================
REM  start.bat - Khoi dong bridge an, khong hien cua so console
REM  Script nay goi bridge_runtime.ps1 de tim Python Windows
REM  va chay bridge qua mot host PowerShell an.
REM ============================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "TASK_NAME=MultiProviderOAuthBridge"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "SCHTASKS_EXE=%SystemRoot%\System32\schtasks.exe"
set "PING_EXE=%SystemRoot%\System32\ping.exe"
set "HEALTH_URL=http://127.0.0.1:12345/health"

if not defined BRIDGE_HOST set BRIDGE_HOST=127.0.0.1
if not defined BRIDGE_PORT set BRIDGE_PORT=12345
set "HEALTH_URL=http://%BRIDGE_HOST%:%BRIDGE_PORT%/health"

if not exist data mkdir data

REM ----- 0) Neu /health da OK thi coi nhu dang chay -----
"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing '%HEALTH_URL%' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 (
    echo [bridge] Da chay san - /health OK.
    echo [bridge] UI:          http://%BRIDGE_HOST%:%BRIDGE_PORT%/
    echo [bridge] OAuth token: http://%BRIDGE_HOST%:%BRIDGE_PORT%/v1/oauth/token
    echo [bridge] Chat API:    http://%BRIDGE_HOST%:%BRIDGE_PORT%/v1/chat/completions
    goto :end
)

REM ----- 1) Kiem tra dang chay chua -----
if exist data\bridge.pid (
    set /p OLDPID=<data\bridge.pid
    "%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -Command "try { Get-Process -Id !OLDPID! -ErrorAction Stop | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
    if not errorlevel 1 (
        echo [bridge] Da chay san - PID !OLDPID!. Dung stop.bat truoc neu can restart.
        goto :end
    ) else (
        del data\bridge.pid >nul 2>nul
    )
)

REM ----- 2) Tim PowerShell de khoi dong hidden/detached -----
if not exist "%POWERSHELL_EXE%" (
    echo [bridge] LOI: khong tim thay Windows PowerShell.
    exit /b 1
)

REM ----- 3) Neu da cai Scheduled Task, uu tien chay dung duong Windows startup -----
if exist "%SCHTASKS_EXE%" (
    "%SCHTASKS_EXE%" /Query /TN "%TASK_NAME%" >nul 2>nul
    if not errorlevel 1 (
        "%SCHTASKS_EXE%" /Run /TN "%TASK_NAME%" >nul 2>nul
        if not errorlevel 1 goto :wait_pid
    )
)

REM ----- 4) Fallback: khoi dong an truc tiep -----
REM  Dung Start-Process de tien trinh khong bi dong khi cmd/task ket thuc.
set "OUT_LOG=%~dp0data\server.out.log"
set "ERR_LOG=%~dp0data\server.err.log"
"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%POWERSHELL_EXE%' -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','%~dp0bridge_runtime.ps1','-Mode','service' -WorkingDirectory '%~dp0' -WindowStyle Hidden"
if errorlevel 1 (
    echo [bridge] LOI: khong khoi dong duoc bridge.
    exit /b 1
)

REM Cho main.py kip ghi PID file va endpoint /health len (toi da 10s)
set /a TRY=0
:wait_pid
if exist data\bridge.pid goto :pid_ok
set /a TRY+=1
if !TRY! GEQ 20 goto :pid_timeout
"%PING_EXE%" -n 2 127.0.0.1 >nul
goto :wait_pid

:pid_ok
set /p PID=<data\bridge.pid
set /a TRY=0
:wait_health
"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing '%HEALTH_URL%' -TimeoutSec 3; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 goto :health_ok
set /a TRY+=1
if !TRY! GEQ 10 goto :health_timeout
"%PING_EXE%" -n 2 127.0.0.1 >nul
goto :wait_health

:health_ok
echo [bridge] Da khoi dong an. PID=!PID!
echo [bridge] UI:          http://%BRIDGE_HOST%:%BRIDGE_PORT%/
echo [bridge] OAuth token: http://%BRIDGE_HOST%:%BRIDGE_PORT%/v1/oauth/token
echo [bridge] Chat API:    http://%BRIDGE_HOST%:%BRIDGE_PORT%/v1/chat/completions
echo [bridge] Stop bang stop.bat
goto :end

:health_timeout
echo [bridge] PID=!PID! da duoc tao nhung /health chua phan hoi. Kiem tra data\server.err.log va data\bridge.log.
exit /b 1

:pid_timeout
echo [bridge] Canh bao: chua thay PID file sau 10s. Bridge co the loi - chay run.bat de xem log truc tiep.
exit /b 1

:end
endlocal
