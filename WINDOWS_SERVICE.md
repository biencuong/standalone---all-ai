# Windows Auto-Start

This app can auto-start with Windows by registering a hidden Scheduled Task.
The task uses `wscript.exe` plus `run_hidden.vbs`, so Windows logon/startup
does not show a black `cmd.exe` or PowerShell console window.

## Files

- `install_service.bat`: install auto-start and start the bridge now
- `remove_service.bat`: stop the bridge and remove auto-start
- `bridge_runtime.ps1`: shared runtime used by manual run and background task
- `run_hidden.vbs`: hidden Windows Script Host launcher used by the startup task

## Behavior

- Run `install_service.bat` as Administrator:
  - creates a boot-time task
  - runs as `SYSTEM`
  - starts with Windows before user logon
  - recommended when you want service-like startup with Windows boot
- Run `install_service.bat` without Administrator:
  - falls back to current-user auto-start at logon
  - still uses a hidden launcher, so it should not show a black cmd window

## Python requirement

The launcher looks for Python in this order:

1. `BRIDGE_PYTHON`
2. `.venv-win\Scripts\python.exe`
3. `.venv\Scripts\python.exe`
4. common Windows Python install paths
5. `python.exe` or `py.exe` from PATH

If your Python is in a custom location, set it before install/start:

```powershell
$env:BRIDGE_PYTHON = "C:\Path\To\python.exe"
```

## Commands

```bat
install_service.bat
start.bat
stop.bat
restart.bat
remove_service.bat
```

## Logs

- `data\server.out.log`
- `data\server.err.log`
- `data\bridge.log`
