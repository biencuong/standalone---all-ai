# Windows Auto-Start

This app can auto-start with Windows by registering a hidden Scheduled Task.

## Files

- `install_service.bat`: install auto-start and start the bridge now
- `remove_service.bat`: stop the bridge and remove auto-start
- `bridge_runtime.ps1`: shared runtime used by manual run and background task

## Behavior

- Run `install_service.bat` as Administrator:
  - creates a boot-time task
  - runs as `SYSTEM`
  - starts with Windows before user logon
- Run `install_service.bat` without Administrator:
  - falls back to current-user auto-start at logon

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
