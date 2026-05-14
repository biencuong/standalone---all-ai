param(
    [ValidateSet("run", "service", "resolve-python")]
    [string]$Mode = "run"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = Join-Path $Root "data"
$OutLog = Join-Path $DataDir "server.out.log"
$ErrLog = Join-Path $DataDir "server.err.log"
$MainPy = Join-Path $Root "main.py"

function Write-BridgeError {
    param([string]$Message)

    if (-not (Test-Path -LiteralPath $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    }
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $ErrLog -Value "[$stamp] $Message"
}

function Test-ReadablePath {
    param([string]$Path)

    if (-not $Path) {
        return $false
    }

    try {
        return (Test-Path -LiteralPath $Path)
    } catch {
        return $false
    }
}

function Resolve-PythonPath {
    if ($env:BRIDGE_PYTHON) {
        if (Test-ReadablePath $env:BRIDGE_PYTHON) {
            return (Resolve-Path -LiteralPath $env:BRIDGE_PYTHON).Path
        }
        throw "BRIDGE_PYTHON points to a missing file: $($env:BRIDGE_PYTHON)"
    }

    $candidates = @(
        (Join-Path $Root ".venv-win\Scripts\python.exe"),
        (Join-Path $Root ".venv\Scripts\python.exe"),
        (Join-Path $Root "venv\Scripts\python.exe"),
        (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"),
        (Join-Path $env:USERPROFILE ".qwenpaw\venv\Scripts\python.exe"),
        (Join-Path $env:LocalAppData "Programs\Python\Python313\python.exe"),
        (Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LocalAppData "Programs\Python\Python311\python.exe"),
        "C:\Program Files\Python313\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Python313\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-ReadablePath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and $pythonCmd.Source -and (Test-ReadablePath $pythonCmd.Source)) {
        if ($pythonCmd.Source -notmatch "WindowsApps\\python(?:3(?:\.\d+)?)?\.exe$") {
            return (Resolve-Path -LiteralPath $pythonCmd.Source).Path
        }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd -and $pyCmd.Source -and (Test-ReadablePath $pyCmd.Source)) {
        return (Resolve-Path -LiteralPath $pyCmd.Source).Path
    }

    return $null
}

function Get-PythonLaunchSpec {
    $pythonExe = Resolve-PythonPath
    if (-not $pythonExe) {
        throw "Cannot find a usable Windows Python. Set BRIDGE_PYTHON to python.exe if needed."
    }

    if ([System.IO.Path]::GetFileName($pythonExe).Equals("py.exe", [System.StringComparison]::OrdinalIgnoreCase)) {
        return @{
            FilePath = $pythonExe
            Arguments = @("-3", $MainPy)
        }
    }

    return @{
        FilePath = $pythonExe
        Arguments = @($MainPy)
    }
}

function Quote-CmdArgument {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }

    return '"' + ($Value -replace '"', '\"') + '"'
}

$script:LastDependencyError = ""

function Test-BridgeDependencies {
    param(
        [string]$PythonExe
    )

    $script:LastDependencyError = ""
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $PythonExe -c "import fastapi, uvicorn, httpx, multipart" 2>&1
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        $script:LastDependencyError = (($output | Out-String).Trim())
        return $false
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
}

try {
    $launch = Get-PythonLaunchSpec

    if ($Mode -eq "resolve-python") {
        Write-Output $launch.FilePath
        exit 0
    }

    if (-not (Test-Path -LiteralPath $MainPy)) {
        throw "Missing main.py at $MainPy"
    }

    if (-not (Test-BridgeDependencies -PythonExe $launch.FilePath)) {
        $depMessage = "Python at $($launch.FilePath) is missing FastAPI bridge dependencies. Install them first with `"$($launch.FilePath)`" -m pip install -r requirements.txt."
        if ($script:LastDependencyError) {
            $depMessage = "$depMessage Last import error: $($script:LastDependencyError)"
        }
        throw $depMessage
    }

    Push-Location $Root
    try {
        if ($Mode -eq "service") {
            if (-not (Test-Path -LiteralPath $DataDir)) {
                New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
            }

            $argumentString = (($launch.Arguments | ForEach-Object { Quote-CmdArgument $_ }) -join " ")
            Start-Process `
                -FilePath $launch.FilePath `
                -ArgumentList $argumentString `
                -WorkingDirectory $Root `
                -WindowStyle Hidden `
                -RedirectStandardOutput $OutLog `
                -RedirectStandardError $ErrLog | Out-Null
            exit 0
        } else {
            & $launch.FilePath @($launch.Arguments)
        }
        $code = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
        exit $code
    } finally {
        Pop-Location
    }
} catch {
    if ($Mode -eq "service") {
        Write-BridgeError $_.Exception.Message
    }
    Write-Error $_.Exception.Message
    exit 1
}
