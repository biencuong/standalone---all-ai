param(
    [string]$TaskName = "MultiProviderOAuthBridge"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimePs = Join-Path $Root "bridge_runtime.ps1"
$HiddenLauncher = Join-Path $Root "run_hidden.vbs"

if (-not (Test-Path -LiteralPath $RuntimePs)) {
    throw "Missing bridge_runtime.ps1 at $RuntimePs"
}

if (-not (Test-Path -LiteralPath $HiddenLauncher)) {
    throw "Missing hidden launcher at $HiddenLauncher"
}

$IsAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).
    IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$WScriptExe = Join-Path $env:SystemRoot "System32\wscript.exe"

if (-not (Test-Path -LiteralPath $PowerShellExe)) {
    throw "Missing Windows PowerShell at $PowerShellExe"
}
if (-not (Test-Path -LiteralPath $WScriptExe)) {
    throw "Missing Windows Script Host at $WScriptExe"
}

$Action = New-ScheduledTaskAction `
    -Execute $WScriptExe `
    -Argument ('//B //NoLogo "{0}"' -f $HiddenLauncher) `
    -WorkingDirectory $Root

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

if ($IsAdmin) {
    $Trigger = New-ScheduledTaskTrigger -AtStartup
    $Principal = New-ScheduledTaskPrincipal `
        -UserId "SYSTEM" `
        -LogonType ServiceAccount `
        -RunLevel Highest
    $ModeLabel = "startup with Windows boot (SYSTEM)"
} else {
    $CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
    $Principal = New-ScheduledTaskPrincipal `
        -UserId $CurrentUser `
        -LogonType Interactive `
        -RunLevel Limited
    $ModeLabel = "logon auto-start for current user"
}

$Task = New-ScheduledTask `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Multi-Provider OAuth Bridge local API startup task."

$Task.Settings.Hidden = $true

Register-ScheduledTask -TaskName $TaskName -InputObject $Task -Force | Out-Null

Write-Host "[bridge] Registered hidden startup task '$TaskName' in mode: $ModeLabel."
