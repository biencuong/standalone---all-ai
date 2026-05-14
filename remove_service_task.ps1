param(
    [string]$TaskName = "MultiProviderOAuthBridge"
)

$ErrorActionPreference = "Stop"

$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $Task) {
    Write-Host "[bridge] Startup task '$TaskName' does not exist."
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "[bridge] Removed startup task '$TaskName'."
