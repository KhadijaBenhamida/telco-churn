# Stop the MLflow tracking server started by scripts/start_tracking_server.ps1.
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $projectRoot "mlflow-server.pid"

if (-not (Test-Path $pidPath)) {
    Write-Output "No mlflow-server.pid found; stop the server with Ctrl+C in its terminal instead."
    exit 0
}

$serverPid = Get-Content $pidPath | Select-Object -First 1
$running = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
if ($null -eq $running) {
    Write-Output "Process $serverPid is not running."
} else {
    Stop-Process -Id $serverPid -Force
    Write-Output "Stopped MLflow server process $serverPid."
}
Remove-Item -LiteralPath $pidPath -ErrorAction SilentlyContinue
