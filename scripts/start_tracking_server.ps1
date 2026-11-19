# Start the local MLflow tracking server from the project root.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/start_tracking_server.ps1 [-Port 5000]
param(
    [int]$Port = 5000
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$logPath = Join-Path $projectRoot "mlflow-server.log"
$errorLogPath = Join-Path $projectRoot "mlflow-server.err.log"
$pidPath = Join-Path $projectRoot "mlflow-server.pid"

$server = Start-Process -FilePath "uv" `
    -ArgumentList @(
        "run", "python", "-m", "mlflow", "server",
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--artifacts-destination", "./mlartifacts"
    ) `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError $errorLogPath `
    -PassThru `
    -WindowStyle Hidden

$server.Id | Set-Content -Path $pidPath
Write-Output "MLflow server starting (PID $($server.Id)) on http://127.0.0.1:$Port"
Write-Output "Backend store: sqlite:///$projectRoot/mlflow.db"
Write-Output "Artifacts    : $projectRoot/mlartifacts"
Write-Output "Logs         : $logPath"
