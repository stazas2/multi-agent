Param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

# Ensure we run from the repository root
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not available in PATH. Activate your virtual environment first."
}

# Enable local mode and provide a sensible default model if none is set
$env:LOCAL_MODE = "1"
if (-not $env:GEMINI_MODEL) {
    $env:GEMINI_MODEL = "gemini-2.5-flash"
}

Write-Host "Starting orchestrator locally with LOCAL_MODE=1 (host=$Host, port=$Port)..." -ForegroundColor Cyan

$uvicornArgs = @(
    "-m", "uvicorn",
    "orchestrator.main:app",
    "--host", $Host,
    "--port", $Port
)

if ($Reload.IsPresent) {
    $uvicornArgs += "--reload"
}

Write-Host "Command: python $($uvicornArgs -join ' ')" -ForegroundColor DarkGray
Write-Host "Press Ctrl+C to stop." -ForegroundColor Cyan

& python @uvicornArgs
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Uvicorn exited with code $LASTEXITCODE."
}
