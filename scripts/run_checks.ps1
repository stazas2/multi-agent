$ErrorActionPreference = "Stop"

# Navigate to repository root
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot

function Invoke-IfAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments = @(),
        [string]$DisplayName = $null
    )

    $display = if ($DisplayName) { $DisplayName } else { $Command }
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        Write-Host "Skipping $display (command not found)" -ForegroundColor Yellow
        return
    }

    Write-Host "Running $display..." -ForegroundColor Cyan
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$display exited with code $LASTEXITCODE"
    }
}

Invoke-IfAvailable -Command "ruff" -Arguments @("check") -DisplayName "ruff lint"
Invoke-IfAvailable -Command "black" -Arguments @("--check", ".") -DisplayName "black format check"
Invoke-IfAvailable -Command "mypy" -Arguments @("orchestrator", "agents", "shared") -DisplayName "mypy type check"
Invoke-IfAvailable -Command "pytest" -Arguments @() -DisplayName "pytest"

Write-Host "All available checks completed successfully." -ForegroundColor Green
