Param(
    [string]$Name = ".venv"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $root $Name
$activate = Join-Path $venvPath "Scripts\Activate.ps1"
$requirements = Join-Path $root "requirements.txt"

if (!(Test-Path $venvPath)) {
    Write-Host "Creating venv at $venvPath"
    py -m venv $venvPath
}

if (!(Test-Path $activate)) {
    throw "Activation script not found at $activate"
}

Write-Host "Activating $Name and installing dependencies..."
. $activate
pip install -r $requirements

Write-Host ""
Write-Host "Done. To activate manually in this shell later, run:"
Write-Host "  . $activate"

