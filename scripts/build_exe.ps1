$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw ".venv is missing. Run .\scripts\setup_env.ps1 first."
}

.venv\Scripts\python.exe -m PyInstaller SOP_PYD.spec --noconfirm

$exePath = Join-Path $projectRoot "dist\SOP_PYD\SOP_PYD.exe"
if (-not (Test-Path $exePath)) {
    throw "Packaging completed without producing $exePath"
}

Write-Host "Package ready: $exePath"
