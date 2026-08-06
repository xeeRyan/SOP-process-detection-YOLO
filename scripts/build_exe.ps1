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

$requiredRuntimePaths = @(
    "dist\SOP_PYD\_internal",
    "dist\SOP_PYD\config\app_config.json",
    "dist\SOP_PYD\models\hand_landmarker.task",
    "dist\SOP_PYD\projects"
)
foreach ($relativePath in $requiredRuntimePaths) {
    $requiredPath = Join-Path $projectRoot $relativePath
    if (-not (Test-Path $requiredPath)) {
        throw "Packaging completed without required runtime asset: $requiredPath"
    }
}

Write-Host "Package ready: $exePath"
