$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install uv, then run this script again."
}

uv python install 3.12
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    uv venv .venv --python 3.12
}

$downloadDir = Join-Path $projectRoot ".downloads"
$torchWheel = Join-Path $downloadDir "torch-2.12.0+cu126-cp312-cp312-win_amd64.whl"
$torchUrl = "https://download-r2.pytorch.org/whl/cu126/torch-2.12.0%2Bcu126-cp312-cp312-win_amd64.whl"
$torchSha256 = "194F5BD0721B968E769777B8AB4DBE51DD7FFDFDF295DB045093B94A1B9765BB"

New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
$downloadRequired = -not (Test-Path $torchWheel)
if (-not $downloadRequired) {
    $downloadRequired = (Get-FileHash -Algorithm SHA256 -LiteralPath $torchWheel).Hash -ne $torchSha256
}

if ($downloadRequired) {
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        curl.exe -L --fail --show-error --continue-at - --output $torchWheel $torchUrl
        if ($LASTEXITCODE -eq 0) {
            break
        }
        Start-Sleep -Seconds 3
    }
}

if ((Get-FileHash -Algorithm SHA256 -LiteralPath $torchWheel).Hash -ne $torchSha256) {
    throw "Torch wheel download is incomplete or has an invalid SHA-256 hash."
}

$env:UV_HTTP_TIMEOUT = "600"
$env:UV_CONCURRENT_DOWNLOADS = "1"
uv pip install --python .venv\Scripts\python.exe $torchWheel -r requirements-build.txt --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match --link-mode copy

Write-Host ""
Write-Host "Environment ready: $projectRoot\.venv"
Write-Host "Activate: .\.venv\Scripts\Activate.ps1"
Write-Host "Check: uv run python scripts\check_env.py"
