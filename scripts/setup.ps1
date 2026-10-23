# One-time setup on Windows: Python virtual environment + web console build.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment (.venv)"
    python -m venv .venv
}
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
# CPU build of PyTorch is sufficient: all twins and sub-solvers run on CPU
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt

Write-Host "Building the web console (web\dist)"
Set-Location (Join-Path $root "web")
npm install
npx vite build
Set-Location $root

Write-Host "Running the test-suite"
& $py -m pytest tests -q
Write-Host "Done. Start the platform with: powershell -File scripts\start.ps1"
