# Start the PLYXIS platform (API + live twin + web console) on http://127.0.0.1:8000
#   powershell -ExecutionPolicy Bypass -File scripts\start.ps1 [-Port 8000] [-Speed 600] [-Autoplay] [-NoNeural]
param(
    [int]$Port = 8000,
    [double]$Speed = 600,
    [switch]$Autoplay,
    [switch]$NoNeural
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run scripts\setup.ps1 first" }
if (-not (Test-Path "web\dist\index.html")) { throw "web\dist missing - run scripts\setup.ps1 (or: cd web; npx vite build)" }

$cli = @("-m", "plyxis", "serve", "--port", "$Port", "--speed", "$Speed")
if ($Autoplay) { $cli += "--autoplay" }
if ($NoNeural) { $cli += "--no-neural" }
Write-Host "PLYXIS console: http://127.0.0.1:$Port  (Ctrl+C to stop)"
& $py @cli
