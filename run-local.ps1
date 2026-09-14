# Roda o app SEM Docker (quando containerd da I/O error)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path "config.toml")) {
    Copy-Item "config.toml.example" "config.toml"
}

$venv = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "Criando venv..." -ForegroundColor Yellow
    python -m venv $venv
}
& "$venv\Scripts\Activate.ps1"

Write-Host "Instalando dependencias (core)..." -ForegroundColor Yellow
pip install -q -U pip
pip install -q -r requirements-core.txt

Write-Host "Opcional ML: pip install -r requirements-ml.txt" -ForegroundColor DarkGray
Write-Host "Opcional RAG: pip install -r requirements.txt" -ForegroundColor DarkGray

python bootstrap.py
Write-Host "`nStreamlit: http://localhost:8501" -ForegroundColor Green
streamlit run app.py --server.port=8501
