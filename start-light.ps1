# Build minimo — evita I/O error quando o disco virtual do Docker esta cheio
# Se .\start-light.ps1 for bloqueado: use .\start-light.cmd  ou:
#   powershell -ExecutionPolicy Bypass -File .\start-light.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:BUILD_PROFILE = "core"
$env:INSTALL_ML = "false"

Write-Host "=== Streamlit (imagem LEVE) ===" -ForegroundColor Cyan
Write-Host "Sem: FLAML, LightGBM, XGBoost, txtai, Gemini" -ForegroundColor DarkGray
Write-Host "Com: Chat Flowise, DuckDB, abas demo" -ForegroundColor DarkGray

docker info 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Abra o Docker Desktop primeiro." -ForegroundColor Red
    exit 1
}

docker compose -f docker-compose.yml -f docker-compose.light.yml down 2>&1 | Out-Host
docker compose -f docker-compose.yml -f docker-compose.light.yml up -d --build 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "Falhou. Rode .\fix-docker-io.ps1 (como Admin) e depois Clean/Purge no Docker Desktop." -ForegroundColor Red
    exit $LASTEXITCODE
}

Start-Sleep -Seconds 15
try {
    $r = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing -TimeoutSec 5
    if ($r.StatusCode -eq 200) {
        Write-Host "OK: http://localhost:8501" -ForegroundColor Green
    }
} catch {
    Write-Host "Aguardando... logs: docker compose logs -f streamlit-chat" -ForegroundColor Yellow
}
docker compose ps
