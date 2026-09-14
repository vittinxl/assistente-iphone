$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Streamlit — Chat + ML Tradicional (FLAML + UCI 350) ===" -ForegroundColor Cyan

$vol = Get-Volume -DriveLetter C -ErrorAction SilentlyContinue
if ($vol) {
    $freeGb = [math]::Round($vol.SizeRemaining / 1GB, 2)
    Write-Host "Disco C: $freeGb GB livres"
    if ($freeGb -lt 5) {
        Write-Host "Menos de 5 GB livres — o build Docker pode falhar." -ForegroundColor Red
    }
}

docker info 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker nao esta rodando. Abra o Docker Desktop e aguarde 'Running'." -ForegroundColor Red
    exit 1
}

Write-Host "Build da imagem (pode levar 5-15 min na primeira vez)..." -ForegroundColor Yellow
docker compose down 2>$null
docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build falhou. Tente: Docker Desktop > Troubleshoot > Clean / Purge data, depois rode de novo." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Aguardando http://localhost:8501 ..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 45; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}

docker compose ps
if ($ready) {
    Write-Host ""
    Write-Host "Abra:  http://localhost:8501" -ForegroundColor Green
    Write-Host "Aba:   ML tradicional  (FLAML + dataset UCI 350)" -ForegroundColor Green
} else {
    Write-Host "Veja logs: docker compose logs -f streamlit-chat" -ForegroundColor Yellow
}
