# Recuperacao Docker apos erro: input/output error no containerd
# Execute no PowerShell (preferencialmente como Administrador)

$ErrorActionPreference = "Continue"

Write-Host "=== Diagnostico ===" -ForegroundColor Cyan

$vol = Get-Volume -DriveLetter C -ErrorAction SilentlyContinue
if ($vol) {
    $freeGb = [math]::Round($vol.SizeRemaining / 1GB, 2)
    $totalGb = [math]::Round($vol.Size / 1GB, 2)
    Write-Host "Disco C: $freeGb GB livres de $totalGb GB"
    if ($freeGb -lt 15) {
        Write-Host "ATENCAO: menos de 15 GB livres. Libere espaco antes de rebuild." -ForegroundColor Red
    }
}

Write-Host "`n=== Limpando cache Docker (libera espaco) ===" -ForegroundColor Yellow
docker compose -f "$PSScriptRoot\docker-compose.yml" down 2>$null
docker stop $(docker ps -aq) 2>$null
docker system prune -af --volumes
docker builder prune -af

Write-Host "`n=== Uso de disco Docker ===" -ForegroundColor Yellow
docker system df

Write-Host "`n=== Proximos passos manuais ===" -ForegroundColor Green
Write-Host "1. Feche o Docker Desktop completamente (Quit na bandeja)."
Write-Host "2. Abra Docker Desktop de novo e aguarde ficar Running."
Write-Host "3. Libere pelo menos 20 GB no disco C: (Lixeira, Arquivos temporarios, Downloads)."
Write-Host "4. Docker Desktop -> Settings -> Resources -> Advanced:"
Write-Host "   - Verifique 'Disk image size' e limpe via 'Clean / Purge data' se necessario."
Write-Host "5. Rode: .\start.ps1"
Write-Host ""
Write-Host "Se o erro persistir: Settings -> Troubleshoot -> Clean / Purge data (apaga imagens locais)."
