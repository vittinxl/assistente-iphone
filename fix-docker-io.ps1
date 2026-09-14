# Corrige erro: input/output error em containerd-mount (disco virtual Docker cheio/corrompido)
# Execute como Administrador se possivel. Feche outros programas pesados.

$ErrorActionPreference = "Continue"
Write-Host "=== Correcao Docker I/O error ===" -ForegroundColor Cyan

$vol = Get-Volume -DriveLetter C -ErrorAction SilentlyContinue
if ($vol) {
    $freeGb = [math]::Round($vol.SizeRemaining / 1GB, 2)
    Write-Host "Disco C: $freeGb GB livres"
    if ($freeGb -lt 8) {
        Write-Host "CRITICO: libere pelo menos 15 GB no C: antes de continuar." -ForegroundColor Red
    }
}

$dockerWsl = Join-Path $env:LOCALAPPDATA "Docker\wsl"
$vhdCandidates = @(
    (Join-Path $dockerWsl "disk\docker_data.vhdx"),
    (Join-Path $dockerWsl "data\ext4.vhdx")
)
Write-Host "`n=== Arquivos do disco virtual Docker ===" -ForegroundColor Yellow
foreach ($p in $vhdCandidates) {
    if (Test-Path $p) {
        $gb = [math]::Round((Get-Item $p).Length / 1GB, 2)
        Write-Host "$gb GB  ->  $p"
    }
}

Write-Host "`n=== Parando WSL / containers ===" -ForegroundColor Yellow
Set-Location $PSScriptRoot
docker compose down 2>$null
wsl --shutdown 2>$null
Start-Sleep -Seconds 5

Write-Host "`n=== Limpeza agressiva (so com Docker Desktop ABERTO depois) ===" -ForegroundColor Yellow
Write-Host "Agora: abra o Docker Desktop, espere 'Running', depois pressione Enter..."
Read-Host

docker system prune -af --volumes 2>&1 | Out-Host
docker builder prune -af 2>&1 | Out-Host
docker system df 2>&1 | Out-Host

Write-Host "`n=== Build LEVE (sem FLAML/txtai/Gemini) ===" -ForegroundColor Green
Write-Host "Rode: .\start-light.ps1"
Write-Host ""
Write-Host "Se AINDA falhar no passo 'exporting to image':"
Write-Host "  Docker Desktop -> Settings -> Troubleshoot -> Clean / Purge data"
Write-Host "  (apaga todas as imagens; libera o VHDX corrompido)"
Write-Host ""
Write-Host "Alternativa sem Docker: .\run-local.ps1"
Write-Host "  Chat Flowise + DuckDB no Python local."
