@echo off
cd /d "%~dp0"
echo.
echo Pasta: %CD%
if not exist "docker-compose.yml" (
    echo ERRO: rode este arquivo dentro da pasta streamlit-chat
    pause
    exit /b 1
)
echo Subindo Streamlit com aba ML Tradicional (Docker)...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
pause
