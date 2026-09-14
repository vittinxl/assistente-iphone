@echo off
chcp 65001 >nul
cd /d "%~dp0"
title ML Tradicional - UCI 350 + FLAML

echo.
echo === ML Tradicional (sem Docker) ===
echo.

set "PYEXE="
set "PYARG="

REM 1) Launcher oficial do Windows (instale de python.org)
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYEXE=py" & set "PYARG=-3.12"
)
if not defined PYEXE (
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYEXE=py" & set "PYARG=-3.11"
)
if not defined PYEXE (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYEXE=py" & set "PYARG=-3"
)

REM 2) python no PATH
if not defined PYEXE (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
        if not errorlevel 1 set "PYEXE=python"
    )
)

REM 3) Instalacoes comuns no Windows
if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
)
if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYEXE=%LocalAppData%\Programs\Python\Python311\python.exe"
)
if not defined PYEXE if exist "C:\Python312\python.exe" set "PYEXE=C:\Python312\python.exe"
if not defined PYEXE if exist "C:\Python311\python.exe" set "PYEXE=C:\Python311\python.exe"

if not defined PYEXE (
    echo.
    echo ERRO: Python 3.10+ nao encontrado neste PC.
    echo.
    echo O atalho "python" da Microsoft Store NAO serve — desative-o:
    echo   Configuracoes ^> Aplicativos ^> Aliases de execucao de aplicativos
    echo   Desligue "Instalador de aplicativos" para python.exe e python3.exe
    echo.
    echo Depois instale Python de verdade:
    echo   https://www.python.org/downloads/
    echo   Na instalacao, marque:  [x] Add python.exe to PATH
    echo.
    echo Feche e abra de novo o INICIAR-ML.bat
    echo.
    pause
    exit /b 1
)

echo Usando: %PYEXE% %PYARG%
%PYEXE% %PYARG% -c "import sys; print('Python', sys.version.split()[0])"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Criando ambiente virtual...
    %PYEXE% %PYARG% -m venv .venv
    if errorlevel 1 (
        echo Falha ao criar .venv
        pause
        exit /b 1
    )
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

echo.
echo Instalando dependencias (primeira vez pode demorar)...
"%VENV_PY%" -m pip install -q -U pip
"%VENV_PY%" -m pip install -q -r requirements-ml-only.txt
if errorlevel 1 (
    echo Falha no pip install
    pause
    exit /b 1
)

echo.
echo Abrindo navegador em http://localhost:8501
echo Feche esta janela ou Ctrl+C para parar.
echo.

"%VENV_PY%" -m streamlit run ml_app.py --server.port=8501 --browser.gatherUsageStats=false

pause
