@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fix-docker-io.ps1"
exit /b %ERRORLEVEL%
