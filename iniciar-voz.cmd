@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente local nao encontrado. Consulte docs\GUIA_COMPLETO.md.
  pause
  exit /b 1
)
set "VOC_MODE=local"
set "HF_HOME=%CD%\.local-cache\huggingface"
set "TOKENIZERS_PARALLELISM=false"
for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Databricks.DatabricksCLI_*") do set "PATH=%%~fD;%PATH%"
echo VOZ - Base local de avaliacoes da Olist
echo Abra http://127.0.0.1:8766 no navegador depois que o servidor iniciar.
echo Mantenha esta janela aberta. Para encerrar, pressione Ctrl+C.
".venv\Scripts\python.exe" app_server.py --port 8766
pause
