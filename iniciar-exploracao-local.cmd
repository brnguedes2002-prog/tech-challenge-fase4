@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente local nao encontrado. Consulte docs\GUIA_COMPLETO.md.
  pause
  exit /b 1
)
set "VOC_MODE=local"
echo VOZ - Exploracao local dos CSVs, TF-IDF e resposta extrativa, SEM LLM.
echo Abra http://127.0.0.1:8766 no navegador depois que o servidor iniciar.
echo Encerre qualquer outra instancia do VOZ antes de usar a mesma porta.
".venv\Scripts\python.exe" app_server.py --port 8766
pause
