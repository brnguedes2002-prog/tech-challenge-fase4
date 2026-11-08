"""Smoke test do backend com recursos existentes; grava resultados só localmente."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from voc_backend import Backend
from app_config import load_env

load_env(ROOT / '.env')
backend = Backend(ROOT)
backend.start()
previous = None
while backend.status()['state'] in ('idle', 'initializing'):
    status = backend.status()
    if status['description'] != previous:
        print(status['description'], flush=True)
        previous = status['description']
    time.sleep(2)
status = backend.status()
out = ROOT / 'output' / ('validacao-' + backend.mode)
out.mkdir(parents=True, exist_ok=True)
(out / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(status, ensure_ascii=True), flush=True)
if not status['ready']:
    raise SystemExit(1)
for i, question in enumerate([
    'O que os clientes relatam sobre atraso na entrega?',
    'Quais problemas são relatados por clientes insatisfeitos?',
    'Qual é a capital da França?',
    'Os clientes reclamam do pagamento em bitcoin?',
]):
    response = backend.query(question)
    (out / f'consulta-{i+1}.json').write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding='utf-8')
    result = response['response']
    print(json.dumps({'question': question, 'abstencao': result['abstencao'],
                      'motivo': result.get('motivo'), 'evidencias': len(result['evidencias']),
                      'avisos': result.get('avisos'), 'latencia_ms': result.get('latencia_ms')}, ensure_ascii=True), flush=True)
