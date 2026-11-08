"""Configuração local declarativa; nunca executa conteúdo de .env."""
import os
import re
from pathlib import Path

ALLOWED = {'VOC_MODE', 'VOC_MIN_SCORE', 'VOC_VS_QUERY_TYPE', 'VOC_GOLD_TABLE',
           'VOC_VS_INDEX', 'VOC_LLM_ENDPOINT', 'DATABRICKS_CONFIG_PROFILE',
           'DATABRICKS_WAREHOUSE_ID', 'DATABRICKS_HOST'}

def load_env(path: Path, environ=None):
    env = os.environ if environ is None else environ
    if not path.exists():
        return
    for number, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r'[A-Z_]+', key) or key not in ALLOWED:
            raise ValueError(f'Configuração inválida na linha {number} do .env; consulte .env.example.')
        if value[:1] in ('"', "'"):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f'Aspas não fechadas na linha {number} do .env.')
            value = value[1:-1]
        if value:
            env.setdefault(key, value)
