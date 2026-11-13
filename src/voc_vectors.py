"""Busca vetorial com os embeddings MiniLM do pipeline, sobre a base local corrigida.

Usa o mesmo modelo do notebook 2.2 (paraphrase-multilingual-MiniLM-L12-v2, vetores normalizados
de 384 dimensões) e devolve o score na mesma escala do Databricks Vector Search:
score = 1 / (1 + d²), com d² = 2 - 2·cos para vetores normalizados. Assim o corte min_score do
pipeline vale sem recalibração. Os vetores ficam em cache em .local-data e só são recalculados
quando a base ou o modelo mudam.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOTE = 2048


def score_indice(cos: np.ndarray) -> np.ndarray:
    """Escala do Vector Search (distância L2) para vetores normalizados."""
    return 1.0 / (3.0 - 2.0 * np.clip(cos, -1.0, 1.0))


def chave_base(tabela: pd.DataFrame, modelo: str) -> str:
    h = hashlib.sha256(modelo.encode("utf-8") + b"\0")
    for rid, texto in zip(tabela["review_id"].astype(str), tabela["review_full_text"].astype(str)):
        h.update(rid.encode("utf-8") + b"\t" + texto.encode("utf-8") + b"\n")
    return h.hexdigest()


def carregar_ou_gerar_vetores(tabela: pd.DataFrame, embedder, modelo: str, pasta: Path,
                              progresso=lambda _texto: None) -> np.ndarray:
    """Vetores alinhados às linhas de `tabela` (mesma ordem). Recalcula se o cache não bater."""
    chave = chave_base(tabela, modelo)
    pasta.mkdir(parents=True, exist_ok=True)
    arq_vetores = pasta / f"vetores-minilm-{chave[:16]}.npy"
    arq_meta = pasta / f"vetores-minilm-{chave[:16]}.json"
    if arq_vetores.exists() and arq_meta.exists():
        try:
            meta = json.loads(arq_meta.read_text(encoding="utf-8"))
            vetores = np.load(arq_vetores)
            if meta.get("chave") == chave and vetores.shape == (len(tabela), meta.get("dimensoes")):
                return vetores
        except (OSError, ValueError):
            pass
    textos = tabela["review_full_text"].astype(str).tolist()
    total = len(textos)
    partes = []
    for inicio in range(0, total, LOTE):
        progresso(f"Gerando embeddings MiniLM das avaliações: {inicio:,} de {total:,}. "
                  "Isso só acontece na primeira execução.".replace(",", "."))
        partes.append(np.asarray(embedder.encode(textos[inicio:inicio + LOTE]), dtype=np.float32))
    vetores = np.vstack(partes) if partes else np.zeros((0, 384), dtype=np.float32)
    temporario = arq_vetores.with_suffix(".tmp.npy")
    np.save(temporario, vetores)
    temporario.replace(arq_vetores)
    arq_meta.write_text(json.dumps({"chave": chave, "modelo": modelo, "avaliacoes": total,
                                    "dimensoes": int(vetores.shape[1])}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return vetores


class LocalVectorRetriever:
    """Busca exata por similaridade; mesma interface do DatabricksVSRetriever."""

    def __init__(self, tabela: pd.DataFrame, embedder, vetores: np.ndarray):
        self.df = tabela.reset_index(drop=True)
        self.embedder = embedder
        self.V = np.asarray(vetores, dtype=np.float32)
        if len(self.df) != len(self.V):
            raise ValueError("Vetores e avaliações estão desalinhados.")

    def encode(self, textos: list[str]) -> np.ndarray:
        return self.embedder.encode(textos)

    def buscar(self, pergunta: str, k: int, filtros: dict) -> list[dict]:
        mask = np.ones(len(self.df), dtype=bool)
        for col, val in (filtros or {}).items():
            mask &= self.df[col].isin(val if isinstance(val, list) else [val]).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        q = np.asarray(self.embedder.encode([pergunta])[0], dtype=np.float32)
        scores = score_indice(self.V[idx] @ q)
        k = min(k, len(idx))
        melhores = np.argpartition(-scores, k - 1)[:k]
        melhores = melhores[np.argsort(-scores[melhores], kind="stable")]
        return [{"review_id": self.df.at[idx[j], "review_id"],
                 "review_full_text": self.df.at[idx[j], "review_full_text"],
                 "review_score": self.df.at[idx[j], "review_score"],
                 "score": float(scores[j])} for j in melhores]
