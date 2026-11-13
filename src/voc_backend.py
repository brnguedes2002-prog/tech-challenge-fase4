"""Application adapter around the team's unmodified RAG pipeline."""
from __future__ import annotations
import math
import os
import threading
from pathlib import Path
import pandas as pd
import rag_pipeline as rp
from voc_data import load_local_gold, load_remote_gold, prepare_gold, utc_now
from voc_vectors import LocalVectorRetriever, carregar_ou_gerar_vetores

EXAMPLES = [
    {"id": "entrega", "title": "Entender atrasos", "question": "O que os clientes relatam sobre atraso na entrega?", "category": "Entrega"},
    {"id": "problemas", "title": "Principais problemas", "question": "Quais são os principais problemas relatados pelos clientes?", "category": "Panorama"},
    {"id": "insatisfeitos", "title": "Clientes insatisfeitos", "question": "Quais problemas são relatados por clientes insatisfeitos?", "category": "Satisfação"},
    {"id": "elogios", "title": "O que gera elogios", "question": "O que os clientes mais elogiam em suas experiências de compra?", "category": "Elogios"},
    {"id": "sem-suporte", "title": "Testar falta de evidências", "question": "Os clientes reclamam do pagamento em bitcoin?", "category": "Limites"},
]


class BackendError(Exception):
    def __init__(self, code, message, status=503):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Backend:
    def __init__(self, root: Path, env=None):
        self.root = root
        self.env = dict(os.environ if env is None else env)
        self.mode = self.env.get("VOC_MODE", "local").lower()
        if self.mode not in ("local", "databricks", "rag"):
            raise ValueError("VOC_MODE deve ser local, databricks ou rag.")
        self.label = {"databricks": "Olist · Llama 3.3", "rag": "Olist · MiniLM + Llama 3.3"}.get(self.mode, "Base local Olist")
        self.pipeline = None
        self.provenance = {}
        self._lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._started = False
        self._status = {"mode": self.mode, "label": self.label, "ready": False, "state": "idle",
                        "description": "Aguardando inicialização.", "corpus": {"reviews": None, "period": "2016–2018", "source": None},
                        "limitations": [], "configuration": {}}

    def status(self):
        with self._lock:
            return json_safe({**self._status, "busy": self._query_lock.locked()})

    def _progress(self, text):
        with self._lock:
            self._status["description"] = text

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            self._status.update(state="initializing", description="Preparando a base e o mecanismo de consulta.")
        threading.Thread(target=self.initialize, name="voc-initialize", daemon=True).start()

    def initialize(self):
        try:
            config = dict(rp.CONFIG)
            if self.mode == "databricks":
                warehouse = self.env.get("DATABRICKS_WAREHOUSE_ID", "")
                if not warehouse:
                    raise BackendError("CONFIGURATION_REQUIRED", "Defina DATABRICKS_WAREHOUSE_ID no servidor local.")
                if not self.env.get("VOC_MIN_SCORE"):
                    raise BackendError("CONFIGURATION_REQUIRED", "Defina VOC_MIN_SCORE explicitamente; não há corte final validado nesta base.")
                threshold = float(self.env["VOC_MIN_SCORE"])
                if not math.isfinite(threshold):
                    raise ValueError("Corte inválido.")
                mode = self.env.get("VOC_VS_QUERY_TYPE", "ANN").upper()
                if mode not in ("ANN", "HYBRID"):
                    raise ValueError("Modo de busca inválido.")
                config.update(min_score=threshold, vs_query_type=mode,
                              gold_table=self.env.get("VOC_GOLD_TABLE", config["gold_table"]),
                              vs_index=self.env.get("VOC_VS_INDEX", config["vs_index"]),
                              llm_endpoint=self.env.get("VOC_LLM_ENDPOINT", config["llm_endpoint"]))
                from databricks.sdk import WorkspaceClient
                client = WorkspaceClient()
                self._progress("Lendo a gold atual do Databricks. Nenhum recurso remoto será modificado.")
                gold, provenance = load_remote_gold(client, warehouse, config["gold_table"], self.root / ".local-data", self._progress)
                gold, quality = prepare_gold(gold)
                self._progress("Construindo temas localmente; preparando o modelo de embeddings.")
                themes = rp.construir_tabela_temas(gold)
                embedder = rp.SentenceEmbedder(config["embedding_model"])
                retriever = rp.DatabricksVSRetriever(embedder, config["vs_index"], mode)
                pipe = rp.RAGPipeline(retriever, embedder, rp.DatabricksLLM(config["llm_endpoint"]), themes, config)
                limitations = [
                    "A base reúne avaliações com mensagem escrita e IDs únicos, com leitura de comentários multilinha.",
                    "Busca no Databricks Vector Search com embeddings MiniLM de 384 dimensões; respostas escritas pelo Llama 3.3 70B.",
                    f"Corte de similaridade de {threshold:.2f} na escala do índice; abaixo dele, um comentário não conta como evidência.".replace("0.", "0,"),
                    "Temas calculados pela aplicação sobre a base lida do Databricks; categorias representam um item por pedido.",
                    *([f"{quality['invalid_or_missing_scores']} avaliações têm nota ausente ou inválida e ficam com nota desconhecida."]
                      if quality['invalid_or_missing_scores'] else []),
                    "As referências [R#] são conferidas automaticamente; leia cada afirmação junto do comentário citado."]
            elif self.mode == "rag":
                if self.env.get("VOC_MIN_SCORE"):
                    threshold = float(self.env["VOC_MIN_SCORE"])
                    if not math.isfinite(threshold):
                        raise ValueError("Corte inválido.")
                    config.update(min_score=threshold)
                config.update(llm_endpoint=self.env.get("VOC_LLM_ENDPOINT", config["llm_endpoint"]))
                self._progress("Lendo as avaliações da Olist.")
                gold, provenance = load_local_gold(self.root / "data", require_message=True)
                gold, quality = prepare_gold(gold)
                self._progress("Calculando os temas das avaliações.")
                themes = rp.construir_tabela_temas(gold)
                self._progress("Carregando o modelo de embeddings MiniLM.")
                embedder = rp.SentenceEmbedder(config["embedding_model"])
                vectors = carregar_ou_gerar_vetores(themes, embedder, config["embedding_model"],
                                                    self.root / ".local-data", self._progress)
                retriever = LocalVectorRetriever(themes, embedder, vectors)
                self._progress("Conectando ao Llama 3.3 no Databricks.")
                pipe = rp.RAGPipeline(retriever, embedder, rp.DatabricksLLM(config["llm_endpoint"]), themes, config)
                config = pipe.cfg
                limitations = [
                    "Busca vetorial com embeddings MiniLM multilíngues de 384 dimensões, o mesmo modelo do índice no Databricks Vector Search; o score segue a escala do índice.",
                    "Respostas escritas pelo Llama 3.3 70B a partir das avaliações recuperadas; toda citação é conferida contra o contexto.",
                    "A base reúne avaliações com mensagem escrita e IDs únicos, com leitura de comentários multilinha.",
                    "Categorias representam um item por pedido; temas por léxico não capturam toda ironia ou negação."]
            else:
                self._progress("Lendo os CSVs locais e preparando busca TF-IDF.")
                gold, provenance = load_local_gold(self.root / "data", require_message=True)
                gold, quality = prepare_gold(gold)
                themes = rp.construir_tabela_temas(gold)
                pipe = rp.pipeline_local(themes)
                config = pipe.cfg
                limitations = ["Modo local: busca TF-IDF e resposta extrativa determinística, sem modelo generativo.",
                               "A base reúne avaliações com mensagem escrita e IDs únicos, com leitura de comentários multilinha.",
                               "Categorias representam um item por pedido; temas por léxico não capturam toda ironia ou negação."]
            dates = pd.to_datetime(gold.order_purchase_timestamp, errors="coerce").dropna()
            period = f"{dates.min():%Y-%m} a {dates.max():%Y-%m}" if len(dates) else "Não informado"
            public_config = {k: config[k] for k in ("min_score", "k_contexto", "min_evidencias", "min_docs_agregada")}
            if self.mode == "rag":
                public_config.update(search_mode="embeddings MiniLM", threshold_status="escala do índice",
                                     llm="Llama 3.3 70B", llm_endpoint=config.get("llm_endpoint"),
                                     embedding_model=config.get("embedding_model"))
            else:
                public_config.update(search_mode=config.get("vs_query_type") if self.mode == "databricks" else "TF-IDF",
                                     threshold_status="escala do índice" if self.mode == "databricks" else "local", llm=config.get("llm_endpoint") if self.mode == "databricks" else "extrativo sem LLM")
            self.pipeline, self.provenance = pipe, provenance
            with self._lock:
                self._status.update(ready=True, state="ready", description="Pronto para consultar a base atual." if self.mode == "databricks" else "Pronto para consultar as avaliações da Olist.",
                                    corpus={"reviews": len(gold), "period": period, "source": provenance["source"],
                                            "score_distribution": {str(int(k)): int(v) for k, v in gold.review_score.value_counts().items()},
                                            "invalid_scores": quality["invalid_or_missing_scores"]},
                                    limitations=limitations, configuration=public_config, loaded_at=provenance["loaded_at"],
                                    dataset_version=provenance["dataset_version"])
        except Exception as exc:
            message = exc.message if isinstance(exc, BackendError) else {
                "local": "Não foi possível preparar a base. Confira os CSVs e as dependências do ambiente.",
                "rag": "Não foi possível preparar a busca ou o acesso ao Llama 3.3. Confira os CSVs, o acesso ao Databricks e as dependências."
            }.get(self.mode, "Não foi possível preparar a base remota. Confira o acesso e as dependências.")
            with self._lock:
                self._status.update(ready=False, state="error", description=message, error_type=type(exc).__name__)
            print("Inicialização falhou:", type(exc).__name__, flush=True)

    def query(self, question):
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise BackendError("INVALID_QUESTION", "Digite uma pergunta entre 1 e 2.000 caracteres.", 400)
        if not self.status()["ready"]:
            raise BackendError("NOT_READY", self.status()["description"])
        if not self._query_lock.acquire(blocking=False):
            raise BackendError("BUSY", "Uma consulta está em andamento. Aguarde a conclusão.", 409)
        try:
            result = self.pipeline.answer(question.strip()).to_dict()
            if self.mode == "local":
                result.setdefault("avisos", []).insert(0, "Exploração local com TF-IDF e resumo extrativo; não é geração por LLM.")

            return json_safe({"response": result, "meta": {"mode": self.mode, "label": self.label,
                              "generated_at": utc_now(), "dataset_version": self.provenance["dataset_version"],
                              "config": self.status()["configuration"], "limitations": self.status()["limitations"]}})
        except BackendError:
            raise
        except Exception as exc:
            print("Consulta falhou:", type(exc).__name__, flush=True)
            raise BackendError("QUERY_FAILED", "A consulta falhou no serviço ou no pipeline. Nenhuma resposta foi substituída por dados de exemplo.", 502) from None
        finally:
            self._query_lock.release()
