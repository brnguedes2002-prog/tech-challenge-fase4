# Databricks notebook source
# MAGIC %md
# MAGIC # 2.2 — Geração de Embeddings e Banco Vetorial
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Este notebook tem como objetivo gerar, armazenar e disponibilizar embeddings das avaliações de clientes da Olist para utilização em um sistema de **Voice of Customer Intelligence com RAG (Retrieval-Augmented Generation)**.
# MAGIC
# MAGIC O modelo utilizado é o `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, selecionado no notebook **2.1 — Comparação de Modelos de Embeddings** após a avaliação de diferentes abordagens de recuperação semântica.
# MAGIC
# MAGIC ## Metodologia
# MAGIC
# MAGIC O processamento está organizado em quatro etapas:
# MAGIC
# MAGIC 1. **Geração dos embeddings:** transformação das avaliações textuais em vetores numéricos normalizados de 384 dimensões.
# MAGIC 2. **Persistência dos dados:** armazenamento dos embeddings e seus metadados em uma tabela Delta no Databricks.
# MAGIC 3. **Indexação e validação:** utilização do Databricks Vector Search para recuperação semântica e aplicação de filtros por metadados.
# MAGIC 4. **Integração com RAG:** disponibilização das configurações necessárias para consulta ao índice vetorial pelos demais componentes da solução.
# MAGIC
# MAGIC ## Dados utilizados
# MAGIC
# MAGIC - **Origem:** `workspace.raw.f4_reviews_rag`
# MAGIC - **Quantidade de avaliações:** 40.739
# MAGIC - **Modelo de embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
# MAGIC - **Tabela de destino:** `workspace.raw.f4_reviews_embeddings`
# MAGIC - **Índice vetorial:** `workspace.raw.f4_reviews_vector_index`
# MAGIC
# MAGIC ## Resultado esperado
# MAGIC
# MAGIC Ao final, teremos uma base vetorial persistente, indexada e validada, capaz de recuperar avaliações semanticamente relevantes e aplicar filtros por metadados.
# MAGIC
# MAGIC Essa estrutura será utilizada como mecanismo de recuperação de informações no pipeline RAG desenvolvido pelo grupo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Geração dos embeddings
# MAGIC
# MAGIC Nesta etapa, utilizamos o modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, selecionado no notebook 2.1, para transformar as avaliações textuais da Olist em representações vetoriais.
# MAGIC
# MAGIC O processamento contempla:
# MAGIC
# MAGIC 1. Preparação do ambiente e carregamento do modelo.
# MAGIC 2. Leitura das avaliações da tabela `workspace.raw.f4_reviews_rag`.
# MAGIC 3. Geração de embeddings normalizados de 384 dimensões.
# MAGIC 4. Preparação dos vetores e metadados para armazenamento no Delta Lake.
# MAGIC
# MAGIC A base contém **40.739 avaliações**, processadas em lotes de 32 registros.
# MAGIC
# MAGIC Os embeddings gerados serão utilizados posteriormente para criar o índice de busca vetorial no Databricks.

# COMMAND ----------

# MAGIC
# MAGIC %pip install sentence-transformers
# MAGIC %restart_python or dbutils.library.restartPython() to use updated packages.

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import numpy as np
import time

MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TABELA_ORIGEM = "workspace.raw.f4_reviews_rag"
TABELA_DESTINO = "workspace.raw.f4_reviews_embeddings"

modelo = SentenceTransformer(MODELO)

df = spark.table(TABELA_ORIGEM)
dados = df.toPandas().reset_index(drop=True)

print(f"Avaliações carregadas: {len(dados):,}")

inicio = time.time()

vetores = modelo.encode(
    dados["review_full_text"].astype(str).tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

tempo = time.time() - inicio

print(f"Embeddings gerados: {len(vetores):,}")
print(f"Dimensões: {vetores.shape[1]}")
print(f"Tempo: {tempo:.2f} segundos")

# Identificador único e conversão dos vetores para armazenamento no Spark
dados["embedding_id"] = [
    str(i) for i in range(1, len(dados) + 1)
]

dados["embedding"] = [
    vetor.astype(np.float32).tolist()
    for vetor in vetores
]

print("Dados preparados para armazenamento.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Persistência e validação dos embeddings
# MAGIC
# MAGIC Após a geração dos embeddings, os vetores e os metadados das avaliações são armazenados em uma tabela Delta no Databricks.
# MAGIC
# MAGIC Essa etapa tem como objetivos:
# MAGIC
# MAGIC 1. Definir o esquema de armazenamento, preservando as colunas originais e adicionando `embedding_id` e `embedding`.
# MAGIC 2. Persistir os 40.739 registros na tabela `workspace.raw.f4_reviews_embeddings`.
# MAGIC 3. Validar a quantidade de registros armazenados e verificar exemplos dos dados.
# MAGIC
# MAGIC A persistência permite reutilizar os embeddings sem executar novamente o modelo de linguagem, além de disponibilizar a tabela para indexação no Databricks Vector Search.

# COMMAND ----------

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType,
    FloatType
)

# Preserva o esquema original e adiciona os campos necessários ao índice vetorial
schema_destino = StructType(
    list(df.schema.fields) + [
        StructField("embedding_id", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), True)
    ]
)

colunas = df.columns + ["embedding_id", "embedding"]

dados_delta = dados[colunas].copy()

df_embeddings = spark.createDataFrame(
    dados_delta,
    schema=schema_destino
)

# Gravação inicial da tabela Delta
df_embeddings.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABELA_DESTINO)

print("Tabela Delta criada com sucesso!")

# Validação dos dados persistidos
df_validacao = spark.table(TABELA_DESTINO)

print(f"Registros persistidos: {df_validacao.count():,}")

display(
    df_validacao.select(
        "embedding_id",
        "review_id",
        "review_full_text",
        "embedding"
    ).limit(5)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Indexação e validação do Vector Search
# MAGIC
# MAGIC Nesta etapa, utilizamos o **Databricks Vector Search** para indexar os embeddings armazenados no Delta Lake e permitir a recuperação semântica das avaliações.
# MAGIC
# MAGIC ### Configuração do índice vetorial
# MAGIC
# MAGIC O índice foi criado pela interface do Databricks, com as seguintes configurações:
# MAGIC
# MAGIC | Parâmetro | Configuração |
# MAGIC |---|---|
# MAGIC | Endpoint | `f4_reviews_endpoint` |
# MAGIC | Índice | `workspace.raw.f4_reviews_vector_index` |
# MAGIC | Tabela de origem | `workspace.raw.f4_reviews_embeddings` |
# MAGIC | Chave primária | `embedding_id` |
# MAGIC | Coluna vetorial | `embedding` |
# MAGIC | Dimensão dos vetores | 384 |
# MAGIC | Tipo de índice | Delta Sync — Hybrid |
# MAGIC | Sincronização | Triggered |
# MAGIC | Registros indexados | 40.739 |
# MAGIC
# MAGIC A sincronização no modo **Triggered** permite atualizar o índice sob demanda, após alterações na tabela Delta.
# MAGIC
# MAGIC ### Validação da recuperação
# MAGIC
# MAGIC Para verificar o funcionamento do banco vetorial, realizamos dois testes:
# MAGIC
# MAGIC 1. **Busca semântica Top-5:** recuperação das cinco avaliações mais similares a uma pergunta sobre atraso na entrega.
# MAGIC 2. **Filtro por metadados:** aplicação de um filtro para recuperar somente avaliações com `review_score = 1`.
# MAGIC
# MAGIC Ambos os testes utilizam embeddings de consulta gerados pelo mesmo modelo empregado na indexação.

# COMMAND ----------

# Criação do endpoint e do índice vetorial (reprodutível; antes era feito pela interface)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointType, VectorIndexType, IndexSubtype, PipelineType,
    DeltaSyncVectorIndexSpecRequest, EmbeddingVectorColumn)
import time

w = WorkspaceClient()
ENDPOINT = "f4_reviews_endpoint"
INDICE = "workspace.raw.f4_reviews_vector_index"
FONTE = TABELA_DESTINO   # workspace.raw.f4_reviews_embeddings

# Delta Sync exige Change Data Feed na tabela de origem
spark.sql(f"ALTER TABLE {FONTE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

if ENDPOINT not in [e.name for e in w.vector_search_endpoints.list_endpoints()]:
    print("Criando endpoint (pode levar uns 10 a 20 min)...")
    w.vector_search_endpoints.create_endpoint_and_wait(name=ENDPOINT, endpoint_type=EndpointType.STANDARD)
print("Endpoint pronto")

try:
    w.vector_search_indexes.get_index(INDICE)
    print("Índice já existe, sincronizando...")
    w.vector_search_indexes.sync_index(INDICE)
except Exception:
    w.vector_search_indexes.create_index(
        name=INDICE, endpoint_name=ENDPOINT, primary_key="embedding_id",
        index_type=VectorIndexType.DELTA_SYNC, index_subtype=IndexSubtype.HYBRID,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=FONTE, pipeline_type=PipelineType.TRIGGERED,
            embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=384)]))
    print("Índice criado, aguardando indexação...")

while True:
    st = w.vector_search_indexes.get_index(INDICE).status
    print(f"pronto={st.ready} | indexados={st.indexed_row_count} | {st.message}")
    if st.ready:
        break
    time.sleep(30)

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

pergunta = "Clientes reclamando de atraso na entrega"

vetor = modelo.encode(
    pergunta,
    normalize_embeddings=True
).tolist()

resultado = w.vector_search_indexes.query_index(
    index_name="workspace.raw.f4_reviews_vector_index",
    columns=["review_full_text", "review_score", "order_status"],
    query_vector=vetor,
    num_results=5,
    query_type="ANN"
)

print(resultado.as_dict())

# COMMAND ----------

import json

resultado_filtrado = w.vector_search_indexes.query_index(
    index_name="workspace.raw.f4_reviews_vector_index",
    columns=["review_full_text", "review_score", "order_status"],
    query_vector=vetor,
    num_results=5,
    query_type="ANN",
    filters_json=json.dumps({"review_score": 1})
)

print(resultado_filtrado.as_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Integração com RAG e conclusão
# MAGIC
# MAGIC Após a geração, persistência e indexação dos embeddings, disponibilizamos as configurações necessárias para integrar o banco vetorial ao pipeline RAG desenvolvido pelo grupo.
# MAGIC
# MAGIC O dicionário `CONFIG_RAG` centraliza os parâmetros de acesso ao índice, incluindo o endpoint, o modelo de embeddings, a dimensionalidade dos vetores e os metadados disponíveis para consulta.
# MAGIC
# MAGIC ### Orientações para integração
# MAGIC
# MAGIC O componente de recuperação do RAG deverá:
# MAGIC
# MAGIC 1. Carregar o mesmo modelo de embeddings utilizado neste notebook.
# MAGIC 2. Converter as perguntas dos usuários em vetores normalizados de 384 dimensões.
# MAGIC 3. Consultar o índice `workspace.raw.f4_reviews_vector_index` utilizando o endpoint configurado.
# MAGIC 4. Recuperar os textos relevantes e, quando necessário, aplicar filtros por metadados.
# MAGIC
# MAGIC ### Resultado final
# MAGIC
# MAGIC Foram gerados e armazenados **40.739 embeddings**, indexados no Databricks Vector Search. O endpoint e o índice são criados por código neste notebook, o que torna a etapa reprodutível em qualquer workspace.
# MAGIC
# MAGIC Os testes confirmaram o funcionamento da recuperação semântica Top-5 e a aplicação de filtros por nota de avaliação. Também evidenciaram que avaliações com texto idêntico (ex.: "atraso na entrega") ocupam várias posições do Top-5; esse efeito é tratado no pipeline RAG (notebook 3) com deduplicação por texto normalizado e contagem de ocorrências.
# MAGIC
# MAGIC O banco vetorial está preparado para ser utilizado como componente de recuperação de informações da solução RAG.

# COMMAND ----------

CONFIG_RAG = {
    "endpoint": "f4_reviews_endpoint",
    "index": "workspace.raw.f4_reviews_vector_index",
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_dimension": 384,
    "text_column": "review_full_text",
    "metadata_columns": [
        "review_score",
        "order_status",
        "product_category_name_english",
        "order_purchase_timestamp"
    ],
    "sync_mode": "TRIGGERED"
}

print(CONFIG_RAG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resultado: Embeddings e Banco Vetorial
# MAGIC
# MAGIC **Modelo selecionado:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
# MAGIC
# MAGIC **Justificativa:** apresentou Precision@5 de 0,40 (empate prático com o E5, a 1 acerto de diferença) e o maior MRR@5 (0,60) no experimento comparativo do notebook 2.1, sem exigir prefixos de instrução.
# MAGIC
# MAGIC **Base processada:** 40.739 avaliações de clientes da Olist.
# MAGIC
# MAGIC **Arquitetura:** embeddings normalizados de 384 dimensões, persistidos na tabela Delta `workspace.raw.f4_reviews_embeddings` e indexados no Databricks Vector Search.
# MAGIC
# MAGIC **Configuração do índice:**
# MAGIC - Endpoint: `f4_reviews_endpoint`
# MAGIC - Índice: `workspace.raw.f4_reviews_vector_index`
# MAGIC - Tipo: Delta Sync, Hybrid
# MAGIC - Sincronização: Triggered
# MAGIC - Criação: por código neste notebook (reprodutível em qualquer workspace)
# MAGIC - Registros indexados: 40.739
# MAGIC - Status: Online
# MAGIC
# MAGIC **Validação:**
# MAGIC - Busca semântica Top-5: executada com sucesso para reclamações de atraso na entrega.
# MAGIC - Filtro por metadados `review_score = 1`: executado com sucesso.
# MAGIC - Comparação de modelos: Precision@5 e MRR@5 calculadas sobre amostra experimental (notebook 2.1).
# MAGIC
# MAGIC **Integração com RAG:** o componente de recuperação deve gerar embeddings das perguntas usando o mesmo modelo selecionado e consultar o índice por similaridade vetorial, podendo aplicar filtros por metadados.