# Databricks notebook source
# MAGIC %md
# MAGIC # 2.1 — Comparação de Modelos de Embeddings
# MAGIC
# MAGIC ## Objetivo
# MAGIC
# MAGIC Este notebook tem como objetivo avaliar diferentes modelos de embeddings para identificar a abordagem mais adequada à recuperação semântica de avaliações de clientes do conjunto de dados Olist.
# MAGIC
# MAGIC Essa etapa faz parte do desenvolvimento de uma solução de **Voice of Customer Intelligence com RAG (Retrieval-Augmented Generation)**, na qual avaliações textuais serão recuperadas conforme sua relevância para as perguntas dos usuários.
# MAGIC
# MAGIC ## Metodologia
# MAGIC
# MAGIC Serão comparados três modelos de embeddings:
# MAGIC
# MAGIC - `intfloat/multilingual-e5-small`
# MAGIC - `BAAI/bge-small-en-v1.5`
# MAGIC - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
# MAGIC
# MAGIC A comparação será realizada utilizando uma amostra de **1.000 avaliações**, extraída da base preparada na etapa de engenharia de dados.
# MAGIC
# MAGIC Os modelos serão avaliados considerando:
# MAGIC
# MAGIC - **Precision@5:** proporção de avaliações relevantes entre as cinco primeiras recuperadas.
# MAGIC - **MRR@5:** posição da primeira avaliação relevante entre os cinco primeiros resultados.
# MAGIC - **Tempo de processamento:** tempo necessário para gerar os embeddings da amostra.
# MAGIC - **Dimensionalidade:** tamanho dos vetores produzidos.
# MAGIC
# MAGIC A avaliação de relevância utilizará um gabarito construído por classificação humana dos resultados recuperados para perguntas representativas.
# MAGIC
# MAGIC ## Resultado esperado
# MAGIC
# MAGIC Ao final, será selecionado o modelo que apresentar o melhor equilíbrio entre qualidade da recuperação semântica e custo computacional.
# MAGIC
# MAGIC O modelo escolhido será utilizado no notebook **2.2 — Geração de Embeddings e Banco Vetorial**, responsável pelo processamento da base completa e pela criação do índice vetorial no Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Preparação da base
# MAGIC
# MAGIC Nesta etapa, preparamos o ambiente e os dados necessários para comparar os modelos de embeddings.
# MAGIC
# MAGIC Utilizamos a tabela `workspace.raw.f4_reviews_rag`, contendo avaliações de clientes da Olist previamente tratadas na etapa de engenharia de dados.
# MAGIC
# MAGIC Após verificar a estrutura e a disponibilidade dos registros, selecionamos uma amostra aleatória de **1.000 avaliações**, utilizando a semente `42` para garantir a reprodutibilidade da seleção.
# MAGIC
# MAGIC Essa mesma amostra será utilizada nos testes dos três modelos, permitindo uma comparação em condições equivalentes.

# COMMAND ----------

#Instalação de dependências
%pip install sentence-transformers

# COMMAND ----------

# Etapa 1 - Validação da base

from pyspark.sql import functions as F

# Carregamento da base preparada
df = spark.table("workspace.raw.f4_reviews_rag")

# Validação dos registros
total = df.count()

df_validos = df.filter(
    F.col("review_full_text").isNotNull() &
    (F.length(F.trim(F.col("review_full_text"))) > 0)
)

validos = df_validos.count()

print(f"Total de avaliações: {total:,}")
print(f"Textos válidos: {validos:,}")
print(f"Textos vazios ou nulos: {total - validos:,}")

# Inspeção da estrutura
df.printSchema()
display(df.limit(5))

# Amostragem reproduzível: ordena por hash do review_id (semente 42).
# F.rand(seed) depende do particionamento e não garante a mesma amostra entre execuções.
amostra = (
    df_validos
    .withColumn("_ordem", F.xxhash64(F.col("review_id"), F.lit(42)))
    .orderBy("_ordem", "review_id")
    .limit(1000)
    .drop("_ordem")
    .toPandas()
    .reset_index(drop=True)
)

print(f"Amostra preparada: {len(amostra)} avaliações")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Comparação dos modelos de embeddings
# MAGIC
# MAGIC Nesta etapa, avaliamos três modelos de embeddings para identificar qual apresenta o melhor desempenho na recuperação semântica de avaliações de clientes em português.
# MAGIC
# MAGIC Os modelos analisados foram:
# MAGIC
# MAGIC - **multilingual-e5-small:** modelo multilíngue voltado à recuperação semântica.
# MAGIC - **bge-small-en-v1.5:** modelo de embeddings treinado principalmente para textos em inglês, utilizado como referência comparativa.
# MAGIC - **paraphrase-multilingual-MiniLM-L12-v2:** modelo multilíngue baseado em Sentence Transformers.
# MAGIC
# MAGIC Para garantir uma comparação consistente, utilizamos a mesma amostra de **1.000 avaliações** e as mesmas perguntas nos três modelos.
# MAGIC
# MAGIC Cada modelo foi responsável por gerar vetores de embeddings e recuperar as cinco avaliações mais similares para cada pergunta.
# MAGIC
# MAGIC Os resultados serão utilizados posteriormente para comparar a qualidade da recuperação e o tempo de processamento de cada abordagem.

# COMMAND ----------

# 1. Geração dos embeddings — E5
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# Carregar modelo
modelo = SentenceTransformer("intfloat/multilingual-e5-small")

# Preparar textos
textos = [
    "passage: " + str(texto)
    for texto in amostra["review_full_text"]
]

# Gerar embeddings
inicio = time.time()

embeddings = modelo.encode(
    textos,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

tempo = time.time() - inicio

print(f"Modelo: multilingual-e5-small")
print(f"Quantidade de embeddings: {len(embeddings)}")
print(f"Dimensões por vetor: {embeddings.shape[1]}")
print(f"Tempo de processamento: {tempo:.2f} segundos")

# COMMAND ----------

# 2. Teste de recuperação semântica — E5
from sentence_transformers import util

# Pergunta de teste
pergunta = "Quais clientes reclamaram de atraso na entrega?"

# Gerar embedding da pergunta
embedding_pergunta = modelo.encode(
    "query: " + pergunta,
    normalize_embeddings=True
)

# Calcular similaridade
similaridades = util.cos_sim(
    embedding_pergunta,
    embeddings
)[0]

# Recuperar os 5 resultados mais relevantes
top5 = similaridades.topk(5)

for posicao, (indice, score) in enumerate(
    zip(top5.indices, top5.values), start=1
):
    print(f"\nResultado {posicao}")
    print(f"Similaridade: {score.item():.4f}")
    print(f"Avaliação: {amostra.iloc[indice.item()]['review_full_text']}")

# COMMAND ----------

#1. Geração dos embeddings — BGE
modelo_bge = SentenceTransformer("BAAI/bge-small-en-v1.5")

inicio = time.time()

embeddings_bge = modelo_bge.encode(
    amostra["review_full_text"].astype(str).tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

tempo_bge = time.time() - inicio

print("Modelo: BAAI/bge-small-en-v1.5")
print(f"Quantidade: {len(embeddings_bge)}")
print(f"Dimensões: {embeddings_bge.shape[1]}")
print(f"Tempo: {tempo_bge:.2f} segundos")

# COMMAND ----------

#2. Teste de recuperação semântica — BGE
pergunta = "Quais clientes reclamaram de atraso na entrega?"

# Embedding da pergunta com BGE
embedding_pergunta_bge = modelo_bge.encode(
    pergunta,
    normalize_embeddings=True
)

# Calcular similaridade
similaridades_bge = util.cos_sim(
    embedding_pergunta_bge,
    embeddings_bge
)[0]

# Recuperar Top-5
top5_bge = similaridades_bge.topk(5)

for posicao, (indice, score) in enumerate(
    zip(top5_bge.indices, top5_bge.values), start=1
):
    print(f"\nResultado {posicao}")
    print(f"Similaridade: {score.item():.4f}")
    print(f"Avaliação: {amostra.iloc[indice.item()]['review_full_text']}")

# COMMAND ----------

# 1. Geração dos embeddings — MiniLM
modelo_minilm = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

inicio = time.time()

embeddings_minilm = modelo_minilm.encode(
    amostra["review_full_text"].astype(str).tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

tempo_minilm = time.time() - inicio

print("Modelo: paraphrase-multilingual-MiniLM-L12-v2")
print(f"Quantidade: {len(embeddings_minilm)}")
print(f"Dimensões: {embeddings_minilm.shape[1]}")
print(f"Tempo: {tempo_minilm:.2f} segundos")

# COMMAND ----------

# 2. Teste de recuperação semântica — MiniLM
pergunta = "Quais clientes reclamaram de atraso na entrega?"

# Gerar embedding da pergunta
embedding_pergunta_minilm = modelo_minilm.encode(
    pergunta,
    normalize_embeddings=True
)

# Calcular similaridades
similaridades_minilm = util.cos_sim(
    embedding_pergunta_minilm,
    embeddings_minilm
)[0]

# Recuperar Top-5
top5_minilm = similaridades_minilm.topk(5)

for posicao, (indice, score) in enumerate(
    zip(top5_minilm.indices, top5_minilm.values), start=1
):
    print(f"\nResultado {posicao}")
    print(f"Similaridade: {score.item():.4f}")
    print(
        "Avaliação:",
        amostra.iloc[indice.item()]["review_full_text"]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Avaliação e seleção do modelo de embeddings
# MAGIC
# MAGIC Após gerar os embeddings e realizar os testes iniciais de recuperação semântica, avaliamos os três modelos utilizando cinco perguntas relacionadas à experiência dos clientes.
# MAGIC
# MAGIC A avaliação foi dividida em três etapas:
# MAGIC
# MAGIC 1. **Construção do conjunto de avaliação:** recuperação dos 10 resultados mais similares por pergunta e modelo, eliminando duplicações entre os candidatos.
# MAGIC 2. **Avaliação quantitativa:** comparação dos resultados com um gabarito de relevância classificado manualmente, utilizando as métricas Precision@5 e MRR@5.
# MAGIC 3. **Análise por pergunta:** verificação da quantidade de avaliações relevantes recuperadas por cada modelo.
# MAGIC
# MAGIC O objetivo é identificar o modelo que oferece o melhor equilíbrio entre qualidade de recuperação semântica e desempenho computacional.

# COMMAND ----------

# Recupera os 10 resultados mais similares de cada modelo para cada pergunta
import os, sys
import pandas as pd
from sentence_transformers import util

perguntas = [
    "Quais clientes reclamaram de atraso na entrega?",
    "Quais clientes receberam produtos com defeito?",
    "O que os clientes elogiam sobre a qualidade dos produtos?",
    "Quais clientes reclamaram do atendimento?",
    "Quais clientes receberam produtos diferentes do solicitado?"
]

modelos = {
    "E5": (modelo, embeddings, "query: "),
    "BGE": (modelo_bge, embeddings_bge, ""),
    "MiniLM": (modelo_minilm, embeddings_minilm, "")
}

candidatos = {}

for pergunta_id, pergunta in enumerate(perguntas, start=1):
    for nome, (modelo_atual, vetores, prefixo) in modelos.items():

        vetor_pergunta = modelo_atual.encode(
            prefixo + pergunta,
            normalize_embeddings=True
        )

        scores = util.cos_sim(vetor_pergunta, vetores)[0]
        top10 = scores.topk(10)

        # Evita classificar mais de uma vez a mesma avaliação para a mesma pergunta
        for indice in top10.indices.tolist():
            chave = (pergunta_id, indice)

            if chave not in candidatos:
                candidatos[chave] = {
                    "pergunta_id": pergunta_id,
                    "pergunta": pergunta,
                    "indice_amostra": indice,
                    "review_id": amostra.iloc[indice]["review_id"],
                    "avaliacao": amostra.iloc[indice]["review_full_text"],
                    "relevante": None
                }

df_avaliacao = pd.DataFrame(candidatos.values())

print(f"Total de avaliações para classificar: {len(df_avaliacao)}")

# ------------------------------------------------------------
# Gabarito versionado no repo (data/), chaveado por (pergunta_id, review_id)
# ------------------------------------------------------------
GABARITO = os.path.abspath("../data/f4_gabarito_relevancia_embeddings.csv")

# Sugestão do léxico só para acelerar a revisão; "relevante" é preenchida por uma pessoa
TEMA_POR_PERGUNTA = {1: "atraso_entrega", 2: "produto_defeito", 3: "elogio_produto", 4: "atendimento", 5: "produto_errado"}
try:
    sys.path.append(os.path.abspath("../src"))
    from rag_pipeline import rotular_temas
    df_avaliacao["sugestao_lexico"] = [
        int(TEMA_POR_PERGUNTA[p] in rotular_temas(t))
        for p, t in zip(df_avaliacao["pergunta_id"], df_avaliacao["avaliacao"])]
except Exception as e:
    print(f"Sugestão do léxico indisponível ({e})")
    df_avaliacao["sugestao_lexico"] = None

# Preserva o que já foi classificado
if os.path.exists(GABARITO):
    anterior = pd.read_csv(GABARITO, sep=";", encoding="utf-8-sig")
    rotulos = anterior.dropna(subset=["relevante"])[["pergunta_id", "review_id", "relevante"]]
    df_avaliacao = df_avaliacao.drop(columns=["relevante"]).merge(rotulos, on=["pergunta_id", "review_id"], how="left")

colunas = ["pergunta_id", "pergunta", "review_id", "indice_amostra", "avaliacao", "sugestao_lexico", "relevante"]
df_avaliacao = df_avaliacao[colunas].sort_values(["pergunta_id", "indice_amostra"])
df_avaliacao.to_csv(GABARITO, sep=";", index=False, encoding="utf-8-sig")

pendentes = int(df_avaliacao["relevante"].isna().sum())
print(f"Gabarito: {GABARITO}")
print(f"Classificados: {len(df_avaliacao) - pendentes} | pendentes: {pendentes}")
if pendentes:
    print("Preencha 'relevante' (1/0) no CSV, faça commit em data/ e rode de novo a partir desta célula.")

display(df_avaliacao)

# COMMAND ----------

# Gabarito de relevância (0 = irrelevante; 1 = relevante)
import pandas as pd
import numpy as np
from sentence_transformers import util

gabarito = pd.read_csv(GABARITO, sep=";", encoding="utf-8-sig")

# Mantém só os pares desta amostra
chaves = set(zip(df_avaliacao["pergunta_id"], df_avaliacao["review_id"]))
gabarito = gabarito[[k in chaves for k in zip(gabarito["pergunta_id"], gabarito["review_id"])]].copy()

# Rótulo humano quando existe; senão, sugestão do léxico (provisório)
gabarito["fonte_rotulo"] = np.where(gabarito["relevante"].isin([0, 1]), "humano", "provisorio_lexico")
gabarito["relevante"] = gabarito["relevante"].where(gabarito["relevante"].isin([0, 1]), gabarito["sugestao_lexico"])
gabarito["relevante"] = gabarito["relevante"].fillna(0).astype(int)

print(f"Pares: {len(gabarito)} | relevantes: {int(gabarito['relevante'].sum())}")
print(gabarito["fonte_rotulo"].value_counts().to_string())
if (gabarito["fonte_rotulo"] == "provisorio_lexico").any():
    print("ATENÇÃO: métricas provisórias. Parte do gabarito veio do léxico, não de classificação humana.")

# Calcula Precision@5 e MRR@5 para cada modelo usando o mesmo gabarito
resultados = []

for nome, (modelo_atual, vetores, prefixo) in modelos.items():

    precisions = []
    mrrs = []

    for pergunta_id, pergunta in enumerate(perguntas, start=1):

        vetor = modelo_atual.encode(prefixo + pergunta, normalize_embeddings=True)

        scores = util.cos_sim(vetor, vetores)[0]
        indices = scores.topk(5).indices.tolist()
        ids_top5 = [amostra.iloc[i]["review_id"] for i in indices]

        relevantes = set(
            gabarito[(gabarito["pergunta_id"] == pergunta_id) & (gabarito["relevante"] == 1)]["review_id"]
        )

        acertos = [1 if rid in relevantes else 0 for rid in ids_top5]

        precisions.append(sum(acertos) / 5)

        primeira_posicao = next((i + 1 for i, acerto in enumerate(acertos) if acerto), None)
        mrrs.append(1 / primeira_posicao if primeira_posicao else 0)

    resultados.append({
        "Modelo": nome,
        "Precision@5": round(np.mean(precisions), 4),
        "MRR@5": round(np.mean(mrrs), 4)
    })

df_metricas = pd.DataFrame(resultados)
display(df_metricas.sort_values("Precision@5", ascending=False))

# COMMAND ----------

# Exibe quantos resultados relevantes cada modelo recuperou no Top-5 de cada pergunta
for pergunta_id, pergunta in enumerate(perguntas, start=1):
    print(f"\nPergunta {pergunta_id}: {pergunta}")

    for nome, (modelo_atual, vetores, prefixo) in modelos.items():
        vetor = modelo_atual.encode(
            prefixo + pergunta,
            normalize_embeddings=True
        )

        scores = util.cos_sim(vetor, vetores)[0]
        indices = scores.topk(5).indices.tolist()

        relevantes = set(
            gabarito[
                (gabarito["pergunta_id"] == pergunta_id) &
                (gabarito["relevante"] == 1)
            ]["indice_amostra"].astype(int)
        )

        acertos = sum(i in relevantes for i in indices)

        print(f"{nome}: {acertos}/5 relevantes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Resultados e conclusão
# MAGIC
# MAGIC Após a comparação dos três modelos de embeddings, foram obtidos os seguintes resultados:
# MAGIC
# MAGIC | Modelo | Precision@5 | MRR@5 | Tempo de processamento |
# MAGIC |---|---:|---:|---:|
# MAGIC | multilingual-e5-small | 0,44 | 0,54 | XX,XX s |
# MAGIC | paraphrase-multilingual-MiniLM-L12-v2 | 0,40 | 0,60 | XX,XX s |
# MAGIC | bge-small-en-v1.5 | 0,16 | 0,34 | XX,XX s |
# MAGIC
# MAGIC > **Métricas provisórias:** o gabarito desta execução veio da sugestão do léxico de temas (`sugestao_lexico`), não de classificação humana. A diferença entre E5 e MiniLM em Precision@5 equivale a 1 acerto em 25 e não é conclusiva.
# MAGIC
# MAGIC ### Modelo selecionado
# MAGIC
# MAGIC O modelo escolhido foi o **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**, considerando os seguintes fatores:
# MAGIC
# MAGIC - **Qualidade da recuperação:** alcançou Precision@5 de 0,40, a 1 acerto do E5 (0,44), o que configura empate prático nesta amostra.
# MAGIC - **Posição do primeiro resultado relevante:** obteve o maior MRR@5 entre os três modelos (0,60).
# MAGIC - **Suporte multilíngue:** é adequado ao processamento de avaliações em português, ao contrário do BGE, treinado principalmente em inglês (Precision@5 de 0,16).
# MAGIC - **Simplicidade de uso:** não exige prefixos de instrução (`query:` / `passage:`), ao contrário do E5, o que reduz o risco de erro na integração com o pipeline RAG.
# MAGIC
# MAGIC ### Análise por pergunta
# MAGIC
# MAGIC O MiniLM teve melhor desempenho nos temas dominantes (atraso: 5/5; elogio à qualidade: 4/5) e pior desempenho em perguntas com termos mais literais (defeito: 0/5; produto diferente: 0/5; atendimento: 1/5). Essa limitação é tratada no pipeline RAG (notebook 3) com reescrita da pergunta pelo tema-alvo, priorização de evidências do tema e comparação entre busca ANN e HYBRID.
# MAGIC
# MAGIC ### Limitações da avaliação
# MAGIC
# MAGIC Os experimentos foram realizados sobre uma amostra de 1.000 avaliações e cinco perguntas, com um gabarito de relevância construído a partir dos resultados recuperados pelos modelos (127 pares pergunta x avaliação).
# MAGIC
# MAGIC Nesta execução, o gabarito foi preenchido pela sugestão do léxico de temas, o que favorece resultados com termos literais e pode subestimar modelos que recuperam paráfrases. O gabarito está versionado em `data/f4_gabarito_relevancia_embeddings.csv` e, quando a coluna `relevante` for revisada por uma pessoa, as métricas passam a refletir a classificação humana sem alteração no código.
# MAGIC
# MAGIC Portanto, as métricas representam o desempenho observado nesse conjunto experimental e não devem ser interpretadas como uma avaliação definitiva da capacidade de generalização dos modelos.
# MAGIC
# MAGIC ### Próxima etapa
# MAGIC
# MAGIC O modelo selecionado será utilizado no notebook **2.2: Geração de Embeddings e Banco Vetorial**, para:
# MAGIC
# MAGIC 1. Gerar os embeddings da base completa de avaliações.
# MAGIC 2. Persistir os vetores em uma tabela Delta.
# MAGIC 3. Criar o índice de busca vetorial no Databricks.
# MAGIC 4. Validar consultas por similaridade semântica e filtros por metadados.
# MAGIC
# MAGIC Essa estrutura permitirá integrar a recuperação de avaliações à solução RAG desenvolvida pelo grupo.