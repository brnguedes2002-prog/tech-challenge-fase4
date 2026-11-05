# Databricks notebook source
# MAGIC %md
# MAGIC # 3: Pipeline RAG (retrieval, contexto, LLM e grounding)
# MAGIC
# MAGIC Este notebook implementa o pipeline RAG sobre a base preparada no notebook **1** (`workspace.raw.f4_reviews_rag`) e o índice vetorial criado no **2.2** (`workspace.raw.f4_reviews_vector_index`, modelo `paraphrase-multilingual-MiniLM-L12-v2`). Toda a lógica está em `src/rag_pipeline.py`; aqui ficam a execução, a calibração e a avaliação.
# MAGIC
# MAGIC ## Fluxo
# MAGIC
# MAGIC ```
# MAGIC pergunta
# MAGIC   -> Router: escopo, intenção (específica | agregada), filtros de metadado, reescrita da pergunta
# MAGIC   -> Camada 0: fora de escopo? abstém
# MAGIC   -> específica: busca vetorial com filtro -> dedup por texto -> camadas 1 e 2 (score e mínimo de evidências) -> MMR
# MAGIC   -> agregada:   contagem de temas no recorte (número real) -> evidências por tema no índice
# MAGIC   -> Contexto: evidências com tags [R1..Rn] + bloco de estatísticas
# MAGIC   -> LLM (Foundation Model API) com citação obrigatória
# MAGIC   -> Camada 3: validação das citações (tag citada precisa existir) + checagem de números sem lastro
# MAGIC   -> resposta + evidências + estatísticas + trace
# MAGIC ```
# MAGIC
# MAGIC ## Decisões técnicas
# MAGIC
# MAGIC | Decisão | Justificativa |
# MAGIC |---|---|
# MAGIC | Dois caminhos: específica e agregada | Top-k traz as avaliações mais parecidas, não as mais frequentes. Pergunta de frequência ("principais problemas") é respondida com contagem de temas sobre o recorte inteiro; o índice entra só para trazer as evidências de cada tema. |
# MAGIC | Temas por léxico determinístico | Auditável, reprodutível e barato (40 mil textos em segundos). Limite: não pega ironia nem toda gíria; o percentual de avaliações sem tema é reportado. |
# MAGIC | Dedup por texto normalizado | O teste do notebook 2.2 trouxe "atraso na entrega" 3 vezes no top-5. Textos idênticos viram uma evidência com contagem de ocorrências. |
# MAGIC | Reescrita da pergunta | Remove meta-palavras ("quais avaliações sustentam a conclusão de que...") e expande com a descrição do tema-alvo, respeitando a polaridade. |
# MAGIC | Filtros de metadado | Nota e categoria vão para o `filters_json` do índice; ano e atraso real são aplicados depois da busca. |
# MAGIC | Atraso real (`dias_atraso`) | Cruza o que o cliente disse com o que aconteceu no pedido (entrega real vs. estimada). |
# MAGIC | Abstenção em camadas | 0: escopo; 1: score mínimo (calibrado neste notebook); 1b: termo da pergunta inexistente na base; 2: mínimo de evidências; 3: citação inválida ou `SEM_EVIDENCIA` do LLM. |
# MAGIC | Reviews tratados como dado | O prompt instrui a ignorar comandos dentro das avaliações, e o texto não consegue fechar as tags do contexto (mitigação de prompt injection indireta). |

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.122.0 idna==3.11
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, sys, json, importlib
import pandas as pd

# src/ fica na raiz do repositório (este notebook está em notebooks/)
sys.path.append(os.path.abspath("../src"))
import rag_pipeline as rp
importlib.reload(rp)

pd.set_option("display.max_colwidth", 120)
print(json.dumps(rp.CONFIG, indent=1, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tabela de temas e atraso real
# MAGIC
# MAGIC Gera `workspace.raw.f4_reviews_temas` a partir da gold do Integrante 1, sem alterar a gold. Colunas novas: `temas` (lista), `dias_atraso`, `atrasado`, `ano_compra` e `texto_norm` (usado na deduplicação).

# COMMAND ----------

gold = spark.table(rp.CONFIG["gold_table"]).toPandas()
tabela_temas = rp.construir_tabela_temas(gold)
rp.salvar_tabela_temas(spark, tabela_temas)

print(f"Avaliações: {len(tabela_temas):,}")
print(f"Sem tema identificado: {(tabela_temas.temas.map(len) == 0).mean():.1%}")
print(f"Atraso real: {tabela_temas.atrasado.value_counts(dropna=False).to_dict()}")

dist = (tabela_temas.temas.explode().value_counts().rename_axis("tema").reset_index(name="avaliacoes"))
dist["label"] = dist.tema.map(lambda t: rp.TEMAS[t]["label"])
dist["polaridade"] = dist.tema.map(lambda t: rp.TEMAS[t]["polaridade"])
display(dist)

# COMMAND ----------

# Inspeção qualitativa: amostra por tema (validar a precisão do léxico antes de confiar nas contagens)
for tema in rp.TEMAS:
    amostra = tabela_temas[tabela_temas.temas.map(lambda l: tema in l)].sample(5, random_state=42)
    print(f"\n== {tema}")
    for _, r in amostra.iterrows():
        print(f"  [{r.review_score}] {r.review_full_text[:140]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Escolha do LLM
# MAGIC
# MAGIC Lista os endpoints de serving do workspace. Escolha um modelo de chat e ajuste `llm_endpoint`.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
for e in w.serving_endpoints.list():
    print(e.name, "|", getattr(e, "task", None))

# COMMAND ----------

rp.CONFIG["llm_endpoint"] = "databricks-meta-llama-3-3-70b-instruct"   # ajuste conforme a lista acima

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Montagem do pipeline

# COMMAND ----------

pipe = rp.pipeline_databricks(spark)
print(rp.formatar(pipe.answer("O que os clientes relatam sobre atraso na entrega?")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Calibração do threshold (camada 1)
# MAGIC
# MAGIC O score do Vector Search tem escala própria, então o corte não pode ser copiado do cosseno do notebook 2.1. A calibração usa a mesma regra do `answer()`: mede o score da 3ª evidência distinta (mínimo de evidências exigido), buscando com a pergunta já transformada pelo router, para perguntas que têm resposta e para perguntas sem resposta. O corte escolhido é o que melhor separa os dois grupos (acurácia balanceada); em empate, fica o corte mais baixo, porque as camadas 1b e 3 ainda barram respostas sem lastro.

# COMMAND ----------

DENTRO = [
    "Clientes reclamando de atraso na entrega",
    "produto chegou com defeito",
    "veio faltando item no pedido",
    "loja não responde as mensagens",
    "produto diferente do anunciado",
    "entrega rápida, chegou antes do prazo",
]
FORA = [
    "entrega por drone",
    "pagamento com pix parcelado",
    "programa de fidelidade com pontos",
    "aplicativo travando no celular",
    "cashback na compra",
    "atendimento por chatbot com inteligência artificial",
]

calibracao = {}
for modo in ["ANN", "HYBRID"]:
    pipe.r.query_type = modo
    calibracao[modo] = rp.calibrar_threshold(pipe, DENTRO, FORA)
    print(modo, json.dumps(calibracao[modo], ensure_ascii=False))
pipe.r.query_type = rp.CONFIG["vs_query_type"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Avaliação no gold set: ANN vs HYBRID
# MAGIC
# MAGIC Cada modo usa o próprio threshold calibrado (as escalas de score são diferentes). Métricas:
# MAGIC
# MAGIC - **acurácia de abstenção**: respondeu quando devia e se absteve quando devia
# MAGIC - **precisão por tema**: fração das evidências rotuladas com o tema esperado (perguntas específicas)
# MAGIC - **citação ok na 1ª tentativa**: o LLM citou só tags existentes sem precisar de retry

# COMMAND ----------

resultados = {}
for modo in ["ANN", "HYBRID"]:
    pipe.r.query_type = modo
    pipe.cfg["min_score"] = calibracao[modo]["threshold_sugerido"]
    df_modo, resumo_modo = rp.avaliar(pipe)
    resultados[modo] = (df_modo, resumo_modo)
    print(modo, json.dumps(resumo_modo, ensure_ascii=False, default=str))

display(pd.DataFrame({m: r[1] for m, r in resultados.items()}))

# COMMAND ----------

# Detalhe por pergunta (para escolher o modo e discutir os erros no vídeo)
modo_escolhido = "ANN"   # ajuste com base na tabela acima
display(resultados[modo_escolhido][0])

# COMMAND ----------

# Fixa a configuração escolhida
pipe.r.query_type = modo_escolhido
pipe.cfg["min_score"] = calibracao[modo_escolhido]["threshold_sugerido"]

# Persiste a avaliação para o README
df_final, resumo_final = resultados[modo_escolhido]
spark.createDataFrame(df_final.astype(str)).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("workspace.raw.f4_rag_avaliacao")
print(json.dumps({"modo": modo_escolhido, "min_score": pipe.cfg["min_score"], **resumo_final}, ensure_ascii=False, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Demonstração: perguntas do edital
# MAGIC
# MAGIC Cada resposta mostra a intenção detectada, os filtros aplicados, as estatísticas (quando agregada), as evidências com `review_id` e os avisos da validação.

# COMMAND ----------

perguntas_demo = [
    "Quais são os principais problemas relatados pelos clientes?",
    "Quais reclamações relacionadas à entrega aparecem com maior frequência?",
    "O que os clientes mais elogiam em suas experiências de compra?",
    "Quais problemas são relatados por clientes insatisfeitos?",
    "Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?",
    "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?",
    # desafio adicional: metadados
    "O que os clientes mais elogiam na categoria cama mesa e banho?",
    "Quais temas aparecem nas avaliações de pedidos entregues com atraso?",
]
for q in perguntas_demo:
    print("=" * 110)
    print(rp.formatar(pipe.answer(q)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Demonstração: abstenção
# MAGIC
# MAGIC Uma pergunta por camada: fora de escopo (0), termo inexistente na base (1b) e um caso que só o LLM consegue barrar (3), porque "drone" existe na base como produto, mas não como forma de entrega.

# COMMAND ----------

for q in ["Qual é a capital da França?",
          "Os clientes reclamam do pagamento em bitcoin?",
          "O que os clientes acham da entrega por drone?"]:
    r = pipe.answer(q)
    print("=" * 110)
    print(rp.formatar(r))
    print("TRACE:", json.dumps({k: v for k, v in r.trace.items() if k != "plano"}, ensure_ascii=False, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Contrato para a interface (Integrante 4)
# MAGIC
# MAGIC A interface chama `pipe.answer(pergunta)` e usa `to_dict()`. Campos principais: `resposta`, `abstencao`, `motivo`, `intencao`, `filtros`, `evidencias` (tag, review_id, review_ids, texto, score, nota, categoria, data_compra, atrasado, ocorrencias, tema_ancora), `estatisticas`, `avisos`, `trace`, `latencia_ms`.

# COMMAND ----------

exemplo = pipe.answer("Quais clientes receberam produtos com defeito?").to_dict()
print(json.dumps(exemplo, indent=1, ensure_ascii=False)[:4000])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Resultados e limitações
# MAGIC
# MAGIC Preencher com os números das seções 1, 4 e 5 depois da execução:
# MAGIC
# MAGIC - Modo de busca escolhido e threshold:
# MAGIC - Acurácia de abstenção / precisão por tema / citação ok na 1ª tentativa:
# MAGIC - Principais temas e o cruzamento "reclamou de atraso" x "atraso real":
# MAGIC
# MAGIC Limitações conhecidas:
# MAGIC
# MAGIC - O léxico de temas não captura ironia e parte das gírias; o percentual sem tema é reportado.
# MAGIC - A categoria vem de um item representativo por pedido (notebook 1), então pedidos com vários itens podem ter categoria imprecisa.
# MAGIC - O MiniLM trunca textos acima de 128 tokens.
# MAGIC - O gold set é pequeno (20 perguntas) e a precisão por tema usa o próprio léxico como referência; serve para comparar configurações, não como medida absoluta.