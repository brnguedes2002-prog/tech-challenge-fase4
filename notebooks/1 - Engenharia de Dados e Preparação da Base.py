# Databricks notebook source
# DBTITLE 1,Visão Geral e Documentação
# MAGIC %md
# MAGIC    
# MAGIC # 1 — Engenharia de Dados e Preparação da Base (Olist VoC)
# MAGIC
# MAGIC Pipeline de preparação da base de conhecimento para o **Tech Challenge Fase 4 — Voice of Customer com RAG**. Este notebook executa as etapas abaixo, na ordem:
# MAGIC
# MAGIC 1. **Carga** — CSVs do dataset Olist (Kaggle) → Delta Tables na camada bronze (`workspace.raw.f4_*`)
# MAGIC 2. **EDA** — qualidade dos dados: nulos, distribuição das notas e cobertura de comentários
# MAGIC 3. **Limpeza textual** — normalização dos comentários (título e mensagem) e tratamento de nulos
# MAGIC 4. **Enriquecimento** — joins com metadados dos pedidos (datas, status), itens e produtos (categoria)
# MAGIC 5. **Dataset final (gold)** — tabela consolidada e deduplicada, pronta para geração de embeddings no pipeline RAG
# MAGIC
# MAGIC **Fonte:** [Kaggle — Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data) (~100 mil pedidos, 2016–2018)

# COMMAND ----------

# DBTITLE 1,Configuração
# ============================================================
# Configuração: paths, schemas e utilidades
# ============================================================

BASE_PATH = "/Workspace/tech-challenge-fase4/data"
TARGET_CATALOG = "workspace"
TARGET_SCHEMA = "raw"

# Garante que o schema existe
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {TARGET_CATALOG}.{TARGET_SCHEMA}")

print(f"Schema {TARGET_CATALOG}.{TARGET_SCHEMA} pronto.")
print(f"Caminho base dos CSVs: {BASE_PATH}")

# COMMAND ----------

# DBTITLE 1,Carga de CSVs como Delta Tables (Bronze)
# ============================================================
# Carga dos 9 CSVs do dataset Olist como Delta Tables (bronze)
# ============================================================
# Os CSVs do dataset Olist estão na pasta data/ do workspace (repo), não no volume

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

# Definição de schemas explícitos para cada arquivo CSV
csv_schemas = {
    "olist_customers_dataset.csv": StructType([
        StructField("customer_id", StringType(), False),
        StructField("customer_unique_id", StringType(), False),
        StructField("customer_zip_code_prefix", IntegerType(), True),
        StructField("customer_city", StringType(), True),
        StructField("customer_state", StringType(), True),
    ]),
    "olist_geolocation_dataset.csv": StructType([
        StructField("geolocation_zip_code_prefix", IntegerType(), True),
        StructField("geolocation_lat", DoubleType(), True),
        StructField("geolocation_lng", DoubleType(), True),
        StructField("geolocation_city", StringType(), True),
        StructField("geolocation_state", StringType(), True),
    ]),
    "olist_order_items_dataset.csv": StructType([
        StructField("order_id", StringType(), False),
        StructField("order_item_id", IntegerType(), True),
        StructField("product_id", StringType(), True),
        StructField("seller_id", StringType(), True),
        StructField("shipping_limit_date", TimestampType(), True),
        StructField("price", DoubleType(), True),
        StructField("freight_value", DoubleType(), True),
    ]),
    "olist_order_payments_dataset.csv": StructType([
        StructField("order_id", StringType(), False),
        StructField("payment_sequential", IntegerType(), True),
        StructField("payment_type", StringType(), True),
        StructField("payment_installments", IntegerType(), True),
        StructField("payment_value", DoubleType(), True),
    ]),
    "olist_order_reviews_dataset.csv": StructType([
        StructField("review_id", StringType(), False),
        StructField("order_id", StringType(), True),
        StructField("review_score", IntegerType(), True),
        StructField("review_comment_title", StringType(), True),
        StructField("review_comment_message", StringType(), True),
        StructField("review_creation_date", TimestampType(), True),
        StructField("review_answer_timestamp", TimestampType(), True),
    ]),
    "olist_orders_dataset.csv": StructType([
        StructField("order_id", StringType(), False),
        StructField("customer_id", StringType(), True),
        StructField("order_status", StringType(), True),
        StructField("order_purchase_timestamp", TimestampType(), True),
        StructField("order_approved_at", TimestampType(), True),
        StructField("order_delivered_carrier_date", TimestampType(), True),
        StructField("order_delivered_customer_date", TimestampType(), True),
        StructField("order_estimated_delivery_date", TimestampType(), True),
    ]),
    "olist_products_dataset.csv": StructType([
        StructField("product_id", StringType(), False),
        StructField("product_category_name", StringType(), True),
        StructField("product_name_lenght", IntegerType(), True),
        StructField("product_description_lenght", IntegerType(), True),
        StructField("product_photos_qty", IntegerType(), True),
        StructField("product_weight_g", IntegerType(), True),
        StructField("product_length_cm", IntegerType(), True),
        StructField("product_height_cm", IntegerType(), True),
        StructField("product_width_cm", IntegerType(), True),
    ]),
    "olist_sellers_dataset.csv": StructType([
        StructField("seller_id", StringType(), False),
        StructField("seller_zip_code_prefix", IntegerType(), True),
        StructField("seller_city", StringType(), True),
        StructField("seller_state", StringType(), True),
    ]),
    "product_category_name_translation.csv": StructType([
        StructField("product_category_name", StringType(), False),
        StructField("product_category_name_english", StringType(), True),
    ]),
}

# Nomes das tabelas Delta no catálogo
table_names = {
    "olist_customers_dataset.csv": "f4_customers",
    "olist_geolocation_dataset.csv": "f4_geolocation",
    "olist_order_items_dataset.csv": "f4_order_items",
    "olist_order_payments_dataset.csv": "f4_order_payments",
    "olist_order_reviews_dataset.csv": "f4_order_reviews",
    "olist_orders_dataset.csv": "f4_orders",
    "olist_products_dataset.csv": "f4_products",
    "olist_sellers_dataset.csv": "f4_sellers",
    "product_category_name_translation.csv": "f4_product_category_translation",
}

for csv_file, schema in csv_schemas.items():
    table_name = table_names[csv_file]
    full_path = f"{BASE_PATH}/{csv_file}"
    full_table = f"{TARGET_CATALOG}.{TARGET_SCHEMA}.{table_name}"
    
    print(f"Carregando {csv_file} -> {full_table} ...")
    
    df = (spark.read
          .csv(full_path, header=True, schema=schema, timestampFormat="yyyy-MM-dd HH:mm:ss")
          .na.fill("", subset=[c.name for c in schema.fields if c.dataType == StringType()]))
    
    # Sobrescreve a tabela Delta (modo batch)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table)
    
    count = spark.table(full_table).count()
    print(f"  -> {count:,} linhas carregadas.")

print("\nTodas as 9 Delta Tables criadas com sucesso!")

# COMMAND ----------

# DBTITLE 1,EDA: qualidade e distribuição
# ============================================================
# EDA: qualidade dos dados e distribuição das avaliações
# ============================================================

from pyspark.sql.functions import col, count, when

reviews_raw = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_order_reviews")
orders_raw = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_orders")

# Nulos por coluna na tabela de reviews
print("Valores nulos por coluna (olist_order_reviews_dataset):")
display(reviews_raw.select([
    count(when(col(c).isNull(), c)).alias(c) for c in reviews_raw.columns
]))

# Cobertura de comentários textuais
total_reviews = reviews_raw.count()
com_msg = reviews_raw.filter(col("review_comment_message") != "").count()
com_title = reviews_raw.filter(col("review_comment_title") != "").count()
print(f"Total de reviews: {total_reviews:,}")
print(f"Com mensagem textual: {com_msg:,} ({com_msg/total_reviews:.1%})")
print(f"Com título: {com_title:,} ({com_title/total_reviews:.1%})")

# Nota média por presença de comentário (clientes insatisfeitos comentam mais)
print("Nota média por presença de comentário:")
display(spark.sql(f"""
    SELECT review_comment_message != '' AS tem_comentario,
           COUNT(*) AS qtd_reviews,
           ROUND(AVG(review_score), 2) AS nota_media
    FROM {TARGET_CATALOG}.{TARGET_SCHEMA}.f4_order_reviews
    GROUP BY review_comment_message != ''
"""))

# Distribuição das notas
print("Distribuição das notas (review_score):")
display(reviews_raw.groupBy("review_score").count().orderBy("review_score"))

# Status dos pedidos (metadado relevante para o RAG)
print("Distribuição do status dos pedidos:")
display(orders_raw.groupBy("order_status").count().orderBy(col("count").desc()))

# COMMAND ----------

# DBTITLE 1,Limpeza textual dos reviews
# ============================================================
# Limpeza textual dos reviews (título e mensagem) + tratamento de nulos
# ============================================================

from pyspark.sql.functions import col, trim, regexp_replace, lower, when, length

reviews = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_order_reviews")

# Aplica limpeza nas colunas textuais
clean_reviews = (reviews
    # Remove espaços extras (tabs, quebras de linha, espaços duplos)
    .withColumn("review_comment_title",
        regexp_replace(trim(col("review_comment_title")), r"\s+", " "))
    .withColumn("review_comment_message",
        regexp_replace(trim(col("review_comment_message")), r"\s+", " "))
    # Remove caracteres de controle e entidades HTML comuns
    .withColumn("review_comment_title",
        regexp_replace(col("review_comment_title"), r"[\x00-\x1f\x7f]|&amp;|&lt;|&gt;|&quot;|&#39;", ""))
    .withColumn("review_comment_message",
        regexp_replace(col("review_comment_message"), r"[\x00-\x1f\x7f]|&amp;|&lt;|&gt;|&quot;|&#39;", ""))
    # Reduz pontuação excessiva
    .withColumn("review_comment_title",
        regexp_replace(col("review_comment_title"), r"[.]{2,}|!{2,}|\?{2,}", "."))
    .withColumn("review_comment_message",
        regexp_replace(col("review_comment_message"), r"[.]{2,}|!{2,}|\?{2,}", "."))
    # Normaliza para lowercase (preserva acentos do português)
    .withColumn("review_comment_title_clean", lower(col("review_comment_title")))
    .withColumn("review_comment_message_clean", lower(col("review_comment_message")))
    # Substitui strings vazias por null (tratamento de nulos)
    .withColumn("review_comment_title_clean",
        when(length(col("review_comment_title_clean")) == 0, None)
        .otherwise(col("review_comment_title_clean")))
    .withColumn("review_comment_message_clean",
        when(length(col("review_comment_message_clean")) == 0, None)
        .otherwise(col("review_comment_message_clean")))
    # Flag indicando se o review tem comentário
    .withColumn("has_comment",
        when(col("review_comment_message_clean").isNotNull(), True).otherwise(False))
)

# Salva como tabela Delta de reviews limpos
clean_table = f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_order_reviews_clean"
clean_reviews.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(clean_table)

print(f"Reviews limpos salvos em {clean_table}")
print(f"Total de reviews: {clean_reviews.count():,}")
print(f"Reviews com comentário textual: {clean_reviews.filter(col('has_comment') == True).count():,}")

# Amostra para verificação
display(clean_reviews.select(
    "review_id", "review_score", "review_comment_title", "review_comment_title_clean",
    "review_comment_message", "review_comment_message_clean", "has_comment"
).filter(col("has_comment") == True).limit(5))

# COMMAND ----------

# DBTITLE 1,Enriquecimento: joins reviews + metadados
# ============================================================
# Enriquecimento: reviews + orders (status/datas) + products (categoria)
# ============================================================

from pyspark.sql.functions import col, min as spark_min, max as spark_max

# Tabelas de origem
reviews_clean = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_order_reviews_clean")
orders = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_orders")
order_items = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_order_items")
products = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_products")
category_translation = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_product_category_translation")

# 1. Agrega order_items para ter uma linha por pedido (item representativo)
order_items_agg = (order_items
    .groupBy("order_id")
    .agg(
        spark_min("product_id").alias("product_id"),
        spark_min("seller_id").alias("seller_id"),
        spark_min("price").alias("min_price"),
        spark_max("price").alias("max_price"),
    )
)

# 2. Join reviews -> orders (status e datas)
# 3. Join -> order_items_agg -> products (categoria)
# 4. Join -> product_category_name_translation (categoria em inglês)
enriched_reviews = (reviews_clean
    .join(orders.select(
        "order_id", "customer_id", "order_status",
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ), on="order_id", how="left")
    .join(order_items_agg, on="order_id", how="left")
    .join(products.select(
        "product_id", "product_category_name"
    ), on="product_id", how="left")
    .join(category_translation, on="product_category_name", how="left")
    # Seleciona e organiza as colunas finais
    .select(
        # Identificadores
        col("review_id"),
        col("order_id"),
        col("customer_id"),
        col("product_id"),
        col("seller_id"),
        # Avaliação
        col("review_score"),
        col("review_comment_title_clean").alias("review_comment_title"),
        col("review_comment_message_clean").alias("review_comment_message"),
        col("has_comment"),
        col("review_creation_date"),
        col("review_answer_timestamp"),
        # Metadados do pedido - status
        col("order_status"),
        # Metadados do pedido - datas
        col("order_purchase_timestamp"),
        col("order_approved_at"),
        col("order_delivered_carrier_date"),
        col("order_delivered_customer_date"),
        col("order_estimated_delivery_date"),
        # Metadados do produto - categoria
        col("product_category_name"),
        col("product_category_name_english"),
        # Metadados do item
        col("min_price"),
        col("max_price"),
    )
)

# Salva como tabela Delta enriquecida
enriched_table = f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_reviews_enriched"
enriched_reviews.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(enriched_table)

print(f"Reviews enriquecidos salvos em {enriched_table}")
print(f"Total de reviews enriquecidos: {enriched_reviews.count():,}")
print(f"Reviews com categoria: {enriched_reviews.filter(col('product_category_name').isNotNull()).count():,}")
print(f"Reviews com order_status: {enriched_reviews.filter(col('order_status').isNotNull()).count():,}")
print(f"Reviews com datas de entrega: {enriched_reviews.filter(col('order_delivered_customer_date').isNotNull()).count():,}")
print(f"Reviews com comentário: {enriched_reviews.filter(col('has_comment') == True).count():,}")

# Amostra final
display(enriched_reviews.filter(col("has_comment") == True).limit(10))

# COMMAND ----------

# DBTITLE 1,Dataset final para o RAG (Gold)
# ============================================================
# Dataset final para o RAG (camada gold)
# ============================================================
# Critérios:
# - apenas reviews com comentário textual (base do RAG)
# - deduplicação por review_id
# - consolidação título + mensagem em um único texto (review_full_text)
# - metadados preservados para filtragem contextual (categoria, datas, status)

from pyspark.sql.functions import col, trim, concat_ws, length as length_col

enriched = spark.table(f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_reviews_enriched")

reviews_rag = (enriched
    .filter(col("has_comment") == True)
    .dropDuplicates(["review_id"])
    .withColumn("review_full_text",
        trim(concat_ws(" ", col("review_comment_title"), col("review_comment_message"))))
    .filter(length_col(col("review_full_text")) > 0)
    .select(
        # Texto consolidado (para embeddings)
        col("review_id"),
        col("review_full_text"),
        col("review_comment_title"),
        col("review_comment_message"),
        # Metadados para o RAG contextual
        col("review_score"),
        col("order_id"),
        col("order_status"),
        col("order_purchase_timestamp"),
        col("order_delivered_customer_date"),
        col("order_estimated_delivery_date"),
        col("product_category_name"),
        col("product_category_name_english"),
        col("min_price"),
        col("max_price"),
    )
)

rag_table = f"{TARGET_CATALOG}.{TARGET_SCHEMA}.f4_reviews_rag"
reviews_rag.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(rag_table)

print(f"Dataset final (gold) salvo em {rag_table}")
print(f"Reviews no dataset RAG: {reviews_rag.count():,}")

# Amostra do dataset final
display(reviews_rag.select(
    "review_id", "review_score", "review_full_text",
    "product_category_name_english", "order_status", "order_purchase_timestamp"
).limit(10))

# COMMAND ----------

# DBTITLE 1,Resumo das tabelas criadas
# MAGIC %sql
# MAGIC     
# MAGIC -- ============================================================
# MAGIC -- Resumo: Delta Tables da Fase 4 (bronze -> silver -> gold)
# MAGIC -- ============================================================
# MAGIC
# MAGIC SHOW TABLES IN workspace.raw LIKE 'f4_*';

# COMMAND ----------

# DBTITLE 1,Documentação da preparação
# MAGIC %md
# MAGIC    
# MAGIC ## Documentação da Preparação da Base
# MAGIC
# MAGIC ### Camadas e tabelas geradas
# MAGIC
# MAGIC | Camada | Tabela | Conteúdo |
# MAGIC |---|---|---|
# MAGIC | Bronze | `workspace.raw.f4_*` (9 tabelas) | Cópia fiel dos CSVs originais, com schemas explícitos e tipos corretos |
# MAGIC | Silver | `workspace.raw.f4_order_reviews_clean` | Reviews com texto normalizado, nulos tratados e flag `has_comment` |
# MAGIC | Silver | `workspace.raw.f4_reviews_enriched` | Reviews limpos + metadados (status, datas, categoria, preços) via joins |
# MAGIC | Gold | `workspace.raw.f4_reviews_rag` | Dataset final do RAG: só reviews com comentário, deduplicado, com `review_full_text` consolidado |
# MAGIC
# MAGIC ### Decisões de preparação
# MAGIC
# MAGIC * **Schemas explícitos** na carga dos CSVs (datas no formato `yyyy-MM-dd HH:mm:ss`), evitando inferência frágil.
# MAGIC * **Tratamento de nulos**: strings vazias normalizadas para `NULL` na limpeza; reviews sem mensagem permanecem nas camadas silver com a flag `has_comment = false` e são excluídos apenas do dataset final do RAG (que exige conteúdo textual para embedding).
# MAGIC * **Limpeza textual**: trim, colapso de espaços/quebras de linha, remoção de caracteres de controle e entidades HTML (`&amp;`, `&quot;` etc.), redução de pontuação repetida e lowercase preservando acentuação do português.
# MAGIC * **Consolidação dos comentários**: `review_full_text` = título + mensagem concatenados — granularidade de 1 documento por `review_id`.
# MAGIC * **Enriquecimento (left joins)**: `reviews -> orders` (status e datas do ciclo do pedido), `-> order_items` agregado por pedido (produto/vendedor representativos, preço mín/máx), `-> products` (categoria em PT) e `-> product_category_name_translation` (categoria em EN). Left joins preservam todos os reviews mesmo sem correspondência.
# MAGIC * **Deduplicação**: `dropDuplicates(["review_id"])` no dataset final.
# MAGIC
# MAGIC ### Uso no pipeline RAG
# MAGIC
# MAGIC O dataset `workspace.raw.f4_reviews_rag` é a fonte para a geração de embeddings (Vector Search). Os metadados (`review_score`, `product_category_name_english`, `order_status`, `order_purchase_timestamp`) permitem filtragem contextual e rastreabilidade das evidências nas respostas.