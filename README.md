# Tech Challenge Fase 4 — Voice of Customer com RAG (Olist)

Repositório para a **Fase 4 do Tech Challenge** — solução de VoC Intelligence com RAG sobre o dataset Olist (e-commerce brasileiro, ~100K pedidos, 2016–2018).

## Estrutura do repositório

```
tech-challenge-fase4/
├── notebooks/
│   └── 1 - Engenharia de Dados e Preparação da Base.ipynb
└── README.md
```

## Notebook: Engenharia de Dados e Preparação da Base

Pipeline completo de preparação da base de conhecimento para o RAG:

1. **Carga** — CSVs do dataset Olist (Kaggle) → Delta Tables na camada bronze (`workspace.raw.f4_*`)
2. **EDA** — qualidade dos dados: nulos, distribuição das notas e cobertura de comentários
3. **Limpeza textual** — normalização dos comentários (título e mensagem) e tratamento de nulos
4. **Enriquecimento** — joins com metadados dos pedidos (datas, status), itens e produtos (categoria)
5. **Dataset final (gold)** — tabela consolidada e deduplicada, pronta para geração de embeddings no pipeline RAG

### Camadas e tabelas geradas

| Camada | Tabela | Conteúdo |
|---|---|---|
| Bronze | `workspace.raw.f4_*` (9 tabelas) | Cópia fiel dos CSVs originais, com schemas explícitos e tipos corretos |
| Silver | `workspace.raw.f4_order_reviews_clean` | Reviews com texto normalizado, nulos tratados e flag `has_comment` |
| Silver | `workspace.raw.f4_reviews_enriched` | Reviews limpos + metadados (status, datas, categoria, preços) via joins |
| Gold | `workspace.raw.f4_reviews_rag` | Dataset final do RAG: só reviews com comentário, deduplicado, com `review_full_text` consolidado |

### Dataset final (`f4_reviews_rag`)

- **40.739 reviews** com comentário textual, deduplicados por `review_id`
- `review_full_text` = título + mensagem concatenados
- Metadados preservados: `review_score`, `product_category_name_english`, `order_status`, `order_purchase_timestamp`, preços
- Pronto para geração de embeddings (Vector Search)

## Fonte dos dados

[Kaggle — Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)

## Próximas fases

- Geração de embeddings + índice Vector Search sobre `workspace.raw.f4_reviews_rag`
- Pipeline RAG com respostas fundamentadas em evidências
- Threshold de similaridade e fallback

## Tecnologias

- Databricks (Delta Lake, Spark, Unity Catalog)
- Python / PySpark
- Git versionado via Databricks Repos
