# Pipeline RAG (Integrante 3)

Arquivos: `src/rag_pipeline.py` (lógica) e `notebooks/3 - Pipeline RAG.ipynb` (execução, calibração, avaliação e demonstração).

## Entradas

| Origem | Objeto |
|---|---|
| Notebook 1 | `workspace.raw.f4_reviews_rag` (gold: avaliações com texto, deduplicadas) |
| Notebook 2.2 | `workspace.raw.f4_reviews_vector_index` (Vector Search, endpoint `f4_reviews_endpoint`) |
| Modelo de embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, o mesmo da indexação |
| LLM | Databricks Foundation Model API (endpoint configurável em `CONFIG["llm_endpoint"]`) |

## Saídas

| Objeto | Conteúdo |
|---|---|
| `workspace.raw.f4_reviews_temas` | temas por avaliação, `dias_atraso`, `atrasado`, `ano_compra`, texto normalizado |
| `workspace.raw.f4_rag_avaliacao` | resultado do gold set por pergunta |
| `RAGPipeline.answer(pergunta)` | contrato com a interface (ver `RAGResponse.to_dict()`) |

## Como a pergunta vira resposta

1. **Router**: detecta escopo, intenção e filtros. Exemplos: "nota baixa" e "insatisfeitos" viram `review_score in [1, 2]`; "cama mesa e banho" vira `bed_bath_table`; "2018" vira filtro de ano; "pedidos entregues com atraso" vira `atrasado = true`.
2. **Transformação da pergunta**: remove meta-palavras e expande com a descrição do tema-alvo. Exemplo: "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?" vira "problemas recorrentes prazo de entrega. entrega atrasada, demorou muito para chegar, fora do prazo".
3. **Pergunta específica**: busca vetorial com filtro, deduplicação de textos idênticos, score mínimo, mínimo de evidências e MMR para diversidade.
4. **Pergunta agregada** ("principais", "mais frequentes", "padrões"): a frequência vem da contagem de temas sobre todas as avaliações do recorte. O índice traz as evidências de cada tema. Top-k sozinho traz o mais parecido, não o mais comum.
5. **Contexto**: evidências com tags `[R1..Rn]` e metadados (nota, categoria, data, atraso real, ocorrências do mesmo texto) e um bloco de estatísticas com os números permitidos.
6. **Geração**: o LLM precisa citar as tags e só pode usar os números do bloco de estatísticas.
7. **Validação**: tag inexistente ou nenhuma citação provoca uma nova tentativa e depois abstenção; número sem lastro no contexto gera aviso.

## Abstenção

| Camada | Regra | Exemplo |
|---|---|---|
| 0 | pergunta sem vocabulário do domínio | "Qual é a capital da França?" |
| 1 | nenhuma evidência acima do score mínimo calibrado | pergunta do domínio sem avaliação parecida |
| 1b | termo da pergunta que não existe em nenhuma avaliação | "pagamento em bitcoin" |
| 2 | menos evidências que o mínimo, ou recorte com poucas avaliações | categoria rara com filtro de nota |
| 3 | LLM responde `SEM_EVIDENCIA` ou cita evidência inexistente | "entrega por drone" (drone existe como produto, não como entrega) |

## Avaliação

Gold set com 20 perguntas: as do edital, as do desafio adicional, perguntas específicas por tema e 5 que exigem abstenção. Métricas: acurácia de abstenção, precisão por tema, citação correta na 1ª tentativa e latência. O notebook compara ANN e HYBRID, cada um com o próprio threshold calibrado.

## Limitações

- Léxico de temas não captura ironia e parte das gírias. O percentual sem tema é reportado.
- Categoria vem de um item representativo por pedido (notebook 1).
- MiniLM trunca textos acima de 128 tokens.
- Gold set pequeno; a precisão por tema usa o próprio léxico como referência, então compara configurações mas não é medida absoluta.
