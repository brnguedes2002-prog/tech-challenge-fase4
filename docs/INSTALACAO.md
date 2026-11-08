# Instalação e uso da aplicação

## 1. Requisitos

Python 3.12 (ambiente testado), acesso ao repositório do grupo e, para o modo Databricks, CLI Databricks instalada no PATH, conta autenticada e permissões de consulta nos recursos existentes. Node.js é necessário somente para testes JavaScript. A aplicação usa um servidor local; não requer publicação na internet.

Os notebooks 1–3 são componentes da solução do grupo para preparação, embeddings e avaliação. Eles contêm operações de escrita. Não os execute apenas para abrir a interface; a aplicação usa recursos existentes.

## 2. Preparar os arquivos

Clone o repositório público do projeto. Na pasta que contém `app_server.py`, execute:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-databricks.txt
Copy-Item .env.example .env
```

Se o ambiente já existe, não precisa recriá-lo. Para o modo local sem LLM, basta instalar `requirements-app.txt`. Em sistemas sem o launcher `py`, use `python3.12 -m venv .venv` e o executável Python correspondente em `.venv/bin/`.

O arquivo `requirements-databricks.txt` fixa as versões diretas do SDK e do sentence-transformers usadas nos testes. `requirements-app.txt` declara os intervalos das dependências de dados. Dependências transitivas não estão congeladas; uma instalação futura deve passar novamente pelos testes. Não se declara instalação limpa em outro computador já validada.

## 3. Autenticar e configurar

Autentique sua conta com OAuth. Substitua a URL e o nome do perfil:

```powershell
databricks auth login --host https://SEU-WORKSPACE.cloud.databricks.com --profile MEU_PERFIL
```

O login abre o navegador e a CLI gerencia as credenciais. Referência: [autenticação da CLI Databricks](https://docs.databricks.com/aws/en/dev-tools/cli/authentication). Não coloque tokens no código, no formulário ou no `.env`.

Edite `.env`:

- `DATABRICKS_CONFIG_PROFILE`: perfil que você autenticou.
- `DATABRICKS_WAREHOUSE_ID`: identificador do SQL Warehouse disponível à sua conta.
- `VOC_GOLD_TABLE`, `VOC_VS_INDEX`, `VOC_LLM_ENDPOINT`: recursos existentes do grupo.
- `VOC_MIN_SCORE=0.60`: corte provisório da execução documentada.
- `VOC_VS_QUERY_TYPE=ANN`: modo usado nessa execução.

O `.env` aceita somente as chaves descritas no exemplo. Ele não executa comandos nem expande variáveis. Valores já definidos no terminal têm prioridade. Os scripts de inicialização escolhem explicitamente o modo. Arquivos de credenciais da CLI permanecem fora do projeto.

## 4. Executar

No Windows, abra `iniciar-voz.cmd`. O script usa `.venv`, carrega a configuração e inicia em http://127.0.0.1:8766. A CLI deve estar no PATH; o script também procura uma instalação padrão via WinGet, sem caminho pessoal fixo.

Alternativa pelo terminal:

```powershell
$env:VOC_MODE='databricks'
.\.venv\Scripts\python.exe app_server.py --port 8766
```

Aguarde o status pronto. A primeira execução lê toda a gold, calcula temas localmente e pode baixar o modelo de embeddings. A cópia paginada verifica contagens e unicidade, mas não é transacional. Origem, data e SHA256 ficam em `.local-data/provenance.json`; os dados ficam em `.local-data/gold.jsonl`.

Mantenha o terminal aberto. Para encerrar, pressione Ctrl+C. Mudanças na configuração ou backend exigem reinício. Mudanças de interface podem ser carregadas com Ctrl+F5.

## 5. Exploração local sem LLM

Abra `iniciar-exploracao-local.cmd`, com os CSVs originais na pasta `data/`. São necessários reviews, orders, order_items, products e product_category_name_translation. O leitor trata campos multiline, inclui título ou mensagem, mantém o primeiro ID e usa a categoria do primeiro item de cada pedido. São 42.370 documentos neste conjunto. A seleção apenas por mensagem produziria 40.641; o total bruto é 99.224 linhas.

Esse modo usa TF-IDF e resumo extrativo determinístico. Não é teste do LLM ou do índice remoto. O sistema não troca de modo automaticamente em caso de erro.

## 6. Uso e interpretação

Pergunte sobre entrega, qualidade, atendimento ou elogios. Clique nas referências para conferir comentários e metadados. Em perguntas agregadas, o denominador é o recorte inteiro, não o número de exemplos. Uma avaliação pode ter vários temas; percentuais não precisam somar 100%. Relato de demora e atraso medido pelas datas são informações diferentes.

A tela mostra aviso curto sobre a base atual e detalhes em “Sobre esta análise”. Dados ausentes permanecem desconhecidos. A consulta é independente da anterior. Exportações permitem preservar pergunta, resposta, configuração e evidências; importações são identificadas como registros, sem nova consulta ou autenticação do conteúdo.

## 7. Verificação

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --test tests/web-contract.test.cjs tests/web-labels.test.cjs
```

Os testes não comprovam fidelidade semântica das respostas. Revise afirmações contra comentários e metadados. Exemplos reais e limites estão em [EXEMPLOS_RESULTADOS.md](EXEMPLOS_RESULTADOS.md).

## 8. Problemas comuns

- **Porta ocupada:** encerre a instância anterior ou use outra porta com `--port`.
- **Falha de preparação:** confira instalação, login OAuth, perfil, warehouse e permissões de leitura/consulta.
- **CLI não encontrada:** confirme `databricks --version` no terminal usado para iniciar.
- **Modelo indisponível:** verifique conectividade para a primeira carga e acesso ao endpoint configurado.
- **Página antiga:** Ctrl+F5. O histórico é da aba e não substitui exportação permanente.
- **Contagens diferentes entre modos:** confira a política de seleção e o problema conhecido de ingestão remota; não misture resultados.
