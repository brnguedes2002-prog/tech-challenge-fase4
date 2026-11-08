(function (root) {
  "use strict";
  // O pipeline do grupo grava rótulos e motivos sem acento. Esta camada só traduz para a tela;
  // o valor original continua no JSON exportado e nos detalhes técnicos.
  const THEMES = {
    atraso_entrega: "Atraso na entrega",
    demora_entrega: "Demora ou prazo de entrega longo",
    nao_recebido: "Produto não recebido até a avaliação",
    entrega_incompleta: "Entrega incompleta (itens faltando)",
    produto_errado: "Produto errado ou diferente do anunciado",
    produto_defeito: "Produto com defeito ou danificado",
    qualidade_ruim: "Qualidade abaixo do esperado",
    atendimento: "Atendimento e falta de retorno",
    troca_devolucao: "Troca, devolução, reembolso ou cancelamento",
    embalagem: "Embalagem",
    insatisfacao_geral: "Insatisfação geral / não recomenda",
    elogio_entrega: "Entrega rápida ou no prazo",
    elogio_produto: "Produto bom / conforme anunciado",
    elogio_vendedor: "Loja / vendedor confiável e bom atendimento",
    recomendacao: "Recomenda / compraria de novo"
  };
  const fold = (value) => String(value ?? "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();
  // Rótulos de versões anteriores do pipeline (JSONs exportados antes da separação de temas).
  const LEGACY = {"Produto nao recebido": "Produto não recebido"};
  const BY_LABEL = new Map([...Object.values(THEMES), ...Object.keys(LEGACY)].map((label) => [fold(label), LEGACY[label] || label]));

  function themeLabel(value) {
    if (value === null || value === undefined || value === "") return "Tema não informado";
    if (Object.prototype.hasOwnProperty.call(THEMES, value)) return THEMES[value];
    return BY_LABEL.get(fold(value)) || String(value);
  }

  function joinPt(items) {
    if (items.length <= 1) return items.join("");
    return `${items.slice(0, -1).join(", ")} e ${items[items.length - 1]}`;
  }

  function describeScores(scores) {
    const list = [...new Set((Array.isArray(scores) ? scores : [scores]).map(Number).filter(Number.isInteger))].sort((a, b) => a - b);
    if (!list.length) return null;
    if (list.length === 1) return `nota ${list[0]}`;
    const contiguous = list.every((value, index) => index === 0 || value === list[index - 1] + 1);
    if (contiguous && list.length > 2) return `notas ${list[0]} a ${list[list.length - 1]}`;
    return `notas ${joinPt(list.map(String))}`;
  }

  // Filtros efetivamente aplicados pelo pipeline, descritos em português.
  function describeFilters(filters, categories = {}) {
    const parts = [];
    if (!filters || typeof filters !== "object" || Array.isArray(filters)) return parts;
    if (filters.review_score !== undefined) {
      const scores = describeScores(filters.review_score);
      if (scores) parts.push(scores);
    }
    const category = filters.product_category_name_english;
    if (category) parts.push(`categoria ${categories[category] || String(category).replaceAll("_", " ")}`);
    if (filters.ano_compra !== undefined) {
      const years = (Array.isArray(filters.ano_compra) ? filters.ano_compra : [filters.ano_compra]).map(String);
      parts.push(`compras em ${joinPt(years)}`);
    }
    if (filters.atrasado === true) parts.push("pedidos entregues após a data estimada");
    return parts;
  }

  const decimal = (value) => String(value).replace(".", ",");

  // Uma mensagem por camada de recusa do pipeline (rag_pipeline.py). Motivos desconhecidos aparecem como vieram.
  function abstentionInfo(reason) {
    const raw = String(reason ?? "").trim();
    const r = fold(raw);
    let m;
    if (r.includes("fora do escopo")) {
      return {kind: "scope", title: "Pergunta fora do escopo",
        text: "A pergunta não trata das avaliações de clientes da Olist. O sistema não buscou comentários nem chamou o modelo de linguagem."};
    }
    if ((m = r.match(/termo\(s\) \[(.*?)\] nao aparece/))) {
      const terms = m[1].split(",").map((term) => term.trim().replace(/^['"]|['"]$/g, "")).filter(Boolean);
      const plural = terms.length > 1;
      return {kind: "absent-term", title: "Termo ausente na base",
        text: `${plural ? "Os termos" : "O termo"} ${joinPt(terms.map((term) => `“${term}”`))} não ${plural ? "aparecem" : "aparece"} em nenhuma avaliação. Responder exigiria informação que a base não tem.`};
    }
    if ((m = r.match(/similaridade >= (\d+(?:\.\d+)?)/))) {
      return {kind: "low-similarity", title: "Nenhum comentário próximo o bastante",
        text: `Nenhuma avaliação passou do corte de similaridade da busca (${decimal(m[1])}). O sistema preferiu não responder.`};
    }
    if ((m = r.match(/apenas (\d+) avaliacao\(oes\) relevante\(s\); o minimo configurado e (\d+)/))) {
      const one = m[1] === "1";
      return {kind: "few-evidence", title: "Evidências insuficientes",
        text: `Só ${m[1]} ${one ? "avaliação passou" : "avaliações passaram"} do corte de similaridade; o mínimo para responder é ${m[2]}.`};
    }
    if ((m = r.match(/recorte pedido tem apenas (\d+) avaliacoes com texto \(minimo (\d+)\)/))) {
      return {kind: "small-slice", title: "Recorte pequeno demais",
        text: `O recorte pedido tem ${m[1]} avaliações com texto; o mínimo para calcular frequências é ${m[2]}.`};
    }
    if (r.includes("nenhum tema do lexico")) {
      return {kind: "no-theme", title: "Nenhum tema identificado",
        text: "Nenhum tema do dicionário de palavras-chave apareceu no recorte pedido."};
    }
    if ((m = r.match(/apenas (\d+) evidencias recuperadas/))) {
      const one = m[1] === "1";
      return {kind: "few-examples", title: "Evidências insuficientes",
        text: `Só ${m[1]} ${one ? "comentário ilustrativo foi recuperado" : "comentários ilustrativos foram recuperados"} para os temas do recorte; o mínimo para responder não foi atingido.`};
    }
    if (r.includes("modelo declarou")) {
      return {kind: "model-refused", title: "O modelo não encontrou suporte",
        text: "O modelo de linguagem avaliou que os comentários recuperados não sustentam uma resposta."};
    }
    if (r.includes("nao citou nenhuma evidencia") || r.includes("citou evidencias inexistentes")) {
      return {kind: "citation-rejected", title: "Resposta descartada na validação",
        text: "O modelo não citou corretamente os comentários recuperados, mesmo após uma nova tentativa. A resposta foi descartada para não exibir afirmações sem referência."};
    }
    return {kind: "other", title: "Sem base suficiente para responder",
      text: raw ? `Motivo informado pelo pipeline: ${raw}` : "O pipeline não informou o motivo."};
  }

  const api = {themeLabel, describeFilters, abstentionInfo};
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.VozLabels = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
