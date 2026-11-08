(function (root) {
  "use strict";
  function normalizeEnvelope(input, imported = false) {
    if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("O JSON precisa conter um objeto de consulta.");
    const response = input.response || input;
    if (typeof response.pergunta !== "string" || typeof response.resposta !== "string" || typeof response.abstencao !== "boolean" || !Array.isArray(response.evidencias)) {
      throw new Error("Formato incompatível. Use uma exportação RAGResponse com pergunta, resposta, abstencao e evidencias.");
    }
    if (response.pergunta.length > 10000 || response.resposta.length > 150000 || response.evidencias.length > 200) throw new Error("O arquivo excede os limites de apresentação.");
    for (const evidence of response.evidencias) {
      if (!evidence || typeof evidence !== "object" || typeof evidence.texto !== "string" || typeof evidence.tag !== "string" || typeof evidence.review_id !== "string") {
        throw new Error("Uma evidência está fora do contrato esperado: tag, review_id e texto precisam ser strings.");
      }
    }
    if (response.avisos && (!Array.isArray(response.avisos) || response.avisos.some((item) => typeof item !== "string"))) throw new Error("O campo avisos deve ser uma lista de textos.");
    const stats = response.estatisticas;
    if (stats !== undefined && stats !== null && (typeof stats !== "object" || Array.isArray(stats))) throw new Error("O campo estatisticas deve ser um objeto ou nulo.");
    if (stats?.temas && (!Array.isArray(stats.temas) || stats.temas.length > 100 || stats.temas.some((item) => !item || typeof item !== "object"))) throw new Error("Lista de temas inválida.");
    const meta = input.meta && typeof input.meta === "object" && !Array.isArray(input.meta) ? input.meta : {};
    return {response, meta, imported: imported || input.imported === true};
  }
  function tokenizeAnswer(value) {
    return String(value).split(/(\*\*[^*\n]+\*\*|\[R\d+\])/g).filter(Boolean).map((value) => {
      if (/^\[R\d+\]$/.test(value)) return {type: "citation", value};
      if (/^\*\*[^*\n]+\*\*$/.test(value)) return {type: "strong", children: tokenizeAnswer(value.slice(2, -2))};
      return {type: "text", value};
    });
  }
  const api = {normalizeEnvelope, tokenizeAnswer};
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.VozContract = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
