"use strict";

// All analysis content is rendered as text nodes. Only these fixed icons are SVG markup.
const ICON_PATHS = {
  spark: '<path d="m12 3 2.2 6.8L21 12l-6.8 2.2L12 21l-2.2-6.8L3 12l6.8-2.2L12 3Z"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5"/>',
  play: '<rect x="3" y="4" width="18" height="15" rx="3"/><path d="m10 8 6 4-6 4V8ZM8 22h8"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 8.5a2.5 2.5 0 1 1 4.7 1.2c-.8 1.1-2.2 1.3-2.2 3M12 16v1"/>',
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  alert: '<path d="m12 3 10 18H2L12 3ZM12 9v5M12 17v1"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  link: '<path d="m10 13 4-4M8 15l-2 2a3 3 0 0 1-4-4l4-4a3 3 0 0 1 4 0m4 0 2-2a3 3 0 0 1 4 4l-4 4a3 3 0 0 1-4 0"/>',
  sliders: '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="8" cy="6" r="2" fill="currentColor"/><circle cx="16" cy="12" r="2" fill="currentColor"/><circle cx="10" cy="18" r="2" fill="currentColor"/>',
  quotes: '<path d="M10 5H4v7h5c0 4-3 6-5 7M20 5h-6v7h5c0 4-3 6-5 7"/>',
  search: '<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6"/>',
  file: '<path d="M13 2H5v20h14V8l-6-6ZM13 2v6h6M8 13h8M8 17h6"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  truck: '<path d="M3 5h11v12H3V5ZM14 9h4l3 4v4h-7M6 17a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM17 17a2 2 0 1 0 0 4 2 2 0 0 0 0-4Z"/>',
  heart: '<path d="M20 5a5 5 0 0 0-8 1 5 5 0 0 0-8-1c-4 5 3 10 8 15 5-5 12-10 8-15Z"/>',
  chart: '<path d="M4 3v18h17M8 17v-4M13 17V9M18 17V5"/>',
  chat: '<path d="M4 4h16v12H9l-5 4V4ZM8 8h8M8 12h5"/>'
};
const $ = (id) => document.getElementById(id);
const text = (value, fallback = "Não informado") => value === null || value === undefined || value === "" ? fallback : String(value);
const number = (value) => value === null || value === undefined || value === "" ? "—" : Number.isFinite(Number(value)) ? new Intl.NumberFormat("pt-BR", {maximumFractionDigits: 1}).format(Number(value)) : "—";
const safeJson = (value) => JSON.stringify(value, null, 2);
const icon = (name) => {
  const span = document.createElement("span");
  span.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON_PATHS[name] || ICON_PATHS.file}</svg>`;
  return span;
};
const node = (tag, className, value) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (value !== undefined) element.textContent = value;
  return element;
};
document.querySelectorAll("[data-icon]").forEach((element) => element.replaceChildren(icon(element.dataset.icon)));

const EXAMPLES = [
  {title: "Entenda as reclamações", question: "Quais são os principais problemas relatados pelos clientes?", caption: "Explore os temas mais frequentes.", icon: "chart"},
  {title: "Investigue a entrega", question: "O que os clientes relatam sobre atraso na entrega?", caption: "Confira relatos sobre prazos.", icon: "truck"},
  {title: "Descubra o que encanta", question: "O que os clientes mais elogiam em suas experiências de compra?", caption: "Encontre experiências positivas.", icon: "heart"},
  {title: "Ouça os insatisfeitos", question: "Quais problemas são relatados por clientes insatisfeitos?", caption: "Investigue avaliações com nota baixa.", icon: "chat"}
];
let state = {status: null, current: null, history: [], busy: false, elapsedTimer: null, lastQuestion: ""};
let toastTimer;
let statusTimer;

function notify(message, action) {
  const toast = $("toast");
  toast.replaceChildren(document.createTextNode(message));
  if (action) {
    const button = node("button", "toast-action", action.label);
    button.type = "button";
    button.addEventListener("click", () => { clearTimeout(toastTimer); toast.hidden = true; action.run(); });
    toast.append(button);
  }
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, action ? 8000 : 4500);
}

const VIEWS = ["explorar", "base", "projeto"];
function view(name) {
  if (!VIEWS.includes(name)) name = "explorar";
  const changed = state.view !== name;
  state.view = name;
  document.querySelectorAll(".view").forEach((section) => { section.hidden = section.id !== `view-${name}`; });
  document.querySelectorAll("[data-view]").forEach((link) => {
    const active = link.dataset.view === name;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
  $("breadcrumb-title").textContent = {explorar: "Explorar avaliações", base: "Base de conhecimento", projeto: "Sobre o projeto"}[name];
  document.title = `Voz — ${$("breadcrumb-title").textContent}`;
  return changed;
}
// Âncoras internas, como o link "Ir para o conteúdo" (#main), não trocam de página.
// Ao trocar de página, a nova começa do topo.
window.addEventListener("hashchange", () => { const name = location.hash.slice(1); if (VIEWS.includes(name) && view(name)) window.scrollTo(0, 0); });
view(location.hash.slice(1));

EXAMPLES.forEach((example) => {
  const button = node("button", "suggestion");
  button.type = "button";
  const top = node("div", "suggestion-top");
  const symbol = node("span", "suggestion-icon"); symbol.append(icon(example.icon));
  top.append(symbol);
  button.append(top, node("strong", "", example.title), node("p", "", example.caption));
  button.setAttribute("aria-label", `${example.title}: ${example.question}`);
  button.addEventListener("click", () => {
    $("question").value = example.question;
    updateCount(); $("question").focus();
  });
  $("example-grid").append(button);
});

function updateCount() { $("character-count").textContent = `${number($("question").value.length)} / 2.000`; }
$("question").addEventListener("input", updateCount);
$("question").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault(); if (!state.busy) $("question-form").requestSubmit();
  }
});

async function fetchJson(url, options = {}, timeout = 20000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, {...options, signal: controller.signal, credentials: "same-origin"});
    const body = await response.json();
    if (!response.ok) throw new Error(body?.error?.message || "O servidor não conseguiu concluir a solicitação.");
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("A consulta excedeu o tempo de espera. O servidor pode ainda estar processando; aguarde antes de tentar novamente.");
    if (error instanceof TypeError) throw new Error("Não foi possível alcançar o servidor local. Confira se a aplicação continua aberta no terminal.");
    throw error;
  } finally { clearTimeout(timer); }
}

function setStatus(status) {
  state.status = status;
  const mode = status.mode === "databricks" ? "Olist · Llama 3.3" : status.mode === "rag" ? "Olist · Llama 3.3" : status.mode === "local" ? "Base local Olist" : "Ambiente indisponível";
  const statusText = status.state === "error" ? "Falha na preparação" : status.ready ? mode : `${mode} · preparando`;
  $("connection-status").replaceChildren(node("i"), document.createTextNode(statusText));
  $("connection-status").classList.toggle("offline", status.state === "error");
  $("submit-question").disabled = state.busy || !status.ready;
  $("mode-chip").textContent = text(status.label, mode);
  $("mode-description").textContent = text(status.description, "Consulte as avaliações da Olist.");
  const corpus = status.corpus || {};
  if (corpus.reviews !== undefined && corpus.reviews !== null) $("corpus-count").textContent = `${number(corpus.reviews)} avaliações`;
  $("dataset-period").textContent = text(corpus.period, "2016–2018");
  renderBase();
  renderArchitecture(status);
}

async function refreshStatus() {
  clearTimeout(statusTimer);
  try { setStatus(await fetchJson("/api/status")); }
  catch (error) {
    $("connection-status").replaceChildren(node("i"), document.createTextNode("Servidor indisponível"));
    $("connection-status").classList.add("offline");
    $("mode-chip").textContent = "Sem conexão";
    $("mode-description").textContent = error.message;
    $("submit-question").disabled = true;
  }
  if (!state.status?.ready && state.status?.state !== "error") statusTimer = setTimeout(refreshStatus, 5000);
}

function renderBase() {
  const status = state.status || {};
  const corpus = status.corpus || {};
  $("base-summary").replaceChildren();
  const facts = [
    ["Avaliações disponíveis", number(corpus.reviews), "Total informado pelo ambiente atual"],
    ["Período da base", text(corpus.period, "2016–2018"), "Base histórica de experiências de compra"],
    status.mode === "rag" ? ["Mecanismo de consulta", "RAG", "Embeddings MiniLM e Llama 3.3 70B"]
      : ["Ambiente de consulta", status.mode === "databricks" ? "Remoto" : "Local", status.mode === "databricks" ? "Recursos existentes; aplicação local" : "Exploração extrativa, sem LLM"]
  ];
  facts.forEach(([label, value, caption]) => {
    const card = node("div", "base-stat");
    card.append(node("span", "", label), node("strong", "", value), node("small", "", caption));
    $("base-summary").append(card);
  });
  const limitations = Array.isArray(status.limitations) ? status.limitations : [];
  $("base-limitations").replaceChildren(...limitations.map((item) => node("li", "", String(item))));
  $("base-config").replaceChildren(node("p", "muted", `Origem: ${text(corpus.source)}`));
  if (status.configuration) {
    const details = node("details");
    details.append(node("summary", "", "Configuração do ambiente"), node("pre", "", safeJson(status.configuration)));
    $("base-config").append(details);
  }
}

// Preenche na página "Sobre o projeto" os parâmetros informados por /api/status.
function renderArchitecture(status) {
  const config = status.configuration || {};
  const local = status.mode === "local";
  const format = {
    min_score: (value) => Number.isFinite(Number(value)) && value !== null ? Number(value).toLocaleString("pt-BR", {minimumFractionDigits: 2}) : null,
    llm: (value) => value ? String(value) : (local ? "nenhum (modo local)" : null)
  };
  document.querySelectorAll("[data-config]").forEach((element) => {
    const key = element.dataset.config;
    const raw = key === "label" ? status.label : config[key];
    const value = format[key] ? format[key](raw) : (raw === null || raw === undefined || raw === "" ? null : String(raw));
    if (value !== null) element.textContent = value;
  });

}

const {normalizeEnvelope, tokenizeAnswer} = globalThis.VozContract;
const {themeLabel, describeFilters, abstentionInfo} = globalThis.VozLabels;
const describeSlice = (filters) => {
  const parts = describeFilters(filters, globalThis.VozCategories || {});
  return parts.length ? parts.join(" · ") : "todas as avaliações com comentário";
};

function setBusy(busy) {
  state.busy = busy;
  $("submit-question").disabled = busy || !state.status?.ready;
  $("question").disabled = busy;
  $("import-json").disabled = busy;
  $("retry-query").disabled = busy;
  $("loading-state").hidden = !busy;
  $("question-form").setAttribute("aria-busy", String(busy));
  if (busy) {
    const started = Date.now();
    $("elapsed-time").textContent = "0s";
    $("loading-detail").textContent = "O primeiro acesso pode levar mais tempo para carregar os dados e o modelo.";
    state.elapsedTimer = setInterval(() => { $("elapsed-time").textContent = `${Math.floor((Date.now() - started) / 1000)}s`; }, 1000);
  } else { clearInterval(state.elapsedTimer); }
}

async function query(question) {
  if (state.busy) return;
  question = question.trim();
  if (!question || question.length > 2000) { notify("Escreva uma pergunta com até 2.000 caracteres."); return; }
  state.lastQuestion = question;
  state.current = null;
  $("results").hidden = true;
  $("error-state").hidden = true;
  $("suggestions").hidden = true;
  setBusy(true);
  try {
    const result = normalizeEnvelope(await fetchJson("/api/query", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({question})}, 300000));
    state.current = result;
    remember(result); renderResult(result);
    $("results").scrollIntoView({behavior: "smooth", block: "start"});
    refreshStatus();
  } catch (error) {
    $("error-message").textContent = error.message;
    $("error-state").hidden = false;
  } finally { setBusy(false); }
}
$("question-form").addEventListener("submit", (event) => { event.preventDefault(); query($("question").value); });
$("retry-query").addEventListener("click", () => query(state.lastQuestion));
$("new-query").addEventListener("click", () => {
  if (state.busy) return;
  $("question").value = ""; updateCount(); $("question").focus();
  $("suggestions").hidden = false;
  $("results").hidden = true;
  state.current = null;
  $("question-form").scrollIntoView({behavior: "smooth", block: "center"});
});

function remember(envelope) {
  state.history.unshift({id: Date.now().toString(36), ...envelope});
  state.history = state.history.slice(0, 12);
  try { sessionStorage.setItem("voz-consultas-v1", JSON.stringify(state.history)); }
  catch { notify("Consulta concluída. O histórico não pôde ser salvo nesta aba."); }
  renderHistory();
}

function renderHistory() {
  const history = $("history"); history.replaceChildren();
  if (!state.history.length) history.append(node("p", "history-empty", "Suas investigações aparecem aqui.\nUma pergunta de cada vez."));
  state.history.forEach((item) => {
    const button = node("button", "history-item");
    button.append(icon("chat"), node("span", "", item.response.pergunta));
    button.title = item.response.pergunta;
    button.addEventListener("click", () => {
      if (state.busy) { notify("Aguarde a consulta atual terminar."); return; }
      location.hash = "explorar"; view("explorar");
      state.current = item; $("question").value = item.response.pergunta; updateCount();
      $("error-state").hidden = true; renderResult(item, true);
      $("results").scrollIntoView({behavior: "smooth"});
    });
    history.append(button);
  });
}
$("clear-history").addEventListener("click", () => {
  if (!state.history.length) { notify("Não há consultas nesta aba."); return; }
  const previous = state.history.slice();
  state.history = [];
  try { sessionStorage.removeItem("voz-consultas-v1"); } catch { /* Storage may be disabled. */ }
  renderHistory();
  notify("Histórico desta aba limpo.", {label: "Desfazer", run: () => {
    state.history = previous;
    try { sessionStorage.setItem("voz-consultas-v1", JSON.stringify(previous)); } catch { /* Storage may be disabled. */ }
    renderHistory(); notify("Histórico restaurado.");
  }});
});
try {
  const history = JSON.parse(sessionStorage.getItem("voz-consultas-v1") || "[]");
  if (Array.isArray(history)) state.history = history.slice(0, 12).map((item) => ({id: String(item.id), ...normalizeEnvelope(item)}));
} catch { state.history = []; }
renderHistory();

function appendAnswer(container, value, evidence) {
  const counts = new Map();
  evidence.forEach((item) => counts.set(item.tag, (counts.get(item.tag) || 0) + 1));
  const targets = new Map();
  evidence.forEach((item, index) => { if (/^\[R\d+\]$/.test(item.tag) && counts.get(item.tag) === 1) targets.set(item.tag, index); });
  container.replaceChildren();
  const missing = new Set();
  let bulletList = null;
  String(value).split(/\n/).forEach((paragraph) => {
    if (!paragraph.trim()) { bulletList = null; return; }
    const bullet = paragraph.match(/^\s*[-*•]\s+(.+)$/);
    const p = node(bullet ? "li" : "p");
    function appendTokens(parent, tokens) {
      tokens.forEach((token) => {
        if (token.type === "citation") {
          if (targets.has(token.value)) {
            const link = node("button", "citation", token.value);
            link.type = "button"; link.setAttribute("aria-label", `Conferir evidência ${token.value}`);
            link.addEventListener("click", () => focusEvidence(targets.get(token.value)));
            parent.append(link);
          } else { parent.append(node("span", "citation invalid-citation", token.value)); missing.add(token.value); }
        } else if (token.type === "strong") {
          const strong = node("strong"); appendTokens(strong, token.children); parent.append(strong);
        } else parent.append(document.createTextNode(token.value));
      });
    }
    appendTokens(p, tokenizeAnswer(bullet ? bullet[1] : paragraph));
    if (bullet) {
      if (!bulletList) { bulletList = node("ul", "answer-findings"); container.append(bulletList); }
      bulletList.append(p);
    } else { bulletList = null; container.append(p); }
  });
  return [...missing];
}

function renderResult(envelope, fromHistory = false) {
  state.current = envelope;
  const {response: r, meta, imported} = envelope;
  $("result-heading").textContent = r.pergunta;
  $("intent-label").textContent = r.abstencao ? "" : r.intencao === "agregada" ? "Visão do recorte" : "Busca de evidências";
  const refusal = r.abstencao ? abstentionInfo(r.motivo) : null;
  $("answer-title").textContent = refusal ? refusal.title : "O que encontramos";
  $("abstention").hidden = !refusal;
  $("abstention").replaceChildren(node("strong", "", "Por que não há resposta"), node("p", "", refusal ? refusal.text : ""));
  // Na recusa, a mensagem genérica do pipeline repetiria o motivo; o texto original fica nos detalhes técnicos.
  $("answer-text").hidden = Boolean(refusal);
  if (refusal) $("answer-text").replaceChildren();
  const missing = refusal ? [] : appendAnswer($("answer-text"), r.resposta, r.evidencias);
  renderBasis(r);
  document.querySelector(".answer-footer").hidden = !$("answer-text").querySelector("button.citation");
  document.querySelector(".evidence-panel").hidden = r.evidencias.length === 0;
  document.querySelector(".evidence-description").textContent = r.intencao === "agregada"
    ? "Exemplos de cada tema do panorama. Eles ilustram os temas; a frequência está nas contagens."
    : "Confira os comentários que sustentam a resposta.";
  const warnings = [...(r.avisos || [])];
  if (missing.length) warnings.push(`As referências ${missing.join(", ")} não identificam uma evidência única no retorno. Revise a resposta.`);
  if (Array.isArray(meta.limitations)) warnings.push(...meta.limitations.map(String));
  if (imported) warnings.unshift("Execução registrada importada. Os resultados foram carregados do arquivo.");
  $("result-warnings").replaceChildren();
  const notices = [];
  if (meta.mode === "local") notices.push("Exploração local: resumo de comentários, sem geração por inteligência artificial.");
  if (imported) notices.push("Registro importado: este resultado não foi consultado novamente nem autenticado.");
  if (missing.length) notices.push("Há referências que não puderam ser verificadas. Confira os detalhes antes de usar esta resposta.");
  (r.avisos || []).filter(w => !/ingest|corte provis|nao citadas|não citadas|explora[çc][aã]o local/i.test(w)).forEach(w => notices.push(w));
  notices.forEach(message => $("result-warnings").append(node("p", "result-note", message)));
  $("result-provenance").replaceChildren();
  const modeLabel = imported ? "Execução registrada · importada" : fromHistory ? "Consulta anterior desta sessão" : ["databricks", "rag"].includes(meta.mode) ? "Resposta do Llama 3.3" : "Consulta à base local";
  $("result-provenance").append(node("span", "source-mode", modeLabel));
  $("result-provenance").append(node("span", "", `${r.evidencias.length} evidências retornadas`));
  if (Number.isFinite(Number(r.latencia_ms)) && r.latencia_ms != null) $("result-provenance").append(node("span", "", `${number(Number(r.latencia_ms) / 1000)}s no pipeline`));
  if (meta.generated_at) {
    const date = new Date(meta.generated_at);
    if (Number.isFinite(date.getTime())) $("result-provenance").append(node("span", "", date.toLocaleString("pt-BR")));
  }
  $("evidence-count").textContent = String(r.evidencias.length);
  $("evidence-search").value = "";
  $("cited-only").checked = /\[R\d+\]/.test(r.resposta);
  renderEvidence(); renderStatistics(r.estatisticas, r.filtros);
  const details = $("technical-content"); details.replaceChildren();
  if (warnings.length) {
    details.append(node("h4", "", "Limitações e verificações"));
    const list = node("ul"); [...new Set(warnings)].forEach(w => list.append(node("li", "", w)));
    details.append(list);
  }
  if (refusal) details.append(node("h4", "", "Mensagem original do pipeline"), node("pre", "", text(r.resposta)));
  details.append(node("h4", "", "Filtros efetivamente aplicados"), node("pre", "", safeJson(r.filtros || {})));
  details.append(node("h4", "", "Origem e configuração desta consulta"), node("pre", "", safeJson(meta)));
  details.append(node("h4", "", "Rastreamento do pipeline"), node("pre", "", safeJson(r.trace || {})));
  $("results").hidden = false;
  // Leva o foco ao resultado para que o leitor de tela anuncie a nova resposta.
  $("result-heading").focus({preventScroll: true});
}

function formatDate(value) {
  if (!value) return "Data não informada";
  const iso = String(value).match(/^(\d{4})-(\d{2})-(\d{2})/);
  return iso ? `${iso[3]}/${iso[2]}/${iso[1]}` : "Data não informada";
}

function renderEvidence() {
  if (!state.current) return;
  const r = state.current.response;
  const search = $("evidence-search").value.toLocaleLowerCase("pt-BR").trim();
  const cited = new Set(r.resposta.match(/\[R\d+\]/g) || []);
  const list = $("evidence-list"); list.replaceChildren();
  let shown = 0;
  r.evidencias.map((evidence, index) => ({evidence, index})).sort((a, b) => Number(cited.has(b.evidence.tag)) - Number(cited.has(a.evidence.tag))).forEach(({evidence, index}) => {
    if ($("cited-only").checked && !cited.has(evidence.tag)) return;
    if (search && !`${evidence.texto} ${evidence.review_id} ${evidence.categoria || ""} ${evidence.tag}`.toLocaleLowerCase("pt-BR").includes(search)) return;
    shown++;
    const card = node("article", "evidence-card"); card.id = `evidence-${index}`; card.tabIndex = -1;
    const top = node("div", "evidence-top");
    const score = Number(evidence.nota);
    top.append(node("span", "evidence-tag", evidence.tag));
    top.append(node("span", "review-score", Number.isInteger(score) && score >= 1 && score <= 5 ? `${"★".repeat(score)}${"☆".repeat(5 - score)} · ${score}/5` : "Nota indisponível"));
    card.append(top, node("blockquote", "", evidence.texto));
    const metadata = node("div", "evidence-meta");
    const category = globalThis.VozCategories?.[evidence.categoria] || text(evidence.categoria, "Categoria não informada").replaceAll("_", " ");
    if (evidence.tema_ancora) metadata.append(node("span", "evidence-theme", `Tema: ${themeLabel(evidence.tema_ancora)}`));
    metadata.append(node("span", "", category), node("span", "", formatDate(evidence.data_compra)));
    metadata.append(node("span", "", evidence.atrasado === true ? "Entrega após o prazo" : evidence.atrasado === false ? "Sem atraso pela data" : "Atraso não verificável"));
    card.append(metadata);
    const details = node("details"); details.append(node("summary", "", "Ver origem e detalhes"));
    const dl = node("dl");
    const pairs = [["Review ID", evidence.review_id], ["Score da busca", `${text(evidence.score)} — não é probabilidade de acerto`], ["Ocorrências do texto", text(evidence.ocorrencias)], ["IDs agrupados", Array.isArray(evidence.review_ids) ? evidence.review_ids.join("\n") : evidence.review_id]];
    pairs.forEach(([key, value]) => dl.append(node("dt", "", key), node("dd", "", text(value))));
    details.append(dl); card.append(details); list.append(card);
  });
  if (!shown) list.append(node("p", "evidence-empty", r.evidencias.length ? "Nenhum comentário corresponde aos filtros de exibição." : "Nenhuma evidência foi retornada nesta consulta."));
  $("evidence-count").textContent = `${shown} de ${r.evidencias.length}`;
}
$("evidence-search").addEventListener("input", renderEvidence);
$("cited-only").addEventListener("change", renderEvidence);

function focusEvidence(index) {
  $("evidence-search").value = ""; $("cited-only").checked = false; renderEvidence();
  const card = $(`evidence-${index}`);
  if (card) {
    card.scrollIntoView({behavior: "smooth", block: "center"}); card.focus({preventScroll: true});
    card.classList.add("highlighted"); setTimeout(() => card.classList.remove("highlighted"), 2500);
  }
}

function renderBasis(r) {
  let basis = $("answer-basis");
  if (!basis) {
    basis = node("p", "answer-basis");
    basis.id = "answer-basis";
    $("abstention").before(basis);
  }
  basis.replaceChildren();
  basis.hidden = Boolean(r.abstencao);
  if (r.abstencao) return;
  const total = r.estatisticas ? Number(r.estatisticas.total_avaliacoes_no_recorte) : NaN;
  if (Number.isFinite(total)) {
    basis.append(node("strong", "", `${number(total)} avaliações no recorte`),
      document.createTextNode(` (recorte: ${describeSlice(r.filtros)}). As frequências estão no panorama abaixo; os comentários citados são exemplos, não a contagem.`));
  } else {
    basis.append(node("strong", "", `${r.evidencias.length} comentários selecionados por similaridade com a pergunta`),
      document.createTextNode(` (recorte: ${describeSlice(r.filtros)}). Eles mostram o que os clientes dizem, não com que frequência.`));
  }
}

function renderStatistics(stats, filters) {
  const area = $("statistics"); area.replaceChildren(); area.hidden = !stats;
  if (!stats) return;
  area.append(node("h3", "", "Panorama do recorte"), node("p", "", `Recorte: ${describeSlice(filters)}. Contagens sobre todas as avaliações do recorte, não só sobre os comentários exibidos.`));
  const totals = node("div", "stat-totals");
  const total = Number(stats.total_avaliacoes_no_recorte);
  const noTheme = Number(stats.avaliacoes_sem_tema_identificado);
  const share = (part, whole) => whole > 0 ? ` (${number(Math.round(1000 * part / whole) / 10)}%)` : "";
  const items = [[number(total), "avaliações no recorte"], [`${number(noTheme)}${share(noTheme, total)}`, "sem nenhum tema identificado"]];
  items.forEach(([value, label]) => {
    const item = node("div"); item.append(node("strong", "", value), node("span", "", label)); totals.append(item);
  });
  area.append(totals);
  const coverage = stats.cobertura_notas_baixas;
  if (coverage && Number.isInteger(coverage.total) && coverage.total > 0) {
    area.append(node("p", "method-note", `Cobertura entre avaliações de notas 1 e 2: ${number(coverage.sem_tema)} de ${number(coverage.total)} (${number(coverage.percentual_sem_tema)}%) não têm tema identificado. ${number(coverage.sem_tema_negativo)} (${number(coverage.percentual_sem_tema_negativo)}%) não têm tema de reclamação identificado. Nota baixa é um indicador de insatisfação, não uma classificação manual do comentário. O ranking cobre apenas os problemas reconhecidos pelo dicionário.`));
  } else if (coverage && coverage.total === 0) {
    area.append(node("p", "method-note", "Não há avaliações de notas 1 e 2 neste recorte para calcular essa cobertura."));
  }

  const chart = node("ul", "bar-chart"); chart.setAttribute("aria-label", "Frequência dos temas identificados no recorte");
  (Array.isArray(stats.temas) ? stats.temas : []).forEach((theme) => {
    const item = node("li", "bar-item"); const label = node("div", "bar-label");
    label.append(node("span", "", themeLabel(theme.tema)), node("strong", "", `${number(theme.avaliacoes)} · ${number(theme.percentual_do_recorte)}%`));
    const track = node("div", "bar-track"); const fill = node("div", "bar-fill");
    const value = Number(theme.percentual_do_recorte);
    fill.style.width = `${Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0}%`;
    track.setAttribute("aria-hidden", "true"); track.append(fill); item.append(label, track); chart.append(item);
  });
  area.append(chart);
  const delay = stats.atraso_confirmado_pelos_dados;
  const slow = stats.demora_x_data_de_entrega;
  // Oculto quando o filtro já seleciona pedidos atrasados: o resultado seria 100% por definição.
  if ((delay || slow) && !(filters && filters.atrasado === true)) {
    const box = node("div", "fact-check");
    box.append(node("strong", "", "O que o cliente disse × o que aconteceu no pedido"));
    if (delay) box.append(node("p", "", `Das ${number(delay.reclamacoes_de_atraso_com_data_de_entrega)} avaliações que relatam atraso e têm data de entrega registrada, ${number(delay.pedidos_entregues_apos_a_data_estimada)} (${number(delay.percentual)}%) foram entregues depois da data estimada.`));
    if (slow) box.append(node("p", "", `Entre as ${number(slow.relatos_de_demora_com_data_de_entrega)} que falam em demora ou prazo longo, ${number(slow.pedidos_entregues_apos_a_data_estimada)} (${number(slow.percentual)}%) chegaram depois da data estimada. Nas demais, a entrega ocorreu até a data estimada.`));
    area.append(box);
  }
  area.append(node("p", "method-note", "Método: frequência dos temas identificados por dicionário de palavras-chave. Reclamações não reconhecidas ficam fora do ranking. Uma avaliação pode ter mais de um tema, por isso os percentuais não somam 100%."));
  const details = node("details"); details.append(node("summary", "muted", "Recorte e estatísticas completas"), node("pre", "", safeJson(stats))); area.append(details);
}

function download(content, filename, type) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const link = node("a"); link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
$("export-json").addEventListener("click", () => {
  if (state.current) download(safeJson(state.current), `voz-consulta-${Date.now()}.json`, "application/json;charset=utf-8");
});
$("export-md").addEventListener("click", () => {
  if (!state.current) return;
  const {response: r, meta, imported} = state.current;
  const report = ["# Voz — relatório de investigação", "", `Pergunta: ${r.pergunta}`, "", `Origem: ${imported ? "Arquivo importado, não autenticado" : text(meta.label, text(meta.mode))}`, `Registro: ${text(meta.generated_at)}`, "", "## Resposta", "", r.resposta, "", `Abstenção: ${r.abstencao ? "Sim" : "Não"}`, `Motivo: ${text(r.motivo, "Não se aplica")}`, "", "## Avisos", "", ...(r.avisos || []).map((item) => `- ${item}`), ...(Array.isArray(meta.limitations) ? meta.limitations.map((item) => `- ${item}`) : []), "", "## Evidências", ""];
  r.evidencias.forEach((e) => report.push(`### ${e.tag} — ${e.review_id}`, "", e.texto, "", `Nota: ${text(e.nota)} | Categoria: ${text(e.categoria)} | Data: ${text(e.data_compra)}`, `Score bruto: ${text(e.score)} (não é confiança)`, ""));
  report.push("## Metadados, estatísticas e configuração", "", "```json", safeJson({filtros: r.filtros, estatisticas: r.estatisticas, meta}), "```", "", "Este relatório registra a consulta; não comprova, por si só, a correção das afirmações. Confira as fontes e a versão dos dados.");
  download(report.join("\n"), `voz-relatorio-${Date.now()}.md`, "text/markdown;charset=utf-8");
});
$("import-json").addEventListener("change", async (event) => {
  const file = event.target.files[0]; if (!file) return;
  try {
    if (file.size > 5 * 1024 * 1024) throw new Error("Use um JSON de consulta com até 5 MB.");
    const envelope = normalizeEnvelope(JSON.parse(await file.text()), true);
    state.current = envelope; remember(envelope); renderResult(envelope);
    $("error-state").hidden = true;
    $("results").scrollIntoView({behavior: "smooth"});
    notify("Execução registrada importada. Nenhuma consulta remota foi feita.");
  } catch (error) { notify(error instanceof SyntaxError ? "O arquivo não contém um JSON válido." : error.message); }
  finally { event.target.value = ""; }
});

$("help-button").addEventListener("click", () => $("help-dialog").showModal());
$("close-help").addEventListener("click", () => $("help-dialog").close());
$("help-dialog").addEventListener("click", (event) => { if (event.target === $("help-dialog")) { const r = event.target.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) event.target.close(); } });
$("submit-question").disabled = true;
refreshStatus();
