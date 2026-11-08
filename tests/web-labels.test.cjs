const {test} = require('node:test');
const assert = require('node:assert/strict');
const {themeLabel, describeFilters, abstentionInfo} = require('../web/labels.js');

test('rótulos do pipeline ganham acentos e chaves também são aceitas', () => {
  assert.equal(themeLabel('Produto nao recebido ate a avaliacao'), 'Produto não recebido até a avaliação');
  assert.equal(themeLabel('Produto nao recebido'), 'Produto não recebido');
  assert.equal(themeLabel('Demora ou prazo de entrega longo'), 'Demora ou prazo de entrega longo');
  assert.equal(themeLabel('demora_entrega'), 'Demora ou prazo de entrega longo');
  assert.equal(themeLabel('Troca, devolucao, reembolso ou cancelamento'), 'Troca, devolução, reembolso ou cancelamento');
  assert.equal(themeLabel('Insatisfacao geral / nao recomenda'), 'Insatisfação geral / não recomenda');
  assert.equal(themeLabel('nao_recebido'), 'Produto não recebido até a avaliação');
  assert.equal(themeLabel('Tema novo'), 'Tema novo');
  assert.equal(themeLabel(null), 'Tema não informado');
});

test('recorte descrito em português', () => {
  assert.deepEqual(describeFilters({review_score: [1, 2]}), ['notas 1 e 2']);
  assert.deepEqual(describeFilters({review_score: [3, 4, 5], product_category_name_english: 'bed_bath_table'}, {bed_bath_table: 'Cama mesa banho'}), ['notas 3 a 5', 'categoria Cama mesa banho']);
  assert.deepEqual(describeFilters({ano_compra: [2017, 2018], atrasado: true}), ['compras em 2017 e 2018', 'pedidos entregues após a data estimada']);
  assert.deepEqual(describeFilters({product_category_name_english: 'cool_stuff'}), ['categoria cool stuff']);
  assert.deepEqual(describeFilters({}), []);
  assert.deepEqual(describeFilters(null), []);
});

test('cada motivo de recusa do pipeline tem mensagem própria', () => {
  const cases = [
    ['a pergunta esta fora do escopo da base (avaliacoes de clientes da Olist).', 'scope'],
    ["o(s) termo(s) ['bitcoin'] nao aparece(m) em nenhuma avaliacao da base.", 'absent-term'],
    ['nenhuma avaliacao com similaridade >= 0.6.', 'low-similarity'],
    ['apenas 1 avaliacao(oes) relevante(s); o minimo configurado e 3.', 'few-evidence'],
    ['o recorte pedido tem apenas 12 avaliacoes com texto (minimo 30).', 'small-slice'],
    ['nenhum tema do lexico foi encontrado no recorte.', 'no-theme'],
    ['apenas 2 evidencias recuperadas para os temas do recorte.', 'few-examples'],
    ['O modelo declarou que as evidencias nao sustentam a resposta.', 'model-refused'],
    ['A resposta nao citou nenhuma evidencia.', 'citation-rejected'],
    ['A resposta citou evidencias inexistentes: [9].', 'citation-rejected'],
  ];
  for (const [reason, kind] of cases) assert.equal(abstentionInfo(reason).kind, kind, reason);
});

test('mensagens de recusa preservam os dados do motivo', () => {
  assert.match(abstentionInfo("o(s) termo(s) ['bitcoin'] nao aparece(m) em nenhuma avaliacao da base.").text, /O termo “bitcoin” não aparece/);
  assert.match(abstentionInfo("o(s) termo(s) ['bitcoin', 'nft'] nao aparece(m) em nenhuma avaliacao da base.").text, /Os termos “bitcoin” e “nft” não aparecem/);
  assert.match(abstentionInfo('apenas 1 avaliacao(oes) relevante(s); o minimo configurado e 3.').text, /Só 1 avaliação passou .* mínimo para responder é 3/);
  assert.match(abstentionInfo('nenhuma avaliacao com similaridade >= 0.6.').text, /\(0,6\)/);
  assert.equal(abstentionInfo(null).kind, 'other');
  assert.match(abstentionInfo('motivo novo').text, /motivo novo/);
});
