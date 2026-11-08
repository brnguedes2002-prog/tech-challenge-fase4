const {test} = require('node:test');
const assert = require('node:assert/strict');
const {normalizeEnvelope, tokenizeAnswer} = require('../web/contract.js');
const response = () => ({pergunta:'Entrega?', resposta:'Relato [R1]', abstencao:false, evidencias:[{tag:'[R1]', review_id:'id', texto:'Entrega atrasada.'}], estatisticas:null, avisos:[]});

test('aceita o contrato original e o envelope do servidor', () => {
  assert.equal(normalizeEnvelope(response()).response.pergunta, 'Entrega?');
  assert.equal(normalizeEnvelope({response:response(), meta:{mode:'databricks'}}).meta.mode, 'databricks');
});
test('importação é sempre marcada mesmo se arquivo alegar consulta ao vivo', () => {
  assert.equal(normalizeEnvelope({response:response(), imported:false, meta:{mode:'databricks'}}, true).imported, true);
});
test('rejeita formas incompletas e abstenção com tipo incorreto', () => {
  for (const value of [null, [], 'texto', {}, {...response(), abstencao:'false'}, {...response(), evidencias:null}]) assert.throws(() => normalizeEnvelope(value));
});
test('rejeita evidência sem origem em vez de exibir falso vínculo', () => {
  assert.throws(() => normalizeEnvelope({...response(), evidencias:[{texto:'texto', tag:'[R1]'}]}));
});
test('rejeita estatísticas e avisos malformados', () => {
  for (const override of [{estatisticas:[]}, {estatisticas:{temas:[null]}}, {avisos:'aviso'}, {avisos:[{}]}]) assert.throws(() => normalizeEnvelope({...response(), ...override}));
});
test('limita importações excessivas', () => {
  assert.throws(() => normalizeEnvelope({...response(), resposta:'x'.repeat(150001)}));
  assert.throws(() => normalizeEnvelope({...response(), evidencias:Array(201).fill(response().evidencias[0])}));
});
test('HTML, links e comandos permanecem texto sem sintaxe executável', () => {
  const payload='<img src=x onerror=alert(1)> [clique](javascript:alert(1))';
  assert.deepEqual(tokenizeAnswer(payload), [{type:'text',value:payload}]);
});
test('referências dentro do negrito continuam rastreáveis', () => {
  assert.deepEqual(tokenizeAnswer('**Atraso [R1]**'), [{type:'strong',children:[{type:'text',value:'Atraso '},{type:'citation',value:'[R1]'}]}]);
});
test('números de referência e texto são preservados', () => {
  const tokens=tokenizeAnswer('A [R1], B [R12].');
  assert.deepEqual(tokens.filter(t=>t.type==='citation').map(t=>t.value), ['[R1]','[R12]']);
});
