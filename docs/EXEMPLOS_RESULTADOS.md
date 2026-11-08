# Exemplos registrados e avaliação

Execução de 07/11/2026: gold com 40.739 avaliações, ANN, corte provisório 0.60, Llama 3.3 70B. Snapshot SHA256: `420fdd4c9499003903d0d04cfc5db6b974bd874c498dc74b6b0a0ee833cfbb67`.

20 perguntas: 15 respostas e 5 recusas esperadas. Foram conferidas 154 evidências, tags e denominadores. Isso não mede fidelidade semântica: houve inferências causais sem suporte e confusão entre relato e dado do pedido. A base possui 105 registros inválidos e defeito de leitura multiline.

## Quais problemas são relatados por clientes insatisfeitos?

Resposta registrada:

Os problemas mais comuns relatados por clientes insatisfeitos incluem a não entrega de produtos, atrasos na entrega, entrega incompleta e dificuldades com trocas, devoluções e reembolsos.
Os principais problemas são:
- Produto não recebido: [R1], [R2], [R3]
- Troca, devolução, reembolso ou cancelamento: [R4], [R5], [R6]
- Atraso na entrega: [R7], [R8], [R9]
- Entrega incompleta: [R10], [R11], [R12]
- Insatisfação geral / não recomenda: [R13], [R14], [R15]
Limitações: As evidências fornecidas são baseadas em avaliações de clientes com notas 1 e 2, o que pode não representar a experiência de todos os clientes. Além disso, as categorias de produtos e datas de compra variam, o que pode influenciar a percepção dos problemas.

Filtros: `{"review_score": [1, 2]}`.

Estatísticas estruturadas:
```json
{
  "recorte": {
    "review_score": [
      1,
      2
    ]
  },
  "total_avaliacoes_no_recorte": 10755,
  "avaliacoes_sem_tema_identificado": 3419,
  "temas": [
    {
      "tema": "Produto nao recebido",
      "avaliacoes": 2922,
      "percentual_do_recorte": 27.2
    },
    {
      "tema": "Troca, devolucao, reembolso ou cancelamento",
      "avaliacoes": 1357,
      "percentual_do_recorte": 12.6
    },
    {
      "tema": "Atraso na entrega",
      "avaliacoes": 1311,
      "percentual_do_recorte": 12.2
    },
    {
      "tema": "Entrega incompleta (itens faltando)",
      "avaliacoes": 1128,
      "percentual_do_recorte": 10.5
    },
    {
      "tema": "Insatisfacao geral / nao recomenda",
      "avaliacoes": 1029,
      "percentual_do_recorte": 9.6
    }
  ],
  "metodo": "contagem por lexico deterministico sobre todas as avaliacoes do recorte; uma avaliacao pode ter mais de um tema",
  "atraso_confirmado_pelos_dados": {
    "reclamacoes_de_atraso_com_data_de_entrega": 1075,
    "pedidos_entregues_apos_a_data_estimada": 677,
    "percentual": 63.0
  }
}
```

- [R1] — `4c2ab64ffc9119884d87d90bf9d630c3`: descaso com o cliente. o produto não foi entregue.

- [R2] — `2f59a88f994c5d6f802f0ddab0dd785b`: produto não entregue o produto ainda não foi entregue. recebi a mensagem de que houve dificuldade na entrega

- [R3] — `ceef71ec8c12263b2435a71d581b4716`: problema com a entrega o produto não foi entregue.

- [R4] — `0d592da0b11f266efd97a16956ff6417`: compra efetuada,debito autorizado ,pedido cancelado sem explicações.extrema falta de respeito e consideração com os clientes.

- [R5] — `21669c0b1d532d754a411e09973d118d`: produto entregue errado, reclamação realizada, solicitado de troca aberta a mais de uma semana e nem sinal da empresa

- [R6] — `26a88078cce2d9bd73845342e6a30d26`: o a loja me entregou um produto trocado ,inferior ao valor q paguei na compra, quero a devoluçao do produto e meu dinheiro de volta. descaso com o cliente !

- [R7] — `81d1e7339c6b790de99d2c27163ee16d`: mercadoria em atraso e nenhuma satisfação foi dada,falta de compromisso com cliente

- [R8] — `91c3523eb116609c5e674ea0fca34bf4`: muita demora na entrega, não dão satisfação

- [R9] — `f02d4978f58368f3346d8450d52f0ba2`: pessimo muita demora na entrega

- [R10] — `d100adcddcef25bfa83bb819d562211c`: pedido incompleto meu pedido chegou faltando um item.

- [R11] — `5148ad2bdca65375e9504355d79c6ad4`: entrega incompleta realizei um pedido com dois itens e apenas 1 deles foi entregue. fiz algumas reclamações e até agora ninguém entrou em contato comigo pra entregar o outro produto ou para devolver o dinheiro.

- [R12] — `42090eea831a1873d7fb9d583a56978f`: pedido entregue incompleto.

- [R13] — `0b9740c80aede334780599d37387228c`: péssimo atendimento, não entregam no prazo, sem qualquer tipo de satisfação ao cliente. horrível!

- [R14] — `10b73c782535c2ca6f5df1188f3785a6`: pessimo atendimento, sem canal de reclamação

- [R15] — `7a93610296d0a660555dbb57d6b9024c`: péssimo * mercadoria super atrasada e nem satisfação

## Qual é a capital da França?

Resposta registrada:

Nao encontrei evidencias suficientes na base de avaliacoes para responder com seguranca. Motivo: a pergunta esta fora do escopo da base (avaliacoes de clientes da Olist).

Filtros: `{}`.

## O que os clientes acham da entrega por drone?

Resposta registrada:

Nao encontrei evidencias suficientes na base de avaliacoes para responder com seguranca. Motivo: apenas 1 avaliacao(oes) relevante(s); o minimo configurado e 3.

Filtros: `{}`.
