"""Pergunta sobre atraso: o relato do cliente e as datas do pedido ficam em blocos separados,
montados pelo código, sem depender do LLM. Casos tirados da consulta real de 13/11/2026."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rag_pipeline as rp  # noqa: E402

# (texto, dias entre a entrega e a data estimada; positivo = depois)
CASOS = {
    "r7": ("tempo de espera produto de ótima qualidade, chegou como descrito. mais a entrega foi mais do que "
           "o esperado. deixou a desejar no tempo de entrega. muito demora", 36),
    "r2": ("atraso na entrega atraso na entrega", -10),
    "r3": ("passou do prazo de entrega, demorou.", 2),
    "r1": ("muita demora o prazo de entrega foi longo demais", -17),
}


def tabela():
    est = pd.Timestamp("2018-03-01")
    linhas = [{"review_id": rid, "review_full_text": txt, "review_comment_message": txt, "review_score": 2,
               "order_status": "delivered", "order_purchase_timestamp": "2018-02-01",
               "order_estimated_delivery_date": est, "order_delivered_customer_date": est + pd.Timedelta(days=d),
               "product_category_name_english": "x"} for rid, (txt, d) in CASOS.items()]
    return rp.construir_tabela_temas(pd.DataFrame(linhas))


class LLMProibido:
    def gerar(self, system, user):
        raise AssertionError("A pergunta de prazo não deveria depender do LLM.")


class Busca:
    """Devolve as avaliações na ordem R7, R2, R3, R1 com score alto."""
    def __init__(self, t):
        self.t = t

    def buscar(self, pergunta, k, filtros):
        return [{"review_id": r.review_id, "review_full_text": r.review_full_text, "review_score": r.review_score,
                 "score": 0.9 - i * 0.01} for i, r in enumerate(self.t.itertuples())][:k]


class Vetores:
    def encode(self, textos):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(len(textos), 8))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class PrazoTests(unittest.TestCase):
    def setUp(self):
        self.t = tabela()

    def test_temas_dos_tres_casos(self):
        temas = dict(zip(self.t.review_id, self.t.temas))
        self.assertIn("demora_entrega", temas["r7"]); self.assertNotIn("atraso_entrega", temas["r7"])
        self.assertIn("atraso_entrega", temas["r2"])
        self.assertIn("atraso_entrega", temas["r3"])

    def resposta(self):
        pipe = rp.RAGPipeline(Busca(self.t), Vetores(), LLMProibido(), self.t, {"min_evidencias": 3})
        return pipe.answer("O que os clientes relatam sobre atraso na entrega?")

    def test_blocos_relato_e_datas(self):
        r = self.resposta()
        self.assertFalse(r.abstencao)
        tag = {e.review_id: e.tag for e in r.evidencias}
        texto = r.resposta
        relato, datas = texto.split("**O que as datas do pedido confirmam**")
        self.assertIn("**O que os clientes relataram**", relato)
        linha = lambda bloco, inicio: next(l for l in bloco.splitlines() if l.startswith(inicio))
        # R7: relata demora, chegou 36 dias depois -> nunca entra como relato de atraso
        self.assertIn(tag["r7"], linha(relato, "- Demora ou prazo longo"))
        self.assertNotIn(tag["r7"], linha(relato, "- Atraso ou prazo descumprido"))
        self.assertIn(f"{tag['r7']} (entregue 36 dias depois da data estimada; o cliente relatou demora)",
                      linha(datas, "- Atraso só nas datas"))
        # R2: relata atraso, mas chegou 10 dias antes
        self.assertIn(tag["r2"], linha(relato, "- Atraso ou prazo descumprido"))
        self.assertIn(f"{tag['r2']} (entregue 10 dias antes da data estimada)",
                      linha(datas, "- Atraso relatado, mas não confirmado"))
        # R3: relata atraso e as datas confirmam
        self.assertIn(f"{tag['r3']} (entregue 2 dias depois da data estimada)",
                      linha(datas, "- Atraso relatado e confirmado"))
        # R1: demora, chegou antes
        self.assertIn(tag["r1"], linha(datas, "- Sem atraso pelas datas"))

    def test_citacoes_validas_e_comentarios_preservados(self):
        r = self.resposta()
        citadas = set(rp._TAG.findall(r.resposta))
        self.assertEqual(citadas, {e.tag[2:-1] for e in r.evidencias})     # todas as fontes citadas, nenhuma inventada
        self.assertEqual({e.texto for e in r.evidencias}, {txt for txt, _ in CASOS.values()})
        self.assertFalse(any("sem lastro" in a for a in r.avisos))
        self.assertIn("resposta_montada_pelo_app", r.trace)

    def test_pergunta_de_outro_tema_continua_no_llm(self):
        class LLMOk:
            chamado = False
            def gerar(self, system, user):
                LLMOk.chamado = True
                return "Os clientes elogiam o produto [R1]."
        pipe = rp.RAGPipeline(Busca(self.t), Vetores(), LLMOk(), self.t, {"min_evidencias": 3})
        pipe.answer("O que os clientes acham da qualidade do produto?")
        self.assertTrue(LLMOk.chamado)


if __name__ == "__main__":
    unittest.main()
