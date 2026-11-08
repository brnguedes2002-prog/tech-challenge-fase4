"""Testes do léxico de temas, da guarda de negação e das estatísticas do pipeline RAG.

Rodam sem Databricks: usam o pipeline local (TF-IDF + LLM extrativo) sobre avaliações sintéticas.
"""
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import rag_pipeline as rp


def temas(texto):
    return set(rp.rotular_temas(texto))


class AtrasoDemoraTests(unittest.TestCase):
    def test_atraso_explicito(self):
        for texto in ["A entrega atrasou 10 dias", "Chegou fora do prazo", "Passou do prazo e nada",
                      "O prazo não foi cumprido", "Não chegou no prazo prometido", "O prazo venceu e não recebi"]:
            with self.subTest(texto=texto):
                self.assertIn("atraso_entrega", temas(texto))

    def test_demora_nao_e_atraso(self):
        t = temas("Demorou bastante, mas chegou dentro do prazo")
        self.assertIn("demora_entrega", t)
        self.assertNotIn("atraso_entrega", t)

    def test_prazo_longo_e_demora(self):
        self.assertIn("demora_entrega", temas("Prazo de entrega muito longo"))

    def test_negacao_proxima_remove_atraso_e_demora(self):
        for texto in ["Não atrasou, chegou certinho", "Sem nenhum atraso", "Não houve muito atraso",
                      "Nem demorou", "O problema não foi a demora e sim a cor"]:
            with self.subTest(texto=texto):
                t = temas(texto)
                self.assertNotIn("atraso_entrega", t)
                self.assertNotIn("demora_entrega", t)

    def test_negacao_distante_nao_apaga_atraso(self):
        self.assertIn("atraso_entrega", temas("Não recebi o aviso e a entrega atrasou muito"))

    def test_atras_adverbio_nao_e_atraso(self):
        self.assertNotIn("atraso_entrega", temas("Comprei há dois meses atrás, vou atrás dos meus direitos"))

    def test_demora_de_atendimento_nao_e_demora_de_entrega(self):
        self.assertNotIn("demora_entrega", temas("A loja demorou para responder meu email"))


class NaoRecebidoTests(unittest.TestCase):
    def test_nao_recebido(self):
        for texto in ["Não recebi o produto", "Ainda não chegou", "Consta como entregue mas não está aqui"]:
            with self.subTest(texto=texto):
                self.assertIn("nao_recebido", temas(texto))

    def test_ainda_nao_recebi_nao_e_atraso(self):
        t = temas("Ainda não recebi meu pedido")
        self.assertIn("nao_recebido", t)
        self.assertNotIn("atraso_entrega", t)

    def test_nao_chegou_no_prazo_e_atraso_nao_extravio(self):
        t = temas("O produto não chegou no prazo")
        self.assertIn("atraso_entrega", t)
        self.assertNotIn("nao_recebido", t)
        self.assertNotIn("elogio_entrega", t)

    def test_nao_recebi_documento_ou_resposta(self):
        for texto in ["Recomendo, só não recebi a nota fiscal", "Não recebi resposta da loja"]:
            with self.subTest(texto=texto):
                self.assertNotIn("nao_recebido", temas(texto))

    def test_rotulo_explica_o_momento(self):
        self.assertIn("ate a avaliacao", rp.TEMAS["nao_recebido"]["label"])


class NegacaoProdutoTests(unittest.TestCase):
    def test_sem_defeito_nao_e_defeito(self):
        for texto in ["Chegou sem nenhum defeito", "Produto sem defeitos", "Não veio quebrado"]:
            with self.subTest(texto=texto):
                self.assertNotIn("produto_defeito", temas(texto))

    def test_defeito_continua_detectado(self):
        for texto in ["Veio quebrado", "Produto com defeito", "Parou de funcionar"]:
            with self.subTest(texto=texto):
                self.assertIn("produto_defeito", temas(texto))

    def test_elogio_negado_continua_fora(self):
        t = temas("Não recomendo, não chegou no prazo")
        self.assertNotIn("recomendacao", t)
        self.assertNotIn("elogio_entrega", t)


class RouterTests(unittest.TestCase):
    def test_pergunta_de_atraso_considera_atraso_e_demora(self):
        plano = rp.rotear("O que os clientes relatam sobre atraso na entrega?")
        self.assertEqual(plano.intencao, "especifica")
        self.assertIn("atraso_entrega", plano.temas_alvo)
        self.assertIn("demora_entrega", plano.temas_alvo)

    def test_precisao_tema_aceita_lista(self):
        evs = [rp.Evidencia(tag="[R1]", review_id="a", texto="x", score=1, temas=["demora_entrega"]),
               rp.Evidencia(tag="[R2]", review_id="b", texto="y", score=1, temas=["embalagem"])]
        self.assertEqual(rp.precisao_tema(["atraso_entrega", "demora_entrega"], evs), 0.5)
        self.assertEqual(rp.precisao_tema("atraso_entrega", evs), 0.0)
        self.assertIsNone(rp.precisao_tema(None, evs))


def _gold_sintetica():
    """40 avaliações: atraso real, demora dentro do prazo e elogios."""
    linhas = []
    base = {"order_status": "delivered", "order_purchase_timestamp": "2018-01-02 10:00:00",
            "product_category_name_english": "housewares", "review_score": 1}
    for i in range(12):   # atraso dito e confirmado pela data
        linhas.append({**base, "review_id": f"a{i:031d}", "review_full_text": f"A entrega atrasou {i + 6} dias, pedido {i}",
                       "order_delivered_customer_date": "2018-01-20", "order_estimated_delivery_date": "2018-01-10"})
    for i in range(12):   # demora, mas entregue antes da data estimada
        linhas.append({**base, "review_id": f"d{i:031d}", "review_full_text": f"Demorou muito para chegar, pedido {i}",
                       "order_delivered_customer_date": "2018-01-08", "order_estimated_delivery_date": "2018-01-10"})
    for i in range(16):   # o vocabulário da pergunta precisa existir no corpus (camada 1b)
        linhas.append({**base, "review_score": 5, "review_id": f"e{i:031d}",
                       "review_full_text": f"Produto excelente, recomendo a loja, compra {i}. Avaliações "
                                           "recorrentes sustentam a conclusão: relatados e comprovados",
                       "order_delivered_customer_date": "2018-01-05", "order_estimated_delivery_date": "2018-01-10"})
    return pd.DataFrame(linhas)


class EstatisticasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import sklearn  # noqa: F401
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("scikit-learn não instalado")
        cls.pipe = rp.pipeline_local(rp.construir_tabela_temas(_gold_sintetica()), config={"min_score": 0.05})

    def test_agregada_cruza_atraso_e_demora_com_a_data(self):
        r = self.pipe.answer("Quais reclamações relacionadas à entrega aparecem com maior frequência?")
        self.assertEqual(r.intencao, "agregada")
        self.assertFalse(r.abstencao, r.motivo)
        atraso = r.estatisticas["atraso_confirmado_pelos_dados"]
        demora = r.estatisticas["demora_x_data_de_entrega"]
        self.assertEqual((atraso["pedidos_entregues_apos_a_data_estimada"], atraso["percentual"]), (12, 100.0))
        self.assertEqual((demora["pedidos_entregues_apos_a_data_estimada"], demora["percentual"]), (0, 0.0))

    def test_pedido_de_evidencia_sobre_recorrencia_leva_estatistica(self):
        r = self.pipe.answer("Quais avaliações sustentam a conclusão de que existem problemas recorrentes "
                             "relacionados ao prazo de entrega?")
        self.assertEqual(r.intencao, "especifica")
        self.assertFalse(r.abstencao, r.motivo)
        self.assertIsNotNone(r.estatisticas)
        nomes = {t["tema"] for t in r.estatisticas["temas"]}
        self.assertEqual(nomes, {rp.TEMAS["atraso_entrega"]["label"], rp.TEMAS["demora_entrega"]["label"]})

    def test_pergunta_de_exemplos_sem_recorrencia_nao_leva_estatistica(self):
        r = self.pipe.answer("O que os clientes relatam sobre atraso na entrega?")
        self.assertEqual(r.intencao, "especifica")
        self.assertIsNone(r.estatisticas)


class ValidacaoTests(unittest.TestCase):
    def setUp(self):
        self.evs = [rp.Evidencia(tag="[R1]", review_id="a", texto="x", score=1)]

    def test_frequencia_sem_estatistica_gera_aviso(self):
        ok, motivo, avisos = rp.validar_resposta("A maioria reclama de atraso [R1]", self.evs, "<evidencias></evidencias>")
        self.assertTrue(ok)
        self.assertTrue(any("frequencia" in a for a in avisos))

    def test_frequencia_com_estatistica_nao_gera_aviso(self):
        ctx = "<estatisticas>{}</estatisticas><evidencias></evidencias>"
        ok, motivo, avisos = rp.validar_resposta("A maioria reclama de atraso [R1]", self.evs, ctx)
        self.assertFalse(any("frequencia" in a for a in avisos))

    def test_prompt_tem_regras_de_metadado_causa_frequencia_e_demora(self):
        p = rp.SYSTEM_PROMPT
        for trecho in ["entrega_atrasada", "causas", "Palavras de frequencia", "Demora nao e o mesmo que atraso"]:
            with self.subTest(trecho=trecho):
                self.assertIn(trecho, p)


if __name__ == "__main__":
    unittest.main()
