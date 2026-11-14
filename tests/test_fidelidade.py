import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_pipeline import Evidencia, _RASGOS, _mencao_afirmativa, rotear, validar_resposta


class FidelidadeTests(unittest.TestCase):
    def test_eventos_concluidos_separados_e_negados(self):
        casos = [
            ("Estorno feito", "não recebi o reembolso", False),
            ("Estorno feito", "devolvi o produto", False),
            ("Estorno feito", "devolvi o produto, mas fizeram o estorno", True),
            ("Caso resolvido", "o problema não foi resolvido", False),
            ("Caso resolvido", "o problema foi resolvido", True),
            ("Produto devolvido", "fizeram o estorno", False),
            ("Produto devolvido", "devolvi o produto", True),
        ]
        for afirmacao, fonte, esperado in casos:
            with self.subTest(afirmacao=afirmacao, fonte=fonte):
                evidencias = [Evidencia("[R1]", "1", fonte, 0.8)]
                self.assertEqual(validar_resposta(f"{afirmacao} [R1].", evidencias, "")[0], esperado)

    def test_rasgos_variantes_e_negacoes(self):
        self.assertFalse(_mencao_afirmativa("se houver avaria, rasgos serão comunicados", _RASGOS))
        self.assertTrue(_mencao_afirmativa("o produto veio rasgado. se precisar, devolvo", _RASGOS))
        self.assertFalse(_mencao_afirmativa("se vier rasgado, vou devolver", _RASGOS))
        self.assertTrue(_mencao_afirmativa("se precisar, devolvo; o produto veio rasgado", _RASGOS))
        self.assertFalse(_mencao_afirmativa("não veio rasgado", _RASGOS))
        self.assertFalse(_mencao_afirmativa("sem nenhum rasgo", _RASGOS))
        self.assertTrue(_mencao_afirmativa("as peças rasgaram", _RASGOS))
        self.assertTrue(_mencao_afirmativa("não veio amassado, mas veio rasgado", _RASGOS))

    def test_transparencia_negada_nao_sustenta_elogio(self):
        fonte = Evidencia("[R1]", "1", "a loja não foi transparente", 0.8)
        self.assertTrue(validar_resposta("Falta de transparência [R1].", [fonte], "")[0])
        for fonte, esperado in (("a loja não foi transparente", False), ("a loja foi transparente", True)):
            with self.subTest(fonte=fonte):
                evidencias = [Evidencia("[R1]", "1", fonte, 0.8)]
                self.assertEqual(validar_resposta("Transparência [R1].", evidencias, "")[0], esperado)

    def test_espera_por_resposta_da_loja(self):
        plano = rotear("O que os compradores dizem sobre esperar uma resposta da loja após reclamar de um produto errado?")
        self.assertIn("atendimento", plano.temas_alvo)
        self.assertNotIn("demora_entrega", plano.temas_alvo)

    def test_resolucao_e_transparencia_exigem_suporte(self):
        fontes = [
            Evidencia("[R1]", "1", "devolvi e fizeram o estorno; loja transparente", 0.8),
            Evidencia("[R2]", "2", "pedi reembolso e espero que seja resolvido; loja comprometida", 0.8),
        ]
        for afirmacao in ("Estorno feito", "Transparência"):
            with self.subTest(afirmacao=afirmacao):
                self.assertFalse(validar_resposta(f"{afirmacao} [R1], [R2].", fontes, "")[0])
                self.assertTrue(validar_resposta(f"{afirmacao} [R1].", fontes, "")[0])

    def test_demora_no_retorno_e_atendimento(self):
        plano = rotear("O que os comentários de notas 1 e 2 dizem sobre demora para receber retorno do vendedor?")
        self.assertEqual(plano.temas_alvo, ["atendimento"])

    def test_produto_em_perfeito_estado_exige_suporte(self):
        fontes = [
            Evidencia("[R1]", "1", "produto excelente e bem embalado", 0.8),
            Evidencia("[R2]", "2", "produto chegou em perfeito estado", 0.8),
        ]
        self.assertFalse(validar_resposta("Produto em perfeito estado [R1], [R2].", fontes, "")[0])
        self.assertTrue(validar_resposta("Produto em perfeito estado [R2].", fontes, "")[0])

    def test_atribuicoes_e_citacoes(self):
        fontes = [
            Evidencia("[R1]", "1", "defeito de fabricação; entrega muito rápida", 0.8),
            Evidencia("[R2]", "2", "produto com defeito; entrega super rápida e descaso", 0.8),
        ]
        casos = {
            "Defeitos de fabricação [R1], [R2].": False,
            "Defeito de fabricação [R1], produto com defeito [R2].": True,
            "Falta de retorno do vendedor [R2].": False,
            "Entrega “muito rápida” [R1], [R2].": False,
            "Entrega “muito rápida” [R1] e “super rápida” [R2].": True,
        }
        for resposta, esperado in casos.items():
            with self.subTest(resposta=resposta):
                self.assertEqual(validar_resposta(resposta, fontes, "")[0], esperado)
