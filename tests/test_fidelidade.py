import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_pipeline import Evidencia, rotear, validar_resposta


class FidelidadeTests(unittest.TestCase):
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
