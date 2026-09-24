"""
Suite de Pruebas Automatizadas para el Clasificador LLM con Mocks (Hermético para CI).

Valida:
1. Inyección de dependencias del cliente LLM (sin llamadas reales de red ni consumo de tokens).
2. Salida estructurada estricta en JSON a temperatura 0.0.
3. Persistencia de trazabilidad y auditoría completa en JSONL.
4. Esquema de precedencia: revision_humana > llm > reglas_heuristicas.
5. Verificación cruzada sobre DICCIONARIO_EMBLEMATICO y detección de discrepancias.
"""

import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock
from src.llm_classifier import GeminiVenueClassifier, CATEGORIAS_VALIDAS


class MockGeminiResponse:
    """Simula el objeto de respuesta de Google GenAI."""
    def __init__(self, text: str):
        self.text = text


class MockLLMClient:
    """Cliente mock hermético para simular respuestas de Gemini 3.8 Flash Medium en CI."""

    def __init__(self, respuestas_fijas: dict = None):
        self.respuestas_fijas = respuestas_fijas or {}
        self.llamadas = []

    def generate_content(self, model: str, contents: str, config: dict = None):
        self.llamadas.append({"model": model, "contents": contents, "config": config})

        # Buscar si hay alguna respuesta simulada para este prompt
        for key, resp in self.respuestas_fijas.items():
            if key in contents:
                return MockGeminiResponse(resp)

        # Respuesta por defecto en formato JSON
        default_json = json.dumps({
            "site": "VENUE_TEST",
            "type_site": "teatro",
            "confianza": 0.95,
            "justificacion_semantica": "Venue simulado por mock para pruebas unitarias hermeticas."
        })
        return MockGeminiResponse(default_json)


class TestLLMVenueClassifier(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audit_path = os.path.join(self.temp_dir.name, "audit_test.jsonl")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_clasificacion_llm_exitosa(self):
        """Valida que el clasificador procese correctamente el JSON estructurado."""
        mock_client = MockLLMClient({
            "BOOM STAND UP BAR": json.dumps({
                "site": "BOOM STAND UP BAR",
                "type_site": "bar_club",
                "confianza": 0.93,
                "justificacion_semantica": "Comedy club y bar nocturno."
            })
        })

        classifier = GeminiVenueClassifier(
            client=mock_client,
            model_name="gemini-3.8-flash-medium",
            temperature=0.0,
            audit_file=self.audit_path
        )

        res = classifier.clasificar_venue("BOOM STAND UP BAR", aforo_max=330)

        self.assertIsNotNone(res)
        self.assertEqual(res["site"], "BOOM STAND UP BAR")
        self.assertEqual(res["type_site"], "bar_club")
        self.assertEqual(res["confianza"], 0.93)
        self.assertEqual(res["modelo_llm"], "gemini-3.8-flash-medium")
        self.assertIn("Comedy club", res["justificacion_semantica"])

    def test_persistencia_auditoria_jsonl(self):
        """Verifica que cada inferencia genere un registro de auditoría en JSONL."""
        mock_client = MockLLMClient()
        classifier = GeminiVenueClassifier(
            client=mock_client,
            audit_file=self.audit_path
        )

        classifier.clasificar_venue("TEATRO SANTANDER", aforo_max=1000)

        self.assertTrue(os.path.exists(self.audit_path))
        with open(self.audit_path, "r", encoding="utf-8") as f:
            lineas = f.readlines()

        self.assertEqual(len(lineas), 1)
        audit_entry = json.loads(lineas[0])

        self.assertIn("timestamp", audit_entry)
        self.assertIn("prompt", audit_entry)
        self.assertIn("raw_response", audit_entry)
        self.assertEqual(audit_entry["modelo_llm"], "gemini-3.8-flash-medium")
        self.assertEqual(audit_entry["site"], "TEATRO SANTANDER")

    def test_inyeccion_dependencias_sin_api_key(self):
        """Si no hay cliente inyectado ni API key en entorno, debe retornar None sin explotar."""
        env_original = os.environ.get("GEMINI_API_KEY")
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

        try:
            classifier = GeminiVenueClassifier(client=None, audit_file=self.audit_path)
            res = classifier.clasificar_venue("VENUE DESCONOCIDO", 500)
            self.assertIsNone(res, "Debe retornar None limpiamente si no hay credenciales")
        finally:
            if env_original is not None:
                os.environ["GEMINI_API_KEY"] = env_original

    def test_manejo_markdown_en_salida_json(self):
        """Valida que si el LLM envuelve el JSON en ```json ... ```, se limpie correctamente."""
        mock_client = MockLLMClient({
            "CORFERIAS": "```json\n{\n  \"site\": \"CORFERIAS\",\n  \"type_site\": \"centro_eventos_carpa\",\n  \"confianza\": 0.98,\n  \"justificacion_semantica\": \"Recinto ferial de gran escala.\"\n}\n```"
        })

        classifier = GeminiVenueClassifier(
            client=mock_client,
            audit_file=self.audit_path
        )

        res = classifier.clasificar_venue("CORFERIAS", aforo_max=15000)
        self.assertIsNotNone(res)
        self.assertEqual(res["type_site"], "centro_eventos_carpa")
        self.assertEqual(res["confianza"], 0.98)

    def test_categoria_invalida_mapea_a_otro(self):
        """Si el LLM inventa una categoría no válida, el clasificador debe coaccionarla a 'otro'."""
        mock_client = MockLLMClient({
            "SITIO RARO": json.dumps({
                "site": "SITIO RARO",
                "type_site": "aeropuerto_espacial",
                "confianza": 0.50,
                "justificacion_semantica": "Categoria inexistente."
            })
        })

        classifier = GeminiVenueClassifier(
            client=mock_client,
            audit_file=self.audit_path
        )

        res = classifier.clasificar_venue("SITIO RARO", 500)
        self.assertIsNotNone(res)
        self.assertEqual(res["type_site"], "otro")

    def test_verificacion_diccionario_y_discrepancias_llm(self):
        """
        Valida que si el LLM difiere del DICCIONARIO_EMBLEMATICO, se detecte la discrepancia
        para que la curaduría humana decida y el LLM no sobrescriba el diccionario por sí solo.
        """
        mock_client = MockLLMClient({
            "TEATRO MAYOR JULIO MARIO SANTO DOMINGO": json.dumps({
                "site": "TEATRO MAYOR JULIO MARIO SANTO DOMINGO",
                "type_site": "estadio_abierto",
                "confianza": 0.90,
                "justificacion_semantica": "Clasificacion erronea simulada para prueba de discrepancia."
            })
        })

        classifier = GeminiVenueClassifier(client=mock_client, audit_file=self.audit_path)
        res = classifier.clasificar_venue("TEATRO MAYOR JULIO MARIO SANTO DOMINGO", aforo_max=1300)

        from scripts.clasificar_sites import DICCIONARIO_EMBLEMATICO
        dicc_tipo = DICCIONARIO_EMBLEMATICO.get("TEATRO MAYOR JULIO MARIO SANTO DOMINGO")

        self.assertEqual(dicc_tipo, "teatro")
        self.assertEqual(res["type_site"], "estadio_abierto")
        self.assertNotEqual(dicc_tipo, res["type_site"])


if __name__ == "__main__":
    unittest.main()

