"""
Tests unitarios y funcionales de endpoints FastAPI con TestClient (Bloque 1).

Valida:
1. GET /health retorna 200 con las versiones de los payloads cargados en memoria.
2. POST /v1/predecir-localidades con evento valido retorna 200 y esquema completo.
3. Evento con precio <= 0 retorna HTTP 400 con mensaje legible.
4. Evento sin localidades retorna HTTP 400 (o 422) con mensaje legible.
5. Error interno no controlado responde 500 con detalle generico sin traza interna.
"""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.main import app


class TestApiEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Usar TestClient con raise_server_exceptions=False para verificar respuestas HTTP 500
        cls.client_cm = TestClient(app, raise_server_exceptions=False)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_health_check_200_con_versiones(self):
        """Valida que /health retorne 200 con las versiones macro y jerárquica."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("2.5", data["modelo_clustering_version"])
        self.assertIn("3.0", data["modelo_jerarquia_version"])
        self.assertIn("2.5", data["modelo_version"])

    def test_predecir_evento_valido_200_y_esquema_completo(self):
        """Valida inferencia exitosa 200 con esquema completo."""
        payload = {
            "t_performance_id": 10230442113886,
            "product": "HANS ZIMMER VS JOHN WILLIAMS",
            "site": "TEATRO JORGE ELIECER GAITAN",
            "performance_quota": 1650.0,
            "localidades": [
                {
                    "logical_seat_category": "PLATEA DELANTERA",
                    "dn_quota": 300.0,
                    "med_base_unit_amt_itx": 250000.0
                },
                {
                    "logical_seat_category": "PLATEA POSTERIOR",
                    "dn_quota": 400.0,
                    "med_base_unit_amt_itx": 180000.0
                },
                {
                    "logical_seat_category": "PALCO PRIMER PISO",
                    "dn_quota": 150.0,
                    "med_base_unit_amt_itx": 280000.0
                },
                {
                    "logical_seat_category": "BALCON",
                    "dn_quota": 400.0,
                    "med_base_unit_amt_itx": 120000.0
                },
                {
                    "logical_seat_category": "PISO 2",
                    "dn_quota": 250.0,
                    "med_base_unit_amt_itx": 90000.0
                },
                {
                    "logical_seat_category": "PISO 3",
                    "dn_quota": 150.0,
                    "med_base_unit_amt_itx": 60000.0
                }
            ]
        }
        response = self.client.post("/v1/predecir-localidades", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("modelo_version", data)
        self.assertEqual(data["filas_in"], 6)
        self.assertEqual(data["filas_out"], 6)
        self.assertIn("reporte_filtro", data)
        self.assertIsInstance(data["predicciones"], list)
        self.assertEqual(len(data["predicciones"]), 6)

        # Validar campos de cada prediccion
        for p in data["predicciones"]:
            self.assertIn("logical_seat_category", p)
            self.assertIn("arquetipo", p)
            self.assertIn("micro_cluster", p)
            self.assertIn("label_auto", p)
            self.assertIn("confianza", p)
            self.assertIn("es_frontera", p)
            self.assertIn("segmento_incierto", p)

    def test_evento_con_precio_cero_retorna_400(self):
        """Valida que una localidad con precio <= 0 sea rechazada con HTTP 400."""
        payload = {
            "t_performance_id": 1,
            "product": "RECITAL",
            "site": "AUDITORIO",
            "localidades": [
                {
                    "logical_seat_category": "GENERAL",
                    "dn_quota": 100.0,
                    "med_base_unit_amt_itx": 0.0
                }
            ]
        }
        response = self.client.post("/v1/predecir-localidades", json=payload)
        self.assertEqual(response.status_code, 400)
        detail = response.json().get("detail", "")
        self.assertTrue("mayor a 0" in detail or "gt=0" in detail or "validaci" in detail.lower())

    def test_evento_sin_localidades_retorna_400_o_422(self):
        """Valida que una solicitud con lista de localidades vacía sea rechazada."""
        payload = {
            "t_performance_id": 1,
            "product": "RECITAL",
            "site": "AUDITORIO",
            "localidades": []
        }
        response = self.client.post("/v1/predecir-localidades", json=payload)
        self.assertIn(response.status_code, [400, 422])
        detail = response.json().get("detail", "")
        self.assertTrue(len(detail) > 0)

    def test_excepcion_pipeline_retorna_500_generico_sin_traza(self):
        """Valida que un fallo imprevisto retorne 500 con mensaje genérico y sin filtrar traceback."""
        payload = {
            "t_performance_id": 1,
            "product": "RECITAL",
            "site": "AUDITORIO",
            "localidades": [
                {
                    "logical_seat_category": "GENERAL",
                    "dn_quota": 100.0,
                    "med_base_unit_amt_itx": 50000.0
                }
            ]
        }
        with patch("src.api.main.predecir_evento", side_effect=RuntimeError("Fallo interno simulado")):
            response = self.client.post("/v1/predecir-localidades", json=payload)
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertEqual(data["detail"], "Error interno del servidor al procesar la prediccion del evento.")
            self.assertNotIn("Fallo interno simulado", str(data))
            self.assertNotIn("Traceback", str(data))


if __name__ == "__main__":
    unittest.main()
