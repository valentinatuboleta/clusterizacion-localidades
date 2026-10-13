"""
Suite de Pruebas Automatizadas y Herméticas para la Gobernanza de Stopwords de Marketing.

Valida:
1. Cadena de resolución de tres niveles:
   - Nivel 1: Descarga desde Azure Blob Storage (mockeado herméticamente).
   - Nivel 2: Lectura de CSV local gobernado.
   - Nivel 3: Fallback a constantes embebidas con emisión de warning.
2. Limpieza léxica de ruido publicitario:
   - Eliminación de nombres de gira y stopwords ("PALCO DESPECHO - SENDE TOUR" -> "PALCO DESPECHO").
   - Preservación íntegra de entradas limpias.
3. Test de inercia ⭐:
   - Demuestra que el modelo congelado en producción es inerte ante la adición de nuevas stopwords
     al CSV oficial hasta el próximo re-entrenamiento (v2.6).
4. Proposer de stopwords (datos sintéticos):
   - Propone candidatos válidos de productos con frecuencia >= 3 eventos distintos.
   - Rechaza categóricamente vocabulario estructural auxiliar y tags (e.g. PALCO, VIP, PLATEA).
5. Consolidación de stopwords:
   - Promueve registros aceptados al CSV oficial con versión incrementada y simula subida a Blob.
"""

import os
import io
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.nlp_utils import (
    cargar_stopwords,
    limpiar_ruido_marketing,
    DEFAULT_STOPWORDS_MARKETING
)
from src import nlp_utils
from src.clustering import cargar_modelo_clustering, predecir_arquetipos_demanda
from scripts.proponer_stopwords import (
    extraer_candidatos_stopwords,
    consolidar_stopwords,
    obtener_vocabulario_prohibido
)


class TestCadenaResolucionStopwords(unittest.TestCase):
    """Valida la cadena de resolución jerárquica de stopwords: Blob -> CSV Local -> Embebido."""

    def test_nivel_1_azure_blob_mock(self):
        """Valida que con AZURE_STOPWORDS_URL definida y cliente Blob mockeado, se cargue desde Blob."""
        csv_simulado = "patron,tipo,fecha_agregado,fuente,version_vocab\n\\bPATRON_BLOB\\b,tour,2026-10-13,blob,v1\n"
        blob_mock = MagicMock()
        blob_mock.download_blob.return_value.readall.return_value = csv_simulado.encode("utf-8")

        with patch.dict(os.environ, {"AZURE_STOPWORDS_URL": "https://storage.blob.core.windows.net/test/sw.csv"}):
            with patch("azure.storage.blob.BlobClient.from_blob_url", return_value=blob_mock):
                stopwords = cargar_stopwords(ruta_local="ruta_inexistente.csv")
                self.assertIn(r"\bPATRON_BLOB\b", stopwords)
                self.assertEqual(len(stopwords), 1)

    def test_nivel_2_csv_local(self):
        """Valida que sin variable de entorno de Blob, cargue correctamente desde el CSV local."""
        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tmp:
            tmp.write("patron,tipo,fecha_agregado,fuente,version_vocab\n\\bPATRON_LOCAL\\b,tour,2026-10-13,local,v1\n")
            tmp_path = tmp.name

        try:
            with patch.dict(os.environ, {}, clear=True):
                # Asegurar que AZURE_STOPWORDS_URL no exista
                os.environ.pop("AZURE_STOPWORDS_URL", None)
                stopwords = cargar_stopwords(ruta_local=tmp_path)
                self.assertIn(r"\bPATRON_LOCAL\b", stopwords)
                self.assertEqual(len(stopwords), 1)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_nivel_3_default_embebido_con_warning(self):
        """Valida que ante fallo de Blob y archivo local ausente, caiga a las constantes embebidas con warning."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AZURE_STOPWORDS_URL", None)
            with self.assertLogs("src.nlp_utils", level="WARNING") as cm:
                stopwords = cargar_stopwords(ruta_local="archivo_absolutamente_inexistente.csv")
                self.assertEqual(stopwords, DEFAULT_STOPWORDS_MARKETING)
                self.assertTrue(any("Nivel 3" in msg for msg in cm.output))


class TestLimpiezaRuidoMarketing(unittest.TestCase):
    """Valida la semántica de limpieza léxica sobre descripciones de localidades."""

    def test_limpieza_patron_sende_tour(self):
        """Valida que 'PALCO DESPECHO - SENDE TOUR' se limpie a 'PALCO DESPECHO'."""
        entrada = "PALCO DESPECHO - SENDE TOUR"
        esperado = "PALCO DESPECHO"
        resultado = limpiar_ruido_marketing(entrada)
        self.assertEqual(resultado, esperado)

    def test_entrada_limpia_sin_cambios(self):
        """Valida que una entrada puramente estructural no sufra alteraciones."""
        entradas_puras = [
            "PALCO DESPECHO",
            "PLATEA VIP",
            "SEGUNDO PISO OCCIDENTAL",
            "GRADA GENERAL NORTE"
        ]
        for entrada in entradas_puras:
            self.assertEqual(limpiar_ruido_marketing(entrada), entrada)


class TestInerciaProduccion(unittest.TestCase):
    """
    Test de inercia ⭐:
    Documenta que agregar un nuevo patrón publicitario al CSV no altera las predicciones
    del modelo congelado actual en producción hasta que se ejecute el re-entrenamiento (v2.6).
    """

    def test_inercia_prediccion_celda_a_celda(self):
        # 1. Cargar modelo congelado de producción v2.5
        ruta_modelo = "data/processed/modelo_clustering_v2_5.joblib"
        if not os.path.exists(ruta_modelo):
            self.skipTest(f"Modelo {ruta_modelo} no encontrado en entorno de pruebas.")

        modelo = cargar_modelo_clustering(ruta_modelo)

        # 2. Fixture sintético representativo de un evento multi-zona
        df_evento = pd.DataFrame({
            "t_performance_id": [9901, 9901, 9901],
            "site": ["MOVISTAR ARENA", "MOVISTAR ARENA", "MOVISTAR ARENA"],
            "logical_seat_category": [
                "PALCO OCCIDENTAL NUEVO TOUR 2026",
                "PLATEA VIP NUEVO TOUR 2026",
                "GENERAL NORTE"
            ],
            "dn_quota": [100, 300, 600],
            "performance_quota": [1000, 1000, 1000],
            "ave_unit_amt_itx": [200000.0, 150000.0, 50000.0],
            "ave_total_unit_amt_itx": [200000.0, 150000.0, 50000.0],
            "ave_base_unit_amt_itx": [200000.0, 150000.0, 50000.0],
            "med_unit_amt_itx": [200000.0, 150000.0, 50000.0],
            "med_total_unit_amt_itx": [200000.0, 150000.0, 50000.0],
            "med_base_unit_amt_itx": [200000.0, 150000.0, 50000.0],
            "net_sold_p_qty": [80.0, 250.0, 500.0],
            "net_sold_c_qty": [5.0, 10.0, 20.0]
        })

        # 3. Predicción A (vocabulario sin el nuevo patrón)
        pred_A = predecir_arquetipos_demanda(df_evento.copy(), modelo)

        # 4. Modificar vocabulario agregando un patrón nuevo
        nuevo_patron = r"\bNUEVO TOUR 2026\b"
        patrones_originales = nlp_utils.STOPWORDS_MARKETING.copy()
        try:
            nlp_utils.STOPWORDS_MARKETING.append(nuevo_patron)

            # 5. Predicción B (vocabulario enriquecido con el nuevo tour)
            pred_B = predecir_arquetipos_demanda(df_evento.copy(), modelo)

            # 6. Verificación estricta: predicción idéntica celda a celda en clusters,
            # arquetipos, scores y frontera
            cols_validacion = [
                "cluster",
                "arquetipo_demanda",
                "score_confianza",
                "es_frontera",
                "segundo_arquetipo"
            ]
            pd.testing.assert_frame_equal(pred_A[cols_validacion], pred_B[cols_validacion])
        finally:
            nlp_utils.STOPWORDS_MARKETING = patrones_originales


class TestProposerStopwordsSintetico(unittest.TestCase):
    """Valida la extracción de candidatos con las 4 rejillas de seguridad y consolidación."""

    def test_candidato_valido_y_rechazo_estructural(self):
        """
        Valida que un término recurrente de producto (ej. 'BICHOTA') sea propuesto como candidato,
        pero vocabulario estructural ('PALCO', 'VIP', 'PLATEA') sea estrictamente bloqueado.
        """
        # Dataset sintético con 4 eventos distintos
        df_sintetico = pd.DataFrame({
            "product": [
                "CONCIERTO BICHOTA GIRA 2026",
                "PALCO VIP BICHOTA TOUR",
                "PLATEA BICHOTA EN VIVO",
                "BICHOTA ESTADIO FESTIVAL"
            ],
            "t_performance_id": [1001, 1002, 1003, 1004]
        })

        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tmp_oficial:
            tmp_oficial.write("patron,tipo,fecha_agregado,fuente,version_vocab\n\\bSENDE\\b,tour,2026-10-13,manual,v1\n")
            ruta_oficial = tmp_oficial.name

        try:
            df_candidatos = extraer_candidatos_stopwords(
                df=df_sintetico,
                col_producto="product",
                col_evento="t_performance_id",
                min_eventos=3,
                ruta_stopwords_oficial=ruta_oficial
            )

            patrones_propuestos = df_candidatos["patron"].tolist()

            # El candidato de gira no estructural 'BICHOTA' debe ser propuesto
            self.assertIn(r"\bBICHOTA\b", patrones_propuestos)

            # El vocabulario estructural bajo ninguna circunstancia debe ser propuesto
            for prohibido in [r"\bPALCO\b", r"\bVIP\b", r"\bPLATEA\b", r"\bGENERAL\b"]:
                self.assertNotIn(prohibido, patrones_propuestos)
        finally:
            if os.path.exists(ruta_oficial):
                os.remove(ruta_oficial)

    def test_consolidar_promueve_y_actualiza_version(self):
        """Valida que --consolidar mueva filas aceptadas incrementando version_vocab y notificando a Blob."""
        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tmp_rev:
            tmp_rev.write(
                "patron,frecuencia,n_eventos,ejemplo_producto,decision\n"
                "\\bNUEVA_GIRA\\b,10,4,EJEMPLO TOUR,aceptar\n"
                "\\bRECHAZADO\\b,5,3,OTRO TOUR,\n"
            )
            ruta_rev = tmp_rev.name

        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tmp_oficial:
            tmp_oficial.write(
                "patron,tipo,fecha_agregado,fuente,version_vocab\n"
                "\\bSENDE\\b,tour,2026-10-13,manual,v1\n"
            )
            ruta_oficial = tmp_oficial.name

        blob_mock = MagicMock()
        try:
            with patch.dict(os.environ, {"AZURE_STOPWORDS_URL": "https://storage.blob.core.windows.net/test/sw.csv"}):
                with patch("azure.storage.blob.BlobClient.from_blob_url", return_value=blob_mock):
                    n_cons, ver = consolidar_stopwords(ruta_revision=ruta_rev, ruta_oficial=ruta_oficial)

                    self.assertEqual(n_cons, 1)
                    self.assertEqual(ver, "v2")

                    # Validar que el CSV oficial contenga la nueva fila con v2
                    df_oficial = pd.read_csv(ruta_oficial)
                    self.assertIn(r"\bNUEVA_GIRA\b", df_oficial["patron"].tolist())
                    fila_nueva = df_oficial[df_oficial["patron"] == r"\bNUEVA_GIRA\b"].iloc[0]
                    self.assertEqual(fila_nueva["version_vocab"], "v2")
                    self.assertEqual(fila_nueva["fuente"], "propuesto")

                    # Validar que se llamó al upload de Blob
                    blob_mock.upload_blob.assert_called_once()

                    # Validar que la fila aceptada fue removida de la revisión
                    df_rev_restante = pd.read_csv(ruta_rev)
                    self.assertEqual(len(df_rev_restante), 1)
                    self.assertNotIn(r"\bNUEVA_GIRA\b", df_rev_restante["patron"].tolist())
        finally:
            if os.path.exists(ruta_rev):
                os.remove(ruta_rev)
            if os.path.exists(ruta_oficial):
                os.remove(ruta_oficial)


if __name__ == "__main__":
    unittest.main()
