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
   - Propone candidatos válidos de productos con frecuencia >= min_eventos.
   - Rechaza categóricamente vocabulario estructural auxiliar y tags (e.g. PALCO, VIP, PLATEA).
   - Subsumción: lattice de n-gramas colapsa al padre compuesto de máxima frecuencia y longitud.
   - Clases sistemáticas: agrupa meses y ciudades en 1 sola fila representativa.
   - No re-proposición: ignora patrones previamente rechazados en el archivo de revisión.
   - Tope de salida: limita la cola a 100 filas y deriva excedentes a observación.
5. Consolidación de stopwords:
   - Valida estrictamente valores de decisión (solo 'aceptar'/'rechazar').
   - Promueve registros aceptados al CSV oficial con versión incrementada, simula subida a Blob,
     y preserva el archivo de revisión como historial permanente.
"""

import os
import io
import tempfile
import warnings
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
    aplicar_subsumcion,
    detectar_clases_sistematicas,
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
        ruta_modelo = "data/processed/modelo_clustering_v2_5.joblib"
        if not os.path.exists(ruta_modelo):
            self.skipTest(f"Modelo {ruta_modelo} no encontrado en entorno de pruebas.")

        modelo = cargar_modelo_clustering(ruta_modelo)

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

        pred_A = predecir_arquetipos_demanda(df_evento.copy(), modelo)

        nuevo_patron = r"\bNUEVO TOUR 2026\b"
        patrones_originales = nlp_utils.STOPWORDS_MARKETING.copy()
        try:
            nlp_utils.STOPWORDS_MARKETING.append(nuevo_patron)
            pred_B = predecir_arquetipos_demanda(df_evento.copy(), modelo)

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
    """Valida la extracción de candidatos con las 4 rejillas de seguridad, subsumción y clases."""

    def test_candidato_valido_y_rechazo_estructural(self):
        """
        Valida que un término recurrente de producto (ej. 'BICHOTA') sea propuesto como candidato,
        pero vocabulario estructural ('PALCO', 'VIP', 'PLATEA') sea estrictamente bloqueado.
        """
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
            df_rev, _, _ = extraer_candidatos_stopwords(
                df=df_sintetico,
                col_producto="product",
                col_evento="t_performance_id",
                min_eventos=3,
                ruta_stopwords_oficial=ruta_oficial
            )

            patrones_propuestos = df_rev["patron"].tolist()
            self.assertIn(r"\bBICHOTA\b", patrones_propuestos)

            for prohibido in [r"\bPALCO\b", r"\bVIP\b", r"\bPLATEA\b", r"\bGENERAL\b"]:
                self.assertNotIn(prohibido, patrones_propuestos)
        finally:
            if os.path.exists(ruta_oficial):
                os.remove(ruta_oficial)

    def test_subsumcion_lattice_colapsa_al_padre(self):
        """
        Valida que el lattice de n-gramas subsumidos ('AHI', 'AFUERA', 'LA BUSQUEDA')
        colapse exactamente en el padre compuesto ('AHI AFUERA LA BUSQUEDA'),
        registrando subsumidos=4.
        """
        candidatos = [
            {"patron": r"\bAHI AFUERA LA BUSQUEDA\b", "raw_text": "AHI AFUERA LA BUSQUEDA", "frecuencia": 100, "n_eventos": 50, "n_tokens": 4},
            {"patron": r"\bAHI AFUERA\b", "raw_text": "AHI AFUERA", "frecuencia": 100, "n_eventos": 50, "n_tokens": 2},
            {"patron": r"\bLA BUSQUEDA\b", "raw_text": "LA BUSQUEDA", "frecuencia": 100, "n_eventos": 50, "n_tokens": 2},
            {"patron": r"\bAHI\b", "raw_text": "AHI", "frecuencia": 100, "n_eventos": 50, "n_tokens": 1},
            {"patron": r"\bAFUERA\b", "raw_text": "AFUERA", "frecuencia": 100, "n_eventos": 50, "n_tokens": 1},
        ]

        sobrevivientes = aplicar_subsumcion(candidatos)

        self.assertEqual(len(sobrevivientes), 1)
        padre = sobrevivientes[0]
        self.assertEqual(padre["raw_text"], "AHI AFUERA LA BUSQUEDA")
        self.assertEqual(padre["subsumidos"], 4)

    def test_clases_sistematicas_meses_agrupa_en_una_fila(self):
        """Valida que productos con meses diferentes agrupen en 1 sola fila con tipo='calendario'."""
        df_meses = pd.DataFrame({
            "product": [
                "CONCIERTO VERANO - MAYO",
                "FESTIVAL JAZZ - JUNIO",
                "OBRA TEATRO - JULIO"
            ],
            "t_performance_id": [201, 202, 203]
        })

        clases = detectar_clases_sistematicas(df_meses, col_producto="product", col_evento="t_performance_id")
        clase_cal = next((c for c in clases if c["tipo"] == "calendario"), None)

        self.assertIsNotNone(clase_cal)
        self.assertEqual(clase_cal["tipo"], "calendario")
        self.assertEqual(clase_cal["n_eventos"], 3)
        self.assertGreaterEqual(clase_cal["subsumidos"], 3)

    def test_rechazados_no_repropuestos(self):
        """Valida que un patrón con decision='rechazar' en revision previa no se vuelva a proponer."""
        df_test = pd.DataFrame({
            "product": [
                "PRODUCTO RECHAZADO EN VIVO",
                "PRODUCTO RECHAZADO TOUR",
                "PRODUCTO RECHAZADO FESTIVAL"
            ],
            "t_performance_id": [301, 302, 303]
        })

        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tmp_rev:
            tmp_rev.write(
                "patron,tipo,frecuencia,n_eventos,subsumidos,ejemplo_producto,decision,motivo\n"
                "\\bRECHAZADO\\b,tour,10,3,0,EJEMPLO,rechazar,es parte del nombre legitimo\n"
            )
            ruta_rev = tmp_rev.name

        try:
            df_rev_nueva, _, _ = extraer_candidatos_stopwords(
                df=df_test,
                col_producto="product",
                col_evento="t_performance_id",
                min_eventos=3,
                ruta_revision_existente=ruta_rev
            )

            patrones_nuevos = df_rev_nueva["patron"].tolist()
            self.assertNotIn(r"\bRECHAZADO\b", patrones_nuevos)
        finally:
            if os.path.exists(ruta_rev):
                os.remove(ruta_rev)

    def test_tope_100_filas(self):
        """Valida que la cola de revisión respete el tope estricto de 100 filas."""
        filas = []
        for i in range(150):
            token_unico = f"ARTISTAX{i:04d}"
            # Cada token aparece en 3 eventos distintos para cumplir min_eventos=3
            for ev in [1000 + i, 2000 + i, 3000 + i]:
                filas.append({
                    "product": f"GIRA {token_unico} EN VIVO",
                    "t_performance_id": ev
                })
        df_150 = pd.DataFrame(filas)

        df_rev, df_obs, metricas = extraer_candidatos_stopwords(
            df=df_150,
            col_producto="product",
            col_evento="t_performance_id",
            min_eventos=3,
            max_salida=100
        )

        self.assertEqual(len(df_rev), 100)
        self.assertGreater(len(df_obs), 0)
        self.assertEqual(metricas["final"], 100)

    def test_consolidar_solo_acepta_valores_validos(self):
        """Valida que consolidar emita warning ante valores inválidos y solo procese 'aceptar'."""
        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tmp_rev:
            tmp_rev.write(
                "patron,tipo,frecuencia,n_eventos,subsumidos,ejemplo_producto,decision,motivo\n"
                "\\bVALIDO\\b,tour,10,5,0,EJEMPLO TOUR,aceptar,aprobado por marketing\n"
                "\\bINVALIDO\\b,tour,8,4,0,OTRO TOUR,si,typo de usuario\n"
                "\\bRECHAZADO\\b,tour,6,3,0,OTRO TOUR 2,rechazar,no es stopword\n"
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
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        n_cons, ver = consolidar_stopwords(ruta_revision=ruta_rev, ruta_oficial=ruta_oficial)

                        # Debe emitir warning por el valor inválido 'si'
                        self.assertTrue(any("valores inválidos en 'decision'" in str(warn.message) for warn in w))

                        # Solo 1 fila consolidada (VALIDO)
                        self.assertEqual(n_cons, 1)
                        self.assertEqual(ver, "v2")

                        df_oficial = pd.read_csv(ruta_oficial)
                        self.assertIn(r"\bVALIDO\b", df_oficial["patron"].tolist())
                        self.assertNotIn(r"\bINVALIDO\b", df_oficial["patron"].tolist())

                        # Historial preservado en revisión (no borrado)
                        df_rev_post = pd.read_csv(ruta_rev)
                        self.assertEqual(len(df_rev_post), 3)
                        fila_val = df_rev_post[df_rev_post["patron"] == r"\bVALIDO\b"].iloc[0]
                        self.assertTrue("consolidado" in str(fila_val["motivo"]).lower())
        finally:
            if os.path.exists(ruta_rev):
                os.remove(ruta_rev)
            if os.path.exists(ruta_oficial):
                os.remove(ruta_oficial)


if __name__ == "__main__":
    unittest.main()
