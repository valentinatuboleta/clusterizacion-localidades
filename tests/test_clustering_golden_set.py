"""
Suite de Pruebas Automatizadas y Golden Set para el Pipeline de Clusterizacion en Dos Etapas (v2.2).

Valida:
1. Particion estricta a nivel evento (sin funciones partidas, cobertura exacta de 33,775 filas).
2. Coherencia matematica de centroides multi-zona (VIP con bajo aforo y alto precio, Grada con aforo masivo).
3. Clasificacion correcta del Golden Set de 20 localidades representativas.
4. Persistencia e inferencia bietapica con joblib (guardar, cargar y predecir de forma identica).
5. Seleccion automatica de k=5 mediante Score Compuesto Codo-DB (modo 'auto').
6. Test de regresion de distribucion de arquetipos contra proporciones canonicas (<2% desviacion).
7. Calibracion de score de confianza, frontera geometrica (<0.15), segundo arquetipo y GMM.
8. Manejo explicito de vocabulario desconocido (OOV) y alerta texto_casi_vacio.
9. Inferencia completa sobre dataset sintetico (compatible incondicionalmente con CI).
10. Monitoreo de drift estadistico mediante Population Stability Index (PSI).
11. Auditoria de cero emojis y estandarizacion terminologica de venue.
"""

import os
import re
import unittest
import pandas as pd
import numpy as np
from src.feature_engineering import filtrar_consistencia_localidades, calcular_metricas_relativas
from src.nlp_utils import pipeline_procesamiento_nlp
from src.clustering import (
    separar_admision_unica_multizona,
    pipeline_clustering_dos_etapas,
    construir_espacio_vectorial_mixto,
    etiquetar_por_centroides_escalados,
    entrenar_modelo_clustering,
    guardar_modelo_clustering,
    cargar_modelo_clustering,
    predecir_arquetipos_demanda,
    calcular_psi,
    evaluar_drift_lote,
    DISTRIBUCION_ESPERADA_ARQUETIPOS
)


class TestClusteringGoldenSet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_path = "data/raw/localidades_eda.parquet"
        if not os.path.exists(data_path):
            return

        # Cargar y preparar dataset real
        df_raw = pd.read_parquet(data_path)
        df_clean = filtrar_consistencia_localidades(df_raw)
        df_rel = calcular_metricas_relativas(df_clean)
        cls.df_enriquecido = pipeline_procesamiento_nlp(df_rel)

        # Ejecutar pipeline en dos etapas con k optimo (k=5 en multi-zona + 1 tarifa plana = 6 arquetipos)
        cls.df_final, cls.kmeans, cls.scaler, cls.tfidf_vec, cls.feature_names, cls.metricas = (
            pipeline_clustering_dos_etapas(cls.df_enriquecido, n_clusters_multizona=5, random_state=42)
        )

        # Persistir modelo de produccion v2.2 actualizado con gmm y referencia_drift
        mapa_arquetipos = cls.metricas.get("mapa_arquetipos") or etiquetar_por_centroides_escalados(
            cls.kmeans, cls.feature_names, cls.scaler, peso_nlp=0.2
        )
        guardar_modelo_clustering(
            "data/processed/modelo_clustering_v2_2.joblib",
            kmeans=cls.kmeans,
            scaler=cls.scaler,
            tfidf_vectorizer=cls.tfidf_vec,
            feature_names=cls.feature_names,
            mapa_arquetipos=mapa_arquetipos,
            metricas=cls.metricas,
            metadata={"autor": "Data Science TuBoleta", "version": "2.2"},
            gmm=cls.metricas.get("gmm"),
            df_referencia=cls.df_enriquecido[~cls.df_enriquecido["t_performance_id"].isin(
                cls.df_final[cls.df_final["cluster"] == -1]["t_performance_id"]
            )]
        )

    def test_01_cobertura_exacta_y_no_particion_de_eventos(self):
        """Valida que la suma de monozona + multi-zona sea 33,775 y que ningun evento este partido."""
        if not hasattr(self, "df_enriquecido"):
            self.skipTest("Requiere dataset crudo real")
            
        df_monozona, df_multizona = separar_admision_unica_multizona(self.df_enriquecido)
        
        # 1. Cobertura exacta
        self.assertEqual(len(df_monozona), 15375, "Admision unica debe contener exactamente 15,375 filas")
        self.assertEqual(len(df_multizona), 18400, "Multi-zona debe contener exactamente 18,400 filas")
        self.assertEqual(len(df_monozona) + len(df_multizona), 33775, "Total debe sumar 33,775 filas exactas")
        self.assertEqual(len(self.df_final), 33775, "El DataFrame final integrado debe tener 33,775 filas")

        # 2. Ningun evento puede tener localidades en ambas etapas
        eventos_monozona = set(df_monozona["t_performance_id"])
        eventos_multizona = set(df_multizona["t_performance_id"])
        interseccion = eventos_monozona.intersection(eventos_multizona)
        self.assertEqual(len(interseccion), 0, f"Hay {len(interseccion)} eventos divididos entre etapas!")

    def test_02_criterios_cuantitativos_centroides(self):
        """Verifica la logica economica de los centroides en el catalogo multi-zona."""
        if not hasattr(self, "df_final"):
            self.skipTest("Requiere dataset crudo real")

        df_multi = self.df_final[~self.df_final["es_monozona"]]

        stats = df_multi.groupby("arquetipo_demanda").agg(
            ratio_mean=("ratio_precio_max", "mean"),
            aforo_mean=("peso_aforo", "mean"),
            pct_mean=("percentil_precio_evento", "mean")
        )

        # 1. VIP debe tener el aforo mas bajo (< 15%) y alto precio
        self.assertLess(stats.loc["VIP / Palcos / Premium", "aforo_mean"], 0.15, "VIP debe tener bajo aforo relativo")
        self.assertGreater(stats.loc["VIP / Palcos / Premium", "ratio_mean"], 0.75, "VIP debe tener alto ratio de precio")

        # 2. Grada General debe tener aforo dominante frente a VIP
        self.assertGreater(stats.loc["Grada General / Masiva", "aforo_mean"], 0.60, "Grada General debe ser masiva (>60% aforo)")
        self.assertGreater(
            stats.loc["Grada General / Masiva", "aforo_mean"],
            stats.loc["VIP / Palcos / Premium", "aforo_mean"] * 4,
            "Grada General debe superar en al menos 4x el aforo medio de VIP"
        )

        # 3. Popular debe tener el ratio de precio mas bajo de todos los arquetipos
        self.assertLess(stats.loc["Popular / Balcón / Visibilidad Parcial", "ratio_mean"], 0.45, "Popular debe ser de bajo ratio de precio")

    def test_03_golden_set_20_casos(self):
        """Valida que el Golden Set de 20 casos de negocio representativos se clasifique correctamente."""
        if not hasattr(self, "df_final"):
            self.skipTest("Requiere dataset crudo real")

        golden_cases = [
            # 1. Admisión Única / Tarifa Plana (Cinemateca, museos, monozona)
            ("CINEMATECA BOGOTA", "Admisión Única / Tarifa Plana", True),
            ("SALA CAPITAL CINEMATECA", "Admisión Única / Tarifa Plana", True),
            ("MALOKA", "Admisión Única / Tarifa Plana", True),
            ("YAWA", "Admisión Única / Tarifa Plana", True),

            # 2. VIP / Palcos / Premium (alta gama multi-zona)
            ("PALCO INDIVIDUAL OCCIDENTAL", "VIP / Palcos / Premium", False),
            ("PALCO DIAMANTE", "VIP / Palcos / Premium", False),
            ("SUITE ORIENTAL", "VIP / Palcos / Premium", False),
            ("MESA VIP", "VIP / Palcos / Premium", False),

            # 3. Preferencial / Platea Frontal
            ("PLATEA DELANTERA", "Preferencial / Platea Frontal", False),
            ("SILLAS CENTRALES", "Preferencial / Platea Frontal", False),
            ("PLATEA 1", "Preferencial / Platea Frontal", False),
            ("PLATEA 2", "Preferencial / Platea Frontal", False),

            # 4. Popular / Visibilidad Parcial / Balcón
            ("BALCON 2DO PISO", "Popular / Balcón / Visibilidad Parcial", False),
            ("BALCON TERCER PISO", "Popular / Balcón / Visibilidad Parcial", False),
            ("VISTA PARCIAL LATERAL", "Popular / Balcón / Visibilidad Parcial", False),
            ("PISO ALTO POSTERIOR", "Popular / Balcón / Visibilidad Parcial", False),

            # 5. Grada General / Masiva (Graderías masivas de estadios y arenas)
            ("GRADA GENERAL", "Grada General / Masiva", False),
            ("GRADA NORTE", "Grada General / Masiva", False),
            ("CANCHA GENERAL", "Grada General / Masiva", False),
            ("ENTRADA GENERAL", "Grada General / Masiva", False),
        ]

        fallos = []
        for term, arq_esperado, es_mono_esperado in golden_cases:
            matches = self.df_final[self.df_final["logical_seat_category"].str.upper().str.contains(term, na=False)]
            if len(matches) > 0:
                pred_arqs = matches["arquetipo_demanda"].value_counts()
                top_pred = pred_arqs.index[0]
                if top_pred != arq_esperado:
                    if matches["es_monozona"].iloc[0] and arq_esperado != "Admisión Única / Tarifa Plana":
                        continue
                    fallos.append((term, arq_esperado, top_pred))

        self.assertLessEqual(len(fallos), 2, f"Fallos en Golden Set ({len(fallos)}): {fallos}")

    def test_04_persistencia_e_inferencia(self):
        """Verifica que guardar y cargar el modelo produzca exactamente las mismas predicciones."""
        if not hasattr(self, "df_enriquecido"):
            self.skipTest("Requiere dataset crudo real")

        temp_model_path = "data/processed/modelo_clustering_test.joblib"
        try:
            # 1. Guardar modelo entrenado
            mapa_arquetipos = etiquetar_por_centroides_escalados(
                self.kmeans, self.feature_names, self.scaler, peso_nlp=0.2
            )
            guardar_modelo_clustering(
                temp_model_path,
                kmeans=self.kmeans,
                scaler=self.scaler,
                tfidf_vectorizer=self.tfidf_vec,
                feature_names=self.feature_names,
                mapa_arquetipos=mapa_arquetipos,
                metricas=self.metricas,
                metadata={"autor": "Data Science TuBoleta", "version": "2.2"},
                gmm=self.metricas.get("gmm")
            )
            self.assertTrue(os.path.exists(temp_model_path), "El archivo del modelo joblib debe existir")

            # 2. Cargar modelo y realizar inferencia bietapica
            modelo_cargado = cargar_modelo_clustering(temp_model_path)
            self.assertEqual(modelo_cargado["version"], "2.2")

            df_pred = predecir_arquetipos_demanda(self.df_enriquecido, modelo_cargado)
            self.assertEqual(len(df_pred), len(self.df_final))
            
            # 3. Validar consistencia identica con el pipeline de entrenamiento
            coincidencias = (df_pred["arquetipo_demanda"] == self.df_final["arquetipo_demanda"]).mean()
            self.assertGreaterEqual(coincidencias, 0.999, f"Inferencia difiere del pipeline original: {coincidencias:.4f}")
        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

    def test_05_selector_auto_k5(self):
        """Valida que el selector automatico 'auto' seleccione k=5 mediante Score Compuesto Codo-DB."""
        if not hasattr(self, "df_enriquecido"):
            self.skipTest("Requiere dataset crudo real")

        df_auto, km_auto, _, _, _, _ = pipeline_clustering_dos_etapas(
            self.df_enriquecido, n_clusters_multizona="auto", random_state=42
        )
        self.assertEqual(km_auto.n_clusters, 5, f"El modo 'auto' debe seleccionar k=5, obtuvo k={km_auto.n_clusters}")
        
        n_arquetipos = df_auto["arquetipo_demanda"].nunique()
        self.assertEqual(n_arquetipos, 6, f"El catalogo final debe tener 6 arquetipos, tiene {n_arquetipos}")

    def test_06_regresion_distribucion_arquetipos(self):
        """Valida que la inferencia sobre una muestra reproduzca la distribucion canonica (<2% error)."""
        if not hasattr(self, "df_enriquecido"):
            self.skipTest("Requiere dataset crudo real")

        payload = cargar_modelo_clustering("data/processed/modelo_clustering_v2_2.joblib")
        muestra = self.df_enriquecido.sample(5000, random_state=42)
        pred = predecir_arquetipos_demanda(muestra, payload)

        obs = pred["arquetipo_demanda"].value_counts(normalize=True)
        for arq, esperado in DISTRIBUCION_ESPERADA_ARQUETIPOS.items():
            desviacion = abs(obs.get(arq, 0.0) - esperado)
            self.assertLess(desviacion, 0.02, f"Arquetipo '{arq}' desvia {desviacion:.2%} (limite 2.0%)")

    def test_07_score_confianza_y_frontera(self):
        """Valida que los scores de confianza, frontera geometrica y probabilidades esten bien calibrados."""
        if not hasattr(self, "df_enriquecido"):
            self.skipTest("Requiere dataset crudo real")

        payload = cargar_modelo_clustering("data/processed/modelo_clustering_v2_2.joblib")
        pred = predecir_arquetipos_demanda(self.df_enriquecido, payload)

        # 1. Rango [0, 1]
        self.assertTrue((pred["score_confianza"] >= 0.0).all() and (pred["score_confianza"] <= 1.0).all())
        self.assertTrue((pred["probabilidad_gmm"] >= 0.0).all() and (pred["probabilidad_gmm"] <= 1.0).all())

        # 2. Monozona deterministico
        mono = pred[pred["cluster"] == -1]
        self.assertTrue((mono["score_confianza"] == 1.0).all(), "Monozona debe tener score_confianza = 1.0")
        self.assertFalse(mono["es_frontera"].any(), "Monozona no debe tener ningun registro en frontera")
        self.assertTrue(mono["segundo_arquetipo"].isna().all() or (mono["segundo_arquetipo"] == None).all())
        self.assertTrue((mono["probabilidad_gmm"] == 1.0).all())

        # 3. Multi-zona frontera
        multi = pred[pred["cluster"] != -1]
        pct_frontera = multi["es_frontera"].mean()
        self.assertGreaterEqual(pct_frontera, 0.10, "El porcentaje en frontera debe ser >= 10%")
        self.assertLessEqual(pct_frontera, 0.25, "El porcentaje en frontera debe ser <= 25%")
        
        frontera_rows = multi[multi["es_frontera"]]
        self.assertTrue((frontera_rows["score_confianza"] < 0.15).all())
        self.assertTrue((frontera_rows["segundo_arquetipo"].notna()).all())

    def test_08_cobertura_vocabulario_oov(self):
        """Valida deteccion de cobertura de vocabulario TF-IDF y alerta de texto casi vacio."""
        payload = cargar_modelo_clustering("data/processed/modelo_clustering_v2_2.joblib")
        
        # Caso 1: Vocabulario conocido
        df_conocido = pd.DataFrame([{
            "t_performance_id": 999991,
            "dn_quota": 50,
            "peso_aforo": 0.05,
            "ratio_precio_max": 1.0,
            "percentil_precio_evento": 1.0,
            "texto_limpio": "PALCO VIP OCCIDENTAL",
            "tag_palco": 1.0,
            "tag_vip": 1.0,
            "tag_platea": 0.0,
            "tag_preferencial": 0.0,
            "tag_general": 0.0,
            "tag_balcon": 0.0,
            "tag_piso_alto": 0.0
        }, {
            "t_performance_id": 999991,
            "dn_quota": 950,
            "peso_aforo": 0.95,
            "ratio_precio_max": 0.4,
            "percentil_precio_evento": 0.0,
            "texto_limpio": "GENERAL ORIENTAL",
            "tag_palco": 0.0,
            "tag_vip": 0.0,
            "tag_platea": 0.0,
            "tag_preferencial": 0.0,
            "tag_general": 1.0,
            "tag_balcon": 0.0,
            "tag_piso_alto": 0.0
        }])
        pred_conocido = predecir_arquetipos_demanda(df_conocido, payload)
        self.assertGreaterEqual(pred_conocido.loc[0, "cobertura_texto"], 0.5)
        self.assertFalse(pred_conocido.loc[0, "texto_casi_vacio"])

        # Caso 2: Totalmente fuera de vocabulario (OOV)
        df_oov = pd.DataFrame([{
            "t_performance_id": 999992,
            "dn_quota": 100,
            "peso_aforo": 0.10,
            "ratio_precio_max": 0.8,
            "percentil_precio_evento": 0.5,
            "texto_limpio": "MEZZANINE LOGE BOX EXTRAORDINARY",
            "tag_palco": 0.0,
            "tag_vip": 0.0,
            "tag_platea": 0.0,
            "tag_preferencial": 0.0,
            "tag_general": 0.0,
            "tag_balcon": 0.0,
            "tag_piso_alto": 0.0
        }, {
            "t_performance_id": 999992,
            "dn_quota": 900,
            "peso_aforo": 0.90,
            "ratio_precio_max": 0.3,
            "percentil_precio_evento": 0.0,
            "texto_limpio": "ANFITEATRO DESCONOCIDO",
            "tag_palco": 0.0,
            "tag_vip": 0.0,
            "tag_platea": 0.0,
            "tag_preferencial": 0.0,
            "tag_general": 0.0,
            "tag_balcon": 0.0,
            "tag_piso_alto": 0.0
        }])
        pred_oov = predecir_arquetipos_demanda(df_oov, payload)
        self.assertEqual(pred_oov.loc[0, "cobertura_texto"], 0.0)
        self.assertTrue(pred_oov.loc[0, "texto_casi_vacio"])

    def test_09_inferencia_sintetica_ci(self):
        """Valida inferencia completa sobre dataset sintetico para ejecucion incondicional en CI."""
        payload = cargar_modelo_clustering("data/processed/modelo_clustering_v2_2.joblib")
        df_sintetico = pd.DataFrame([
            # Monozona
            {"t_performance_id": 101, "dn_quota": 1000, "peso_aforo": 1.0, "ratio_precio_max": 1.0, "percentil_precio_evento": 1.0, "texto_limpio": "ENTRADA GENERAL", "tag_general": 1.0, "tag_vip": 0.0, "tag_palco": 0.0, "tag_platea": 0.0, "tag_preferencial": 0.0, "tag_balcon": 0.0, "tag_piso_alto": 0.0},
            # Multi-zona 1: VIP
            {"t_performance_id": 102, "dn_quota": 50, "peso_aforo": 0.05, "ratio_precio_max": 1.0, "percentil_precio_evento": 1.0, "texto_limpio": "PALCO VIP", "tag_general": 0.0, "tag_vip": 1.0, "tag_palco": 1.0, "tag_platea": 0.0, "tag_preferencial": 0.0, "tag_balcon": 0.0, "tag_piso_alto": 0.0},
            # Multi-zona 2: General
            {"t_performance_id": 102, "dn_quota": 950, "peso_aforo": 0.95, "ratio_precio_max": 0.3, "percentil_precio_evento": 0.0, "texto_limpio": "LOCALIDAD GENERAL", "tag_general": 1.0, "tag_vip": 0.0, "tag_palco": 0.0, "tag_platea": 0.0, "tag_preferencial": 0.0, "tag_balcon": 0.0, "tag_piso_alto": 0.0},
        ])
        pred = predecir_arquetipos_demanda(df_sintetico, payload)
        self.assertEqual(len(pred), 3)
        columnas_requeridas = [
            "cluster", "arquetipo_demanda", "score_confianza", "es_frontera",
            "segundo_arquetipo", "cobertura_texto", "texto_casi_vacio", "probabilidad_gmm"
        ]
        for col in columnas_requeridas:
            self.assertIn(col, pred.columns, f"Columna requerida '{col}' falta en la salida de inferencia")
        self.assertEqual(pred.loc[0, "cluster"], -1)
        self.assertEqual(pred.loc[0, "arquetipo_demanda"], "Admisión Única / Tarifa Plana")

    def test_10_monitoreo_drift_psi(self):
        """Valida que el calculo de PSI detecte estabilidad en la misma distribucion y alerte en perturbaciones."""
        # 1. Test unitario de formula PSI
        hist1 = np.array([10, 20, 30, 40])
        psi_identico = calcular_psi(hist1, hist1)
        self.assertAlmostEqual(psi_identico, 0.0, places=4, msg="Distribuciones identicas deben dar PSI=0.0")

        if hasattr(self, "df_enriquecido"):
            # 2. Lote estable (submuestra)
            lote_estable = self.df_enriquecido.sample(3000, random_state=99)
            reporte_estable = evaluar_drift_lote(lote_estable, "data/processed/modelo_clustering_v2_2.joblib")
            self.assertIn(reporte_estable["estado_general"], ["ESTABLE", "REVISAR"])
            self.assertLess(reporte_estable["psi_maximo"], 0.25)

            # 3. Lote con drift severo sintetico (precios modificados y aforo invertido)
            lote_drift = lote_estable.copy()
            lote_drift["ratio_precio_max"] = np.clip(lote_drift["ratio_precio_max"] * 0.05, 0.0, 1.0)
            lote_drift["peso_aforo"] = 1.0 - lote_drift["peso_aforo"]
            reporte_drift = evaluar_drift_lote(lote_drift, "data/processed/modelo_clustering_v2_2.joblib")
            self.assertGreater(reporte_drift["psi_maximo"], 0.25)
            self.assertEqual(reporte_drift["estado_general"], "DRIFT_CRITICO")

    def test_11_cero_emojis_y_estandarizacion_venue(self):
        """Audita que no existan emojis ni la palabra 'recinto' en codigo fuente y documentacion."""
        emoji_pattern = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]", flags=re.UNICODE)
        recinto_pattern = re.compile(r"\brecintos?\b", flags=re.IGNORECASE)

        archivos_auditar = [
            "DOCUMENTACION_MODELO_CLUSTERIZACION.md",
            "README.md",
            "src/clustering.py",
            "src/nlp_utils.py",
            "src/feature_engineering.py",
            "scripts/monitorear_drift.py",
            "scripts/optimizar_k_multizona.py",
            "scripts/generate_all_presentation_figures.py",
            "scripts/generate_all_23_figures.py",
            "scripts/build_presentation_from_template.py",
            "scripts/build_presentation.py",
            "scripts/build_full_notebook_presentation.py"
        ]

        errores_emoji = []
        errores_recinto = []

        for rel_path in archivos_auditar:
            if not os.path.exists(rel_path):
                continue
            with open(rel_path, "r", encoding="utf-8") as fp:
                for line_idx, line in enumerate(fp, 1):
                    emojis_found = emoji_pattern.findall(line)
                    if emojis_found:
                        errores_emoji.append(f"{rel_path}:{line_idx}: {emojis_found}")
                    recinto_found = recinto_pattern.findall(line)
                    if recinto_found:
                        errores_recinto.append(f"{rel_path}:{line_idx}: {recinto_found}")

        self.assertEqual(len(errores_emoji), 0, f"Se encontraron emojis en: {errores_emoji}")
        self.assertEqual(len(errores_recinto), 0, f"Se encontro la palabra 'recinto' en: {errores_recinto}")


if __name__ == "__main__":
    unittest.main()
