"""
Pruebas unitarias para la integracion de type_site y percentil de precio por tipo (v2.3).

Cubre:
(a) Merge de type_site con fallback y normalizacion de nombres de venue.
(b) Reproduccion identica del percentil en inferencia contra referencia persistida (round-trip con payload).
(c) Cobertura de casos: monozona, multi-zona, venue desconocido y type_site con < 50 localidades (cold start).
"""

import os
import unittest
import numpy as np
import pandas as pd
from src.nlp_utils import normalizar_venue
from src.feature_engineering import (
    enriquecer_type_site,
    calcular_percentil_precio_absoluto_dentro_tipo,
    generar_referencia_percentil_tipo
)
from src.clustering import (
    construir_espacio_vectorial_mixto,
    pipeline_clustering_dos_etapas,
    guardar_modelo_clustering,
    cargar_modelo_clustering,
    predecir_arquetipos_demanda,
    MODEL_VERSION,
    CANONICAL_TYPE_SITE_CATEGORIES
)


class TestFeatureTypeSite(unittest.TestCase):

    def test_a_merge_type_site_con_normalizacion_y_fallback(self):
        """Valida normalizacion canonica de venue, merge de lookup y fallback a 'desconocido'."""
        df_test = pd.DataFrame({
            "site": [
                "  TEATRO  MAYOR  JULIO  MARIO  SANTO DOMINGO  ",
                "Teatro Jorge Eliécer Gaitán",
                "   MOVISTAR    ARENA   ",
                "RECINTO_COMPLETAMENTE_INVENTADO_XYZ_999"
            ],
            "logical_seat_category": ["Platea", "Balcon", "VIP", "General"]
        })

        df_enr = enriquecer_type_site(df_test, site_column="site")
        self.assertIn("type_site", df_enr.columns)
        self.assertIn("flag_site_desconocido", df_enr.columns)

        # Venues conocidos normalizados deben mapear a sus tipos correctos
        self.assertEqual(df_enr.loc[0, "type_site"], "teatro")
        self.assertEqual(df_enr.loc[0, "flag_site_desconocido"], 0)

        self.assertEqual(df_enr.loc[1, "type_site"], "teatro")
        self.assertEqual(df_enr.loc[1, "flag_site_desconocido"], 0)

        self.assertEqual(df_enr.loc[2, "type_site"], "arena_cubierta")
        self.assertEqual(df_enr.loc[2, "flag_site_desconocido"], 0)

        # Venue desconocido debe recibir fallback y activar bandera
        self.assertEqual(df_enr.loc[3, "type_site"], "desconocido")
        self.assertEqual(df_enr.loc[3, "flag_site_desconocido"], 1)

    def test_b_percentil_reproduce_identico_en_inferencia_roundtrip(self):
        """Valida que el percentil por tipo se reproduce de forma identica en inferencia contra el payload."""
        # 1. Crear un dataset de entrenamiento sintetico con suficientes localidades para superar cold start
        rng = np.random.RandomState(42)
        n_locs = 80
        sites = [f"TEATRO_MUESTRA_{i}" for i in range(n_locs)]
        prices = np.linspace(20000, 250000, n_locs)

        df_train = pd.DataFrame({
            "t_performance_id": [1000 + i for i in range(n_locs)],
            "site": sites,
            "logical_seat_category": [f"LOC_{i}" for i in range(n_locs)],
            "type_site": ["teatro"] * n_locs,
            "med_unit_amt_itx": prices,
            "dn_quota": [500] * n_locs,
            "performance_quota": [500] * n_locs,
            "peso_aforo": [1.0] * n_locs,
            "ratio_precio_max": [1.0] * n_locs,
            "percentil_precio_evento": [1.0] * n_locs,
            "texto_limpio": ["LOCALIDAD GENERAL"] * n_locs,
            "tag_general": [1.0] * n_locs,
            "tag_vip": [0.0] * n_locs,
            "tag_palco": [0.0] * n_locs,
            "tag_platea": [0.0] * n_locs,
            "tag_preferencial": [0.0] * n_locs,
            "tag_balcon": [0.0] * n_locs,
            "tag_piso_alto": [0.0] * n_locs
        })

        # Calcular percentil en modo entrenamiento
        df_train_calc = calcular_percentil_precio_absoluto_dentro_tipo(df_train)
        ref_distribucion = generar_referencia_percentil_tipo(df_train)

        self.assertIn("teatro", ref_distribucion)
        self.assertFalse(ref_distribucion["teatro"]["es_cold_start"])
        self.assertEqual(len(ref_distribucion["teatro"]["precios_referencia"]), n_locs)

        # 2. Inferencia evaluada con la referencia guardada sobre una submuestra
        df_inferencia = df_train.sample(20, random_state=7).copy()
        # Eliminar columna calculada para obligar a usar la referencia
        df_inferencia = df_inferencia.drop(columns=["percentil_precio_absoluto_dentro_tipo", "flag_cold_start_tipo"], errors="ignore")

        df_pred = calcular_percentil_precio_absoluto_dentro_tipo(
            df_inferencia, referencia_distribucion=ref_distribucion
        )

        # 3. Comprobar reproduccion exacta
        for idx in df_inferencia.index:
            pct_esperado = df_train_calc.loc[idx, "percentil_precio_absoluto_dentro_tipo"]
            pct_obtenido = df_pred.loc[idx, "percentil_precio_absoluto_dentro_tipo"]
            self.assertAlmostEqual(
                pct_esperado,
                pct_obtenido,
                places=4,
                msg=f"Discrepancia en percentil: {pct_esperado} != {pct_obtenido} para indice {idx}"
            )

    def test_c_cobertura_monozona_multizona_site_desconocido_cold_start(self):
        """Valida inferencia con monozona, multi-zona, site desconocido y type_site con < 50 localidades."""
        # Configurar referencia de distribucion con un tipo normal y un tipo cold start
        ref_distribucion = {
            "teatro": {
                "conteo": 100,
                "es_cold_start": False,
                "bin_edges": list(np.linspace(10000, 100000, 101)),
                "precios_referencia": list(np.linspace(10000, 100000, 100))
            },
            "cine_sala_cultural": {
                "conteo": 15,
                "es_cold_start": True,
                "bin_edges": [],
                "precios_referencia": []
            }
        }

        # Dataset que contiene los cuatro escenarios requeridos
        df_casos = pd.DataFrame([
            # 1. Monozona: 1 sola localidad con 100% de aforo
            {
                "t_performance_id": 501,
                "site": "TEATRO MAYOR JULIO MARIO SANTO DOMINGO",
                "logical_seat_category": "ENTRADA GENERAL MONOZONA",
                "med_unit_amt_itx": 50000,
                "dn_quota": 1000,
                "performance_quota": 1000,
                "peso_aforo": 1.0,
                "ratio_precio_max": 1.0,
                "percentil_precio_evento": 1.0,
                "texto_limpio": "GENERAL",
                "type_site": "teatro"
            },
            # 2. Multi-zona VIP
            {
                "t_performance_id": 502,
                "site": "TEATRO MAYOR JULIO MARIO SANTO DOMINGO",
                "logical_seat_category": "PALCO VIP CENTRAL",
                "med_unit_amt_itx": 95000,
                "dn_quota": 50,
                "performance_quota": 1000,
                "peso_aforo": 0.05,
                "ratio_precio_max": 1.0,
                "percentil_precio_evento": 1.0,
                "texto_limpio": "PALCO VIP",
                "type_site": "teatro"
            },
            # 3. Multi-zona General
            {
                "t_performance_id": 502,
                "site": "TEATRO MAYOR JULIO MARIO SANTO DOMINGO",
                "logical_seat_category": "BALCON GENERAL",
                "med_unit_amt_itx": 20000,
                "dn_quota": 950,
                "performance_quota": 1000,
                "peso_aforo": 0.95,
                "ratio_precio_max": 0.21,
                "percentil_precio_evento": 0.0,
                "texto_limpio": "BALCON",
                "type_site": "teatro"
            },
            # 4. Venue desconocido
            {
                "t_performance_id": 503,
                "site": "SITIO_TOTALMENTE_NUEVO_SIN_LOOKUP",
                "logical_seat_category": "GENERAL ENTRADA",
                "med_unit_amt_itx": 40000,
                "dn_quota": 1000,
                "performance_quota": 1000,
                "peso_aforo": 1.0,
                "ratio_precio_max": 1.0,
                "percentil_precio_evento": 1.0,
                "texto_limpio": "GENERAL",
                "type_site": "desconocido"
            },
            # 5. type_site con < 50 localidades (cold start cine_sala_cultural)
            {
                "t_performance_id": 504,
                "site": "CINEMATECA SALA 1",
                "logical_seat_category": "GENERAL CINEMATECA",
                "med_unit_amt_itx": 15000,
                "dn_quota": 100,
                "performance_quota": 100,
                "peso_aforo": 1.0,
                "ratio_precio_max": 1.0,
                "percentil_precio_evento": 1.0,
                "texto_limpio": "GENERAL",
                "type_site": "cine_sala_cultural"
            }
        ])

        # Enriquecer y calcular percentil
        df_eval = calcular_percentil_precio_absoluto_dentro_tipo(
            df_casos, referencia_distribucion=ref_distribucion
        )

        # Comprobar caso 4 (site desconocido): fallback a 0.50 y flag_cold_start_tipo=1
        self.assertEqual(df_eval.loc[3, "type_site"], "desconocido")
        self.assertEqual(df_eval.loc[3, "percentil_precio_absoluto_dentro_tipo"], 0.50)
        self.assertEqual(df_eval.loc[3, "flag_cold_start_tipo"], 1)

        # Comprobar caso 5 (cold start < 50 localidades): percentil 0.50 y flag_cold_start_tipo=1
        self.assertEqual(df_eval.loc[4, "type_site"], "cine_sala_cultural")
        self.assertEqual(df_eval.loc[4, "percentil_precio_absoluto_dentro_tipo"], 0.50)
        self.assertEqual(df_eval.loc[4, "flag_cold_start_tipo"], 1)

        # Comprobar caso 2 (teatro con > 50 localidades): percentil dinamico > 0.80 y flag 0
        self.assertGreater(df_eval.loc[1, "percentil_precio_absoluto_dentro_tipo"], 0.80)
        self.assertEqual(df_eval.loc[1, "flag_cold_start_tipo"], 0)

        # Verificar espacio vectorial estructural sin TF-IDF: 4 continuas + 7 tags + 9 type_site = 20 dimensiones
        X_base, _, _, feat_names_base = construir_espacio_vectorial_mixto(
            df_eval,
            peso_type_site=0.5,
            usar_tfidf_texto=False,
            categorias_type_site=CANONICAL_TYPE_SITE_CATEGORIES
        )
        self.assertEqual(X_base.shape[0], len(df_eval))
        self.assertEqual(X_base.shape[1], 20)

        # La categoria 'desconocido' no debe figurar en feature_names
        self.assertNotIn("type_site_desconocido", feat_names_base)
        for cat in CANONICAL_TYPE_SITE_CATEGORIES:
            self.assertIn(f"type_site_{cat}", feat_names_base)

        # Con vectorizador de 15 features: 20 + 15 = 35 dimensiones
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_mock = TfidfVectorizer(max_features=15)
        # Ajustar con corpus sintetico de 15 tokens
        corpus_15 = [" ".join([f"token_{i}" for i in range(15)])]
        tfidf_mock.fit(corpus_15)

        X_35, _, _, feat_names_35 = construir_espacio_vectorial_mixto(
            df_eval,
            peso_type_site=0.5,
            tfidf_vectorizer=tfidf_mock,
            categorias_type_site=CANONICAL_TYPE_SITE_CATEGORIES
        )
        self.assertEqual(X_35.shape[1], 35)



if __name__ == "__main__":
    unittest.main()
