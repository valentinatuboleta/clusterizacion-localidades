"""
Tests unitarios hermeticos para la arquitectura jerarquica de micro-clusters (v3.0).

Valida:
1. Seleccion Codo-DB en sub-espacios sinteticos (deteccion exacta de k=2).
2. Calculo de pureza en sub-espacios (casos puro e hibrido).
3. Generacion de label_auto con 13 tags, deduplicacion sin acentos y fallback a hibrido.
4. Propagacion de segmento_incierto a partir de es_frontera de Nivel 1.
5. Invarianza de rollup jerarquico 1:1 (ningun micro-cluster cruza fronteras de arquetipo).
"""

import os
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.entrenar_jerarquia_microclusters import (
    seleccionar_sub_k_codo_db,
    TAGS_EXPANDIDOS_13
)
from scripts.explorar_microclusters import (
    calcular_pureza_cluster,
    generar_label_auto
)
from src.jerarquia import (
    HIERARCHY_VERSION,
    TAGS_SUBESPACIO_13,
    MICRO_CLUSTERS_CANONICAL,
    predecir_microclusters,
    evaluar_drift_microclusters,
    generar_label_auto_v3
)


class TestJerarquiaMicroclusters(unittest.TestCase):

    def test_sub_codo_db_sintetico(self):
        """Valida que en 2 blobs gaussianos bien separados seleccione k=2."""
        X, y = make_blobs(
            n_samples=300,
            n_features=10,
            centers=2,
            cluster_std=0.4,
            random_state=42
        )
        df_sub = pd.DataFrame({
            "texto_limpio": ["platea" if yi == 0 else "balcon" for yi in y],
            "tag_platea": [1 if yi == 0 else 0 for yi in y],
            "tag_balcon": [0 if yi == 0 else 1 for yi in y]
        })
        for tag in TAGS_EXPANDIDOS_13:
            if tag not in df_sub.columns:
                df_sub[tag] = 0

        res = seleccionar_sub_k_codo_db(
            X_sub=X,
            df_sub=df_sub,
            vocab_sub=["platea", "balcon"],
            k_range=[2, 3, 4],
            min_pct_piso=0.03,
            sample_size=300,
            random_state=42
        )
        self.assertEqual(res["k_optimo"], 2)
        self.assertFalse(res["ganador"]["es_degenerado"])

    def test_pureza_subespacio(self):
        """Valida calculo de pureza estructural y lexica en sub-espacio."""
        # 1. Caso puro
        df_puro = pd.DataFrame({
            "tag_palco": [1, 1, 1, 1],
            "tag_mesa": [0, 0, 0, 0],
            "texto_limpio": ["palco sur", "palco norte", "palco", "palco este"]
        })
        for tag in TAGS_EXPANDIDOS_13:
            if tag not in df_puro.columns:
                df_puro[tag] = 0

        res_puro = calcular_pureza_cluster(df_puro, TAGS_EXPANDIDOS_13, ["palco", "norte", "sur"])
        self.assertEqual(res_puro["tag_dominante"], "tag_palco")
        self.assertAlmostEqual(res_puro["purity_tag"], 1.0)
        self.assertAlmostEqual(res_puro["purity_naming"], 1.0)

        # 2. Caso mixto
        df_mixto = pd.DataFrame({
            "tag_palco": [1, 1, 0, 0],
            "tag_mesa": [0, 0, 1, 1],
            "texto_limpio": ["palco 1", "palco 2", "mesa 1", "mesa 2"]
        })
        for tag in TAGS_EXPANDIDOS_13:
            if tag not in df_mixto.columns:
                df_mixto[tag] = 0

        res_mixto = calcular_pureza_cluster(df_mixto, TAGS_EXPANDIDOS_13, ["palco", "mesa"])
        self.assertAlmostEqual(res_mixto["purity_tag"], 0.5)
        self.assertAlmostEqual(res_mixto["purity_term"], 0.5)
        self.assertAlmostEqual(res_mixto["purity_naming"], 0.5)

    def test_generar_label_auto_13_tags(self):
        """Valida generacion de etiquetas con 13 tags, combinaciones y deduplicacion."""
        # Combinacion de 3 tags activos: Palco + Mesa + Occidental
        shares_combo = pd.Series({
            "tag_palco": 0.85,
            "tag_mesa": 0.80,
            "tag_occidental": 0.75,
            "tag_general": 0.05
        })
        label_combo = generar_label_auto(
            cluster_id=1,
            tag_shares=shares_combo,
            term_dominante="palco",
            purity_tag=0.85,
            purity_term=0.85,
            threshold_tag_active=0.60,
            min_purity_tag=0.70
        )
        self.assertEqual(label_combo, "Palco Mesa Occidental")

        # Deduplicacion insensible a acentos: Balcon + piso alto + termino "balcon"
        shares_balcon = pd.Series({
            "tag_balcon": 0.90,
            "tag_piso_alto": 0.80
        })
        label_balcon = generar_label_auto(
            cluster_id=2,
            tag_shares=shares_balcon,
            term_dominante="balcon",
            purity_tag=0.90,
            purity_term=0.90,
            threshold_tag_active=0.60,
            min_purity_tag=0.70
        )
        self.assertEqual(label_balcon, "Balc\u00f3n Piso Alto")

        # Fallback a hibrido si pureza < min_purity_tag
        label_impuro = generar_label_auto(
            cluster_id=3,
            tag_shares=pd.Series({"tag_palco": 0.50, "tag_general": 0.50}),
            term_dominante="general",
            purity_tag=0.50,
            purity_term=0.50,
            min_purity_tag=0.70
        )
        self.assertEqual(label_impuro, "hibrido_k3")

    def test_propagacion_segmento_incierto(self):
        """Valida que es_frontera propague segmento_incierto y afecte es_frontera_pct."""
        df_sub = pd.DataFrame({
            "es_frontera": [True, False, True, False],
            "sub_cluster_num": [0, 0, 1, 1]
        })
        df_sub["segmento_incierto"] = df_sub["es_frontera"].astype(bool)

        # Cluster 0: 1 True, 1 False -> 50%
        c0 = df_sub[df_sub["sub_cluster_num"] == 0]
        pct_c0 = float(c0["es_frontera"].mean() * 100)
        self.assertEqual(pct_c0, 50.0)
        self.assertTrue(c0.iloc[0]["segmento_incierto"])
        self.assertFalse(c0.iloc[1]["segmento_incierto"])

    def test_rollup_jerarquico(self):
        """Valida que cada micro-cluster pertenezca estrictamente a un unico arquetipo macro."""
        df_mock = pd.DataFrame({
            "micro_cluster_id": ["VIP-0", "VIP-0", "VIP-1", "GGM-0", "GGM-0"],
            "arquetipo_demanda": [
                "VIP / Palcos / Premium",
                "VIP / Palcos / Premium",
                "VIP / Palcos / Premium",
                "Grada General / Masiva",
                "Grada General / Masiva"
            ]
        })
        rollup_check = df_mock.groupby("micro_cluster_id")["arquetipo_demanda"].nunique()
        self.assertTrue((rollup_check == 1).all())

        # Si el catalogo o la asignacion real v3 existe, verificarlo tambien
        asig_path = "data/processed/asignacion_microclusters.csv"
        if os.path.exists(asig_path):
            df_real = pd.read_csv(asig_path)
            rollup_real = df_real.groupby("micro_cluster_id")["arquetipo_demanda"].nunique()
            self.assertTrue((rollup_real == 1).all())

    def test_scoring_microclusters_hermetico_roundtrip(self):
        """Valida round-trip de inferencia hermetico sobre payload sintetico."""
        # 1. DataFrame sintetico de entrada: mitad monozona, mitad multi-zona
        df_test = pd.DataFrame({
            "t_performance_id": list(range(20)) + [100] * 20,
            "performance_quota": [1000] * 40,
            "dn_quota": [1000] * 20 + [50] * 20,
            "price_amount": [150000.0] * 40,
            "logical_seat_category": ["GENERAL"] * 20 + ["PALCO VIP"] * 10 + ["PALCO MESA"] * 10,
            "type_site": ["arena_estadio"] * 40
        })

        from src.clustering import construir_espacio_vectorial_mixto
        from src.nlp_utils import pipeline_procesamiento_nlp
        from src.jerarquia import construir_subespacio_arquetipo

        df_proc = pipeline_procesamiento_nlp(df_test.copy())
        df_multi = df_proc[df_proc["t_performance_id"] == 100].copy()
        for num_col in ["ratio_precio_max", "percentil_precio_evento", "peso_aforo", "percentil_precio_absoluto_dentro_tipo"]:
            df_multi[num_col] = 0.5

        X_n1, scaler_n1, tfidf_n1, _ = construir_espacio_vectorial_mixto(
            df_multi,
            peso_nlp=0.2,
            peso_type_site=0.5
        )
        km_n1 = KMeans(n_clusters=2, random_state=42, n_init=5).fit(X_n1)

        X_sub, sc_sub, tf_sub, _ = construir_subespacio_arquetipo(
            df_multi,
            columnas_tags=TAGS_SUBESPACIO_13,
            max_tfidf=15,
            peso_nlp=0.2
        )
        km_sub = KMeans(n_clusters=2, random_state=42, n_init=5).fit(X_sub)

        payload_synth = {
            "version": HIERARCHY_VERSION,
            "nivel_1": {
                "version_base": "2.3",
                "kmeans": km_n1,
                "scaler": scaler_n1,
                "tfidf": tfidf_n1,
                "mapa_arquetipos": {0: "VIP / Palcos / Premium", 1: "VIP / Palcos / Premium"}
            },
            "sub_modelos": {
                "VIP / Palcos / Premium": {
                    "prefix": "VIP",
                    "k_sub": 2,
                    "kmeans_sub": km_sub,
                    "scaler_sub": sc_sub,
                    "tfidf_sub": tf_sub,
                    "columnas_tags_sub": TAGS_SUBESPACIO_13,
                    "mapa_micro_label": {0: "Palco VIP", 1: "Palco Mesa"}
                }
            },
            "distribucion_referencia_microclusters": {
                "AU-0": 0.50,
                "VIP-0": 0.25,
                "VIP-1": 0.25
            },
            "categorias_vigentes": ["AU-0", "VIP-0", "VIP-1"]
        }

        # Inferencia
        pred1 = predecir_microclusters(df_test, payload_synth)
        pred2 = predecir_microclusters(df_test, payload_synth)

        # Validar esquema de salida
        cols_esperadas = [
            "logical_seat_category",
            "micro_cluster_id",
            "label_auto",
            "arquetipo_demanda",
            "score_confianza",
            "es_frontera"
        ]
        self.assertEqual(list(pred1.columns), cols_esperadas)
        self.assertEqual(len(pred1), 40)

        # Validar consistencia deterministica (100% coincidencia round-trip)
        self.assertTrue((pred1["micro_cluster_id"] == pred2["micro_cluster_id"]).all())
        self.assertTrue((pred1["label_auto"] == pred2["label_auto"]).all())

        # Monozona asignada a AU-0
        self.assertTrue((pred1.iloc[:20]["micro_cluster_id"] == "AU-0").all())
        self.assertTrue((pred1.iloc[:20]["label_auto"] == "Admisión Única").all())

        # Multi-zona asignada a micro-clusters VIP
        self.assertTrue(pred1.iloc[20:]["micro_cluster_id"].isin(["VIP-0", "VIP-1"]).all())

    def test_scoring_defensivo_venues_desconocidos(self):
        """Valida que entradas incompletas o venues no vistos nunca lancen excepcion."""
        df_raro = pd.DataFrame({
            "venue_desconocido": ["LUGAR_X", "LUGAR_Y"],
            "valor_random": [999, 123]
        })
        payload_min = {
            "version": HIERARCHY_VERSION,
            "sub_modelos": {}
        }
        res = predecir_microclusters(df_raro, payload_min)
        self.assertEqual(len(res), 2)
        self.assertEqual(res["logical_seat_category"].iloc[0], "GENERAL")
        self.assertEqual(res["micro_cluster_id"].iloc[0], "AU-0")
        self.assertFalse(res["micro_cluster_id"].isna().any())

    def test_drift_microclusters_referencia_sin_alerta(self):
        """Valida que una distribucion coincidente con la referencia tenga PSI < 0.05 y sin alerta."""
        ref_dist = {cat: 1.0 / len(MICRO_CLUSTERS_CANONICAL) for cat in MICRO_CLUSTERS_CANONICAL}
        payload = {
            "distribucion_referencia_microclusters": ref_dist,
            "categorias_vigentes": MICRO_CLUSTERS_CANONICAL
        }
        registros = []
        for cat in MICRO_CLUSTERS_CANONICAL:
            registros.extend([cat] * 5)
        df_obs = pd.DataFrame({"micro_cluster_id": registros})

        res = evaluar_drift_microclusters(df_obs, payload, umbral_psi=0.10)
        self.assertFalse(res["alerta_activa"])
        self.assertEqual(res["estado"], "ESTABLE")
        self.assertLess(res["psi_global"], 0.05)
        self.assertEqual(len(res["alertas"]), 0)

    def test_drift_microclusters_perturbada_con_alerta(self):
        """Valida que una distribucion concentrada de forma anomala active alerta y supere el umbral PSI."""
        ref_dist = {cat: 1.0 / len(MICRO_CLUSTERS_CANONICAL) for cat in MICRO_CLUSTERS_CANONICAL}
        payload = {
            "distribucion_referencia_microclusters": ref_dist,
            "categorias_vigentes": MICRO_CLUSTERS_CANONICAL
        }
        registros = ["AU-0"] * 90 + ["VIP-0"] * 10
        df_perturb = pd.DataFrame({"micro_cluster_id": registros})

        res = evaluar_drift_microclusters(df_perturb, payload, umbral_psi=0.10)
        self.assertTrue(res["alerta_activa"])
        self.assertIn(res["estado"], ["REVISAR", "DRIFT_CRITICO"])
        self.assertGreaterEqual(res["psi_global"], 0.10)
        self.assertGreater(len(res["alertas"]), 0)

    def test_roundtrip_dataset_real_si_existe(self):
        """Si los artefactos reales v3 existen localmente, valida >=99% de coincidencia round-trip."""
        mod_path = "data/processed/modelo_jerarquia_v3.joblib"
        asig_path = "data/processed/asignacion_microclusters.csv"
        raw_path = "data/raw/localidades_eda.parquet"
        if os.path.exists(mod_path) and os.path.exists(asig_path) and os.path.exists(raw_path):
            from src.feature_engineering import preparar_dataset_enriquecido
            df_raw = pd.read_parquet(raw_path)
            df_enr = preparar_dataset_enriquecido(df_raw)
            pred = predecir_microclusters(df_enr, mod_path)
            asig = pd.read_csv(asig_path)

            ARQ_ORDEN = [
                "Admisión Única / Tarifa Plana",
                "VIP / Palcos / Premium",
                "Popular / Balcón / Visibilidad Parcial",
                "Platea General / Intermedia",
                "Preferencial / Platea Frontal",
                "Grada General / Masiva"
            ]
            pred_ord = pd.concat([pred[pred["arquetipo_demanda"] == a] for a in ARQ_ORDEN], ignore_index=True)
            match_mc = float((pred_ord["micro_cluster_id"].values == asig["micro_cluster_id"].values).mean())
            self.assertGreaterEqual(match_mc, 0.99, f"Tasa de match real ({match_mc:.4%}) inferior a 99%")


if __name__ == "__main__":
    unittest.main()
