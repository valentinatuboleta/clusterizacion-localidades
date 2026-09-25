"""
Tests unitarios herméticos para el módulo de exploración de micro-clusters.

Valida:
1. Función de pureza estructural y léxica (casos puro y mixto).
2. Función de estabilidad bootstrap-ARI sobre datos sintéticos separables.
3. Generación determinística de label_auto (Title Case y fallback a hibrido_k{n}).
4. Generación y esquema del catálogo consolidado de micro-clusters.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from scripts.explorar_microclusters import (
    calcular_pureza_cluster,
    generar_label_auto,
    calcular_estabilidad_bootstrap_ari,
    generar_catalogo_clusters
)


class TestMicroclustersExploracion(unittest.TestCase):

    def test_pureza_naming_caso_puro(self):
        """Valida que un cluster homogéneo tenga pureza 1.0 en tags y términos."""
        df_puro = pd.DataFrame({
            "tag_palco": [1, 1, 1, 1, 1],
            "tag_vip": [0, 0, 0, 0, 0],
            "texto_limpio": [
                "palco lateral 1",
                "palco lateral 2",
                "palco lateral",
                "palco norte",
                "palco sur"
            ]
        })
        res = calcular_pureza_cluster(
            df_cluster=df_puro,
            tag_cols=["tag_palco", "tag_vip"],
            tfidf_vocab=["palco", "lateral", "vip"]
        )
        self.assertEqual(res["tag_dominante"], "tag_palco")
        self.assertAlmostEqual(res["purity_tag"], 1.0)
        self.assertEqual(res["term_dominante"], "palco")
        self.assertAlmostEqual(res["purity_term"], 1.0)
        self.assertAlmostEqual(res["purity_naming"], 1.0)

    def test_pureza_naming_caso_mixto(self):
        """Valida que un cluster heterogéneo refleje purezas reducidas."""
        df_mixto = pd.DataFrame({
            "tag_palco": [1, 1, 0, 0],
            "tag_vip": [0, 0, 1, 1],
            "texto_limpio": [
                "palco 1",
                "palco 2",
                "vip 1",
                "vip 2"
            ]
        })
        res = calcular_pureza_cluster(
            df_cluster=df_mixto,
            tag_cols=["tag_palco", "tag_vip"],
            tfidf_vocab=["palco", "vip"]
        )
        self.assertAlmostEqual(res["purity_tag"], 0.5)
        self.assertAlmostEqual(res["purity_term"], 0.5)
        self.assertAlmostEqual(res["purity_naming"], 0.5)

    def test_estabilidad_bootstrap_ari_sintetico(self):
        """Valida que clusters sintéticos bien separados tengan estabilidad ARI > 0.90."""
        X, y_true = make_blobs(
            n_samples=600,
            n_features=5,
            centers=[[0, 0, 0, 0, 0], [50, 50, 50, 50, 50], [100, 100, 100, 100, 100]],
            cluster_std=1.0,
            random_state=42
        )
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_full = km.fit_predict(X)

        def _estimador(X_sub, seed):
            return KMeans(n_clusters=3, random_state=seed, n_init=5).fit_predict(X_sub)

        ari_mean, ari_std, ari_vals = calcular_estabilidad_bootstrap_ari(
            X=X,
            labels_full=labels_full,
            estimador_fn=_estimador,
            n_iter=10,
            frac=0.80,
            random_state=42
        )
        self.assertGreater(ari_mean, 0.90)
        self.assertLess(ari_std, 0.10)
        self.assertEqual(len(ari_vals), 10)

    def test_generar_label_auto_puro(self):
        """Valida composición en Title Case de tags activos + término dominante."""
        tag_shares = pd.Series({
            "tag_palco": 0.85,
            "tag_occidental": 0.75,
            "tag_general": 0.05
        })
        label = generar_label_auto(
            cluster_id=2,
            tag_shares=tag_shares,
            term_dominante="alto",
            purity_tag=0.85,
            purity_term=0.60,
            threshold_tag_active=0.60,
            min_purity_tag=0.70
        )
        self.assertEqual(label, "Palco Occidental Alto")

    def test_generar_label_auto_deduplicacion_acentos(self):
        """Valida que una colisión acentuada ('Balcón' vs 'balcon') se deduplique correctamente."""
        tag_shares = pd.Series({
            "tag_balcon": 0.90,
            "tag_piso_alto": 0.80
        })
        label = generar_label_auto(
            cluster_id=1,
            tag_shares=tag_shares,
            term_dominante="balcon",
            purity_tag=0.90,
            purity_term=0.80,
            threshold_tag_active=0.60,
            min_purity_tag=0.70
        )
        self.assertEqual(label, "Balcón Piso Alto")

        tag_shares_solo_balcon = pd.Series({
            "tag_balcon": 0.95
        })
        label_solo = generar_label_auto(
            cluster_id=6,
            tag_shares=tag_shares_solo_balcon,
            term_dominante="balcon",
            purity_tag=0.95,
            purity_term=0.85,
            threshold_tag_active=0.60,
            min_purity_tag=0.70
        )
        self.assertEqual(label_solo, "Balcón")

    def test_generar_label_auto_impuro_hibrido(self):
        """Valida fallback estricto a 'hibrido_k{id}' si la pureza es < 0.70 o sin término."""
        tag_shares = pd.Series({
            "tag_palco": 0.55,
            "tag_platea": 0.45
        })
        label_baja_pureza = generar_label_auto(
            cluster_id=7,
            tag_shares=tag_shares,
            term_dominante="platea",
            purity_tag=0.55,
            purity_term=0.40,
            min_purity_tag=0.70
        )
        self.assertEqual(label_baja_pureza, "hibrido_k7")

        # Sin término dominante
        label_sin_termino = generar_label_auto(
            cluster_id=4,
            tag_shares=tag_shares,
            term_dominante="",
            purity_tag=0.80,
            purity_term=0.0,
            min_purity_tag=0.70
        )
        self.assertEqual(label_sin_termino, "hibrido_k4")

    def test_generar_catalogo_clusters(self):
        """Valida que el catálogo contenga exactamente las columnas requeridas y n de filas."""
        df_dummy = pd.DataFrame({
            "logical_seat_category": ["PALCO 1", "PALCO 2", "PLATEA A", "PLATEA B"],
            "texto_limpio": ["palco 1", "palco 2", "platea a", "platea b"],
            "tag_palco": [1, 1, 0, 0],
            "tag_platea": [0, 0, 1, 1],
            "arquetipo_demanda": ["VIP / Palcos / Premium", "VIP / Palcos / Premium", "Preferencial / Platea Frontal", "Preferencial / Platea Frontal"]
        })
        labels = np.array([0, 0, 1, 1])

        cat = generar_catalogo_clusters(
            df_multi=df_dummy,
            labels=labels,
            tfidf_vocab=["palco", "platea"],
            tag_cols=["tag_palco", "tag_platea"]
        )

        columnas_esperadas = {
            "cluster_id", "n", "purity_tag", "tag_dominante",
            "purity_term", "term_dominante", "label_auto", "arquetipo_v23_rollup"
        }
        self.assertTrue(columnas_esperadas.issubset(set(cat.columns)))
        self.assertEqual(len(cat), 2)
        self.assertEqual(cat.loc[cat["cluster_id"] == 0, "n"].iloc[0], 2)
        self.assertEqual(cat.loc[cat["cluster_id"] == 1, "n"].iloc[0], 2)


if __name__ == "__main__":
    unittest.main()
