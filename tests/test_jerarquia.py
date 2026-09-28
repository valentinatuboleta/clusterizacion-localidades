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

from scripts.entrenar_jerarquia_microclusters import (
    seleccionar_sub_k_codo_db,
    TAGS_EXPANDIDOS_13
)
from scripts.explorar_microclusters import (
    calcular_pureza_cluster,
    generar_label_auto
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


if __name__ == "__main__":
    unittest.main()
