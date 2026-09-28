"""
Tests unitarios hermeticos para el modulo de diagnostico de sub-espacios (v3.0-hier).

Valida:
1. Calculo de cobertura de identidad (deteccion exacta con tags de identidad y terminos distintivos).
2. Pureza alcanzable teorica (Oracle): caso limpio (>= 0.85) vs caso sin identidad (< 0.60).
3. Regla deterministica de veredicto: senal_ahogada vs senial_inexistente.
4. Extraccion de top terminos TF-IDF del centroide textual por micro-cluster.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.diagnosticar_subespacios import (
    determinar_veredicto,
    calcular_top_terminos_microcluster
)
from scripts.explorar_microclusters import calcular_pureza_cluster


class TestDiagnosticoSubespacios(unittest.TestCase):

    def test_determinar_veredicto(self):
        """Valida las reglas de decision para clasificar el tipo de senal."""
        # 1. Senal ahogada: cobertura >= 50%, oracle >= 0.85, actual < 0.85
        v1 = determinar_veredicto(cobertura=80.0, pureza_oracle=0.88, pureza_actual=0.52)
        self.assertEqual(v1, "senal_ahogada")

        # 2. Cobertura baja (< 50%) -> senial inexistente
        v2 = determinar_veredicto(cobertura=40.0, pureza_oracle=0.90, pureza_actual=0.50)
        self.assertEqual(v2, "senial_inexistente")

        # 3. Pureza oracle baja (< 0.85) -> senial inexistente
        v3 = determinar_veredicto(cobertura=75.0, pureza_oracle=0.78, pureza_actual=0.45)
        self.assertEqual(v3, "senial_inexistente")

        # 4. Pureza actual ya aprobada (>= 0.85) -> senial inexistente (no requiere intervencion)
        v4 = determinar_veredicto(cobertura=90.0, pureza_oracle=0.95, pureza_actual=0.92)
        self.assertEqual(v4, "senial_inexistente")

    def test_cobertura_identidad(self):
        """Valida medicion exacta de filas con tags de identidad y terminos distintivos."""
        df_test = pd.DataFrame({
            "texto_limpio": [
                "palco mesa 1",     # tiene tag_mesa y termino 'mesa'
                "palco lateral",    # tiene tag_lateral
                "palco central",    # sin tags de identidad, tiene termino distintivo 'central'
                "palco regular",    # sin tag de identidad ni termino distintivo
                "palco vip"         # sin tag de identidad ni termino distintivo
            ],
            "micro_cluster_id": ["VIP-0", "VIP-1", "VIP-2", "VIP-0", "VIP-1"],
            "tag_mesa": [1, 0, 0, 0, 0],
            "tag_lateral": [0, 1, 0, 0, 0]
        })
        tags_id = ["tag_mesa", "tag_lateral"]
        map_term_dom = {"VIP-0": "mesa", "VIP-1": "lateral", "VIP-2": "central"}

        has_tag_id = (df_test[tags_id].sum(axis=1) > 0)
        # Filas 0 y 1 tienen tag_id -> 2/5 = 40%
        self.assertAlmostEqual(has_tag_id.mean() * 100, 40.0)

        def _has_term(row):
            term = map_term_dom.get(row["micro_cluster_id"], "")
            return term.lower() in str(row["texto_limpio"]).lower().split()

        has_term = df_test.apply(_has_term, axis=1)
        # Fila 0 ('mesa'), Fila 1 ('lateral'), Fila 2 ('central') -> 3/5 = 60%
        self.assertAlmostEqual(has_term.mean() * 100, 60.0)

        # Union: Filas 0, 1, 2 -> 3/5 = 60%
        cobertura = (has_tag_id | has_term).mean() * 100
        self.assertAlmostEqual(cobertura, 60.0)

    def test_pureza_oracle(self):
        """Valida que el oraculo alcance >= 0.85 con datos limpios y < 0.60 sin identidad."""
        vocab = ["mesa", "lateral", "palco"]
        tags_id = ["tag_mesa", "tag_lateral"]

        # Caso A: Particion limpia donde 90% de filas tienen identidad pura
        df_limpio = pd.DataFrame({
            "texto_limpio": (
                ["mesa vip"] * 50 +
                ["lateral vip"] * 40 +
                ["palco vip"] * 10
            ),
            "tag_mesa": [1] * 50 + [0] * 40 + [0] * 10,
            "tag_lateral": [0] * 50 + [1] * 40 + [0] * 10
        })

        def _get_id(row):
            active = [t for t in tags_id if row[t] == 1]
            return "+".join(sorted(active)) if active else "sin_identidad"

        df_limpio["identidad"] = df_limpio.apply(_get_id, axis=1)
        clusters_a = df_limpio["identidad"].unique()
        p_list_a = []
        w_list_a = []
        for c in clusters_a:
            sub_c = df_limpio[df_limpio["identidad"] == c]
            res = calcular_pureza_cluster(sub_c, tags_id, vocab)
            p_list_a.append(res["purity_naming"])
            w_list_a.append(len(sub_c))
        pureza_oracle_a = float(np.sum(np.array(w_list_a) * np.array(p_list_a)) / len(df_limpio))
        # 50 rows tag_mesa (purity=1.0) + 40 rows tag_lateral (purity=1.0) + 10 rows sin_id (purity=0) = 0.90
        self.assertGreaterEqual(pureza_oracle_a, 0.85)

        # Caso B: Todas las filas sin identidad (tags de identidad en 0)
        df_sin_id = pd.DataFrame({
            "texto_limpio": ["general a", "general b", "standard 1", "entrada 2"] * 25,
            "tag_mesa": [0] * 100,
            "tag_lateral": [0] * 100
        })
        df_sin_id["identidad"] = df_sin_id.apply(_get_id, axis=1)
        clusters_b = df_sin_id["identidad"].unique()
        self.assertEqual(len(clusters_b), 1)
        self.assertEqual(clusters_b[0], "sin_identidad")

        res_b = calcular_pureza_cluster(df_sin_id, tags_id, vocab)
        pureza_oracle_b = res_b["purity_naming"]
        self.assertLess(pureza_oracle_b, 0.60)

    def test_top_terminos_microcluster(self):
        """Valida que se extraigan correctamente los top terminos del centroide textual."""
        textos = [
            "mesa occidental vip",
            "mesa lateral vip",
            "mesa vip",
            "mesa frontal"
        ]
        df_mc = pd.DataFrame({"texto_limpio": textos})
        tfidf = TfidfVectorizer(max_features=10)
        tfidf.fit(textos)

        top_terms = calcular_top_terminos_microcluster(df_mc, tfidf, top_n=3)
        self.assertTrue(len(top_terms) > 0)
        terms_only = [w for w, _ in top_terms]
        # 'mesa' y 'vip' deben liderar
        self.assertIn("mesa", terms_only)
        self.assertIn("vip", terms_only)


if __name__ == "__main__":
    unittest.main()
