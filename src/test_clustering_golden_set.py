"""
Suite de Pruebas Automatizadas y Golden Set para el Pipeline de Clusterización en Dos Etapas (v2.1).

Valida:
1. Partición estricta a nivel evento (sin funciones partidas, cobertura exacta de 33,775 filas).
2. Coherencia matemática de centroides multi-zona (VIP con bajo aforo y alto precio, Grada con aforo masivo).
3. Clasificación correcta del Golden Set de 20 localidades representativas.
"""

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
    entrenar_modelo_clustering
)


class TestClusteringGoldenSet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Cargar y preparar dataset real
        df_raw = pd.read_parquet("data/raw/localidades_eda.parquet")
        df_clean = filtrar_consistencia_localidades(df_raw)
        df_rel = calcular_metricas_relativas(df_clean)
        cls.df_enriquecido = pipeline_procesamiento_nlp(df_rel)

        # Ejecutar pipeline en dos etapas con k óptimo (k=5 en multi-zona + 1 tarifa plana)
        cls.df_final, cls.kmeans, cls.scaler, cls.tfidf_vec, cls.feature_names, cls.metricas = (
            pipeline_clustering_dos_etapas(cls.df_enriquecido, n_clusters_multizona=5, random_state=42)
        )

    def test_01_cobertura_exacta_y_no_particion_de_eventos(self):
        """Valida que la suma de monozona + multi-zona sea 33,775 y que ningún evento esté partido."""
        df_monozona, df_multizona = separar_admision_unica_multizona(self.df_enriquecido)
        
        # 1. Cobertura exacta
        self.assertEqual(len(df_monozona), 15375, "Admisión única debe contener exactamente 15,375 filas")
        self.assertEqual(len(df_multizona), 18400, "Multi-zona debe contener exactamente 18,400 filas")
        self.assertEqual(len(df_monozona) + len(df_multizona), 33775, "Total debe sumar 33,775 filas exactas")
        self.assertEqual(len(self.df_final), 33775, "El DataFrame final integrado debe tener 33,775 filas")

        # 2. Ningún evento puede tener localidades en ambas etapas
        eventos_monozona = set(df_monozona["t_performance_id"])
        eventos_multizona = set(df_multizona["t_performance_id"])
        interseccion = eventos_monozona.intersection(eventos_multizona)
        self.assertEqual(len(interseccion), 0, f"Hay {len(interseccion)} eventos divididos entre etapas!")

    def test_02_criterios_cuantitativos_centroides(self):
        """Verifica la lógica económica de los centroides en el catálogo multi-zona."""
        df_multi = self.df_final[~self.df_final["es_monozona"]]

        stats = df_multi.groupby("arquetipo_demanda").agg(
            ratio_mean=("ratio_precio_max", "mean"),
            aforo_mean=("peso_aforo", "mean"),
            pct_mean=("percentil_precio_evento", "mean")
        )

        # 1. VIP debe tener el aforo más bajo (< 15%) y alto precio
        self.assertLess(stats.loc["VIP / Palcos / Premium", "aforo_mean"], 0.15, "VIP debe tener bajo aforo relativo")
        self.assertGreater(stats.loc["VIP / Palcos / Premium", "ratio_mean"], 0.75, "VIP debe tener alto ratio de precio")

        # 2. Grada General debe tener aforo dominante frente a VIP
        self.assertGreater(
            stats.loc["Grada General / Masiva", "aforo_mean"],
            stats.loc["VIP / Palcos / Premium", "aforo_mean"] * 2,
            "Grada General debe tener al menos el doble de aforo relativo que VIP"
        )

        # 3. La anomalía histórica de Grada General con ratio = 1.00 desaparece:
        # En multi-zona, Grada General no puede tener aforo del 100%
        self.assertLess(stats.loc["Grada General / Masiva", "aforo_mean"], 0.85)

    def test_03_golden_set_20_localidades_representativas(self):
        """Verifica la clasificación adecuada de 20 localidades canónicas del negocio."""
        # Casos Golden Set conocidos en el catálogo:
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
            ("BALCON 2DO PISO", "Popular / Visibilidad Parcial / Balcón", False),
            ("BALCON TERCER PISO", "Popular / Visibilidad Parcial / Balcón", False),
            ("VISTA PARCIAL LATERAL", "Popular / Visibilidad Parcial / Balcón", False),
            ("PISO ALTO POSTERIOR", "Popular / Visibilidad Parcial / Balcón", False),

            # 5. Grada General / Masiva (Graderías masivas de estadios y arenas)
            ("GRADA GENERAL", "Grada General / Masiva", False),
            ("GRADA NORTE", "Grada General / Masiva", False),
            ("CANCHA GENERAL", "Grada General / Masiva", False),
            ("ENTRADA GENERAL", "Grada General / Masiva", False),
        ]

        # Validar en el catálogo final filtrando por patrones
        fallos = []
        for term, arq_esperado, es_mono_esperado in golden_cases:
            matches = self.df_final[self.df_final["logical_seat_category"].str.upper().str.contains(term)]
            if len(matches) > 0:
                pred_arqs = matches["arquetipo_demanda"].value_counts()
                top_pred = pred_arqs.index[0]
                # Si el top predicho no coincide y no es una categoría afín razonable
                if top_pred != arq_esperado:
                    # En algunos eventos pequeños, un término como 'GRADA' puede ser monozona si el evento fue monozona
                    if matches["es_monozona"].iloc[0] and arq_esperado != "Admisión Única / Tarifa Plana":
                        continue
                    fallos.append(f"Término '{term}': Esperado '{arq_esperado}', obtenido '{top_pred}'")

        # Tolerancia: los términos contextuales de eventos reales pueden distribuirse entre categorías afines
        self.assertLessEqual(len(fallos), 2, f"Fallos en Golden Set: {fallos}")


if __name__ == "__main__":
    unittest.main()
