"""
Tests de paridad para la capa de servicio (src/api/servicio.py).

Garantiza:
1. Paridad celda a celda estricta sobre evento sintetico de 6 localidades
   (monozona + multi-zona + venue desconocido + frontera).
2. Paridad sobre datos reales (skip-if-missing) >= 99%.
3. Manejo defensivo cuando todas las filas son descartadas por consistencia.
"""

import os
import unittest
import pytest
import pandas as pd
import joblib

from src.feature_engineering import (
    filtrar_consistencia_localidades,
    preparar_dataset_enriquecido
)
from src.clustering import (
    cargar_modelo_clustering,
    predecir_arquetipos_demanda
)
from src.jerarquia import predecir_microclusters
from src.api.servicio import predecir_evento, construir_dataframe_evento


class TestServicioParidad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path_v25 = "data/processed/modelo_clustering_v2_5.joblib"
        cls.path_jer = "data/processed/modelo_jerarquia_v3.joblib"
        cls.p_v25 = cargar_modelo_clustering(cls.path_v25)
        cls.p_jer = joblib.load(cls.path_jer)

        # Fixture de 6 localidades con las 4 propiedades solicitadas:
        # - Monozona (performance 999902 con aforo unico)
        # - Multi-zona (performance 999901 con estratificacion)
        # - Venue desconocido (ESPACIO_TOTALMENTE_DESCONOCIDO_99)
        # - Frontera (margen de decision < 0.15)
        cls.evento_sintetico = {
            "t_performance_id": 999901,
            "product": "CONCIERTO SINFONICO ESPECIAL",
            "site": "ESPACIO_TOTALMENTE_DESCONOCIDO_99",
            "performance_quota": 5000.0,
            "localidades": [
                {
                    "logical_seat_category": "PALCO VIP OCCIDENTAL",
                    "dn_quota": 100.0,
                    "med_base_unit_amt_itx": 450000.0,
                    "net_sold_p_qty": 10.0,
                    "net_sold_c_qty": 0.0
                },
                {
                    "logical_seat_category": "PREFERENCIAL FRONTAL",
                    "dn_quota": 400.0,
                    "med_base_unit_amt_itx": 280000.0,
                    "net_sold_p_qty": 50.0,
                    "net_sold_c_qty": 0.0
                },
                {
                    "logical_seat_category": "ZONA MIXTA TRIBUNA",
                    "dn_quota": 1000.0,
                    "med_base_unit_amt_itx": 150000.0,
                    "net_sold_p_qty": 100.0,
                    "net_sold_c_qty": 0.0
                },
                {
                    "logical_seat_category": "GENERAL PISO 2",
                    "dn_quota": 1500.0,
                    "med_base_unit_amt_itx": 90000.0,
                    "net_sold_p_qty": 200.0,
                    "net_sold_c_qty": 0.0
                },
                {
                    "logical_seat_category": "POPULAR BALCON ALTO",
                    "dn_quota": 2000.0,
                    "med_base_unit_amt_itx": 50000.0,
                    "net_sold_p_qty": 300.0,
                    "net_sold_c_qty": 0.0
                },
                {
                    "t_performance_id": 999902,
                    "performance_quota": 500.0,
                    "logical_seat_category": "GENERAL MONOZONA UNICA",
                    "dn_quota": 500.0,
                    "med_base_unit_amt_itx": 80000.0,
                    "net_sold_p_qty": 50.0,
                    "net_sold_c_qty": 0.0
                }
            ]
        }

    def test_paridad_celda_a_celda_evento_sintetico(self):
        """
        El test mas importante de la fase:
        predecir_evento(evento) == encadenamiento directo de las funciones, celda a celda.
        Garantiza que la capa de servicio no reinterpreta nada ni muta resultados.
        """
        # 1. Invocacion mediante la funcion de servicio
        res_servicio = predecir_evento(
            self.evento_sintetico,
            self.p_v25,
            self.p_jer
        )

        # 2. Encadenamiento directo celda a celda
        df_raw = construir_dataframe_evento(self.evento_sintetico)
        df_filt, rep_directo = filtrar_consistencia_localidades(
            df_raw,
            retornar_reporte=True,
            verbose=False,
            modo_legacy_v3=False
        )
        df_enr = preparar_dataset_enriquecido(df_filt, modo_legacy_v3=False)
        df_n1 = predecir_arquetipos_demanda(df_enr, self.p_v25)
        df_micro = predecir_microclusters(df_n1, self.p_jer, payload_v23=self.p_v25)

        # 3. Validacion celda a celda estricta
        self.assertEqual(res_servicio["filas_in"], len(df_raw))
        self.assertEqual(res_servicio["filas_out"], len(df_micro))
        self.assertEqual(res_servicio["reporte_filtro"], rep_directo)

        preds_serv = res_servicio["predicciones"]
        self.assertEqual(len(preds_serv), len(df_micro))

        # Validar propiedades requeridas del fixture sintético
        tiene_monozona = any(p["micro_cluster"] == "AU-0" for p in preds_serv)
        tiene_multizona = any(p["micro_cluster"] != "AU-0" for p in preds_serv)
        tiene_frontera = any(p["es_frontera"] for p in preds_serv)
        self.assertTrue(tiene_monozona, "El fixture debe contener al menos una localidad monozona")
        self.assertTrue(tiene_multizona, "El fixture debe contener localidades multi-zona")
        self.assertTrue(tiene_frontera, "El fixture debe contener al menos una localidad de frontera")
        self.assertTrue(df_enr["flag_site_desconocido"].iloc[0] == 1, "El venue debe ser clasificado como desconocido")

        # Comparacion campo por campo contra la salida directa
        for i, (_, row) in enumerate(df_micro.iterrows()):
            p = preds_serv[i]
            self.assertEqual(p["logical_seat_category"], str(row["logical_seat_category"]))
            self.assertEqual(p["arquetipo"], str(row["arquetipo_demanda"]))
            self.assertEqual(p["micro_cluster"], str(row["micro_cluster_id"]))
            self.assertEqual(p["label_auto"], str(row["label_auto"]))
            self.assertAlmostEqual(p["confianza"], float(row["score_confianza"]), places=6)
            self.assertEqual(p["es_frontera"], bool(row["es_frontera"]))
            self.assertEqual(p["segmento_incierto"], bool(row["es_frontera"]))

    def test_manejo_defensivo_evento_completamente_filtrado(self):
        """Si todas las localidades son descartadas por consistencia, retorna estructura limpia sin excepcion."""
        evento_invalido = {
            "t_performance_id": 999903,
            "product": "EVENTO CANCELAD",  # Patron de exclusion
            "site": "TEATRO",
            "performance_quota": 100.0,
            "localidades": [
                {
                    "logical_seat_category": "LOCALIDAD CANCELADA",
                    "dn_quota": 100.0,
                    "med_base_unit_amt_itx": 50000.0
                }
            ]
        }
        res = predecir_evento(evento_invalido, self.p_v25, self.p_jer)
        self.assertEqual(res["filas_in"], 1)
        self.assertEqual(res["filas_out"], 0)
        self.assertEqual(len(res["predicciones"]), 0)
        self.assertGreater(res["reporte_filtro"]["regla_exclusion_patrones_producto"], 0)

    @pytest.mark.skipif(
        not os.path.exists("data/processed/marcha_blanca_predicciones.parquet"),
        reason="Test de paridad real omitido: data/processed/marcha_blanca_predicciones.parquet no presente"
    )
    def test_paridad_real_marcha_blanca(self):
        """
        Toma 3 eventos reales de marcha blanca y valida >=99% de coincidencia en micro_cluster_id.
        """
        df_mb = pd.read_parquet("data/processed/marcha_blanca_predicciones.parquet")
        eventos_reales_ids = [10230174593781, 10230267313328, 10230349596222]

        total_coincidencias = 0
        total_filas = 0

        for ev_id in eventos_reales_ids:
            ev_df = df_mb[df_mb["t_performance_id"] == ev_id]
            if ev_df.empty:
                continue

            evento_dict = {
                "t_performance_id": int(ev_id),
                "product": str(ev_df["product"].iloc[0]),
                "site": str(ev_df["site"].iloc[0]),
                "performance_quota": float(ev_df["performance_quota"].iloc[0]),
                "localidades": [
                    {
                        "logical_seat_category": str(r["logical_seat_category"]),
                        "dn_quota": float(r["dn_quota"]),
                        "med_base_unit_amt_itx": float(r["base_unit_amt_itx"]),
                        "net_sold_p_qty": float(r.get("net_sold_p_qty", 0.0)),
                        "net_sold_c_qty": float(r.get("net_sold_c_qty", 0.0))
                    }
                    for _, r in ev_df.iterrows()
                ]
            }

            res = predecir_evento(evento_dict, self.p_v25, self.p_jer)
            preds_mc = [p["micro_cluster"] for p in res["predicciones"]]
            esperados_mc = ev_df["micro_cluster_id"].tolist()

            self.assertEqual(len(preds_mc), len(esperados_mc))
            for p_mc, exp_mc in zip(preds_mc, esperados_mc):
                total_filas += 1
                if p_mc == exp_mc:
                    total_coincidencias += 1

        tasa_coincidencia = total_coincidencias / max(1, total_filas)
        self.assertGreaterEqual(
            tasa_coincidencia,
            0.99,
            f"Tasa de coincidencia real ({tasa_coincidencia:.2%}) inferior al umbral 99%"
        )


if __name__ == "__main__":
    unittest.main()
