"""
Capa de servicio de inferencia de localidades (Bloque 0).

La regla de oro de la capa de servicio: solo encadena funciones
preexistentes del pipeline (feature_engineering, clustering, jerarquia);
cero logica de modelo nueva.
"""

from typing import Any, Dict, List, Union
import numpy as np
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


def construir_dataframe_evento(evento: Dict[str, Any]) -> pd.DataFrame:
    """
    Construye un pd.DataFrame defensivo a partir de un diccionario de evento.
    Soporta formato anidado con clave 'localidades' o listas planas por columna.
    """
    if "localidades" in evento and isinstance(evento["localidades"], list):
        ev_id = evento.get("t_performance_id", 1)
        ev_prod = evento.get("product") or evento.get("event_name", "EVENTO")
        ev_site = evento.get("site") or evento.get("venue_name", "VENUE")
        ev_quota = evento.get("performance_quota")

        # Si el evento no trae quota total, calcular la suma de aforos de sus localidades
        total_quota_localidades = sum(
            float(loc.get("dn_quota", 0.0)) for loc in evento["localidades"]
        )

        rows: List[Dict[str, Any]] = []
        for loc in evento["localidades"]:
            r = dict(loc)
            if "t_performance_id" not in r:
                r["t_performance_id"] = ev_id
            if "product" not in r and "event_name" not in r:
                r["product"] = ev_prod
            elif "event_name" in r and "product" not in r:
                r["product"] = r["event_name"]
            if "site" not in r and "venue_name" not in r:
                r["site"] = ev_site
            elif "venue_name" in r and "site" not in r:
                r["site"] = r["venue_name"]
            if "performance_quota" not in r:
                r["performance_quota"] = float(ev_quota) if ev_quota is not None else total_quota_localidades
            if "med_base_unit_amt_itx" not in r:
                col_p = next((c for c in ["precio", "price_amount", "unit_amount"] if c in r), None)
                if col_p:
                    r["med_base_unit_amt_itx"] = r[col_p]
            if "net_sold_p_qty" not in r:
                r["net_sold_p_qty"] = 0.0
            if "net_sold_c_qty" not in r:
                r["net_sold_c_qty"] = 0.0
            rows.append(r)
        df_raw = pd.DataFrame(rows)
    else:
        df_raw = pd.DataFrame(evento)
        if "precio" in df_raw.columns and "med_base_unit_amt_itx" not in df_raw.columns:
            df_raw["med_base_unit_amt_itx"] = df_raw["precio"]
        if "event_name" in df_raw.columns and "product" not in df_raw.columns:
            df_raw["product"] = df_raw["event_name"]
        if "venue_name" in df_raw.columns and "site" not in df_raw.columns:
            df_raw["site"] = df_raw["venue_name"]
        if "performance_quota" not in df_raw.columns and "dn_quota" in df_raw.columns:
            if "t_performance_id" in df_raw.columns:
                df_raw["performance_quota"] = df_raw.groupby("t_performance_id")["dn_quota"].transform("sum")
            else:
                df_raw["performance_quota"] = df_raw["dn_quota"].sum()
        if "net_sold_p_qty" not in df_raw.columns:
            df_raw["net_sold_p_qty"] = 0.0
        if "net_sold_c_qty" not in df_raw.columns:
            df_raw["net_sold_c_qty"] = 0.0

    return df_raw


def predecir_evento(
    evento: Dict[str, Any],
    payload_v25: Union[str, Dict[str, Any]],
    payload_jer: Union[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Servicio de inferencia que encadena las funciones existentes del pipeline:
    1. Construir DataFrame crudo del evento
    2. filtrar_consistencia_localidades (modo estricto)
    3. preparar_dataset_enriquecido
    4. predecir_arquetipos_demanda (Nivel 1 v2.5)
    5. predecir_microclusters (Nivel 2 jerarquico v3.0)
    6. Formatear y retornar diccionario estandarizado de respuesta.
    """
    if isinstance(payload_v25, str):
        p_v25 = cargar_modelo_clustering(payload_v25)
    else:
        p_v25 = payload_v25

    if isinstance(payload_jer, str):
        p_jer = joblib.load(payload_jer)
    else:
        p_jer = payload_jer

    v25_ver = p_v25.get("version", "2.5") if isinstance(p_v25, dict) else "2.5"
    jer_ver = p_jer.get("version", "3.0-hier.2") if isinstance(p_jer, dict) else "3.0-hier.2"
    modelo_version = f"v{v25_ver} / {jer_ver}"

    # 1. Construir DataFrame
    df_raw = construir_dataframe_evento(evento)
    filas_in = len(df_raw)

    # 2. Filtrado de consistencia en modo estricto
    df_filtrado, reporte_filtro = filtrar_consistencia_localidades(
        df_raw,
        retornar_reporte=True,
        verbose=False,
        modo_legacy_v3=False
    )

    if len(df_filtrado) == 0:
        return {
            "modelo_version": modelo_version,
            "filas_in": filas_in,
            "filas_out": 0,
            "reporte_filtro": reporte_filtro,
            "predicciones": []
        }

    # 3. Enriquecimiento de variables
    df_enriquecido = preparar_dataset_enriquecido(df_filtrado, modo_legacy_v3=False)

    # 4. Prediccion Nivel 1 (v2.5)
    df_n1 = predecir_arquetipos_demanda(df_enriquecido, p_v25)

    # 5. Prediccion Nivel 2 (Micro-clusters)
    df_micro = predecir_microclusters(df_n1, p_jer, payload_v23=p_v25)

    # 6. Reensamblaje de respuesta estandarizada
    predicciones: List[Dict[str, Any]] = []
    for _, row in df_micro.iterrows():
        es_frontera_bool = bool(row["es_frontera"])
        predicciones.append({
            "logical_seat_category": str(row["logical_seat_category"]),
            "arquetipo": str(row["arquetipo_demanda"]),
            "micro_cluster": str(row["micro_cluster_id"]),
            "label_auto": str(row["label_auto"]),
            "confianza": float(row["score_confianza"]),
            "es_frontera": es_frontera_bool,
            "segmento_incierto": es_frontera_bool,
            # Campos canonicos adicionales para maxima compatibilidad
            "arquetipo_demanda": str(row["arquetipo_demanda"]),
            "micro_cluster_id": str(row["micro_cluster_id"]),
            "score_confianza": float(row["score_confianza"])
        })

    return {
        "modelo_version": modelo_version,
        "filas_in": filas_in,
        "filas_out": len(df_micro),
        "reporte_filtro": reporte_filtro,
        "predicciones": predicciones
    }
