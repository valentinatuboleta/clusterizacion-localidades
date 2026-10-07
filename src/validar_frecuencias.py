"""
Módulo de Validación de Frecuencias de Taxonomía de Venues (Bloque 1.5) — FUENTE DE VERDAD ANALÍTICA.

Provee las funciones oficiales para auditar la distribución de frecuencias de type_site tanto
a nivel venue único como a nivel localidad (frecuencia efectiva de entrenamiento).
Consumido programáticamente por la suite de pruebas y por el wrapper CLI `scripts/validar_frecuencias_taxonomy.py`.
Genera métricas de concentración, flags automáticos de sanidad y comparativa de migración.
"""

from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
from src.nlp_utils import normalizar_venue
from src.clustering import CANONICAL_TYPE_SITE_CATEGORIES


def calcular_reporte_frecuencias(
    df_raw: pd.DataFrame,
    df_lookup: pd.DataFrame,
    site_column: str = "site"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calcula la tabla de frecuencias de type_site v2 a nivel localidad y a nivel venue.
    Aplica flags automáticos:
    - flag_vacia: conteo de localidades == 0 (alerta / flag de parada).
    - flag_cold_start: 0 < conteo de localidades < 50 (activa cold start de percentil por tipo).
    - flag_dominancia: porcentaje de localidades > 60.0% (concentración patológica).

    Retorna:
    - df_frecuencias: DataFrame detallado por categoría.
    - diagnostico: diccionario con resumen de flags y estado general.
    """
    total_localidades = len(df_raw)
    if total_localidades == 0:
        raise ValueError("df_raw no contiene registros.")

    # Mapa canonico de lookup normalizado
    lookup_map = {}
    if "site" in df_lookup.columns and "type_site" in df_lookup.columns:
        for s, t in zip(df_lookup["site"], df_lookup["type_site"]):
            sn = normalizar_venue(s)
            if sn:
                lookup_map[sn] = str(t)

    # Conteo de venues unicos por categoria en lookup
    conteo_sites_lookup = df_lookup["type_site"].value_counts().to_dict() if "type_site" in df_lookup.columns else {}

    # Mapeo a nivel localidad
    sites_norm = df_raw[site_column].apply(normalizar_venue) if site_column in df_raw.columns else pd.Series(["desconocido"] * total_localidades)
    type_site_efectivo = sites_norm.map(lookup_map).fillna("desconocido")
    conteo_locs = type_site_efectivo.value_counts().to_dict()

    # Evaluar todas las categorias canonicas mas el bucket interno 'desconocido'
    todas_categorias = list(CANONICAL_TYPE_SITE_CATEGORIES) + ["desconocido"]
    registros = []

    flags_rojos = []
    flags_amarillos = []

    for cat in todas_categorias:
        n_locs = int(conteo_locs.get(cat, 0))
        pct_locs = round((n_locs / total_localidades) * 100.0, 2)
        n_sites = int(conteo_sites_lookup.get(cat, 0))

        flag_vacia = (n_locs == 0)
        flag_cold_start = (0 < n_locs < 50)
        flag_dominancia = (pct_locs > 60.0)

        if flag_vacia:
            flags_rojos.append(f"CATEGORIA_VACIA:{cat}")
        if flag_dominancia:
            flags_rojos.append(f"DOMINANCIA_EXCESIVA:{cat}({pct_locs}%)")
        if flag_cold_start:
            flags_amarillos.append(f"COLD_START_ACTIVO:{cat}({n_locs}_locs)")

        registros.append({
            "categoria": cat,
            "conteo_localidades": n_locs,
            "porcentaje_localidades": pct_locs,
            "conteo_sites_unicos": n_sites,
            "flag_vacia": flag_vacia,
            "flag_cold_start": flag_cold_start,
            "flag_dominancia": flag_dominancia
        })

    df_frecuencias = pd.DataFrame(registros).sort_values(by="conteo_localidades", ascending=False).reset_index(drop=True)

    diagnostico = {
        "total_localidades": total_localidades,
        "total_sites_lookup": len(df_lookup),
        "flags_rojos": flags_rojos,
        "flags_amarillos": flags_amarillos,
        "pasa_compuerta": len(flags_rojos) == 0
    }

    return df_frecuencias, diagnostico
