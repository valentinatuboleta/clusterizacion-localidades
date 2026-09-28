"""
Script de Migración de Taxonomía v1 -> v2 para site_type_lookup.csv y site_type_revision_humana.csv.

Aplica:
1. Mapeo mecánico explícito v1 -> v2 (Bloque 1a).
2. Curaduría y reglas prioritarias para MUSEO, RESTAURANTE, PARQUEADERO, ARENA, COLISEO (Bloque 1c).
3. Preservación absoluta de la trazabilidad original (fuente, fecha_clasificacion, aforo_max, funciones).
4. Inclusión obligatoria de taxonomia_version="v2" y fecha_reclasificacion (Bloque 1b).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import re
import datetime
import pandas as pd
from src.nlp_utils import normalizar_venue
from scripts.clasificar_sites import (
    DICCIONARIO_NORMALIZADO,
    pasaje_a_clasificar,
    pasaje_b_clasificar,
    CATEGORIAS_VALIDAS
)
from src.clustering import MAPEO_TAXONOMIA_V1_A_V2, CANONICAL_TYPE_SITE_CATEGORIES


def migrar_registro_lookup(site: str, old_type: str) -> str:
    site_norm = normalizar_venue(site)
    
    # 1. Chequeo prioritario en DICCIONARIO_EMBLEMATICO
    if site_norm in DICCIONARIO_NORMALIZADO:
        return DICCIONARIO_NORMALIZADO[site_norm]
    for k_norm, v in DICCIONARIO_NORMALIZADO.items():
        if len(k_norm.split()) >= 2 and len(k_norm) >= 8:
            if re.search(r"\b" + re.escape(k_norm) + r"\b", site_norm):
                return v

    # 2. Casos especiales de re-clasificación del residual y nuevas categorías (1c)
    if re.search(r"\b(PARQUEADERO|PARKING|ESTACIONAMIENTO)\b", site_norm):
        return "PARQUEADERO"
    if re.search(r"\b(COLISEO|PALACIO\s+DE\s+LOS\s+DEPORTES)\b", site_norm):
        return "COLISEO"
    if re.search(r"\b(MUSEO)\b", site_norm):
        return "MUSEO"
    if site_norm in ["MALOKA", "PLANETARIO DE BOGOTA", "YAWA, CENTRO DE CIENCIA, ARTE Y TECNOLOGIA - CALI", "BIBLIOTECA NACIONAL DE COLOMBIA"]:
        return "MUSEO"
    if re.search(r"\b(CINEMATECA|CINE\s+COLOMBO)\b", site_norm):
        return "CINEMATECA"
    if re.search(r"\b(MOVISTAR\s*ARENA|ARENA\s+CA[NÑ]AVERALEJO|NECTAR\s+ARENA|INDIGO\s*-\s*MOVISTAR\s*ARENA|ARENA\s+LAS\s+MEJORES)\b", site_norm):
        return "ARENA"
    if re.search(r"\b(RESTAURANTE|GASTROBAR|BAR|CLUB|DISCOTECA|PUB|FONDA|CANTINA|RESTOBAR)\b", site_norm):
        return "RESTAURANTE"

    # 3. Mapeo mecánico 1a
    destino_mecanico = MAPEO_TAXONOMIA_V1_A_V2.get(old_type)
    if destino_mecanico and destino_mecanico != "re-clasificar":
        return destino_mecanico

    # 4. Fallback léxico ortogonal
    t_a, _ = pasaje_a_clasificar(site, 500)
    return t_a


def ejecutar_migracion(
    ruta_lookup: str = "data/lookup/site_type_lookup.csv",
    ruta_revision: str = "data/lookup/site_type_revision_humana.csv"
):
    fecha_hoy = datetime.date.today().isoformat()

    # --- 1. Migrar site_type_lookup.csv ---
    print(f"Cargando {ruta_lookup}...")
    df_lookup = pd.read_csv(ruta_lookup)
    total_previo = len(df_lookup)
    print(f"Total registros previos: {total_previo}")

    tipos_nuevos = []
    for _, row in df_lookup.iterrows():
        nuevo_tipo = migrar_registro_lookup(str(row["site"]), str(row.get("type_site", "")))
        tipos_nuevos.append(nuevo_tipo)

    df_lookup["type_site"] = tipos_nuevos
    df_lookup["taxonomia_version"] = "v2"
    df_lookup["fecha_reclasificacion"] = fecha_hoy
    if "modelo_llm" not in df_lookup.columns:
        df_lookup["modelo_llm"] = None

    # Validar que todas las categorías sean canónicas
    invalidos = df_lookup[~df_lookup["type_site"].isin(CANONICAL_TYPE_SITE_CATEGORIES)]
    if len(invalidos) > 0:
        raise ValueError(f"Categorías no canónicas detectadas en lookup: {invalidos[['site', 'type_site']]}")

    # Ordenar columnas con trazabilidad completa
    cols_orden = [
        "site", "type_site", "confianza", "aforo_max", "funciones",
        "fecha_clasificacion", "fuente", "modelo_llm",
        "taxonomia_version", "fecha_reclasificacion"
    ]
    df_lookup = df_lookup[cols_orden]
    df_lookup.to_csv(ruta_lookup, index=False)
    print(f"[OK] {ruta_lookup} regenerado exitosamente ({len(df_lookup)} registros).")
    print(df_lookup["type_site"].value_counts())

    # --- 2. Migrar site_type_revision_humana.csv ---
    if os.path.exists(ruta_revision):
        print(f"\nCargando {ruta_revision}...")
        df_rev = pd.read_csv(ruta_revision)
        tipos_rev = []
        for _, row in df_rev.iterrows():
            orig_sug = str(row.get("type_site_sugerido", ""))
            site_val = str(row["site"])
            # Aplicar reglas actualizadas
            tipo_migrado = migrar_registro_lookup(site_val, orig_sug)
            tipos_rev.append(tipo_migrado)
        
        df_rev["type_site_sugerido"] = tipos_rev
        if "propuesta_a" in df_rev.columns:
            df_rev["propuesta_a"] = [migrar_registro_lookup(str(s), str(pa)) for s, pa in zip(df_rev["site"], df_rev["propuesta_a"])]
        if "propuesta_b" in df_rev.columns:
            df_rev["propuesta_b"] = [migrar_registro_lookup(str(s), str(pb)) for s, pb in zip(df_rev["site"], df_rev["propuesta_b"])]
        
        df_rev.to_csv(ruta_revision, index=False)
        print(f"[OK] {ruta_revision} actualizado exitosamente ({len(df_rev)} registros).")
        print(df_rev["type_site_sugerido"].value_counts())


if __name__ == "__main__":
    ejecutar_migracion()
