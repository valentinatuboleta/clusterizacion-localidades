"""
Módulo de Ingeniería de Características (Feature Engineering) para Localidades.

Este módulo se encarga de:
1. Filtrar registros inválidos o inconsistentes.
2. Calcular métricas relativas contextualizadas por evento (percentil de precio, ratio de precio, peso de aforo).
3. Calcular métricas de comportamiento histórico de absorción de ventas (tasa de ocupación, ventas pagas vs cortesías).
4. Integrar la extracción de atributos NLP limpios para alimentar el espacio vectorial mixto.
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from src.nlp_utils import pipeline_procesamiento_nlp, normalizar_venue


def normalizar_recinto(texto: str) -> str:
    """Alias de compatibilidad para normalizacion de venue."""
    return normalizar_venue(texto)


def enriquecer_type_site(
    df: pd.DataFrame,
    site_column: str = "site",
    lookup_path: str = "data/lookup/site_type_lookup.csv"
) -> pd.DataFrame:
    """
    Enriquece el DataFrame con la categoria estandarizada de venue (type_site) y flag_site_desconocido.
    Aplica matching canonico mediante normalizar_venue. Fallback type_site = 'desconocido'
    para sites no encontrados o lookup inexistente. El pipeline nunca falla ante un venue nuevo.
    """
    df_res = df.copy()
    if site_column not in df_res.columns:
        df_res["type_site"] = "desconocido"
        df_res["flag_site_desconocido"] = 1
        return df_res

    # Resolver ruta relativa tanto desde raiz como desde subdirectorios
    ruta_efectiva = lookup_path
    if not os.path.exists(ruta_efectiva):
        alt_ruta = os.path.join("..", lookup_path)
        if os.path.exists(alt_ruta):
            ruta_efectiva = alt_ruta

    lookup_map = {}
    if os.path.exists(ruta_efectiva):
        try:
            df_lookup = pd.read_csv(ruta_efectiva)
            if "site" in df_lookup.columns and "type_site" in df_lookup.columns:
                for s_val, t_val in zip(df_lookup["site"], df_lookup["type_site"]):
                    norm_s = normalizar_venue(s_val)
                    if norm_s:
                        lookup_map[norm_s] = str(t_val)
        except Exception:
            pass

    sites_norm = df_res[site_column].apply(normalizar_venue)
    df_res["type_site"] = sites_norm.map(lookup_map).fillna("desconocido")
    df_res["flag_site_desconocido"] = (df_res["type_site"] == "desconocido").astype(int)

    return df_res


def adjuntar_tipo_venue(df: pd.DataFrame, ruta_lookup: str = "data/lookup/site_type_lookup.csv") -> pd.DataFrame:
    """Wrapper de compatibilidad retroactiva para enriquecer_type_site."""
    return enriquecer_type_site(df, site_column="site", lookup_path=ruta_lookup)


def generar_referencia_percentil_tipo(
    df: pd.DataFrame,
    site_column: str = "site",
    localidad_column: Optional[str] = None,
    precio_column: str = "med_unit_amt_itx",
    min_localidades_historicas: int = 50
) -> Dict[str, Any]:
    """
    Genera la distribucion de referencia del percentil historico de precios por type_site:
    conteo de localidades unicas, flag cold start, bordes de bins y lista de precios ordenados.
    """
    df_work = df.copy()
    if "type_site" not in df_work.columns:
        df_work = enriquecer_type_site(df_work, site_column=site_column)

    if localidad_column is None:
        cands = ["logical_seat_category", "product", "translation_name", "cd_name", "nombre_localidad", "texto_limpio"]
        loc_col = next((c for c in cands if c in df_work.columns), "logical_seat_category")
    else:
        loc_col = localidad_column

    col_p = precio_column if precio_column in df_work.columns else "ave_unit_amt_itx"
    if col_p not in df_work.columns:
        return {}

    # Precio promedio por localidad historica unica (site, localidad, type_site)
    loc_stats = (
        df_work.groupby([site_column, loc_col, "type_site"], as_index=False)[col_p]
        .mean()
        .rename(columns={col_p: "_precio_prom_hist"})
    )

    referencia = {}
    for tipo, grp in loc_stats.groupby("type_site"):
        precios = grp["_precio_prom_hist"].dropna().values
        n_locs = len(precios)
        es_cold = (n_locs < min_localidades_historicas) or (str(tipo) == "desconocido")
        if es_cold:
            referencia[str(tipo)] = {
                "conteo": int(n_locs),
                "es_cold_start": True,
                "bin_edges": [],
                "precios_referencia": []
            }
        else:
            sorted_p = np.sort(precios).astype(float)
            referencia[str(tipo)] = {
                "conteo": int(n_locs),
                "es_cold_start": False,
                "bin_edges": np.percentile(sorted_p, np.linspace(0, 100, 101)).tolist(),
                "precios_referencia": sorted_p.tolist()
            }
    return referencia


def calcular_percentil_precio_absoluto_dentro_tipo(
    df: pd.DataFrame,
    site_column: str = "site",
    localidad_column: Optional[str] = None,
    precio_column: str = "med_unit_amt_itx",
    referencia_distribucion: Optional[Dict[str, Any]] = None,
    min_localidades_historicas: int = 50
) -> pd.DataFrame:
    """
    Calcula el percentil del precio promedio historico de la localidad dentro de su type_site.
    - Modo entrenamiento: precio promedio por (site, localidad), luego groupby('type_site').rank(pct=True).
      Cold start: si el tipo tiene < 50 localidades historicas -> valor 0.50 con flag_cold_start_tipo=1.
    - Modo inferencia: evalua contra la distribucion de referencia persistida (round-trip deterministico).
    """
    df_res = df.copy()
    if "type_site" not in df_res.columns:
        df_res = enriquecer_type_site(df_res, site_column=site_column)

    if localidad_column is None:
        cands = ["logical_seat_category", "product", "translation_name", "cd_name", "nombre_localidad", "texto_limpio"]
        loc_col = next((c for c in cands if c in df_res.columns), "logical_seat_category")
    else:
        loc_col = localidad_column

    col_p = precio_column if precio_column in df_res.columns else "ave_unit_amt_itx"
    if col_p not in df_res.columns:
        df_res["percentil_precio_absoluto_dentro_tipo"] = 0.50
        df_res["flag_cold_start_tipo"] = 1
        return df_res

    # Inferencia con referencia persistida
    if referencia_distribucion is not None:
        pcts = []
        flags = []
        for _, row in df_res.iterrows():
            t_val = str(row.get("type_site", "desconocido"))
            info = referencia_distribucion.get(t_val)
            val_p = row.get(col_p)
            val_p = float(val_p) if (val_p is not None and pd.notna(val_p)) else 0.0

            if info is None or info.get("es_cold_start", True) or not info.get("precios_referencia"):
                pcts.append(0.50)
                flags.append(1)
            else:
                ref_p = np.array(info["precios_referencia"])
                pct_val = float(percentileofscore(ref_p, val_p, kind="rank") / 100.0)
                pcts.append(pct_val)
                flags.append(0)

        df_res["percentil_precio_absoluto_dentro_tipo"] = pcts
        df_res["flag_cold_start_tipo"] = flags
        return df_res

    # Calculo directo sobre el dataset (entrenamiento)
    if site_column in df_res.columns and loc_col in df_res.columns:
        loc_stats = (
            df_res.groupby([site_column, loc_col, "type_site"], as_index=False)[col_p]
            .mean()
            .rename(columns={col_p: "_precio_prom_loc"})
        )
        conteos = loc_stats.groupby("type_site")["_precio_prom_loc"].transform("count")
        es_cold = (conteos < min_localidades_historicas) | (loc_stats["type_site"] == "desconocido")
        ranks = loc_stats.groupby("type_site")["_precio_prom_loc"].rank(pct=True)

        loc_stats["percentil_precio_absoluto_dentro_tipo"] = np.where(es_cold, 0.50, ranks)
        loc_stats["flag_cold_start_tipo"] = np.where(es_cold, 1, 0)

        cols_merge = [site_column, loc_col, "type_site"]
        cols_val = cols_merge + ["percentil_precio_absoluto_dentro_tipo", "flag_cold_start_tipo"]
        df_res = df_res.merge(loc_stats[cols_val], on=cols_merge, how="left")
        df_res["percentil_precio_absoluto_dentro_tipo"] = df_res["percentil_precio_absoluto_dentro_tipo"].fillna(0.50)
        df_res["flag_cold_start_tipo"] = df_res["flag_cold_start_tipo"].fillna(1).astype(int)
    else:
        df_res["percentil_precio_absoluto_dentro_tipo"] = 0.50
        df_res["flag_cold_start_tipo"] = 1

    return df_res


def filtrar_consistencia_localidades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica filtros de consistencia para el análisis y modelado de localidades:
    - dn_quota > 0 (la localidad debe tener capacidad física asignada).
    - performance_quota > 0 (el evento debe tener aforo registrado).
    - med_unit_amt_itx >= 0 (precios no negativos).
    - net_sold_p_qty >= 0 y net_sold_c_qty >= 0 (cantidades no negativas).
    - Consistencia física del evento: suma(dn_quota) == performance_quota por t_performance_id.
    """
    # 1. Filtros básicos de validez física y monetaria
    df_clean = df[
        (df["dn_quota"] > 0) &
        (df["performance_quota"] > 0) &
        (df["med_unit_amt_itx"] >= 0) &
        (df["net_sold_p_qty"] >= 0) &
        (df["net_sold_c_qty"] >= 0)
    ].copy()
    
    # 2. Validar que la suma del aforo de las localidades activas sea igual al aforo total del evento
    suma_quota_evento = df_clean.groupby("t_performance_id")["dn_quota"].transform("sum")
    df_clean = df_clean[suma_quota_evento == df_clean["performance_quota"]].copy()
    
    return df_clean


def calcular_metricas_relativas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula variables relativas normalizadas dentro de cada evento (t_performance_id):
    - peso_aforo: Capacidad relativa de la localidad frente al aforo total del evento.
    - ratio_precio_max: Precio relativo frente a la localidad más cara del evento.
    - ratio_precio_mean: Precio relativo frente al promedio del evento.
    - percentil_precio_evento: Posición relativa de precio (0.0 a 1.0) en la función.
    - tasa_ocupacion: Absorción total de aforo (pagas + cortesías / aforo).
    - tasa_venta_paga: Absorción comercial (pagas / aforo).
    - ratio_cortesias: Proporción de cortesías sobre el total vendido.
    """
    df_res = df.copy()
    
    # 1. Peso de aforo relativo (% de capacidad del evento)
    df_res["peso_aforo"] = np.clip(df_res["dn_quota"] / df_res["performance_quota"], 0.0, 1.0)
    
    # 2. Métricas de precio relativo dentro de cada función/evento
    evento_max_precio = df_res.groupby("t_performance_id")["med_unit_amt_itx"].transform("max")
    evento_mean_precio = df_res.groupby("t_performance_id")["med_unit_amt_itx"].transform("mean")
    
    # Ratio frente al precio máximo
    df_res["ratio_precio_max"] = np.where(
        evento_max_precio > 0, 
        df_res["med_unit_amt_itx"] / evento_max_precio, 
        0.0
    )
    
    # Ratio frente al precio promedio
    df_res["ratio_precio_mean"] = np.where(
        evento_mean_precio > 0, 
        df_res["med_unit_amt_itx"] / evento_mean_precio, 
        1.0
    )
    
    # Percentil relativo de precio dentro de la misma función (0 a 1)
    df_res["percentil_precio_evento"] = df_res.groupby("t_performance_id")["med_unit_amt_itx"].rank(pct=True)
    
    # 3. Métricas de absorción de demanda y ocupación
    total_vendido = df_res["net_sold_p_qty"] + df_res["net_sold_c_qty"]
    df_res["tasa_ocupacion"] = np.clip(total_vendido / df_res["dn_quota"], 0.0, 1.0)
    df_res["tasa_venta_paga"] = np.clip(df_res["net_sold_p_qty"] / df_res["dn_quota"], 0.0, 1.0)
    
    df_res["ratio_cortesias"] = np.where(
        total_vendido > 0,
        df_res["net_sold_c_qty"] / total_vendido,
        0.0
    )
    
    return df_res


def preparar_dataset_enriquecido(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el pipeline de enriquecimiento completo:
    1. Filtrado de consistencia.
    2. Adjuntar categoría estandarizada de venue (type_site).
    3. Cálculo de métricas relativas por evento.
    4. Extracción de variables semánticas, espaciales y limpieza de texto NLP.
    """
    print("1. Aplicando filtros de consistencia...")
    df_clean = filtrar_consistencia_localidades(df)
    
    print("2. Adjuntando tipología estandarizada de venue (type_site)...")
    df_site = adjuntar_tipo_venue(df_clean)

    print("3. Calculando métricas numéricas relativas por evento...")
    df_rel = calcular_metricas_relativas(df_site)

    print("4. Calculando percentil de precio absoluto dentro de type_site...")
    df_pct = calcular_percentil_precio_absoluto_dentro_tipo(df_rel)
    
    print("5. Extrayendo variables estructurales y limpiando texto NLP...")
    df_final = pipeline_procesamiento_nlp(df_pct, col_nombre="logical_seat_category")
    
    print(f" Dataset enriquecido listo: {len(df_final):,} filas y {len(df_final.columns)} columnas.")
    return df_final

