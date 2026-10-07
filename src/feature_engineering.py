"""
Módulo de Ingeniería de Características (Feature Engineering) para Localidades.

Este módulo se encarga de:
1. Filtrar registros inválidos o inconsistentes.
2. Calcular métricas relativas contextualizadas por evento (percentil de precio, ratio de precio, peso de aforo).
3. Calcular métricas de comportamiento histórico de absorción de ventas (tasa de ocupación, ventas pagas vs cortesías).
4. Integrar la extracción de atributos NLP limpios para alimentar el espacio vectorial mixto.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from src.nlp_utils import pipeline_procesamiento_nlp, normalizar_venue

logger = logging.getLogger(__name__)


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
    precio_column: Optional[str] = None,
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

    if precio_column is not None:
        col_p = precio_column if precio_column in df_work.columns else "med_base_unit_amt_itx"
    else:
        col_p = "med_base_unit_amt_itx" if "med_base_unit_amt_itx" in df_work.columns else "ave_unit_amt_itx"
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
    precio_column: Optional[str] = None,
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

    if precio_column is not None and precio_column in df_res.columns:
        col_p = precio_column
    else:
        cands_p = ["med_base_unit_amt_itx", "base_unit_amt_itx", "ave_unit_amt_itx", "unit_amt_itx", "precio", "price_amount"]
        col_p = next((c for c in cands_p if c in df_res.columns), None)

    if not col_p or col_p not in df_res.columns:
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


def filtrar_consistencia_localidades(
    df: pd.DataFrame,
    col_precio: Optional[str] = None,
    permitir_precio_cero: bool = False,
    modo_legacy_v3: bool = False,
    retornar_reporte: bool = False,
    verbose: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Aplica filtros de consistencia física, monetaria y de negocio sobre el dataset de localidades:

    Reglas aplicadas (modo estándar):
    1. Categoría lógica no vacía: elimina filas con logical_seat_category nula, vacía o 'nan'.
    2. Aforo de localidad positivo: dn_quota > 0.
    3. Aforo de evento positivo: performance_quota > 0.
    4. Precio estrictamente positivo: col_precio > 0 y no nulo (precios 0, negativos o NaN se eliminan).
    5. Cantidades no negativas: net_sold_p_qty >= 0 y net_sold_c_qty >= 0.
    6. Deduplicación exacta por clave de negocio: (t_performance_id, site, logical_seat_category, dn_quota, precio).
    7. Consistencia física de evento: suma(dn_quota) == performance_quota por t_performance_id.

    Modo legacy:
        Si modo_legacy_v3=True, ejecuta la lógica histórica utilizada durante el entrenamiento original
        de v2.5 / v3.0 preservando exactamente las 33,775 filas.

    Retorna:
        pd.DataFrame filtrado (por defecto) o (pd.DataFrame, dict con conteos por regla) si retornar_reporte=True.
        df.attrs['reporte_filtro'] contiene siempre el diccionario de métricas.
    """
    if df.empty:
        reporte_vacio = {
            "total_inicial": 0,
            "regla_categoria_no_vacia": 0,
            "regla_dn_quota_positiva": 0,
            "regla_performance_quota_positiva": 0,
            "regla_precio_positivo": 0,
            "regla_cantidades_no_negativas": 0,
            "regla_deduplicacion_clave_negocio": 0,
            "regla_consistencia_aforo_evento": 0,
            "total_eliminadas": 0,
            "total_final": 0
        }
        df_vacio = df.copy()
        df_vacio.attrs["reporte_filtro"] = reporte_vacio
        return (df_vacio, reporte_vacio) if retornar_reporte else df_vacio

    if modo_legacy_v3:
        if col_precio is not None:
            precio_col = col_precio if col_precio in df.columns else "med_base_unit_amt_itx"
        else:
            precio_col = "med_base_unit_amt_itx" if "med_base_unit_amt_itx" in df.columns else None

        precio_valido = df[precio_col].abs() >= 0 if (precio_col and precio_col in df.columns) else True
        df_clean = df[
            (df["dn_quota"] > 0) &
            (df["performance_quota"] > 0) &
            precio_valido &
            (df["net_sold_p_qty"] >= 0) &
            (df["net_sold_c_qty"] >= 0)
        ].copy()
        if "t_performance_id" in df_clean.columns and "dn_quota" in df_clean.columns and "performance_quota" in df_clean.columns:
            suma_quota_evento = df_clean.groupby("t_performance_id")["dn_quota"].transform("sum")
            df_clean = df_clean[suma_quota_evento == df_clean["performance_quota"]].copy()
        rep_leg = {
            "total_inicial": len(df),
            "modo": "legacy_v3",
            "total_eliminadas": len(df) - len(df_clean),
            "total_final": len(df_clean)
        }
        df_clean.attrs["reporte_filtro"] = rep_leg
        return (df_clean, rep_leg) if retornar_reporte else df_clean

    df_curr = df.copy()
    n_inicial = len(df_curr)
    reporte = {"total_inicial": n_inicial}

    # 1. Categoría lógica no vacía
    n_prev = len(df_curr)
    if "logical_seat_category" in df_curr.columns:
        s_cat = df_curr["logical_seat_category"]
        mask_cat = (
            s_cat.notna() &
            (s_cat.astype(str).str.strip() != "") &
            (~s_cat.astype(str).str.strip().str.lower().isin(["nan", "none", "null"]))
        )
        df_curr = df_curr[mask_cat].copy()
    reporte["regla_categoria_no_vacia"] = n_prev - len(df_curr)

    # 2. dn_quota > 0 (capacidad física asignada)
    n_prev = len(df_curr)
    if "dn_quota" in df_curr.columns:
        df_curr = df_curr[df_curr["dn_quota"] > 0].copy()
    reporte["regla_dn_quota_positiva"] = n_prev - len(df_curr)

    # 3. performance_quota > 0 (evento con aforo registrado)
    n_prev = len(df_curr)
    if "performance_quota" in df_curr.columns:
        df_curr = df_curr[df_curr["performance_quota"] > 0].copy()
    reporte["regla_performance_quota_positiva"] = n_prev - len(df_curr)

    # 4. Precio positivo y no nulo (> 0 por defecto; >= 0 si permitir_precio_cero=True)
    n_prev = len(df_curr)
    if col_precio is not None and col_precio in df_curr.columns:
        precio_col = col_precio
    else:
        candidatos_precio = ["med_base_unit_amt_itx", "base_unit_amt_itx", "unit_amt_itx", "precio", "price_amount"]
        precio_col = next((c for c in candidatos_precio if c in df_curr.columns), None)

    if precio_col:
        s_precio = df_curr[precio_col]
        if permitir_precio_cero:
            mask_precio = s_precio.notna() & (s_precio >= 0)
        else:
            mask_precio = s_precio.notna() & (s_precio > 0)
        df_curr = df_curr[mask_precio].copy()
    else:
        logger.warning(
            "Columna de precio no encontrada en el DataFrame. Se omite la regla de precio positivo."
        )
    reporte["regla_precio_positivo"] = n_prev - len(df_curr)

    # 5. Cantidades no negativas
    n_prev = len(df_curr)
    cond_cant = pd.Series(True, index=df_curr.index)
    if "net_sold_p_qty" in df_curr.columns:
        cond_cant &= (df_curr["net_sold_p_qty"] >= 0)
    if "net_sold_c_qty" in df_curr.columns:
        cond_cant &= (df_curr["net_sold_c_qty"] >= 0)
    df_curr = df_curr[cond_cant].copy()
    reporte["regla_cantidades_no_negativas"] = n_prev - len(df_curr)

    # 6. Deduplicación exacta por clave de negocio: (t_performance_id, site, logical_seat_category, dn_quota, precio)
    n_prev = len(df_curr)
    candidatos_clave = ["t_performance_id", "site", "logical_seat_category", "dn_quota"]
    if precio_col and precio_col in df_curr.columns:
        candidatos_clave.append(precio_col)
    cols_clave = [c for c in candidatos_clave if c in df_curr.columns]

    if cols_clave:
        df_curr = df_curr.drop_duplicates(subset=cols_clave).copy()
    reporte["regla_deduplicacion_clave_negocio"] = n_prev - len(df_curr)

    # 7. Consistencia física del evento: suma(dn_quota) == performance_quota por t_performance_id
    n_prev = len(df_curr)
    if "t_performance_id" in df_curr.columns and "dn_quota" in df_curr.columns and "performance_quota" in df_curr.columns:
        suma_quota_evento = df_curr.groupby("t_performance_id")["dn_quota"].transform("sum")
        df_curr = df_curr[suma_quota_evento == df_curr["performance_quota"]].copy()
    reporte["regla_consistencia_aforo_evento"] = n_prev - len(df_curr)

    # Totales y metadatos
    n_final = len(df_curr)
    total_eliminadas = n_inicial - n_final
    reporte["total_eliminadas"] = total_eliminadas
    reporte["total_final"] = n_final

    df_curr.attrs["reporte_filtro"] = reporte

    if verbose:
        print("=== REPORTE DE FILTRADO DE CONSISTENCIA (filtrar_consistencia_localidades) ===")
        print(f" • Registros iniciales               : {n_inicial:,}")
        print(f" • Regla 1 (Categoría no vacía)      : -{reporte['regla_categoria_no_vacia']:,}")
        print(f" • Regla 2 (dn_quota > 0)            : -{reporte['regla_dn_quota_positiva']:,}")
        print(f" • Regla 3 (performance_quota > 0)   : -{reporte['regla_performance_quota_positiva']:,}")
        print(f" • Regla 4 (Precio > 0)              : -{reporte['regla_precio_positivo']:,}")
        print(f" • Regla 5 (Cantidades >= 0)         : -{reporte['regla_cantidades_no_negativas']:,}")
        print(f" • Regla 6 (Deduplicación clave)     : -{reporte['regla_deduplicacion_clave_negocio']:,}")
        print(f" • Regla 7 (Consistencia aforo)      : -{reporte['regla_consistencia_aforo_evento']:,}")
        print(f" • Total registros eliminados        : -{total_eliminadas:,} ({total_eliminadas/n_inicial*100:.2f}%)")
        print(f" • Registros limpios finales         : {n_final:,}")
        print("==============================================================================")

    if retornar_reporte:
        return df_curr, reporte
    return df_curr


def calcular_metricas_relativas(
    df: pd.DataFrame,
    col_precio: Optional[str] = None
) -> pd.DataFrame:
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
    if col_precio is not None and col_precio in df_res.columns:
        col_p = col_precio
    else:
        cands_p = ["med_base_unit_amt_itx", "base_unit_amt_itx", "ave_unit_amt_itx", "unit_amt_itx", "precio", "price_amount"]
        col_p = next((c for c in cands_p if c in df_res.columns), "med_base_unit_amt_itx")
    s_precio = df_res[col_p].abs() if col_p in df_res.columns else pd.Series(0.0, index=df_res.index)
    
    evento_max_precio = df_res.groupby("t_performance_id")[col_p].transform(lambda x: x.abs().max())
    evento_mean_precio = df_res.groupby("t_performance_id")[col_p].transform(lambda x: x.abs().mean())
    
    # Ratio frente al precio máximo
    df_res["ratio_precio_max"] = np.where(
        evento_max_precio > 0, 
        s_precio / evento_max_precio, 
        0.0
    )
    
    # Ratio frente al precio promedio
    df_res["ratio_precio_mean"] = np.where(
        evento_mean_precio > 0, 
        s_precio / evento_mean_precio, 
        1.0
    )
    
    # Percentil relativo de precio dentro de la misma función (0 a 1)
    df_res["percentil_precio_evento"] = df_res.groupby("t_performance_id")[col_p].rank(pct=True)
    
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


def preparar_dataset_enriquecido(
    df: pd.DataFrame,
    col_precio: Optional[str] = None,
    permitir_precio_cero: bool = False,
    modo_legacy_v3: bool = False
) -> pd.DataFrame:
    """
    Ejecuta el pipeline de enriquecimiento completo:
    1. Filtrado de consistencia.
    2. Adjuntar categoría estandarizada de venue (type_site).
    3. Cálculo de métricas relativas por evento (soporta med_base_unit_amt_itx).
    4. Cálculo de percentil de precio absoluto dentro de type_site.
    5. Extracción de variables semánticas, espaciales y limpieza de texto NLP.
    """
    print("1. Aplicando filtros de consistencia...")
    df_clean = filtrar_consistencia_localidades(
        df,
        col_precio=col_precio,
        permitir_precio_cero=permitir_precio_cero,
        modo_legacy_v3=modo_legacy_v3
    )
    
    print("2. Adjuntando tipología estandarizada de venue (type_site)...")
    df_site = adjuntar_tipo_venue(df_clean)

    print("3. Calculando métricas numéricas relativas por evento...")
    df_rel = calcular_metricas_relativas(df_site, col_precio=col_precio)

    print("4. Calculando percentil de precio absoluto dentro de type_site...")
    df_pct = calcular_percentil_precio_absoluto_dentro_tipo(df_rel, precio_column=col_precio)
    
    print("5. Extrayendo variables estructurales y limpiando texto NLP...")
    df_final = pipeline_procesamiento_nlp(df_pct, col_nombre="logical_seat_category")
    
    print(f" Dataset enriquecido listo: {len(df_final):,} filas y {len(df_final.columns)} columnas.")
    return df_final

