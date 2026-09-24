"""
Módulo de Ingeniería de Características (Feature Engineering) para Localidades.

Este módulo se encarga de:
1. Filtrar registros inválidos o inconsistentes.
2. Calcular métricas relativas contextualizadas por evento (percentil de precio, ratio de precio, peso de aforo).
3. Calcular métricas de comportamiento histórico de absorción de ventas (tasa de ocupación, ventas pagas vs cortesías).
4. Integrar la extracción de atributos NLP limpios para alimentar el espacio vectorial mixto.
"""

import os
import numpy as np
import pandas as pd
from src.nlp_utils import pipeline_procesamiento_nlp


def adjuntar_tipo_venue(df: pd.DataFrame, ruta_lookup: str = "data/lookup/site_type_lookup.csv") -> pd.DataFrame:
    """
    Enriquece el DataFrame con la categoría estandarizada de venue (type_site).
    Si ya existe la columna 'type_site', retorna una copia sin alterar.
    """
    df_res = df.copy()
    if "type_site" in df_res.columns:
        return df_res

    # Permitir resolución de ruta relativa tanto desde raíz como desde notebooks/
    if not os.path.exists(ruta_lookup):
        alt_ruta = os.path.join("..", ruta_lookup)
        if os.path.exists(alt_ruta):
            ruta_lookup = alt_ruta

    if os.path.exists(ruta_lookup):
        df_lookup = pd.read_csv(ruta_lookup)
        df_res = df_res.merge(df_lookup[["site", "type_site"]], on="site", how="left")
        df_res["type_site"] = df_res["type_site"].fillna("otro")
    else:
        df_res["type_site"] = "otro"

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
    
    print("4. Extrayendo variables estructurales y limpiando texto NLP...")
    df_final = pipeline_procesamiento_nlp(df_rel, col_nombre="logical_seat_category")
    
    print(f" Dataset enriquecido listo: {len(df_final):,} filas y {len(df_final.columns)} columnas.")
    return df_final
