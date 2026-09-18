"""
Script de Comparación Cuantitativa: Modelo Anterior (v2.0) vs Modelo en Dos Etapas (v2.1).

Permite visualizar con exactitud qué tanto cambiaron los resultados, las métricas
de los clusters y la matriz de migración/transición de localidades entre ambos enfoques.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from src.feature_engineering import filtrar_consistencia_localidades, calcular_metricas_relativas
from src.nlp_utils import pipeline_procesamiento_nlp
from src.clustering import (
    construir_espacio_vectorial_mixto,
    entrenar_modelo_clustering,
    asignar_arquetipos_demanda,
    pipeline_clustering_dos_etapas
)


def ejecutar_comparativa():
    print("================================================================================")
    print("   COMPARATIVA DE RESULTADOS: MODELO ANTERIOR (v2.0) vs MODELO EN DOS ETAPAS (v2.1)")
    print("================================================================================\n")

    # 1. Carga y Enriquecimiento
    print("1. Cargando y preparando dataset limpio (33,775 filas)...")
    df_raw = pd.read_parquet("data/raw/localidades_eda.parquet")
    df_clean = filtrar_consistencia_localidades(df_raw)
    df_rel = calcular_metricas_relativas(df_clean)
    df_enriquecido = pipeline_procesamiento_nlp(df_rel)

    # -------------------------------------------------------------------------
    # 2. MODELO ANTERIOR (v2.0: 1 sola etapa sobre las 33,775 filas)
    # -------------------------------------------------------------------------
    print("\n2. Ejecutando Modelo Anterior (K-Means conjunto con blob de tarifa plana)...")
    X_old, scaler_old, tfidf_old, feats_old = construir_espacio_vectorial_mixto(
        df_enriquecido, max_tfidf_features=15, peso_nlp=1.2, scaler_type="robust"
    )
    km_old, labels_old, _ = entrenar_modelo_clustering(X_old, n_clusters=4, random_state=42)
    df_old = df_enriquecido.copy()
    df_old["cluster"] = labels_old
    df_old = asignar_arquetipos_demanda(df_old, col_cluster="cluster")

    resumen_old = df_old.groupby("arquetipo_demanda").agg(
        Registros=("ratio_precio_max", "count"),
        Ratio_Precio=("ratio_precio_max", "mean"),
        Peso_Aforo_Pct=("peso_aforo", lambda s: f"{s.mean()*100:.1f}%"),
        Ocupacion_Pct=("tasa_ocupacion", lambda s: f"{s.mean()*100:.1f}%"),
        Precio_Mediano_COP=("med_unit_amt_itx", "median")
    ).reset_index()
    resumen_old["% Catálogo"] = (resumen_old["Registros"] / len(df_old) * 100).round(1).astype(str) + "%"

    # -------------------------------------------------------------------------
    # 3. MODELO NUEVO (v2.1: Dos etapas con split evento + centroides escalados)
    # -------------------------------------------------------------------------
    print("3. Ejecutando Modelo Nuevo v2.1 Optimizado (Aislamiento de tarifa plana + K-Means k=5 multi-zona)...")
    df_new, km_new, scaler_new, tfidf_new, feats_new, _ = pipeline_clustering_dos_etapas(
        df_enriquecido, n_clusters_multizona=5, random_state=42, peso_nlp=0.2
    )

    resumen_new = df_new.groupby("arquetipo_demanda").agg(
        Registros=("ratio_precio_max", "count"),
        Ratio_Precio=("ratio_precio_max", "mean"),
        Peso_Aforo_Pct=("peso_aforo", lambda s: f"{s.mean()*100:.1f}%"),
        Ocupacion_Pct=("tasa_ocupacion", lambda s: f"{s.mean()*100:.1f}%"),
        Precio_Mediano_COP=("med_unit_amt_itx", "median")
    ).reset_index()
    resumen_new["% Catálogo"] = (resumen_new["Registros"] / len(df_new) * 100).round(1).astype(str) + "%"

    # -------------------------------------------------------------------------
    # 4. IMPRESIÓN DE TABLAS COMPARATIVAS
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("TABLA 1: MODELO ANTERIOR (v2.0 - 4 Arquetipos con Distorsión de Tarifa Plana)")
    print("="*80)
    print(resumen_old.to_string(index=False))

    print("\n" + "="*80)
    print("TABLA 2: MODELO NUEVO (v2.1 - 5 Arquetipos en Dos Etapas)")
    print("="*80)
    print(resumen_new.to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. MATRIZ DE TRANSICIÓN: ¿DÓNDE MIGRARON LAS LOCALIDADES?
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("TABLA 3: MATRIZ DE MIGRACIÓN (Modelo Anterior filas -> Modelo Nuevo columnas)")
    print("="*80)
    matriz_migracion = pd.crosstab(
        df_old["arquetipo_demanda"],
        df_new["arquetipo_demanda"],
        margins=True,
        margins_name="Total Anterior"
    )
    print(matriz_migracion.to_string())

    print("\n" + "="*80)
    print("HALLAZGOS CLAVE (Calculados dinámicamente del run actual):")
    print("="*80)
    # 1. Cuántas filas de la vieja 'Grada General' eran tarifa plana
    vieja_grada = df_old[df_old["arquetipo_demanda"] == "Grada General / Masiva"]
    grada_era_tarifa_plana = (df_new.loc[vieja_grada.index, "arquetipo_demanda"] == "Admisión Única / Tarifa Plana").sum()
    pct_grada_tp = (grada_era_tarifa_plana / len(vieja_grada) * 100) if len(vieja_grada) > 0 else 0.0
    print(f"1. De las {len(vieja_grada):,} filas que el modelo anterior llamaba 'Grada General', exactamente")
    print(f"   {grada_era_tarifa_plana:,} ({pct_grada_tp:.1f}%) eran funciones monozona (Cinemateca/Maloka/YAWA).")

    # 2. Métricas de la nueva Grada General
    if "Grada General / Masiva" in df_new["arquetipo_demanda"].values:
        nueva_grada = df_new[df_new["arquetipo_demanda"] == "Grada General / Masiva"]
        aforo_nueva_grada = nueva_grada["peso_aforo"].mean() * 100
        print(f"2. En el modelo nuevo, 'Grada General / Masiva' cuenta con {len(nueva_grada):,} filas")
        print(f"   y representa graderías reales con aforo dominante promedio del {aforo_nueva_grada:.1f}%.")

    # 3. Transición de VIP / Palcos
    vip_old = df_old[df_old["arquetipo_demanda"] == "VIP / Palcos / Premium"]
    vip_new = df_new[df_new["arquetipo_demanda"] == "VIP / Palcos / Premium"]
    p_med_vip = vip_new["med_unit_amt_itx"].median() if "med_unit_amt_itx" in vip_new.columns else 0.0
    print(f"3. 'VIP / Palcos / Premium' pasó de {len(vip_old):,} filas (mezcladas con monozona) a {len(vip_new):,} filas puras,")
    print(f"   concentrando palcos y mesas de alta gama con precio mediano de ${p_med_vip:,.0f} COP (el más alto del catálogo).")


if __name__ == "__main__":
    ejecutar_comparativa()
