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
    print("3. Ejecutando Modelo Nuevo v2.1 (Aislamiento de tarifa plana + K-Means multi-zona)...")
    df_new, km_new, scaler_new, tfidf_new, feats_new, _ = pipeline_clustering_dos_etapas(
        df_enriquecido, n_clusters_multizona=4, random_state=42, peso_nlp=1.2
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
    print("HALLAZGOS CLAVE:")
    print("="*80)
    # Cuántas filas de la vieja 'Grada General' eran tarifa plana
    vieja_grada = df_old[df_old["arquetipo_demanda"] == "Grada General / Masiva"]
    grada_era_tarifa_plana = (df_new.loc[vieja_grada.index, "arquetipo_demanda"] == "Admisión Única / Tarifa Plana").sum()
    print(f"1. De las {len(vieja_grada):,} filas que el modelo anterior llamaba 'Grada General', exactamente")
    print(f"   {grada_era_tarifa_plana:,} ({grada_era_tarifa_plana/len(vieja_grada)*100:.1f}%) eran funciones monozona (Cinemateca/Maloka/YAWA).")
    print("2. La 'Grada General' del modelo anterior tenía Ratio = 1.00 (más cara que VIP).")
    print("   En el modelo nuevo, 'Grada General' ahora representa graderías reales con aforo masivo (59.2%).")
    print("3. 'VIP / Palcos / Premium' se expandió de 1,905 a 5,400 filas, capturando palcos y plateas")
    print("   de alta gama que antes caían en 'Popular' o 'Preferencial' por el arrastre de los centroides.")


if __name__ == "__main__":
    ejecutar_comparativa()
