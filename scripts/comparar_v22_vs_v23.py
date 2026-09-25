"""
Script de Comparacion Cuantitativa y Validacion de Migracion: Modelo v2.2 vs v2.3 (3 Brazos).

Evalua la ablacion honesta y descompuesta en tres brazos metodologicos:
1. Brazo 1: v2.2 Base (25D) -> 3 numericas ex-ante + 7 tags estructurales + 15 TF-IDF (peso_type_site=0.0).
2. Brazo 2: v2.2 + Percentil Tipo (26D) -> 4 numericas (con percentil_precio_absoluto_dentro_tipo) + 7 tags + 15 TF-IDF (peso_type_site=0.0).
3. Brazo 3: v2.3 Completo (35D) -> 4 numericas + 7 tags + 9 one-hot venue (peso_type_site=0.5) + 15 TF-IDF.

Adicionalmente evalua:
- Matrices de migracion descompuestas (v2.2 -> +Percentil -> v2.3 Completo).
- Caso de negocio: General cara en teatro pequeno que migra a Platea/Preferencial.
- Exploracion de granularidad k in {3..12} en el espacio 35D.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from src.feature_engineering import preparar_dataset_enriquecido
from src.clustering import (
    pipeline_clustering_dos_etapas,
    construir_espacio_vectorial_mixto,
    CANONICAL_TYPE_SITE_CATEGORIES,
    guardar_modelo_clustering,
    evaluar_rango_k
)


def ejecutar_comparativa_v22_vs_v23():
    print("=" * 85)
    print("   COMPARATIVA DE ABLACION METODOLOGICA: v2.2 (25D) -> +PERCENTIL (26D) -> v2.3 (35D)")
    print("=" * 85)

    # 1. Dataset enriquecido
    print("\n1. Cargando y preparando dataset completo...")
    df_raw = pd.read_parquet("data/raw/localidades_eda.parquet")
    df_enr = preparar_dataset_enriquecido(df_raw)
    total = len(df_enr)

    # 2. Brazo 1: Modelo v2.2 Base Real (25D: 3 numericas, 7 tags, 15 tf-idf, peso_type_site=0.0)
    print("\n2. Entrenando Brazo 1: v2.2 Base Real (25D: 3 numericas, sin venue)...")
    cols_v22 = ["ratio_precio_max", "percentil_precio_evento", "peso_aforo"]
    df_v22, km_v22, sc_v22, tf_v22, fn_v22, met_v22 = pipeline_clustering_dos_etapas(
        df_enr,
        n_clusters_multizona=5,
        random_state=42,
        peso_nlp=0.2,
        peso_type_site=0.0,
        columnas_numericas=cols_v22
    )

    # 3. Brazo 2: v2.2 + Percentil Tipo (26D: 4 numericas, 7 tags, 15 tf-idf, peso_type_site=0.0)
    print("\n3. Entrenando Brazo 2: v2.2 + Percentil Tipo (26D: 4 numericas, sin one-hot venue)...")
    cols_v26 = ["ratio_precio_max", "percentil_precio_evento", "peso_aforo", "percentil_precio_absoluto_dentro_tipo"]
    df_v26, km_v26, sc_v26, tf_v26, fn_v26, met_v26 = pipeline_clustering_dos_etapas(
        df_enr,
        n_clusters_multizona=5,
        random_state=42,
        peso_nlp=0.2,
        peso_type_site=0.0,
        columnas_numericas=cols_v26
    )

    # 4. Brazo 3: Modelo v2.3 Completo (35D: 4 numericas + 7 tags + 9 one-hot venue peso=0.5 + 15 tf-idf)
    print("\n4. Entrenando Brazo 3: v2.3 Completo (35D: 4 numericas + one-hot venue peso=0.5)...")
    df_v23, km_v23, sc_v23, tf_v23, fn_v23, met_v23 = pipeline_clustering_dos_etapas(
        df_enr,
        n_clusters_multizona=5,
        random_state=42,
        peso_nlp=0.2,
        peso_type_site=0.5,
        columnas_numericas=cols_v26
    )

    # Guardar modelo de produccion v2.3 con metadatos completos y consistentes
    mapa_arq = met_v23["mapa_arquetipos"]
    guardar_modelo_clustering(
        "data/processed/modelo_clustering_v2_3.joblib",
        kmeans=km_v23,
        scaler=sc_v23,
        tfidf_vectorizer=tf_v23,
        feature_names=fn_v23,
        mapa_arquetipos=mapa_arq,
        metricas=met_v23,
        metadata={
            "autor": "Data Science TuBoleta",
            "version": "2.3",
            "peso_nlp": 0.2,
            "peso_type_site": 0.5
        },
        gmm=met_v23.get("gmm"),
        peso_nlp=0.2,
        peso_type_site=0.5,
        df_referencia=df_enr[~df_enr["t_performance_id"].isin(
            df_v23[df_v23["cluster"] == -1]["t_performance_id"]
        )]
    )

    # 5. Tabla de metricas lado a lado de 3 brazos
    print("\n" + "=" * 85)
    print("TABLA 1: ABLACION DE METRICAS INTRINSECAS DE AGRUPAMIENTO (MULTI-ZONA)")
    print("=" * 85)
    tabla_metricas = pd.DataFrame([
        {
            "Brazo Metodológico": "1. v2.2 Base (3 num, 7 tags, 15 tfidf)",
            "Dimensiones": len(fn_v22),
            "Silhouette Score": round(met_v22.get("silhouette_score", 0), 4),
            "Davies-Bouldin": round(met_v22.get("davies_bouldin", 0), 4),
            "Calinski-Harabasz": round(met_v22.get("calinski_harabasz", 0), 1),
            "Inercia": round(met_v22.get("inercia", 0), 1)
        },
        {
            "Brazo Metodológico": "2. v2.2 + Percentil Tipo (4 num, 7 tags, 15 tfidf)",
            "Dimensiones": len(fn_v26),
            "Silhouette Score": round(met_v26.get("silhouette_score", 0), 4),
            "Davies-Bouldin": round(met_v26.get("davies_bouldin", 0), 4),
            "Calinski-Harabasz": round(met_v26.get("calinski_harabasz", 0), 1),
            "Inercia": round(met_v26.get("inercia", 0), 1)
        },
        {
            "Brazo Metodológico": "3. v2.3 Completo (4 num, 7 tags, 9 venue, 15 tfidf)",
            "Dimensiones": len(fn_v23),
            "Silhouette Score": round(met_v23.get("silhouette_score", 0), 4),
            "Davies-Bouldin": round(met_v23.get("davies_bouldin", 0), 4),
            "Calinski-Harabasz": round(met_v23.get("calinski_harabasz", 0), 1),
            "Inercia": round(met_v23.get("inercia", 0), 1)
        }
    ])
    print(tabla_metricas.to_string(index=False))

    # 6. Distribucion de Arquetipos en los 3 brazos
    print("\n" + "=" * 85)
    print("TABLA 2: DISTRIBUCION DE ARQUETIPOS EN LOS 3 BRAZOS")
    print("=" * 85)
    dist_v22 = df_v22["arquetipo_demanda"].value_counts()
    dist_v26 = df_v26["arquetipo_demanda"].value_counts()
    dist_v23 = df_v23["arquetipo_demanda"].value_counts()
    df_dist = pd.DataFrame({
        "v2.2 Base (25D)": dist_v22,
        "Share v2.2": (dist_v22 / total * 100).round(2).astype(str) + "%",
        "+Percentil (26D)": dist_v26,
        "Share 26D": (dist_v26 / total * 100).round(2).astype(str) + "%",
        "v2.3 Full (35D)": dist_v23,
        "Share v2.3": (dist_v23 / total * 100).round(2).astype(str) + "%",
        "Delta Total (v2.3-v2.2)": dist_v23 - dist_v22
    }).fillna(0)
    print(df_dist.to_string())

    # 7. Matriz de Migracion Total: v2.2 Base (25D) -> v2.3 Completo (35D)
    print("\n" + "=" * 85)
    print("TABLA 3A: MATRIZ DE MIGRACION TOTAL (Filas: v2.2 Base 25D -> Columnas: v2.3 Full 35D)")
    print("=" * 85)
    matriz_total = pd.crosstab(
        df_v22["arquetipo_demanda"],
        df_v23["arquetipo_demanda"],
        margins=True,
        margins_name="Total v2.2"
    )
    print(matriz_total.to_string())

    # 7b. Matriz de Descomposicion: v2.2 Base (25D) -> +Percentil Tipo (26D)
    print("\n" + "=" * 85)
    print("TABLA 3B: DESCOMPOSICION 1: EFECTO DEL PERCENTIL DE PRECIO POR TIPO (25D -> 26D)")
    print("=" * 85)
    matriz_p1 = pd.crosstab(
        df_v22["arquetipo_demanda"],
        df_v26["arquetipo_demanda"],
        margins=True,
        margins_name="Total v2.2"
    )
    print(matriz_p1.to_string())

    # 7c. Matriz de Descomposicion: +Percentil Tipo (26D) -> v2.3 Completo (35D)
    print("\n" + "=" * 85)
    print("TABLA 3C: DESCOMPOSICION 2: EFECTO DEL ONE-HOT DE TYPE_SITE (26D -> 35D)")
    print("=" * 85)
    matriz_p2 = pd.crosstab(
        df_v26["arquetipo_demanda"],
        df_v23["arquetipo_demanda"],
        margins=True,
        margins_name="Total 26D"
    )
    print(matriz_p2.to_string())

    # 8. Caso Especial Cualitativo: General cara en teatro pequeno
    print("\n" + "=" * 85)
    print("CASO ESPECIAL: LOCALIDADES 'GENERAL' EN TEATROS / AUDITORIOS")
    print("=" * 85)
    mask_teatro_gen = (
        (df_enr["type_site"].isin(["teatro", "auditorio"])) &
        (df_enr["texto_limpio"].str.contains(r"\bGENERAL\b", regex=True, na=False)) &
        (df_enr["percentil_precio_absoluto_dentro_tipo"] > 0.60) &
        (df_v22["cluster"] != -1)
    )
    casos_teatro_gen = pd.DataFrame({
        "site": df_enr.loc[mask_teatro_gen, "site"],
        "localidad": df_enr.loc[mask_teatro_gen, "logical_seat_category"],
        "precio": df_enr.loc[mask_teatro_gen, "med_unit_amt_itx"],
        "percentil_tipo": df_enr.loc[mask_teatro_gen, "percentil_precio_absoluto_dentro_tipo"].round(3),
        "arquetipo_v22": df_v22.loc[mask_teatro_gen, "arquetipo_demanda"],
        "arquetipo_v26": df_v26.loc[mask_teatro_gen, "arquetipo_demanda"],
        "arquetipo_v23": df_v23.loc[mask_teatro_gen, "arquetipo_demanda"]
    })
    print(f"Total casos detectados: {len(casos_teatro_gen)}")
    migrados_total = (casos_teatro_gen["arquetipo_v22"].str.contains("Popular") & 
                      casos_teatro_gen["arquetipo_v23"].isin(["Preferencial / Platea Frontal", "Platea General / Intermedia", "VIP / Palcos / Premium"]))
    print(f"Casos que migraron de Popular hacia Platea/Preferencial/VIP en v2.3: {migrados_total.sum()} ({migrados_total.mean()*100:.1f}%)")
    if len(casos_teatro_gen) > 0:
        print("\nMuestra de 5 transiciones a traves de los 3 brazos:")
        print(casos_teatro_gen[migrados_total].head(5).to_string(index=False))

    # 9. Exploracion de k in {3..12} en Espacio 35D
    print("\n" + "=" * 85)
    print("EXPLORACION METRICA DE K IN {3..12} (MULTI-ZONA v2.3 - 35D)")
    print("=" * 85)
    df_multi = df_enr[df_v23["cluster"] != -1]
    X_multi, _, _, _ = construir_espacio_vectorial_mixto(
        df_multi,
        columnas_numericas=cols_v26,
        peso_type_site=0.5,
        categorias_type_site=CANONICAL_TYPE_SITE_CATEGORIES
    )
    eval_k = evaluar_rango_k(X_multi, k_min=3, k_max=12, random_state=42, sample_size=10000)
    print(eval_k.to_string())

    return {
        "metricas": tabla_metricas,
        "distribucion": df_dist,
        "matriz_total": matriz_total,
        "eval_k": eval_k
    }


if __name__ == "__main__":
    ejecutar_comparativa_v22_vs_v23()
