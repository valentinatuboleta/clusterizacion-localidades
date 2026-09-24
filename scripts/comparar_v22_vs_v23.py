"""
Script de Comparacion Cuantitativa y Validacion de Migracion: Modelo v2.2 vs v2.3.

Evalua:
1. Metricas intrinsecas de agrupamiento (Silueta, Davies-Bouldin, Calinski-Harabasz).
2. Ablacion del peso de type_site: peso=0.0 vs peso=0.5.
3. Matriz de migracion / transicion de localidades entre v2.2 y v2.3.
4. Validacion cualitativa de casos clave (General cara en teatro pequeno que migra hacia Platea/Preferencial).
5. Exploracion de k in {3..12} como evidencia para la fase posterior.
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
    entrenar_modelo_clustering,
    etiquetar_por_centroides_escalados,
    CANONICAL_TYPE_SITE_CATEGORIES,
    guardar_modelo_clustering,
    evaluar_rango_k
)


def ejecutar_comparativa_v22_vs_v23():
    print("=" * 80)
    print("   COMPARATIVA DE RESULTADOS: MODELO v2.2 vs MODELO v2.3 (TYPE_SITE INTEGRADO)")
    print("=" * 80)

    # 1. Dataset enriquecido
    print("\n1. Cargando y preparando dataset completo...")
    df_raw = pd.read_parquet("data/raw/localidades_eda.parquet")
    df_enr = preparar_dataset_enriquecido(df_raw)
    total = len(df_enr)

    # 2. Modelo v2.2 (sin type_site: peso_type_site=0.0, 3 numericas ex-ante)
    print("\n2. Entrenando representacion v2.2 (Ablacion peso_type_site=0.0, 3 numericas)...")
    cols_v22 = ["ratio_precio_max", "percentil_precio_evento", "peso_aforo"]
    df_v22, km_v22, sc_v22, tf_v22, fn_v22, met_v22 = pipeline_clustering_dos_etapas(
        df_enr,
        n_clusters_multizona=5,
        random_state=42,
        peso_nlp=0.2,
        peso_type_site=0.0
    )

    # 3. Modelo v2.3 (con percentil_precio_absoluto_dentro_tipo + type_site one-hot peso=0.5)
    print("\n3. Entrenando representacion v2.3 (4 numericas + one-hot type_site peso=0.5)...")
    df_v23, km_v23, sc_v23, tf_v23, fn_v23, met_v23 = pipeline_clustering_dos_etapas(
        df_enr,
        n_clusters_multizona=5,
        random_state=42,
        peso_nlp=0.2,
        peso_type_site=0.5
    )

    # Guardar modelo de produccion v2.3
    mapa_arq = met_v23["mapa_arquetipos"]
    guardar_modelo_clustering(
        "data/processed/modelo_clustering_v2_3.joblib",
        kmeans=km_v23,
        scaler=sc_v23,
        tfidf_vectorizer=tf_v23,
        feature_names=fn_v23,
        mapa_arquetipos=mapa_arq,
        metricas=met_v23,
        metadata={"autor": "Data Science TuBoleta", "version": "2.3", "peso_type_site": 0.5},
        gmm=met_v23.get("gmm"),
        df_referencia=df_enr[~df_enr["t_performance_id"].isin(
            df_v23[df_v23["cluster"] == -1]["t_performance_id"]
        )]
    )

    # 4. Tabla de metricas lado a lado
    print("\n" + "=" * 80)
    print("TABLA 1: METRICAS INTRINSECAS DE AGRUPAMIENTO (MULTI-ZONA)")
    print("=" * 80)
    tabla_metricas = pd.DataFrame([
        {
            "Modelo": "v2.2 (Sin type_site / peso=0.0)",
            "Dimensiones": len(fn_v22),
            "Silhouette Score": met_v22.get("silhouette_score"),
            "Davies-Bouldin": met_v22.get("davies_bouldin"),
            "Calinski-Harabasz": met_v22.get("calinski_harabasz"),
            "Inercia": met_v22.get("inercia")
        },
        {
            "Modelo": "v2.3 (Con type_site / peso=0.5)",
            "Dimensiones": len(fn_v23),
            "Silhouette Score": met_v23.get("silhouette_score"),
            "Davies-Bouldin": met_v23.get("davies_bouldin"),
            "Calinski-Harabasz": met_v23.get("calinski_harabasz"),
            "Inercia": met_v23.get("inercia")
        }
    ])
    print(tabla_metricas.to_string(index=False))

    # 5. Distribucion de Arquetipos
    print("\n" + "=" * 80)
    print("TABLA 2: DISTRIBUCION DE ARQUETIPOS (v2.2 vs v2.3)")
    print("=" * 80)
    dist_v22 = df_v22["arquetipo_demanda"].value_counts()
    dist_v23 = df_v23["arquetipo_demanda"].value_counts()
    df_dist = pd.DataFrame({
        "Registros v2.2": dist_v22,
        "Share v2.2": (dist_v22 / total * 100).round(2).astype(str) + "%",
        "Registros v2.3": dist_v23,
        "Share v2.3": (dist_v23 / total * 100).round(2).astype(str) + "%",
        "Diferencia": dist_v23 - dist_v22
    }).fillna(0)
    print(df_dist.to_string())

    # 6. Matriz de Migracion
    print("\n" + "=" * 80)
    print("TABLA 3: MATRIZ DE MIGRACION (Filas: v2.2 -> Columnas: v2.3)")
    print("=" * 80)
    matriz = pd.crosstab(
        df_v22["arquetipo_demanda"],
        df_v23["arquetipo_demanda"],
        margins=True,
        margins_name="Total v2.2"
    )
    print(matriz.to_string())

    # 7. Caso Especial: General cara en teatro pequeno
    print("\n" + "=" * 80)
    print("CASO ESPECIAL: GENERALES EN TEATROS / AUDITORIOS")
    print("=" * 80)
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
        "arquetipo_v23": df_v23.loc[mask_teatro_gen, "arquetipo_demanda"]
    })
    print(f"Total casos detectados: {len(casos_teatro_gen)}")
    migrados = (casos_teatro_gen["arquetipo_v22"].str.contains("Popular") & 
                casos_teatro_gen["arquetipo_v23"].isin(["Preferencial / Platea Frontal", "Platea General / Intermedia", "VIP / Palcos / Premium"]))
    print(f"Casos que migraron de Popular hacia Platea/Preferencial/VIP: {migrados.sum()} ({migrados.mean()*100:.1f}%)")
    if len(casos_teatro_gen) > 0:
        print("\nMuestra de 5 transiciones:")
        print(casos_teatro_gen[migrados].head(5).to_string(index=False))

    # 8. Exploracion de k in {3..12}
    print("\n" + "=" * 80)
    print("EXPLORACION METRICA DE K IN {3..12} (MULTI-ZONA v2.3)")
    print("=" * 80)
    X_multi_23 = km_v23.cluster_centers_  # o evaluar sobre X
    # Separar multi-zona
    df_mono, df_multi = df_enr[df_v23["cluster"] == -1], df_enr[df_v23["cluster"] != -1]
    X_multi, _, _, _ = construir_espacio_vectorial_mixto(
        df_multi,
        peso_type_site=0.5,
        categorias_type_site=CANONICAL_TYPE_SITE_CATEGORIES
    )
    eval_k = evaluar_rango_k(X_multi, k_min=3, k_max=12, random_state=42, sample_size=10000)
    print(eval_k.to_string())


if __name__ == "__main__":
    ejecutar_comparativa_v22_vs_v23()
