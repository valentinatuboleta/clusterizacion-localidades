"""
Script de Re-entrenamiento y Comparación v2.3 -> v2.4 (Bloque 2).

Re-entrena el pipeline_clustering_dos_etapas bajo la nueva taxonomía de 10 categorías:
- k=5, omega_nlp=0.2, omega_venue=0.5
- One-hot venue de 10 categorías (espacio estructural 21D + 15 NLP = 36D)
- Referencia de percentil por tipo regenerada con 10 categorías
- MODEL_VERSION = "2.4"
- Payload persistido en data/processed/modelo_clustering_v2_5.joblib

Compara cuantitativamente contra v2.3:
- Silueta y Davies-Bouldin
- Benchmark ARI entre v2.3 y v2.4
- Distribución de arquetipos y matriz de transición
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from src.feature_engineering import preparar_dataset_enriquecido
from src.clustering import (
    pipeline_clustering_dos_etapas,
    guardar_modelo_clustering,
    cargar_modelo_clustering,
    predecir_arquetipos_demanda,
    CANONICAL_TYPE_SITE_CATEGORIES,
    MODEL_VERSION,
    DISTRIBUCION_ESPERADA_ARQUETIPOS
)


def main():
    print("=" * 85)
    print("      RE-ENTRENAMIENTO DEL MODELO DE CLUSTERIZACION v2.4 (TAXONOMIA 10 CATEGORIAS)")
    print("=" * 85)

    # 1. Dataset enriquecido con el lookup v2
    print("\n1. Cargando y preparando dataset enriquecido con taxonomía v2...")
    df_raw = pd.read_parquet("data/raw/localidades_eda.parquet")
    df_enr = preparar_dataset_enriquecido(df_raw)
    total_locs = len(df_enr)
    print(f"Total localidades enriquecidas: {total_locs:,}")

    # 2. Entrenar pipeline v2.4
    print("\n2. Entrenando pipeline dos etapas v2.4 (k=5, w_nlp=0.2, w_venue=0.5, 36D)...")
    cols_num = ["ratio_precio_max", "percentil_precio_evento", "peso_aforo", "percentil_precio_absoluto_dentro_tipo"]
    df_v24, km_v24, sc_v24, tf_v24, fn_v24, met_v24 = pipeline_clustering_dos_etapas(
        df_enr,
        n_clusters_multizona=5,
        random_state=42,
        peso_nlp=0.2,
        peso_type_site=0.5,
        columnas_numericas=cols_num,
        categorias_type_site=CANONICAL_TYPE_SITE_CATEGORIES
    )

    print(f"Dimensiones del espacio vectorial v2.4: {len(fn_v24)}")
    print(f"Silueta v2.4 (multi-zona): {met_v24['silhouette_score']:.4f}")
    print(f"Davies-Bouldin v2.4 (multi-zona): {met_v24['davies_bouldin']:.4f}")

    # 3. Guardar modelo v2.4
    ruta_modelo_v24 = "data/processed/modelo_clustering_v2_5.joblib"
    print(f"\n3. Persistiendo payload v2.4 en {ruta_modelo_v24}...")
    mapa_arq = met_v24["mapa_arquetipos"]
    guardar_modelo_clustering(
        ruta_modelo_v24,
        kmeans=km_v24,
        scaler=sc_v24,
        tfidf_vectorizer=tf_v24,
        feature_names=fn_v24,
        mapa_arquetipos=mapa_arq,
        metricas=met_v24,
        metadata={
            "autor": "Data Science TuBoleta",
            "version": "2.4",
            "peso_nlp": 0.2,
            "peso_type_site": 0.5,
            "taxonomia_version": "v2"
        },
        gmm=met_v24.get("gmm"),
        peso_nlp=0.2,
        peso_type_site=0.5,
        df_referencia=df_enr[~df_enr["t_performance_id"].isin(
            df_v24[df_v24["cluster"] == -1]["t_performance_id"]
        )],
        categorias_type_site=CANONICAL_TYPE_SITE_CATEGORIES
    )
    print(f"[OK] Payload v2.4 guardado exitosamente.")

    # 4. Comparación con v2.3 si existe
    ruta_modelo_historico = "artifacts/archive/modelo_legacy_v23.joblib"
    if os.path.exists(ruta_modelo_historico):
        print("\n4. Comparando cuantitativamente v2.3 vs v2.4...")
        payload_v23 = cargar_modelo_clustering(ruta_modelo_historico)
        met_v23 = payload_v23.get("metricas", {})

        pred_v23 = predecir_arquetipos_demanda(df_enr, payload_v23)
        pred_v24 = predecir_arquetipos_demanda(df_enr, cargar_modelo_clustering(ruta_modelo_v24))

        # ARI Benchmark
        ari = adjusted_rand_score(pred_v23["arquetipo_demanda"], pred_v24["arquetipo_demanda"])
        tasa_coincidencia = (pred_v23["arquetipo_demanda"] == pred_v24["arquetipo_demanda"]).mean() * 100.0

        print("\n" + "=" * 85)
        print("          TABLA COMPARATIVA DE METRICAS: MODELO v2.3 vs MODELO v2.4")
        print("=" * 85)
        print(f"{'Metrica':<35} | {'v2.3 (9 cats, 35D)':<20} | {'v2.4 (10 cats, 36D)':<20}")
        print("-" * 85)
        print(f"{'Silueta Multi-Zona':<35} | {met_v23.get('silhouette_score', 0):<20.4f} | {met_v24['silhouette_score']:<20.4f}")
        print(f"{'Davies-Bouldin Multi-Zona':<35} | {met_v23.get('davies_bouldin', 0):<20.4f} | {met_v24['davies_bouldin']:<20.4f}")
        print(f"{'Calinski-Harabasz Multi-Zona':<35} | {met_v23.get('calinski_harabasz', 0):<20.1f} | {met_v24.get('calinski_harabasz', 0):<20.1f}")
        print(f"{'Inercia K-Means':<35} | {met_v23.get('inercia', 0):<20.1f} | {met_v24['inercia']:<20.1f}")
        print(f"{'ARI vs Version Anterior':<35} | {'1.0000 (base)':<20} | {ari:<20.4f}")
        print(f"{'Tasa de Coincidencia Exacta':<35} | {'100.00%':<20} | {tasa_coincidencia:<19.2f}%")
        print("=" * 85)

        # Distribución de arquetipos
        print("\n=== DISTRIBUCION DE ARQUETIPOS: v2.3 vs v2.4 ===")
        cnt_v23 = pred_v23["arquetipo_demanda"].value_counts()
        cnt_v24 = pred_v24["arquetipo_demanda"].value_counts()
        todos_arqs = sorted(list(set(cnt_v23.index).union(set(cnt_v24.index))))

        print(f"{'Arquetipo':<42} | {'v2.3 Locs':<10} | {'v2.3 %':<8} | {'v2.4 Locs':<10} | {'v2.4 %':<8}")
        print("-" * 85)
        for a in todos_arqs:
            n23 = cnt_v23.get(a, 0)
            p23 = (n23 / total_locs) * 100
            n24 = cnt_v24.get(a, 0)
            p24 = (n24 / total_locs) * 100
            print(f"{a:<42} | {n23:<10d} | {p23:<7.2f}% | {n24:<10d} | {p24:<7.2f}%")
        print("-" * 85)

        # Matriz de migración
        print("\n=== MATRIZ DE MIGRACION DE ARQUETIPOS (v2.3 Filas -> v2.4 Columnas) ===")
        matriz = pd.crosstab(
            pred_v23["arquetipo_demanda"],
            pred_v24["arquetipo_demanda"],
            rownames=["v2.3"],
            colnames=["v2.4"]
        )
        print(matriz.to_string())

        return df_v24, met_v24, ari, matriz

    return df_v24, met_v24, None, None


if __name__ == "__main__":
    main()
