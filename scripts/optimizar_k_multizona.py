"""
Script de optimización formal de K para el subset multi-zona con peso_nlp=0.2.
Evalúa k en [3, 10] calculando:
- Inercia y punto de codo (método de la distancia ortogonal a la cuerda)
- Silhouette Score
- Davies-Bouldin Index (mínimo es mejor)
- Calinski-Harabasz
- Tamaño del cluster más pequeño (para evitar clusters degenerados)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import filtrar_consistencia_localidades, calcular_metricas_relativas
from src.nlp_utils import pipeline_procesamiento_nlp
from src.clustering import separar_admision_unica_multizona, construir_espacio_vectorial_mixto

def optimizar_k():
    df_raw = pd.read_parquet("data/raw/localidades_eda.parquet")
    df_clean = filtrar_consistencia_localidades(df_raw)
    df_rel = calcular_metricas_relativas(df_clean)
    df_enriquecido = pipeline_procesamiento_nlp(df_rel)
    df_mono, df_multi = separar_admision_unica_multizona(df_enriquecido)

    # Espacio con peso_nlp=0.2 (el óptimo encontrado en el sweep)
    X_multi, scaler, tfidf, feats = construir_espacio_vectorial_mixto(
        df_multi, peso_nlp=0.2, scaler_type="robust"
    )

    np.random.seed(42)
    sample_size = min(10000, len(df_multi))
    idx_sample = np.random.choice(len(df_multi), sample_size, replace=False)
    X_eval = X_multi[idx_sample]

    k_range = list(range(3, 11))
    inertias = []
    results = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = km.fit_predict(X_multi)
        inertias.append(km.inertia_)

        labels_eval = labels[idx_sample]
        sil = silhouette_score(X_eval, labels_eval)
        db = davies_bouldin_score(X_eval, labels_eval)
        ch = calinski_harabasz_score(X_eval, labels_eval)

        # Tamaño de clusters
        unique, counts = np.unique(labels, return_counts=True)
        min_cluster_pct = (counts.min() / len(labels)) * 100
        max_cluster_pct = (counts.max() / len(labels)) * 100

        results.append({
            "k": k,
            "Inercia": round(km.inertia_, 1),
            "Silhouette": round(sil, 4),
            "Davies-Bouldin": round(db, 4),
            "Calinski-Harabasz": round(ch, 1),
            "Min Cluster (%)": f"{min_cluster_pct:.1f}%",
            "Max Cluster (%)": f"{max_cluster_pct:.1f}%"
        })

    # Codo ortogonal
    P1 = np.array([k_range[0], inertias[0]])
    P2 = np.array([k_range[-1], inertias[-1]])
    distancias_codo = []
    for i, k in enumerate(k_range):
        P0 = np.array([k, inertias[i]])
        d = np.abs(np.cross(P2 - P1, P1 - P0)) / np.linalg.norm(P2 - P1)
        distancias_codo.append(d)

    k_codo_idx = np.argmax(distancias_codo)
    k_codo = k_range[k_codo_idx]

    df_res = pd.DataFrame(results)
    df_res["Distancia Codo"] = np.round(distancias_codo, 2)
    print("==================================================================")
    print("RESULTADOS DE OPTIMIZACIÓN DE K EN MULTI-ZONA (peso_nlp = 0.2)")
    print("==================================================================")
    print(df_res.to_string(index=False))
    print(f"\n-> Punto de Codo Matemático (Distancia máxima a la cuerda): k = {k_codo}")
    
    # Ranking compuesto: normalizar Silueta (max) y Davies-Bouldin (min)
    sil_arr = np.array([r["Silhouette"] for r in results])
    db_arr = np.array([r["Davies-Bouldin"] for r in results])
    
    # Score combinado normalizado: (Sil - min)/(max - min) + (max_db - DB)/(max_db - min_db)
    norm_sil = (sil_arr - sil_arr.min()) / (sil_arr.max() - sil_arr.min())
    norm_db = (db_arr.max() - db_arr) / (db_arr.max() - db_arr.min())
    score_compuesto = norm_sil + norm_db
    
    df_res["Score Compuesto"] = np.round(score_compuesto, 3)
    k_optimo_score = k_range[np.argmax(score_compuesto)]
    print(f"-> K Óptimo por Score Compuesto (Silueta + Davies-Bouldin): k = {k_optimo_score}")

if __name__ == "__main__":
    optimizar_k()
