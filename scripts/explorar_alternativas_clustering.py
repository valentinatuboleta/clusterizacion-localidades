"""
Script de exploración de alternativas de clustering sobre el subset multi-zona.
Evalúa:
1. K-Means con k más alto (k=4 a 12)
2. Gaussian Mixture Models (GMM) con diferentes k
3. PCA + K-Means (reducción de dimensionalidad en 25D)
4. HDBSCAN con diferentes ajustes de densidad
5. K-Means en Espacio Numérico Puro (3D) vs Espacio Mixto (25D)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Asegurar path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import filtrar_consistencia_localidades, calcular_metricas_relativas
from src.nlp_utils import pipeline_procesamiento_nlp
from src.clustering import (
    separar_admision_unica_multizona,
    construir_espacio_vectorial_mixto
)

def run():
    print("Cargando y preparando dataset...")
    df_raw = pd.read_parquet("data/raw/localidades_eda.parquet")
    df_clean = filtrar_consistencia_localidades(df_raw)
    df_rel = calcular_metricas_relativas(df_clean)
    df_enriquecido = pipeline_procesamiento_nlp(df_rel)
    
    df_mono, df_multi = separar_admision_unica_multizona(df_enriquecido)
    
    X_multi, scaler, tfidf, feature_names = construir_espacio_vectorial_mixto(
        df_multi,
        peso_nlp=1.2
    )
    
    print(f"Dataset multi-zona: {X_multi.shape[0]} filas, {X_multi.shape[1]} dimensiones.")
    
    np.random.seed(42)
    sample_size = min(10000, len(X_multi))
    idx_sample = np.random.choice(len(X_multi), sample_size, replace=False)
    X_sample = X_multi[idx_sample]
    
    # 1. Rango amplio de K-Means
    print("\n--- 1. K-MEANS CON K AMPLIO ---")
    res_km = []
    for k in [4, 5, 6, 7, 8, 9, 10, 12]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_multi)
        sil = silhouette_score(X_sample, labels[idx_sample])
        db = davies_bouldin_score(X_multi, labels)
        ch = calinski_harabasz_score(X_multi, labels)
        res_km.append({"k": k, "Silhouette": round(sil, 4), "Davies-Bouldin": round(db, 4), "Calinski-Harabasz": round(ch, 1), "Inertia": round(km.inertia_, 1)})
    df_res_km = pd.DataFrame(res_km)
    print(df_res_km.to_string(index=False))
    
    # 2. GMM
    print("\n--- 2. GAUSSIAN MIXTURE MODEL (GMM) ---")
    res_gmm = []
    for k in [4, 5, 6, 7, 8]:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3, covariance_type="diag")
        labels = gmm.fit_predict(X_multi)
        sil = silhouette_score(X_sample, labels[idx_sample])
        db = davies_bouldin_score(X_multi, labels)
        ch = calinski_harabasz_score(X_multi, labels)
        res_gmm.append({"k": k, "Silhouette": round(sil, 4), "Davies-Bouldin": round(db, 4), "Calinski-Harabasz": round(ch, 1)})
    df_res_gmm = pd.DataFrame(res_gmm)
    print(df_res_gmm.to_string(index=False))
    
    # 3. PCA + K-Means
    print("\n--- 3. REDUCCIÓN PCA + K-MEANS ---")
    pca = PCA(random_state=42)
    pca.fit(X_multi)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_90 = np.argmax(cum_var >= 0.90) + 1
    print(f"Componentes para 90% varianza: {n_90} de {X_multi.shape[1]}")
    
    for n_comp in [3, 5, 8, n_90]:
        pca_n = PCA(n_components=n_comp, random_state=42)
        X_pca = pca_n.fit_transform(X_multi)
        for k in [5, 6, 7, 8]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_pca)
            sil_pca = silhouette_score(X_pca[idx_sample], labels[idx_sample])
            sil_orig = silhouette_score(X_sample, labels[idx_sample])
            db_orig = davies_bouldin_score(X_multi, labels)
            print(f"PCA (comps={n_comp}), k={k} -> Sil(Espacio PCA)={sil_pca:.4f}, Sil(Espacio 25D)={sil_orig:.4f}, DB(25D)={db_orig:.4f}")

    # 4. HDBSCAN con diferentes escalas
    print("\n--- 4. HDBSCAN ---")
    res_hdb = []
    for min_size in [50, 100, 180, 300]:
        hdb = HDBSCAN(min_cluster_size=min_size, min_samples=15)
        labels = hdb.fit_predict(X_multi)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).mean() * 100
        mask = labels != -1
        idx_eval = [i for i in idx_sample if labels[i] != -1]
        if len(idx_eval) > 50 and n_clusters > 1:
            sil = silhouette_score(X_multi[idx_eval], labels[idx_eval])
            db = davies_bouldin_score(X_multi[mask], labels[mask])
        else:
            sil, db = np.nan, np.nan
        res_hdb.append({"min_cluster_size": min_size, "n_clusters": n_clusters, "Ruido (%)": round(noise_pct, 1), "Silhouette": round(sil, 4), "Davies-Bouldin": round(db, 4)})
    df_res_hdb = pd.DataFrame(res_hdb)
    print(df_res_hdb.to_string(index=False))

    # 5. Exploración de Features: Espacio Numérico Puro (3D)
    print("\n--- 5. CLUSTERING EN ESPACIO NUMÉRICO PURO (3D) ---")
    X_num = X_multi[:, :3]
    for k in [4, 5, 6, 7]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_num)
        sil = silhouette_score(X_num[idx_sample], labels[idx_sample])
        db = davies_bouldin_score(X_num, labels)
        print(f"K-Means en 3D Numéricas Puras: k={k} -> Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}")

    # 6. Clustering Jerárquico Aglomerativo
    print("\n--- 6. AGGLOMERATIVE CLUSTERING (Ward) ---")
    for k in [5, 6, 7]:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels_sample = agg.fit_predict(X_sample)
        sil_agg = silhouette_score(X_sample, labels_sample)
        db_agg = davies_bouldin_score(X_sample, labels_sample)
        print(f"Jerárquico (Ward): k={k} -> Silhouette={sil_agg:.4f}, Davies-Bouldin={db_agg:.4f}")

if __name__ == "__main__":
    run()
