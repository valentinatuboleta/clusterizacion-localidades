"""
Diagnóstico previo y benchmark avanzado de alternativas de clustering:
1. Silueta por cluster en K-Means (k=4 y k=5)
2. Sweep de pesos de NLP (peso_nlp in {0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0})
3. GMM con cálculo de BIC/AIC (k=4 a 10)
4. K-Prototypes (kmodes)
5. HDBSCAN con grid de min_cluster_size (0.5%, 1%, 2%)
6. UMAP (8D) + HDBSCAN (evaluado en 25D)
7. Sobrequipping: K=8 y K=10 mapeados a arquetipos con Hungarian matching
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
import umap
from kmodes.kprototypes import KPrototypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import filtrar_consistencia_localidades, calcular_metricas_relativas
from src.nlp_utils import pipeline_procesamiento_nlp
from src.clustering import (
    separar_admision_unica_multizona,
    construir_espacio_vectorial_mixto,
    etiquetar_por_centroides_escalados,
    construir_perfiles_ideales_escalados,
    DEFAULT_TAG_FEATURES
)

def run():
    print("==================================================================")
    print("1. CARGANDO DATOS Y PREPARANDO SUBSET MULTI-ZONA")
    print("==================================================================")
    df_raw = pd.read_parquet("data/raw/localidades_eda.parquet")
    df_clean = filtrar_consistencia_localidades(df_raw)
    df_rel = calcular_metricas_relativas(df_clean)
    df_enriquecido = pipeline_procesamiento_nlp(df_rel)
    df_mono, df_multi = separar_admision_unica_multizona(df_enriquecido)

    print(f"Total multi-zona: {len(df_multi)} filas.")

    # Base fija para evaluación justa y comparable
    np.random.seed(42)
    sample_size = min(10000, len(df_multi))
    idx_sample = np.random.choice(len(df_multi), sample_size, replace=False)

    # ------------------------------------------------------------------
    # EXPERIMENTO 1: SWEEP DE PESOS NLP (peso_nlp in {0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0})
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print("EXPERIMENTO 1: SWEEP DE PESOS NLP (Efecto de la dilución dimensional)")
    print("==================================================================")
    sweep_res = []
    for p_nlp in [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        X_p, scaler_p, tfidf_p, feats_p = construir_espacio_vectorial_mixto(
            df_multi, peso_nlp=p_nlp, scaler_type="robust"
        )
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = km.fit_predict(X_p)
        
        sil = silhouette_score(X_p[idx_sample], labels[idx_sample])
        db = davies_bouldin_score(X_p[idx_sample], labels[idx_sample])
        ch = calinski_harabasz_score(X_p[idx_sample], labels[idx_sample])
        sweep_res.append({
            "peso_nlp": p_nlp,
            "Silhouette": round(sil, 4),
            "Davies-Bouldin": round(db, 4),
            "Calinski-Harabasz": round(ch, 1),
            "Inercia": round(km.inertia_, 1)
        })
    df_sweep = pd.DataFrame(sweep_res)
    print(df_sweep.to_string(index=False))

    # Fijamos el espacio calibrado de producción (peso_nlp=0.2) para el resto del diagnóstico
    X_std, scaler_std, tfidf_std, feats_std = construir_espacio_vectorial_mixto(
        df_multi, peso_nlp=0.2, scaler_type="robust"
    )
    X_sample_std = X_std[idx_sample]

    # ------------------------------------------------------------------
    # EXPERIMENTO 2: SILUETA POR CLUSTER (Identificar el cluster débil)
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print("EXPERIMENTO 2: SILUETA POR CLUSTER (K=4 y K=5 en 25D)")
    print("==================================================================")
    for k in [4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_std)
        sil_vals = silhouette_samples(X_sample_std, labels[idx_sample])
        sample_labels = labels[idx_sample]
        print(f"\n--- Desglose de Silueta para k={k} (Global: {sil_vals.mean():.4f}) ---")
        for c in range(k):
            c_mask = sample_labels == c
            c_size = c_mask.sum()
            c_sil = sil_vals[c_mask].mean()
            # Métricas descriptivas del cluster
            ratio_p = df_multi.iloc[idx_sample][c_mask]["ratio_precio_max"].mean()
            aforo_p = df_multi.iloc[idx_sample][c_mask]["peso_aforo"].mean()
            print(f"  Cluster {c} (N={c_size}, {c_size/len(sample_labels)*100:.1f}%): Silueta={c_sil:.4f} | RatioPrecio={ratio_p:.2f} | PesoAforo={aforo_p*100:.1f}%")

    # ------------------------------------------------------------------
    # EXPERIMENTO 3: GMM CON SELECCIÓN DE K POR BIC / AIC
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print("EXPERIMENTO 3: GMM (BIC y AIC para selección de k)")
    print("==================================================================")
    gmm_res = []
    for k in [4, 5, 6, 7, 8, 9, 10]:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3, covariance_type="diag")
        gmm.fit(X_std)
        bic = gmm.bic(X_std)
        aic = gmm.aic(X_std)
        labels = gmm.predict(X_std)
        sil = silhouette_score(X_sample_std, labels[idx_sample])
        db = davies_bouldin_score(X_sample_std, labels[idx_sample])
        gmm_res.append({
            "k": k, "BIC (en miles)": round(bic/1000, 1), "AIC (en miles)": round(aic/1000, 1),
            "Silhouette": round(sil, 4), "Davies-Bouldin": round(db, 4)
        })
    df_gmm = pd.DataFrame(gmm_res)
    print(df_gmm.to_string(index=False))

    # ------------------------------------------------------------------
    # EXPERIMENTO 4: K-PROTOTYPES (Datos numéricos + categóricos)
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print("EXPERIMENTO 4: K-PROTOTYPES (Tratamiento nativo mixto)")
    print("==================================================================")
    # Derivación dinámica de columnas numéricas y tags categóricos a partir de feats_std
    # (Evita fragilidad posicional si cambia el orden o número de features)
    cols_num_idx = [i for i, f in enumerate(feats_std) if not f.startswith("tag_") and not f.startswith("tfidf_")]
    cols_cat_idx_global = [i for i, f in enumerate(feats_std) if f.startswith("tag_")]
    cols_num_cat = cols_num_idx + cols_cat_idx_global
    X_num_cat = X_std[:, cols_num_cat]
    cols_cat_idx = [i for i, idx in enumerate(cols_num_cat) if feats_std[idx].startswith("tag_")]
    
    kproto_res = []
    for gamma in [0.5, 1.0, 2.0]:
        for k in [4, 5, 6]:
            kp = KPrototypes(n_clusters=k, gamma=gamma, random_state=42, n_init=3, n_jobs=-1)
            # Entrenamos en la submuestra de 10,000
            labels_kp = kp.fit_predict(X_num_cat[idx_sample], categorical=cols_cat_idx)
            sil_25d = silhouette_score(X_sample_std, labels_kp)
            sil_10d = silhouette_score(X_num_cat[idx_sample], labels_kp)
            db_25d = davies_bouldin_score(X_sample_std, labels_kp)
            kproto_res.append({
                "gamma (peso cat)": gamma, "k": k,
                "Silueta (en 25D)": round(sil_25d, 4),
                "Silueta (en 10D)": round(sil_10d, 4),
                "Davies-Bouldin": round(db_25d, 4)
            })
    df_kproto = pd.DataFrame(kproto_res)
    print(df_kproto.to_string(index=False))

    # ------------------------------------------------------------------
    # EXPERIMENTO 5: HDBSCAN CON GRID DE DENSIDAD (0.5%, 1%, 2%)
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print("EXPERIMENTO 5: HDBSCAN AFINADO (Grid de densidad sobre 25D)")
    print("==================================================================")
    hdb_res = []
    n_total = len(df_multi)
    for pct, min_pts in [(0.005, int(0.005*n_total)), (0.01, int(0.01*n_total)), (0.02, int(0.02*n_total))]:
        hdb = HDBSCAN(min_cluster_size=min_pts, min_samples=max(5, int(min_pts*0.2)))
        labels_hdb = hdb.fit_predict(X_std)
        n_clusters = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
        noise_pct = (labels_hdb == -1).mean() * 100
        
        sample_labels_hdb = labels_hdb[idx_sample]
        mask_eval = sample_labels_hdb != -1
        if mask_eval.sum() > 100 and n_clusters > 1:
            sil_val = silhouette_score(X_sample_std[mask_eval], sample_labels_hdb[mask_eval])
            db_val = davies_bouldin_score(X_sample_std[mask_eval], sample_labels_hdb[mask_eval])
        else:
            sil_val, db_val = np.nan, np.nan
        hdb_res.append({
            "Porcentaje": f"{pct*100:.1f}%", "min_cluster_size": min_pts,
            "N Clusters": n_clusters, "Ruido (%)": round(noise_pct, 1),
            "Silueta (25D sin ruido)": round(sil_val, 4),
            "Davies-Bouldin (25D sin ruido)": round(db_val, 4)
        })
    df_hdb = pd.DataFrame(hdb_res)
    print(df_hdb.to_string(index=False))

    # ------------------------------------------------------------------
    # EXPERIMENTO 6: UMAP (8D) + HDBSCAN (Evaluado en 25D original)
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print("EXPERIMENTO 6: UMAP (8D) + HDBSCAN (Evaluado en 25D original)")
    print("==================================================================")
    reducer = umap.UMAP(n_components=8, n_neighbors=35, min_dist=0.0, random_state=42)
    X_umap_8 = reducer.fit_transform(X_std)
    
    umap_hdb_res = []
    for min_pts in [90, 180, 360]:
        hdb = HDBSCAN(min_cluster_size=min_pts, min_samples=10)
        labels_uh = hdb.fit_predict(X_umap_8)
        n_clusters = len(set(labels_uh)) - (1 if -1 in labels_uh else 0)
        noise_pct = (labels_uh == -1).mean() * 100
        
        sample_labels_uh = labels_uh[idx_sample]
        mask_eval = sample_labels_uh != -1
        if mask_eval.sum() > 100 and n_clusters > 1:
            sil_orig = silhouette_score(X_sample_std[mask_eval], sample_labels_uh[mask_eval])
            sil_umap = silhouette_score(X_umap_8[idx_sample][mask_eval], sample_labels_uh[mask_eval])
            db_orig = davies_bouldin_score(X_sample_std[mask_eval], sample_labels_uh[mask_eval])
        else:
            sil_orig, sil_umap, db_orig = np.nan, np.nan, np.nan
        umap_hdb_res.append({
            "min_cluster_size": min_pts, "N Clusters": n_clusters, "Ruido (%)": round(noise_pct, 1),
            "Silueta (en Espacio 25D)": round(sil_orig, 4),
            "Silueta (en Embedding UMAP)": round(sil_umap, 4),
            "Davies-Bouldin (25D)": round(db_orig, 4)
        })
    df_umap_hdb = pd.DataFrame(umap_hdb_res)
    print(df_umap_hdb.to_string(index=False))

    # ------------------------------------------------------------------
    # EXPERIMENTO 7: SOBREQUIPPING (K=8 y K=10 consolidado a 5 arquetipos)
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print("EXPERIMENTO 7: SOBREQUIPPING (K fino 8-10 mapeado a 5 Arquetipos)")
    print("==================================================================")
    for k_fine in [8, 10]:
        km_fine = KMeans(n_clusters=k_fine, random_state=42, n_init=10)
        labels_fine = km_fine.fit_predict(X_std)
        sil_fine = silhouette_score(X_sample_std, labels_fine[idx_sample])
        db_fine = davies_bouldin_score(X_sample_std, labels_fine[idx_sample])
        
        # Mapeo real de micro-clusters a arquetipos con correspondencia húngara
        mapa_fino = etiquetar_por_centroides_escalados(
            kmeans=km_fine,
            feature_names=feats_std,
            scaler=scaler_std,
            peso_nlp=0.2
        )
        arquetipos_consolidados = pd.Series(labels_fine).map(mapa_fino)
        conteo = arquetipos_consolidados.value_counts()
        print(f"\nK-Means fino k={k_fine}: Silueta={sil_fine:.4f}, Davies-Bouldin={db_fine:.4f}")
        print("  Distribución consolidada en arquetipos:")
        for arq, cnt in conteo.items():
            print(f"    - {arq}: {cnt} ({cnt/len(labels_fine)*100:.1f}%)")

if __name__ == "__main__":
    run()
