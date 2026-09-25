#!/usr/bin/env python3
"""
Script de Exploracion de Micro-Clusters (k > 10) para Normalizacion de Nombres en Backend.

Objetivo de Negocio:
Evaluar si particiones finas (k > 10) sobre los ~18,400 registros multi-zona en el espacio 35D
(v2.3) son viables para la normalizacion sistematica de localidades por detras, preservando
siempre intacto el nombre comercial de la boleta (logical_seat_category).
El cluster_id resultante funciona como identificador de backend y cualquier etiqueta legible
(label_auto) es autogenerada sin requerir aprobacion humana.

Metodología:
1. Reutiliza el dataset enriquecido y el espacio vectorial mixto 35D calibrado en v2.3
   (peso_nlp=0.2, peso_type_site=0.5, scaler=robust, random_state=42).
2. Algoritmos candidatos:
   - K-Means con k in {8, 10, 12, 14, 16} (n_init=15, random_state=42).
   - GMM con k in {8, 10, 12, 14, 16} (covariance_type='full', n_init=3) y BIC.
   - HDBSCAN con grid min_cluster_size in {1.0%, 1.5%, 2.0%} del dataset (min_samples=20%).
3. Metricas evaluadas sobre muestra fija de 5,000 registros para comparabilidad estricta:
   - Silueta, Davies-Bouldin, Calinski-Harabasz.
   - Estabilidad bootstrap-ARI: 20 remuestreos al 80% frente a la particion completa.
   - Pureza de naming: share del tag dominante y share del termino TF-IDF dominante.
   - Distribucion de tamanos y deteccion de clusters degenerados (<2% del dataset).
4. Criterios de Aceptacion:
   - pureza_naming >= 0.85 Y bootstrap_ari >= 0.85 Y sin clusters degenerados.
   - Si varios pasan, gana el de mayor silueta.
"""

import os
import sys
import time
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

# Asegurar path raiz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

from src.feature_engineering import preparar_dataset_enriquecido
from src.clustering import (
    separar_admision_unica_multizona,
    construir_espacio_vectorial_mixto,
    cargar_modelo_clustering,
    predecir_arquetipos_demanda,
    CANONICAL_TYPE_SITE_CATEGORIES
)
from src.nlp_utils import normalizar_texto

# Mapeo limpio de tags a etiquetas Title Case
TAG_LABEL_MAP = {
    "tag_palco": "Palco",
    "tag_vip": "VIP",
    "tag_platea": "Platea",
    "tag_preferencial": "Preferencial",
    "tag_general": "General",
    "tag_balcon": "Balcón",
    "tag_piso_alto": "Piso Alto",
    "tag_piso_bajo": "Piso Bajo",
    "tag_occidental": "Occidental",
    "tag_oriental": "Oriental",
    "tag_norte": "Norte",
    "tag_sur": "Sur",
    "tag_lateral": "Lateral",
    "tag_vista_parcial": "Vista Parcial",
    "tag_familiar": "Familiar",
    "tag_menores": "Menores",
    "tag_movilidad_reducida": "Movilidad Reducida"
}


def calcular_pureza_cluster(
    df_cluster: pd.DataFrame,
    tag_cols: List[str],
    tfidf_vocab: List[str]
) -> Dict[str, Any]:
    """
    Calcula la pureza estructural y léxica de un cluster individual.
    - purity_tag: share del tag estructural dominante (máx activación).
    - purity_term: share del término TF-IDF dominante en el texto limpio.
    - purity_naming: combinación de ambas señales max(purity_tag, purity_term).
    """
    n_cluster = len(df_cluster)
    if n_cluster == 0:
        return {
            "purity_tag": 0.0,
            "tag_dominante": "ninguno",
            "purity_term": 0.0,
            "term_dominante": "ninguno",
            "purity_naming": 0.0,
            "tag_shares": pd.Series(dtype=float)
        }

    # 1. Tags estructurales
    tag_shares = df_cluster[tag_cols].mean() if tag_cols else pd.Series(dtype=float)
    if len(tag_shares) > 0 and tag_shares.max() > 0:
        tag_dom = str(tag_shares.idxmax())
        purity_tag = float(tag_shares.max())
    else:
        tag_dom = "ninguno"
        purity_tag = 0.0

    # 2. Términos léxicos TF-IDF
    texts_tokens = df_cluster["texto_limpio"].astype(str).str.lower().str.split()
    term_counts = {}
    for term in tfidf_vocab:
        term_clean = term.lower()
        cnt = texts_tokens.apply(lambda toks: term_clean in toks).sum()
        term_counts[term] = cnt / n_cluster

    if term_counts and max(term_counts.values()) > 0:
        term_dom = max(term_counts, key=term_counts.get)
        purity_term = float(term_counts[term_dom])
    else:
        term_dom = "ninguno"
        purity_term = 0.0

    purity_naming = max(purity_tag, purity_term)

    return {
        "purity_tag": purity_tag,
        "tag_dominante": tag_dom,
        "purity_term": purity_term,
        "term_dominante": term_dom,
        "purity_naming": purity_naming,
        "tag_shares": tag_shares
    }


def generar_label_auto(
    cluster_id: int,
    tag_shares: pd.Series,
    term_dominante: str,
    purity_tag: float,
    purity_term: float,
    threshold_tag_active: float = 0.60,
    min_purity_tag: float = 0.70
) -> str:
    """
    Genera automáticamente el nombre representativo del micro-cluster.
    Regla estricta:
    - Si purity_tag < min_purity_tag (0.70) o no hay término dominante -> 'hibrido_k{cluster_id}'.
    - En caso contrario, compone los tags activos (activación >= 0.60) + término dominante en Title Case.
    """
    if purity_tag < min_purity_tag or not term_dominante or term_dominante == "ninguno" or purity_term < 0.10:
        return f"hibrido_k{cluster_id}"

    # Identificar tags activos con >= threshold_tag_active
    active_tags = []
    if len(tag_shares) > 0:
        for tag_col, val in tag_shares.items():
            if val >= threshold_tag_active:
                clean_name = TAG_LABEL_MAP.get(tag_col, tag_col.replace("tag_", "").capitalize())
                active_tags.append(clean_name)

    # Si ningún tag alcanza 0.60 pero purity_tag >= 0.70, usar el dominante
    if not active_tags and len(tag_shares) > 0 and tag_shares.max() > 0:
        dom_col = tag_shares.idxmax()
        active_tags.append(TAG_LABEL_MAP.get(dom_col, dom_col.replace("tag_", "").capitalize()))

    # Agregar término léxico dominante si no es redundante (con stripping de acentos)
    partes = []
    seen = set()
    for tag_name in active_tags:
        for word in tag_name.split():
            w_norm = normalizar_texto(word)
            if w_norm and w_norm not in seen:
                clean_word = word if word.isupper() and len(word) > 1 else word.capitalize()
                partes.append(clean_word)
                seen.add(w_norm)

    term_clean = term_dominante.capitalize()
    term_norm = normalizar_texto(term_clean)
    if term_norm and term_norm not in seen and term_norm != "NINGUNO":
        partes.append(term_clean)

    if not partes:
        return f"hibrido_k{cluster_id}"

    return " ".join(partes)


def calcular_estabilidad_bootstrap_ari(
    X: np.ndarray,
    labels_full: np.ndarray,
    estimador_fn: Any,
    n_iter: int = 20,
    frac: float = 0.80,
    random_state: int = 42
) -> Tuple[float, float, List[float]]:
    """
    Evalúa la estabilidad de la partición mediante remuestreos al 80%.
    Calcula el Adjusted Rand Index (ARI) de cada remuestreo contra las etiquetas completas.
    """
    n_samples = len(X)
    n_sub = int(frac * n_samples)
    ari_scores = []
    rng_master = np.random.RandomState(random_state)

    for i in range(n_iter):
        sub_seed = rng_master.randint(0, 100000)
        rng_sub = np.random.RandomState(sub_seed)
        idx_sub = rng_sub.choice(n_samples, size=n_sub, replace=False)

        X_sub = X[idx_sub]
        labels_sub = estimador_fn(X_sub, sub_seed)

        # Evaluar ARI contra la partición completa en las mismas posiciones
        ari = adjusted_rand_score(labels_full[idx_sub], labels_sub)
        ari_scores.append(float(ari))

    return float(np.mean(ari_scores)), float(np.std(ari_scores)), ari_scores


def generar_catalogo_clusters(
    df_multi: pd.DataFrame,
    labels: np.ndarray,
    tfidf_vocab: List[str],
    tag_cols: List[str],
    modelo_v23: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Genera el catálogo consolidado de micro-clusters para el modelo ganador.
    Columnas: cluster_id, n, purity_tag, tag_dominante, purity_term, term_dominante, label_auto, arquetipo_v23_rollup.
    """
    df_work = df_multi.copy()
    df_work["cluster_eval"] = labels

    # Rollup contra modelo v2.3 si se provee
    if modelo_v23 is not None and "arquetipo_demanda" not in df_work.columns:
        df_pred = predecir_arquetipos_demanda(df_work, modelo_v23)
        df_work["arquetipo_v23"] = df_pred["arquetipo_demanda"]
    elif "arquetipo_demanda" in df_work.columns:
        df_work["arquetipo_v23"] = df_work["arquetipo_demanda"]
    else:
        df_work["arquetipo_v23"] = "Desconocido"

    unique_clusters = sorted([c for c in np.unique(labels) if c != -1])
    filas_cat = []

    for c_id in unique_clusters:
        sub = df_work[df_work["cluster_eval"] == c_id]
        p_info = calcular_pureza_cluster(sub, tag_cols, tfidf_vocab)
        label_str = generar_label_auto(
            cluster_id=c_id,
            tag_shares=p_info["tag_shares"],
            term_dominante=p_info["term_dominante"],
            purity_tag=p_info["purity_tag"],
            purity_term=p_info["purity_term"]
        )

        # Arquetipo mayoritario de rollup
        arqs_counts = sub["arquetipo_v23"].value_counts()
        rollup = arqs_counts.index[0] if len(arqs_counts) > 0 else "Sin Datos"

        filas_cat.append({
            "cluster_id": c_id,
            "n": len(sub),
            "purity_tag": round(p_info["purity_tag"], 4),
            "tag_dominante": p_info["tag_dominante"],
            "purity_term": round(p_info["purity_term"], 4),
            "term_dominante": p_info["term_dominante"],
            "label_auto": label_str,
            "arquetipo_v23_rollup": rollup
        })

    return pd.DataFrame(filas_cat)


def ejecutar_exploracion_microclusters(
    guardar_csv: bool = True,
    output_resultados: str = "reports/microclusters_resultados.csv",
    output_catalogo: str = "data/processed/cluster_catalog.csv"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Ejecuta el protocolo experimental completo de micro-clusters sobre multi-zona (35D).
    """
    print("=" * 90)
    print("   EXPLORACION DE MICRO-CLUSTERS (k > 10) PARA NORMALIZACION DE LOCALIDADES (35D)")
    print("=" * 90)

    # 1. Cargar y enriquecer dataset
    print("\n1. Cargando y preparando dataset completo...")
    df_raw = pd.read_parquet("data/raw/localidades_eda.parquet")
    df_enr = preparar_dataset_enriquecido(df_raw)

    # 2. Separar multi-zona
    _, df_multi = separar_admision_unica_multizona(df_enr)
    n_multi = len(df_multi)
    print(f" Dataset multi-zona certificado: {n_multi:,} localidades.")

    # 3. Construir espacio mixto 35D con parámetros calibrados v2.3
    print("2. Construyendo espacio mixto 35D (RobustScaler, peso_nlp=0.2, peso_venue=0.5)...")
    X_multi, scaler, tfidf_vec, feature_names = construir_espacio_vectorial_mixto(
        df_multi,
        max_tfidf_features=15,
        peso_nlp=0.2,
        peso_type_site=0.5,
        categorias_type_site=CANONICAL_TYPE_SITE_CATEGORIES
    )
    print(f" Matriz multi-zona lista: {X_multi.shape[0]:,} filas x {X_multi.shape[1]} dimensiones.")

    tfidf_vocab = list(tfidf_vec.get_feature_names_out())
    tag_cols = [c for c in df_multi.columns if c.startswith("tag_")]

    # 4. Muestra fija de 5,000 para evaluación métrica imparcial
    sample_size = min(5000, n_multi)
    rng_eval = np.random.RandomState(42)
    idx_eval = rng_eval.choice(n_multi, size=sample_size, replace=False)
    X_eval = X_multi[idx_eval]

    # Cargar modelo v2.3 para rollup de arquetipos
    modelo_v23 = None
    if os.path.exists("data/processed/modelo_clustering_v2_3.joblib"):
        try:
            modelo_v23 = cargar_modelo_clustering("data/processed/modelo_clustering_v2_3.joblib")
        except Exception as e:
            print(f" Nota: No se pudo cargar modelo v2.3 para rollup ({e}).")

    resultados = []
    modelos_entrenados = {}

    # Umbral de clusters degenerados: <2% del dataset
    umbral_degenerado = int(0.02 * n_multi)
    print(f" Umbral de cluster degenerado (<2% de N): {umbral_degenerado} localidades.")

    # =========================================================================
    # A. CANDIDATOS K-MEANS: k in {8, 10, 12, 14, 16}
    # =========================================================================
    print("\n3. Evaluando familia K-Means (k in {8, 10, 12, 14, 16})...")
    k_vals = [8, 10, 12, 14, 16]
    for k in k_vals:
        t0 = time.time()
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels_km = km.fit_predict(X_multi)
        tiempo = time.time() - t0

        # Metricas intrinsecas en submuestra fija
        labels_km_eval = labels_km[idx_eval]
        sil = float(silhouette_score(X_eval, labels_km_eval))
        db = float(davies_bouldin_score(X_eval, labels_km_eval))
        ch = float(calinski_harabasz_score(X_eval, labels_km_eval))

        # Estabilidad bootstrap-ARI (20 remuestreos al 80%)
        def _estimador_km(X_sub, seed):
            return KMeans(n_clusters=k, random_state=seed, n_init=5).fit_predict(X_sub)

        ari_mean, ari_std, _ = calcular_estabilidad_bootstrap_ari(
            X_multi, labels_km, _estimador_km, n_iter=20, frac=0.80, random_state=42
        )

        # Pureza de naming y distribucion de tamanos
        df_temp = df_multi.copy()
        df_temp["c"] = labels_km
        counts = df_temp["c"].value_counts()
        min_size = int(counts.min())
        median_size = float(counts.median())
        pct_min = float(min_size / n_multi * 100)
        n_deg = int((counts < umbral_degenerado).sum())

        p_tags = []
        p_terms = []
        p_namings = []
        for c_id in range(k):
            sub = df_temp[df_temp["c"] == c_id]
            res_p = calcular_pureza_cluster(sub, tag_cols, tfidf_vocab)
            p_tags.append(res_p["purity_tag"])
            p_terms.append(res_p["purity_term"])
            p_namings.append(res_p["purity_naming"])

        weights = np.array([len(df_temp[df_temp["c"] == c_id]) for c_id in range(k)])
        pureza_tag_ponderada = float(np.sum(weights * np.array(p_tags)) / np.sum(weights))
        pureza_term_ponderada = float(np.sum(weights * np.array(p_terms)) / np.sum(weights))
        pureza_naming_ponderada = float(np.sum(weights * np.array(p_namings)) / np.sum(weights))
        pct_puros = float(sum(p >= 0.85 for p in p_namings) / k * 100)

        # Criterio de aceptación estricto
        pasa = bool(pureza_naming_ponderada >= 0.85 and ari_mean >= 0.85 and n_deg == 0)

        row_res = {
            "algoritmo": "K-Means",
            "k_param": k,
            "k_efectivo": k,
            "outliers_pct": 0.0,
            "bic": np.nan,
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
            "calinski_harabasz": round(ch, 1),
            "bootstrap_ari_mean": round(ari_mean, 4),
            "bootstrap_ari_std": round(ari_std, 4),
            "pureza_tag_ponderada": round(pureza_tag_ponderada, 4),
            "pureza_term_ponderada": round(pureza_term_ponderada, 4),
            "pureza_naming_ponderada": round(pureza_naming_ponderada, 4),
            "pct_clusters_puros": round(pct_puros, 1),
            "min_size": min_size,
            "median_size": median_size,
            "pct_min_size": round(pct_min, 2),
            "clusters_degenerados": n_deg,
            "pasa_candidato": pasa,
            "tiempo_seg": round(tiempo, 2)
        }
        resultados.append(row_res)
        modelos_entrenados[f"K-Means_k{k}"] = {"labels": labels_km, "res": row_res}
        print(f"  [K-Means k={k:2d}] Silueta: {sil:.4f} | DB: {db:.4f} | ARI: {ari_mean:.4f} | Pureza: {pureza_naming_ponderada:.4f} | Degenerados: {n_deg} | Pasa: {pasa}")

    # =========================================================================
    # B. CANDIDATOS GMM (Gaussian Mixture): k in {8, 10, 12, 14, 16}
    # =========================================================================
    print("\n4. Evaluando familia Gaussian Mixture (GMM, covariance full, n_init=3)...")
    for k in k_vals:
        t0 = time.time()
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=3)
        labels_gmm = gmm.fit_predict(X_multi)
        tiempo = time.time() - t0
        bic_val = float(gmm.bic(X_multi))

        labels_gmm_eval = labels_gmm[idx_eval]
        sil = float(silhouette_score(X_eval, labels_gmm_eval))
        db = float(davies_bouldin_score(X_eval, labels_gmm_eval))
        ch = float(calinski_harabasz_score(X_eval, labels_gmm_eval))

        def _estimador_gmm(X_sub, seed):
            return GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=1).fit_predict(X_sub)

        ari_mean, ari_std, _ = calcular_estabilidad_bootstrap_ari(
            X_multi, labels_gmm, _estimador_gmm, n_iter=20, frac=0.80, random_state=42
        )

        df_temp = df_multi.copy()
        df_temp["c"] = labels_gmm
        counts = df_temp["c"].value_counts()
        min_size = int(counts.min())
        median_size = float(counts.median())
        pct_min = float(min_size / n_multi * 100)
        n_deg = int((counts < umbral_degenerado).sum())

        p_tags = []
        p_terms = []
        p_namings = []
        for c_id in range(k):
            sub = df_temp[df_temp["c"] == c_id]
            res_p = calcular_pureza_cluster(sub, tag_cols, tfidf_vocab)
            p_tags.append(res_p["purity_tag"])
            p_terms.append(res_p["purity_term"])
            p_namings.append(res_p["purity_naming"])

        weights = np.array([len(df_temp[df_temp["c"] == c_id]) for c_id in range(k)])
        pureza_tag_ponderada = float(np.sum(weights * np.array(p_tags)) / np.sum(weights))
        pureza_term_ponderada = float(np.sum(weights * np.array(p_terms)) / np.sum(weights))
        pureza_naming_ponderada = float(np.sum(weights * np.array(p_namings)) / np.sum(weights))
        pct_puros = float(sum(p >= 0.85 for p in p_namings) / k * 100)

        pasa = bool(pureza_naming_ponderada >= 0.85 and ari_mean >= 0.85 and n_deg == 0)

        row_res = {
            "algoritmo": "GMM",
            "k_param": k,
            "k_efectivo": k,
            "outliers_pct": 0.0,
            "bic": round(bic_val, 1),
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
            "calinski_harabasz": round(ch, 1),
            "bootstrap_ari_mean": round(ari_mean, 4),
            "bootstrap_ari_std": round(ari_std, 4),
            "pureza_tag_ponderada": round(pureza_tag_ponderada, 4),
            "pureza_term_ponderada": round(pureza_term_ponderada, 4),
            "pureza_naming_ponderada": round(pureza_naming_ponderada, 4),
            "pct_clusters_puros": round(pct_puros, 1),
            "min_size": min_size,
            "median_size": median_size,
            "pct_min_size": round(pct_min, 2),
            "clusters_degenerados": n_deg,
            "pasa_candidato": pasa,
            "tiempo_seg": round(tiempo, 2)
        }
        resultados.append(row_res)
        modelos_entrenados[f"GMM_k{k}"] = {"labels": labels_gmm, "res": row_res}
        print(f"  [GMM k={k:2d}] BIC: {bic_val:,.0f} | Silueta: {sil:.4f} | DB: {db:.4f} | ARI: {ari_mean:.4f} | Pureza: {pureza_naming_ponderada:.4f} | Degenerados: {n_deg} | Pasa: {pasa}")

    # =========================================================================
    # C. CANDIDATOS HDBSCAN: grid min_cluster_size in {1%, 1.5%, 2%}
    # =========================================================================
    print("\n5. Evaluando familia HDBSCAN (min_cluster_size in {1.0%, 1.5%, 2.0%})...")
    hdb_grid = [0.010, 0.015, 0.020]
    for frac_hdb in hdb_grid:
        t0 = time.time()
        mcs = max(25, int(frac_hdb * n_multi))
        ms = max(5, int(0.20 * mcs))
        hdb = HDBSCAN(min_cluster_size=mcs, min_samples=ms, allow_single_cluster=False)
        labels_hdb = hdb.fit_predict(X_multi)
        tiempo = time.time() - t0

        unique_clusters = set(labels_hdb) - {-1}
        k_hdb = len(unique_clusters)
        outliers_pct = float((labels_hdb == -1).mean() * 100)

        # Silueta sobre la submuestra fija excluyendo outliers
        labels_hdb_eval = labels_hdb[idx_eval]
        mask_valid = labels_hdb_eval != -1
        if mask_valid.sum() > 100 and k_hdb > 1:
            sil = float(silhouette_score(X_eval[mask_valid], labels_hdb_eval[mask_valid]))
            db = float(davies_bouldin_score(X_eval[mask_valid], labels_hdb_eval[mask_valid]))
            ch = float(calinski_harabasz_score(X_eval[mask_valid], labels_hdb_eval[mask_valid]))
        else:
            sil, db, ch = np.nan, np.nan, np.nan

        def _estimador_hdb(X_sub, seed):
            return HDBSCAN(min_cluster_size=mcs, min_samples=ms, allow_single_cluster=False).fit_predict(X_sub)

        ari_mean, ari_std, _ = calcular_estabilidad_bootstrap_ari(
            X_multi, labels_hdb, _estimador_hdb, n_iter=20, frac=0.80, random_state=42
        )

        df_temp = df_multi.copy()
        df_temp["c"] = labels_hdb
        valid_df = df_temp[df_temp["c"] != -1]
        if len(valid_df) > 0 and k_hdb > 0:
            counts = valid_df["c"].value_counts()
            min_size = int(counts.min())
            median_size = float(counts.median())
            pct_min = float(min_size / n_multi * 100)
            n_deg = int((counts < umbral_degenerado).sum())

            p_tags = []
            p_terms = []
            p_namings = []
            for c_id in sorted(unique_clusters):
                sub = df_temp[df_temp["c"] == c_id]
                res_p = calcular_pureza_cluster(sub, tag_cols, tfidf_vocab)
                p_tags.append(res_p["purity_tag"])
                p_terms.append(res_p["purity_term"])
                p_namings.append(res_p["purity_naming"])

            weights = np.array([len(df_temp[df_temp["c"] == c_id]) for c_id in sorted(unique_clusters)])
            pureza_tag_ponderada = float(np.sum(weights * np.array(p_tags)) / np.sum(weights))
            pureza_term_ponderada = float(np.sum(weights * np.array(p_terms)) / np.sum(weights))
            pureza_naming_ponderada = float(np.sum(weights * np.array(p_namings)) / np.sum(weights))
            pct_puros = float(sum(p >= 0.85 for p in p_namings) / k_hdb * 100)
        else:
            min_size, median_size, pct_min, n_deg = 0, 0, 0.0, 0
            pureza_tag_ponderada, pureza_term_ponderada, pureza_naming_ponderada, pct_puros = 0.0, 0.0, 0.0, 0.0

        pasa = bool(pureza_naming_ponderada >= 0.85 and ari_mean >= 0.85 and n_deg == 0 and outliers_pct < 5.0)

        grid_name = f"{frac_hdb*100:.1f}%"
        row_res = {
            "algoritmo": "HDBSCAN",
            "k_param": grid_name,
            "k_efectivo": k_hdb,
            "outliers_pct": round(outliers_pct, 2),
            "bic": np.nan,
            "silhouette": round(sil, 4) if not np.isnan(sil) else np.nan,
            "davies_bouldin": round(db, 4) if not np.isnan(db) else np.nan,
            "calinski_harabasz": round(ch, 1) if not np.isnan(ch) else np.nan,
            "bootstrap_ari_mean": round(ari_mean, 4),
            "bootstrap_ari_std": round(ari_std, 4),
            "pureza_tag_ponderada": round(pureza_tag_ponderada, 4),
            "pureza_term_ponderada": round(pureza_term_ponderada, 4),
            "pureza_naming_ponderada": round(pureza_naming_ponderada, 4),
            "pct_clusters_puros": round(pct_puros, 1),
            "min_size": min_size,
            "median_size": median_size,
            "pct_min_size": round(pct_min, 2),
            "clusters_degenerados": n_deg,
            "pasa_candidato": pasa,
            "tiempo_seg": round(tiempo, 2)
        }
        resultados.append(row_res)
        modelos_entrenados[f"HDBSCAN_{grid_name}"] = {"labels": labels_hdb, "res": row_res}
        print(f"  [HDBSCAN min={grid_name}] k={k_hdb} | Outliers: {outliers_pct:.1f}% | Silueta: {sil:.4f} | ARI: {ari_mean:.4f} | Pureza: {pureza_naming_ponderada:.4f} | Degenerados: {n_deg} | Pasa: {pasa}")

    # =========================================================================
    # D. CONSOLIDACION Y SELECCION DEL GANADOR
    # =========================================================================
    df_resultados = pd.DataFrame(resultados)

    print("\n" + "=" * 90)
    print("RESUMEN GENERAL DE METRICAS Y ACEPTACION DE CANDIDATOS")
    print("=" * 90)
    cols_print = [
        "algoritmo", "k_param", "k_efectivo", "silhouette", "davies_bouldin",
        "bootstrap_ari_mean", "pureza_naming_ponderada", "pct_clusters_puros",
        "clusters_degenerados", "pasa_candidato"
    ]
    print(df_resultados[cols_print].to_string(index=False))

    if guardar_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_resultados)), exist_ok=True)
        df_resultados.to_csv(output_resultados, index=False)
        print(f"\n Resultados consolidados guardados en: {output_resultados}")

    # Evaluar ganadores
    candidatos_validos = df_resultados[df_resultados["pasa_candidato"] == True]
    df_catalogo = None

    if len(candidatos_validos) > 0:
        # Mayor silueta entre los que cumplen criterios
        ganador_row = candidatos_validos.sort_values(by="silhouette", ascending=False).iloc[0]
        alg_win = ganador_row["algoritmo"]
        k_win = ganador_row["k_param"]
        key_win = f"{alg_win}_k{k_win}" if alg_win != "HDBSCAN" else f"HDBSCAN_{k_win}"
        print(f"\n CANDIDATO GANADOR SELECCIONADO: {alg_win} (k={k_win})")
        print(f"   Silueta: {ganador_row['silhouette']} | ARI: {ganador_row['bootstrap_ari_mean']} | Pureza Naming: {ganador_row['pureza_naming_ponderada']}")

        labels_ganador = modelos_entrenados[key_win]["labels"]
        df_catalogo = generar_catalogo_clusters(
            df_multi=df_multi,
            labels=labels_ganador,
            tfidf_vocab=tfidf_vocab,
            tag_cols=tag_cols,
            modelo_v23=modelo_v23
        )
        if guardar_csv:
            os.makedirs(os.path.dirname(os.path.abspath(output_catalogo)), exist_ok=True)
            df_catalogo.to_csv(output_catalogo, index=False)
            print(f" Catálogo de micro-clusters del ganador guardado en: {output_catalogo}")
            print("\nPrimeras 5 filas del catálogo generado:")
            print(df_catalogo.head(5).to_string(index=False))
    else:
        print("\n CONCLUSION METODOLOGICA: NINGUN CANDIDATO CUMPLE SIMULTANEAMENTE:")
        print("   - Pureza de naming >= 0.85")
        print("   - Bootstrap-ARI >= 0.85")
        print("   - 0 clusters degenerados (<2% del dataset)")
        print("\n Se reporta la evidencia cuantitativa formal. No se promueve ningún micro-cluster a producción.")

        # Guardar de forma exploratoria el catálogo del modelo con mayor pureza para auditoría
        top_candidato = df_resultados.sort_values(by=["pureza_naming_ponderada", "silhouette"], ascending=False).iloc[0]
        alg_top = top_candidato["algoritmo"]
        k_top = top_candidato["k_param"]
        key_top = f"{alg_top}_k{k_top}" if alg_top != "HDBSCAN" else f"HDBSCAN_{k_top}"
        labels_top = modelos_entrenados[key_top]["labels"]
        df_catalogo = generar_catalogo_clusters(
            df_multi=df_multi,
            labels=labels_top,
            tfidf_vocab=tfidf_vocab,
            tag_cols=tag_cols,
            modelo_v23=modelo_v23
        )
        if guardar_csv:
            os.makedirs(os.path.dirname(os.path.abspath(output_catalogo)), exist_ok=True)
            df_catalogo.to_csv(output_catalogo, index=False)
            print(f" Catálogo de referencia del candidato con mayor pureza ({alg_top} k={k_top}) guardado en: {output_catalogo}")
            print("\nMuestra del catálogo exploratorio:")
            print(df_catalogo.head(5).to_string(index=False))

    return df_resultados, df_catalogo


if __name__ == "__main__":
    ejecutar_exploracion_microclusters()
