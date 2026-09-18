"""
Módulo de Modelado y Clusterización de Localidades con Espacio Vectorial Mixto.

Este módulo se encarga de:
1. Construir el espacio vectorial mixto (características numéricas relativas + representaciones NLP estructuradas).
2. Evaluar el número óptimo de clusters (k) mediante métodos de Silueta, Inercia y Davies-Bouldin.
3. Entrenar modelos de segmentación (K-Means, GMM).
4. Mapear y etiquetar automáticamente los arquetipos estandarizados de demanda de negocio.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score
)
from scipy.optimize import linear_sum_assignment
from src.nlp_utils import vectorizar_texto_limpio


# Lista por defecto de variables estructurales numéricas ex-ante (sin tasa de ocupación)
DEFAULT_NUMERIC_FEATURES = [
    "ratio_precio_max",
    "percentil_precio_evento",
    "peso_aforo"
]

DEFAULT_TAG_FEATURES = [
    "tag_palco",
    "tag_vip",
    "tag_platea",
    "tag_preferencial",
    "tag_general",
    "tag_balcon",
    "tag_piso_alto"
]


def separar_admision_unica_multizona(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Particiona el catálogo a nivel EVENTO (t_performance_id) en dos ramas disjuntas:
    1. Eventos Monozona / Admisión Única: Funciones con 1 sola localidad activa
       O donde alguna localidad concentra el >= 99% del aforo (tarifa plana de facto).
    2. Eventos Multi-Zona: Funciones estratificadas donde coexisten y compiten múltiples localidades.
    
    Garantiza consistencia física a nivel evento (no parte una función entre dos etapas)
    y verifica cobertura matemática exacta: len(df_monozona) + len(df_multizona) == len(df).
    """
    df_work = df.copy()
    
    # 1. Métricas agrupadas a nivel evento
    n_localidades_evento = df_work.groupby("t_performance_id")["dn_quota"].transform("count")
    max_peso_aforo_evento = df_work.groupby("t_performance_id")["peso_aforo"].transform("max")
    
    # 2. Criterio primario: nunique == 1 por evento. Red de seguridad: max_peso_aforo >= 0.99
    es_monozona = (n_localidades_evento == 1) | (max_peso_aforo_evento >= 0.99)
    
    df_work["es_monozona"] = es_monozona
    df_work["segmento_etapa"] = np.where(es_monozona, "admision_unica", "multizona")
    
    df_monozona = df_work[es_monozona].copy()
    df_multizona = df_work[~es_monozona].copy()
    
    assert len(df_monozona) + len(df_multizona) == len(df), (
        f"Inconsistencia en partición: {len(df_monozona)} + {len(df_multizona)} != {len(df)}"
    )
    
    return df_monozona, df_multizona


def construir_espacio_vectorial_mixto(
    df: pd.DataFrame,
    columnas_numericas: List[str] = DEFAULT_NUMERIC_FEATURES,
    columnas_tags: List[str] = DEFAULT_TAG_FEATURES,
    usar_tfidf_texto: bool = True,
    max_tfidf_features: int = 15,
    peso_nlp: float = 1.0,
    scaler_type: str = "robust",
    scaler: Optional[Any] = None,
    tfidf_vectorizer: Optional[Any] = None
) -> Tuple[np.ndarray, Any, Any, List[str]]:
    """
    Construye el espacio vectorial mixto de 25 dimensiones combinando:
    1. Métricas numéricas relativas ex-ante escaladas (3 variables: ratio_precio_max, percentil_precio_evento, peso_aforo).
       Nota: 'tasa_ocupacion' se excluye deliberadamente al ser una métrica ex-post de absorción comercial.
    2. Tags estructurales densos (7 variables: 5 de jerarquía comercial + 2 de nivel vertical).
       Nota: De los 17 tags extraídos en el pipeline NLP, se seleccionan estos 7 para modelado
       para evitar alta dimensionalidad y dispersión causada por orientaciones y restricciones ultra-escasas.
    3. Embeddings/TF-IDF del texto limpio de la localidad (15 n-gramas) ponderados por peso_nlp.
    
    Total de dimensiones por defecto: 3 + 7 + 15 = 25 dimensiones.
    """
    # 1. Variables numéricas continuas
    cols_num_presentes = [c for c in columnas_numericas if c in df.columns]
    X_num = df[cols_num_presentes].values
    
    if len(cols_num_presentes) > 0:
        if scaler is None:
            scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
            X_num_scaled = scaler.fit_transform(X_num)
        else:
            X_num_scaled = scaler.transform(X_num)
    else:
        X_num_scaled = np.empty((len(df), 0))
        
    feature_names = list(cols_num_presentes)
    
    # 2. Variables binarias (Tags) en su escala natural [0, 1]
    cols_tags_presentes = [c for c in columnas_tags if c in df.columns]
    if len(cols_tags_presentes) > 0:
        X_tags = df[cols_tags_presentes].values.astype(float)
        feature_names.extend(cols_tags_presentes)
    else:
        X_tags = np.empty((len(df), 0))
        
    # 3. Concatenación base
    bloques = []
    if X_num_scaled.shape[1] > 0:
        bloques.append(X_num_scaled)
    if X_tags.shape[1] > 0:
        bloques.append(X_tags)
    
    # 4. TF-IDF sobre texto limpio
    if usar_tfidf_texto and "texto_limpio" in df.columns:
        X_tfidf, tfidf_vectorizer = vectorizar_texto_limpio(
            df["texto_limpio"],
            max_features=max_tfidf_features,
            vectorizer=tfidf_vectorizer
        )
        X_tfidf_weighted = X_tfidf * peso_nlp
        bloques.append(X_tfidf_weighted)
        tfidf_vocab = [f"tfidf_{w}" for w in tfidf_vectorizer.get_feature_names_out()]
        feature_names.extend(tfidf_vocab)
        
    X_mixto = np.hstack(bloques) if len(bloques) > 0 else np.empty((len(df), 0))
        
    return X_mixto, scaler, tfidf_vectorizer, feature_names


def evaluar_rango_k(
    X: np.ndarray, 
    k_min: int = 3, 
    k_max: int = 7, 
    random_state: int = 42,
    sample_size: int = 10000
) -> pd.DataFrame:
    """
    Evalúa múltiples valores de k calculando Inercia, Silueta, Calinski-Harabasz y Davies-Bouldin.
    """
    metricas = []
    
    if len(X) > sample_size:
        np.random.seed(random_state)
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        labels_eval = labels[idx] if len(X) > sample_size else labels
        
        sil = silhouette_score(X_eval, labels_eval)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        
        metricas.append({
            "k": k,
            "Inercia": kmeans.inertia_,
            "Silhouette Score": sil,
            "Calinski-Harabasz": ch,
            "Davies-Bouldin": db
        })
        
    return pd.DataFrame(metricas).set_index("k")


def entrenar_modelo_clustering(
    X: np.ndarray, 
    n_clusters: int = 4, 
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray, Dict[str, float]]:
    """
    Entrena el modelo K-Means final y retorna el modelo, las etiquetas asignadas y las métricas.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    labels = kmeans.fit_predict(X)
    
    sample_size = min(10000, len(X))
    idx = np.random.RandomState(random_state).choice(len(X), sample_size, replace=False)
    
    metricas = {
        "silhouette_score": float(silhouette_score(X[idx], labels[idx])),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "inercia": float(kmeans.inertia_)
    }
    
    return kmeans, labels, metricas


def construir_perfiles_ideales_escalados(
    feature_names: List[str],
    scaler: Any = None,
    peso_nlp: float = 1.2
) -> Dict[str, np.ndarray]:
    """
    Construye las representaciones vectoriales ideales para cada arquetipo multi-zona
    dentro del espacio geométrico escalado de 25 dimensiones.
    
    1. Bloque numérico (3 variables): Se especifica en unidades relativas naturales y se transforma
       mediante scaler.transform() para proyectarlo al espacio RobustScaler.
    2. Bloque tags (7 variables): Se especifica la tasa de activación ideal en [0, 1].
    3. Bloque TF-IDF (15 variables): Se ponderan los términos afines por peso_nlp.
    """
    arquetipos_multi = [
        "VIP / Palcos / Premium",
        "Preferencial / Platea Frontal",
        "Popular / Visibilidad Parcial / Balcón",
        "Grada General / Masiva"
    ]
    
    perfiles_config = {
        "VIP / Palcos / Premium": {
            "num": [0.90, 0.85, 0.08],
            "tags": {"tag_palco": 0.6, "tag_vip": 0.4},
            "words": {"tfidf_palco": 0.5, "tfidf_mesa": 0.3}
        },
        "Preferencial / Platea Frontal": {
            "num": [0.80, 0.75, 0.20],
            "tags": {"tag_platea": 0.8, "tag_preferencial": 0.3},
            "words": {"tfidf_platea": 0.5, "tfidf_central": 0.3}
        },
        "Popular / Visibilidad Parcial / Balcón": {
            "num": [0.35, 0.30, 0.15],
            "tags": {"tag_balcon": 0.4, "tag_piso_alto": 0.4},
            "words": {"tfidf_balcon": 0.4, "tfidf_piso": 0.3, "tfidf_posterior": 0.3}
        },
        "Grada General / Masiva": {
            "num": [0.60, 0.50, 0.60],
            "tags": {"tag_general": 0.6},
            "words": {"tfidf_general": 0.5}
        }
    }
    
    perfiles_vectores = {}
    for nombre in arquetipos_multi:
        cfg = perfiles_config[nombre]
        
        if scaler is not None and hasattr(scaler, "transform"):
            num_scaled = scaler.transform([cfg["num"]])[0]
        else:
            num_scaled = np.array(cfg["num"], dtype=float)
            
        vec = list(num_scaled)
        
        for feat in feature_names[3:]:
            if feat.startswith("tag_"):
                vec.append(cfg["tags"].get(feat, 0.0))
            elif feat.startswith("tfidf_"):
                vec.append(cfg["words"].get(feat, 0.0) * peso_nlp)
            else:
                vec.append(0.0)
                
        perfiles_vectores[nombre] = np.array(vec, dtype=float)
        
    return perfiles_vectores


def etiquetar_por_centroides_escalados(
    kmeans: KMeans,
    feature_names: List[str],
    scaler: Any = None,
    peso_nlp: float = 1.2
) -> Dict[int, str]:
    """
    Asigna arquetipos a los clusters evaluando la distancia euclidiana entre los centroides
    reales del modelo (en el espacio transformado de 25D) y los perfiles ideales de negocio.
    
    Aplica el algoritmo de asignación óptima 1 a 1 (Hungarian / Munkres) para garantizar
    una correspondencia biyectiva estricta sin duplicidades ni heurísticas frágiles de ordenamiento.
    """
    centroids = kmeans.cluster_centers_
    k = len(centroids)
    
    perfiles_dict = construir_perfiles_ideales_escalados(feature_names, scaler, peso_nlp=peso_nlp)
    nombres_perfiles = list(perfiles_dict.keys())
    perfiles_matriz = np.array([perfiles_dict[nom] for nom in nombres_perfiles])
    
    diff = centroids[:, np.newaxis, :] - perfiles_matriz[np.newaxis, :, :]
    D = np.linalg.norm(diff, axis=2)
    
    row_ind, col_ind = linear_sum_assignment(D)
    
    mapping = {}
    for c_id, p_id in zip(row_ind, col_ind):
        mapping[int(c_id)] = nombres_perfiles[p_id]
        
    for c_id in range(k):
        if c_id not in mapping:
            best_p = int(np.argmin(D[c_id]))
            mapping[c_id] = f"{nombres_perfiles[best_p]} (Variante #{c_id})"
            
    return mapping


def asignar_arquetipos_demanda(
    df_clustered: pd.DataFrame, 
    col_cluster: str = "cluster",
    kmeans: Optional[KMeans] = None,
    feature_names: Optional[List[str]] = None,
    scaler: Optional[Any] = None,
    peso_nlp: float = 1.2
) -> pd.DataFrame:
    """
    Interpreta los centroides de cada cluster en términos de precio relativo, peso de aforo y semántica,
    asignando nombres de arquetipos estandarizados de negocio.
    Si se suministra el objeto kmeans y feature_names, utiliza el motor de asignación geométrica
    en el espacio escalado 25D. En caso contrario, recurre al clasificador heurístico.
    """
    df_res = df_clustered.copy()
    
    # 1. Asignación geométrica basada en centroides escalados 25D
    if kmeans is not None and feature_names is not None:
        mapa_arquetipos = etiquetar_por_centroides_escalados(
            kmeans=kmeans,
            feature_names=feature_names,
            scaler=scaler,
            peso_nlp=peso_nlp
        )
        df_res["arquetipo_demanda"] = df_res[col_cluster].map(mapa_arquetipos)
        return df_res
        
    # 2. Clasificador heurístico de fallback
    clusters_info = []
    for c_id in sorted(df_res[col_cluster].unique()):
        sub = df_res[df_res[col_cluster] == c_id]
        clusters_info.append({
            "cluster": c_id,
            "precio_prom": sub["ratio_precio_max"].mean() if "ratio_precio_max" in sub.columns else 0.0,
            "aforo_prom": sub["peso_aforo"].mean() if "peso_aforo" in sub.columns else 0.0,
            "ocupacion_prom": sub["tasa_ocupacion"].mean() if "tasa_ocupacion" in sub.columns else 0.0,
            "general_share": sub["tag_general"].mean() if "tag_general" in sub.columns else 0.0,
            "palco_vip_share": (sub["tag_palco"].mean() + sub["tag_vip"].mean()) if ("tag_palco" in sub.columns and "tag_vip" in sub.columns) else 0.0,
            "platea_share": sub["tag_platea"].mean() if "tag_platea" in sub.columns else 0.0
        })
        
    df_info = pd.DataFrame(clusters_info)
    mapa_arquetipos = {}
    
    gen_c = df_info.sort_values(by=["aforo_prom", "general_share"], ascending=False).iloc[0]["cluster"]
    mapa_arquetipos[int(gen_c)] = "Grada General / Masiva"
    
    restantes = df_info[df_info["cluster"] != gen_c].copy()
    vip_c = restantes.sort_values(by=["palco_vip_share", "precio_prom"], ascending=False).iloc[0]["cluster"]
    mapa_arquetipos[int(vip_c)] = "VIP / Palcos / Premium"
    
    restantes_2 = restantes[restantes["cluster"] != vip_c].sort_values(by="precio_prom", ascending=False)
    if len(restantes_2) > 0:
        pref_c = restantes_2.iloc[0]["cluster"]
        mapa_arquetipos[int(pref_c)] = "Preferencial / Platea Frontal"
    if len(restantes_2) > 1:
        pop_c = restantes_2.iloc[1]["cluster"]
        mapa_arquetipos[int(pop_c)] = "Popular / Visibilidad Parcial / Balcón"
        
    for _, row in df_info.iterrows():
        c = int(row["cluster"])
        if c not in mapa_arquetipos:
            mapa_arquetipos[c] = f"Segmento #{c}"
            
    df_res["arquetipo_demanda"] = df_res[col_cluster].map(mapa_arquetipos)
    return df_res


def pipeline_clustering_dos_etapas(
    df: pd.DataFrame,
    n_clusters_multizona: int = 4,
    random_state: int = 42,
    peso_nlp: float = 1.2,
    max_tfidf_features: int = 15,
    scaler_type: str = "robust"
) -> Tuple[pd.DataFrame, KMeans, Any, Any, List[str], Dict[str, Any]]:
    """
    Ejecuta el pipeline de clustering en dos etapas (Modelo v2.1):
    1. Etapa 1 (Determinística): Aísla funciones monozona / tarifa plana (~45.5%).
       Se asignan directamente a 'Admisión Única / Tarifa Plana' con cluster = -1.
    2. Etapa 2 (Machine Learning): Construye el espacio vectorial mixto de 25D sobre multi-zona (~54.5%),
       ajusta K-Means y etiqueta los arquetipos mediante geometría de centroides en espacio escalado.
    3. Integración: Reensambla el catálogo unificado asegurando cobertura exacta de filas.
    """
    # 1. Separación a nivel evento
    df_monozona, df_multizona = separar_admision_unica_multizona(df)
    
    df_monozona["cluster"] = -1
    df_monozona["arquetipo_demanda"] = "Admisión Única / Tarifa Plana"
    
    # 2. Espacio mixto sobre multi-zona
    X_multizona, scaler, tfidf_vec, feature_names = construir_espacio_vectorial_mixto(
        df_multizona,
        max_tfidf_features=max_tfidf_features,
        peso_nlp=peso_nlp,
        scaler_type=scaler_type
    )
    
    # 3. K-Means en multi-zona
    kmeans, labels_multi, metricas = entrenar_modelo_clustering(
        X_multizona,
        n_clusters=n_clusters_multizona,
        random_state=random_state
    )
    df_multizona["cluster"] = labels_multi
    
    # 4. Etiquetado por centroides escalados
    mapa_arquetipos = etiquetar_por_centroides_escalados(
        kmeans=kmeans,
        feature_names=feature_names,
        scaler=scaler,
        peso_nlp=peso_nlp
    )
    df_multizona["arquetipo_demanda"] = df_multizona["cluster"].map(mapa_arquetipos)
    
    # 5. Reensamblaje y validación
    df_final = pd.concat([df_monozona, df_multizona], axis=0).sort_index()
    assert len(df_final) == len(df), f"Pérdida de filas: {len(df_final)} != {len(df)}"
    
    return df_final, kmeans, scaler, tfidf_vec, feature_names, metricas


def ejecutar_benchmark_modelos(
    X: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
    sample_size: int = 10000
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta un benchmark comparativo entre 4 familias de algoritmos de clustering:
    1. K-Means (Centroides duros)
    2. Gaussian Mixture Models (GMM - Probabilístico / Soft)
    3. Clustering Jerárquico Aglomerativo (Ward)
    4. HDBSCAN (Basado en densidad y detección de ruido)

    Retorna:
    - df_metricas: Tabla comparativa de métricas de calidad de clustering.
    - df_ari: Matriz de consenso / acuerdo entre modelos (Adjusted Rand Index).
    - dict_modelos: Diccionario con modelos entrenados y sus arrays de etiquetas.
    """
    import time

    np.random.seed(random_state)
    if len(X) > sample_size:
        idx_eval = np.random.choice(len(X), sample_size, replace=False)
        X_eval = X[idx_eval]
    else:
        idx_eval = np.arange(len(X))
        X_eval = X

    dict_modelos = {}
    metricas = []

    # 1. K-Means
    t0 = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    labels_km = kmeans.fit_predict(X)
    t_km = time.time() - t0
    sil_km = float(silhouette_score(X_eval, labels_km[idx_eval]))
    db_km = float(davies_bouldin_score(X, labels_km))
    ch_km = float(calinski_harabasz_score(X, labels_km))

    dict_modelos["kmeans"] = {"modelo": kmeans, "labels": labels_km}
    metricas.append({
        "Modelo": "1. K-Means",
        "Familia": "Centroides (Hard)",
        "N° Clusters": n_clusters,
        "Outliers (%)": "0.0%",
        "Silhouette Score": sil_km,
        "Davies-Bouldin": db_km,
        "Calinski-Harabasz": ch_km,
        "Tiempo (s)": round(t_km, 2)
    })

    # 2. Gaussian Mixture Model (GMM)
    t0 = time.time()
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=5, covariance_type="diag")
    labels_gmm = gmm.fit_predict(X)
    t_gmm = time.time() - t0
    sil_gmm = float(silhouette_score(X_eval, labels_gmm[idx_eval]))
    db_gmm = float(davies_bouldin_score(X, labels_gmm))
    ch_gmm = float(calinski_harabasz_score(X, labels_gmm))

    dict_modelos["gmm"] = {"modelo": gmm, "labels": labels_gmm}
    metricas.append({
        "Modelo": "2. Gaussian Mixture (GMM)",
        "Familia": "Probabilístico (Soft)",
        "N° Clusters": n_clusters,
        "Outliers (%)": "0.0%",
        "Silhouette Score": sil_gmm,
        "Davies-Bouldin": db_gmm,
        "Calinski-Harabasz": ch_gmm,
        "Tiempo (s)": round(t_gmm, 2)
    })

    # 3. Clustering Jerárquico Aglomerativo (Ward)
    t0 = time.time()
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels_agg_sample = agg.fit_predict(X_eval)
    t_agg = time.time() - t0
    sil_agg = float(silhouette_score(X_eval, labels_agg_sample))
    db_agg = float(davies_bouldin_score(X_eval, labels_agg_sample))
    ch_agg = float(calinski_harabasz_score(X_eval, labels_agg_sample))

    dict_modelos["jerarquico"] = {"modelo": agg, "labels_sample": labels_agg_sample}
    metricas.append({
        "Modelo": "3. Jerárquico (Ward)",
        "Familia": "Jerárquico Ascendente",
        "N° Clusters": n_clusters,
        "Outliers (%)": "0.0%",
        "Silhouette Score": sil_agg,
        "Davies-Bouldin": db_agg,
        "Calinski-Harabasz": ch_agg,
        "Tiempo (s)": round(t_agg, 2)
    })

    # 4. HDBSCAN (con parámetros relativos al tamaño del conjunto de datos)
    t0 = time.time()
    min_cluster_size_rel = max(30, int(0.01 * len(X)))
    min_samples_rel = max(10, int(0.002 * len(X)))
    hdb = HDBSCAN(min_cluster_size=min_cluster_size_rel, min_samples=min_samples_rel)
    labels_hdb = hdb.fit_predict(X)
    t_hdb = time.time() - t0

    mask_no_noise = labels_hdb != -1
    n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
    outliers_pct = float((labels_hdb == -1).mean() * 100)

    idx_eval_hdb = [i for i in idx_eval if labels_hdb[i] != -1]
    if len(idx_eval_hdb) > 100 and n_clusters_hdb > 1:
        sil_hdb = float(silhouette_score(X[idx_eval_hdb], labels_hdb[idx_eval_hdb]))
        db_hdb = float(davies_bouldin_score(X[mask_no_noise], labels_hdb[mask_no_noise]))
        ch_hdb = float(calinski_harabasz_score(X[mask_no_noise], labels_hdb[mask_no_noise]))
    else:
        sil_hdb, db_hdb, ch_hdb = np.nan, np.nan, np.nan

    dict_modelos["hdbscan"] = {"modelo": hdb, "labels": labels_hdb}
    metricas.append({
        "Modelo": "4. HDBSCAN",
        "Familia": "Densidad no paramétrica",
        "N° Clusters": n_clusters_hdb,
        "Outliers (%)": f"{outliers_pct:.1f}%",
        "Silhouette Score": sil_hdb,
        "Davies-Bouldin": db_hdb,
        "Calinski-Harabasz": ch_hdb,
        "Tiempo (s)": round(t_hdb, 2)
    })

    df_metricas = pd.DataFrame(metricas)

    # Matriz de Consenso / Acuerdo (Adjusted Rand Index)
    ari_matrix = pd.DataFrame(
        [
            [1.0, float(adjusted_rand_score(labels_km, labels_gmm)), float(adjusted_rand_score(labels_km[idx_eval], labels_agg_sample))],
            [float(adjusted_rand_score(labels_gmm, labels_km)), 1.0, float(adjusted_rand_score(labels_gmm[idx_eval], labels_agg_sample))],
            [float(adjusted_rand_score(labels_agg_sample, labels_km[idx_eval])), float(adjusted_rand_score(labels_agg_sample, labels_gmm[idx_eval])), 1.0]
        ],
        index=["K-Means", "GMM", "Jerárquico"],
        columns=["K-Means", "GMM", "Jerárquico"]
    )

    return df_metricas, ari_matrix, dict_modelos
