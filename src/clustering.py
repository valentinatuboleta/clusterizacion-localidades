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
from typing import Dict, List, Tuple, Any, Optional, Union
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
    peso_nlp: float = 0.2,
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
    Metodología rigurosa: Todas las métricas se evalúan sobre la misma muestra X_eval usando RandomState local.
    """
    metricas = []
    
    if len(X) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), sample_size, replace=False)
        X_eval = X[idx]
    else:
        idx = np.arange(len(X))
        X_eval = X

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        labels_eval = labels[idx]
        
        sil = float(silhouette_score(X_eval, labels_eval))
        ch = float(calinski_harabasz_score(X_eval, labels_eval))
        db = float(davies_bouldin_score(X_eval, labels_eval))
        
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
    peso_nlp: float = 0.2,
    n_clusters: int = 5
) -> Dict[str, np.ndarray]:
    """
    Construye las representaciones vectoriales ideales para cada arquetipo multi-zona
    dentro del espacio geométrico escalado de 25 dimensiones.
    
    Refactor robusto: Mapea cada dimensión explícitamente por nombre en lugar de
    asumir posiciones fijas o slices posicionales [3:].
    """
    if n_clusters == 4:
        perfiles_config = {
            "VIP / Palcos / Premium": {
                "num": {"ratio_precio_max": 0.90, "percentil_precio_evento": 0.85, "peso_aforo": 0.08},
                "tags": {"tag_palco": 0.6, "tag_vip": 0.4},
                "words": {"tfidf_palco": 0.5, "tfidf_mesa": 0.3}
            },
            "Preferencial / Platea Frontal": {
                "num": {"ratio_precio_max": 0.80, "percentil_precio_evento": 0.75, "peso_aforo": 0.20},
                "tags": {"tag_platea": 0.8, "tag_preferencial": 0.3},
                "words": {"tfidf_platea": 0.5, "tfidf_central": 0.3}
            },
            "Popular / Visibilidad Parcial / Balcón": {
                "num": {"ratio_precio_max": 0.35, "percentil_precio_evento": 0.30, "peso_aforo": 0.15},
                "tags": {"tag_balcon": 0.4, "tag_piso_alto": 0.4},
                "words": {"tfidf_balcon": 0.4, "tfidf_piso": 0.3, "tfidf_posterior": 0.3}
            },
            "Grada General / Masiva": {
                "num": {"ratio_precio_max": 0.60, "percentil_precio_evento": 0.50, "peso_aforo": 0.60},
                "tags": {"tag_general": 0.6},
                "words": {"tfidf_general": 0.5}
            }
        }
    else:
        # Configuración para k=5 (óptimo de codo y mínimo Davies-Bouldin en multi-zona)
        perfiles_config = {
            "VIP / Palcos / Premium": {
                "num": {"ratio_precio_max": 0.90, "percentil_precio_evento": 0.85, "peso_aforo": 0.08},
                "tags": {"tag_palco": 0.6, "tag_vip": 0.4},
                "words": {"tfidf_palco": 0.5, "tfidf_mesa": 0.3}
            },
            "Preferencial / Platea Frontal": {
                "num": {"ratio_precio_max": 0.85, "percentil_precio_evento": 0.82, "peso_aforo": 0.10},
                "tags": {"tag_platea": 0.8, "tag_preferencial": 0.3},
                "words": {"tfidf_platea": 0.5, "tfidf_central": 0.3}
            },
            "Platea General / Intermedia": {
                "num": {"ratio_precio_max": 0.80, "percentil_precio_evento": 0.70, "peso_aforo": 0.35},
                "tags": {"tag_platea": 0.4},
                "words": {"tfidf_platea": 0.3}
            },
            "Grada General / Masiva": {
                "num": {"ratio_precio_max": 0.75, "percentil_precio_evento": 0.60, "peso_aforo": 0.75},
                "tags": {"tag_general": 0.6},
                "words": {"tfidf_general": 0.5}
            },
            "Popular / Balcón / Visibilidad Parcial": {
                "num": {"ratio_precio_max": 0.35, "percentil_precio_evento": 0.30, "peso_aforo": 0.12},
                "tags": {"tag_balcon": 0.4, "tag_piso_alto": 0.4},
                "words": {"tfidf_balcon": 0.4, "tfidf_piso": 0.3, "tfidf_posterior": 0.3}
            }
        }
    
    # Identificar nombres y orden de columnas numéricas que maneja el scaler
    cols_num_scaler = []
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        cols_num_scaler = list(scaler.feature_names_in_)
    else:
        cols_num_scaler = [f for f in feature_names if not f.startswith("tag_") and not f.startswith("tfidf_")]
        
    perfiles_vectores = {}
    for nombre, cfg in perfiles_config.items():
        num_dict = cfg.get("num", {})
        if len(cols_num_scaler) > 0:
            num_row = [num_dict.get(c, 0.0) for c in cols_num_scaler]
            if scaler is not None and hasattr(scaler, "transform"):
                num_scaled_vals = scaler.transform([num_row])[0]
            else:
                num_scaled_vals = np.array(num_row, dtype=float)
            scaled_num_map = dict(zip(cols_num_scaler, num_scaled_vals))
        else:
            scaled_num_map = {}

        vec = []
        for feat in feature_names:
            if feat in scaled_num_map:
                vec.append(scaled_num_map[feat])
            elif feat.startswith("tag_"):
                vec.append(cfg.get("tags", {}).get(feat, 0.0))
            elif feat.startswith("tfidf_"):
                vec.append(cfg.get("words", {}).get(feat, 0.0) * peso_nlp)
            else:
                vec.append(num_dict.get(feat, 0.0))
                
        perfiles_vectores[nombre] = np.array(vec, dtype=float)
        
    return perfiles_vectores


def etiquetar_por_centroides_escalados(
    kmeans: KMeans,
    feature_names: List[str],
    scaler: Any = None,
    peso_nlp: float = 0.2
) -> Dict[int, str]:
    """
    Asigna arquetipos a los clusters evaluando la distancia euclidiana entre los centroides
    reales del modelo (en el espacio transformado de 25D) y los perfiles ideales de negocio.
    
    Aplica el algoritmo de asignación óptima 1 a 1 (Hungarian / Munkres) para garantizar
    una correspondencia biyectiva estricta sin duplicidades ni heurísticas frágiles de ordenamiento.
    """
    centroids = kmeans.cluster_centers_
    k = len(centroids)
    
    perfiles_dict = construir_perfiles_ideales_escalados(feature_names, scaler, peso_nlp=peso_nlp, n_clusters=k)
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
    peso_nlp: float = 0.2
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
        if c_id == -1:
            continue
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
    if -1 in df_res[col_cluster].values:
        mapa_arquetipos[-1] = "Admisión Única / Tarifa Plana"
        
    if len(df_info) > 0:
        # 1. Grada General / Masiva (mayor aforo o share general)
        gen_c = df_info.sort_values(by=["aforo_prom", "general_share"], ascending=False).iloc[0]["cluster"]
        mapa_arquetipos[int(gen_c)] = "Grada General / Masiva"
        
        # 2. VIP / Palcos (mayor activación de palcos/VIP)
        restantes = df_info[df_info["cluster"] != gen_c].copy()
        if len(restantes) > 0:
            vip_c = restantes.sort_values(by=["palco_vip_share", "precio_prom"], ascending=False).iloc[0]["cluster"]
            mapa_arquetipos[int(vip_c)] = "VIP / Palcos / Premium"
            
            restantes_2 = restantes[restantes["cluster"] != vip_c].sort_values(by="precio_prom", ascending=False)
            if len(restantes_2) == 1:
                mapa_arquetipos[int(restantes_2.iloc[0]["cluster"])] = "Preferencial / Platea Frontal"
            elif len(restantes_2) == 2:
                mapa_arquetipos[int(restantes_2.iloc[0]["cluster"])] = "Preferencial / Platea Frontal"
                mapa_arquetipos[int(restantes_2.iloc[1]["cluster"])] = "Popular / Visibilidad Parcial / Balcón"
            elif len(restantes_2) >= 3:
                # Caso estándar k=5 multi-zona
                mapa_arquetipos[int(restantes_2.iloc[0]["cluster"])] = "Preferencial / Platea Frontal"
                mapa_arquetipos[int(restantes_2.iloc[1]["cluster"])] = "Platea General / Intermedia"
                mapa_arquetipos[int(restantes_2.iloc[-1]["cluster"])] = "Popular / Balcón / Visibilidad Parcial"
                
        for _, row in df_info.iterrows():
            c = int(row["cluster"])
            if c not in mapa_arquetipos:
                mapa_arquetipos[c] = f"Segmento #{c}"
                
    df_res["arquetipo_demanda"] = df_res[col_cluster].map(mapa_arquetipos)
    return df_res


def pipeline_clustering_dos_etapas(
    df: pd.DataFrame,
    n_clusters_multizona: Union[int, str] = 5,
    random_state: int = 42,
    peso_nlp: float = 0.2,
    max_tfidf_features: int = 15,
    scaler_type: str = "robust"
) -> Tuple[pd.DataFrame, KMeans, Any, Any, List[str], Dict[str, Any]]:
    """
    Ejecuta el pipeline de clustering en dos etapas (Modelo v2.1 optimizado):
    1. Etapa 1 (Determinística): Aísla funciones monozona / tarifa plana (~45.5%).
       Se asignan directamente a 'Admisión Única / Tarifa Plana' con cluster = -1.
    2. Etapa 2 (Machine Learning): Construye el espacio vectorial mixto de 25D sobre multi-zona (~54.5%)
       con peso_nlp calibrado en 0.2 para evitar dilución dimensional.
       Ajusta K-Means con k óptimo (k=5 por defecto, determinado por codo ortogonal y mínimo Davies-Bouldin,
       o selección automática balanceada mediante n_clusters_multizona='auto') y etiqueta mediante geometría húngara.
    3. Integración: Reensambla el catálogo unificado asegurando cobertura exacta del 100% de filas.
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
    
    # Selección automática de k si se solicita 'auto' mediante score compuesto (Silueta + Davies-Bouldin)
    if isinstance(n_clusters_multizona, str) and n_clusters_multizona.lower() == "auto":
        cand_ks = [4, 5, 6, 7]
        sil_cands = []
        db_cands = []
        sample_eval_sz = min(5000, len(X_multizona))
        for cand_k in cand_ks:
            km_cand = KMeans(n_clusters=cand_k, random_state=random_state, n_init=10)
            lbls = km_cand.fit_predict(X_multizona)
            sil_cands.append(silhouette_score(X_multizona[:sample_eval_sz], lbls[:sample_eval_sz]))
            db_cands.append(davies_bouldin_score(X_multizona, lbls))
        s_arr = np.array(sil_cands)
        d_arr = np.array(db_cands)
        norm_s = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-8)
        norm_d = (d_arr.max() - d_arr) / (d_arr.max() - d_arr.min() + 1e-8)
        best_idx = int(np.argmax(norm_s + norm_d))
        n_clusters_multizona = cand_ks[best_idx]
    else:
        n_clusters_multizona = int(n_clusters_multizona)
    
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
    n_clusters: int = 5,
    random_state: int = 42,
    sample_size: int = 10000
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta un benchmark comparativo entre 4 familias de algoritmos de clustering:
    1. K-Means (Centroides duros)
    2. Gaussian Mixture Models (GMM - Probabilístico / Soft)
    3. Clustering Jerárquico Aglomerativo (Ward)
    4. HDBSCAN (Basado en densidad y detección de ruido)

    Metodología rigurosa: Todas las métricas (Silhouette, Davies-Bouldin, Calinski-Harabasz)
    se calculan sobre la misma submuestra idéntica X_eval para garantizar comparabilidad estricta,
    usando un generador RandomState local.

    Retorna:
    - df_metricas: Tabla comparativa de métricas de calidad de clustering.
    - df_ari: Matriz de consenso / acuerdo 4x4 entre todos los modelos (Adjusted Rand Index).
    - dict_modelos: Diccionario con modelos entrenados y sus arrays de etiquetas.
    """
    import time

    rng = np.random.RandomState(random_state)
    if len(X) > sample_size:
        idx_eval = rng.choice(len(X), sample_size, replace=False)
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
    labels_km_eval = labels_km[idx_eval]
    sil_km = float(silhouette_score(X_eval, labels_km_eval))
    db_km = float(davies_bouldin_score(X_eval, labels_km_eval))
    ch_km = float(calinski_harabasz_score(X_eval, labels_km_eval))

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
    labels_gmm_eval = labels_gmm[idx_eval]
    sil_gmm = float(silhouette_score(X_eval, labels_gmm_eval))
    db_gmm = float(davies_bouldin_score(X_eval, labels_gmm_eval))
    ch_gmm = float(calinski_harabasz_score(X_eval, labels_gmm_eval))

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

    labels_hdb_eval = labels_hdb[idx_eval]
    mask_eval_hdb = labels_hdb_eval != -1
    n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
    outliers_pct = float((labels_hdb == -1).mean() * 100)

    if mask_eval_hdb.sum() > 100 and n_clusters_hdb > 1:
        sil_hdb = float(silhouette_score(X_eval[mask_eval_hdb], labels_hdb_eval[mask_eval_hdb]))
        db_hdb = float(davies_bouldin_score(X_eval[mask_eval_hdb], labels_hdb_eval[mask_eval_hdb]))
        ch_hdb = float(calinski_harabasz_score(X_eval[mask_eval_hdb], labels_hdb_eval[mask_eval_hdb]))
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

    # Matriz de Consenso / Acuerdo 4x4 (Adjusted Rand Index) evaluada sobre puntos válidos de X_eval
    if mask_eval_hdb.sum() > 10:
        ari_km_hdb = float(adjusted_rand_score(labels_km_eval[mask_eval_hdb], labels_hdb_eval[mask_eval_hdb]))
        ari_gmm_hdb = float(adjusted_rand_score(labels_gmm_eval[mask_eval_hdb], labels_hdb_eval[mask_eval_hdb]))
        ari_agg_hdb = float(adjusted_rand_score(labels_agg_sample[mask_eval_hdb], labels_hdb_eval[mask_eval_hdb]))
    else:
        ari_km_hdb, ari_gmm_hdb, ari_agg_hdb = np.nan, np.nan, np.nan

    ari_matrix = pd.DataFrame(
        [
            [1.0, float(adjusted_rand_score(labels_km_eval, labels_gmm_eval)), float(adjusted_rand_score(labels_km_eval, labels_agg_sample)), ari_km_hdb],
            [float(adjusted_rand_score(labels_gmm_eval, labels_km_eval)), 1.0, float(adjusted_rand_score(labels_gmm_eval, labels_agg_sample)), ari_gmm_hdb],
            [float(adjusted_rand_score(labels_agg_sample, labels_km_eval)), float(adjusted_rand_score(labels_agg_sample, labels_gmm_eval)), 1.0, ari_agg_hdb],
            [ari_km_hdb, ari_gmm_hdb, ari_agg_hdb, 1.0]
        ],
        index=["K-Means", "GMM", "Jerárquico", "HDBSCAN"],
        columns=["K-Means", "GMM", "Jerárquico", "HDBSCAN"]
    )

    return df_metricas, ari_matrix, dict_modelos
