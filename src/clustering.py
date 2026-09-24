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
import joblib
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


MODEL_VERSION = "2.3"

# Categorias canonicas estandarizadas de type_site (excluyendo 'desconocido')
CANONICAL_TYPE_SITE_CATEGORIES = [
    "arena_cubierta",
    "auditorio",
    "bar_club",
    "centro_eventos_carpa",
    "cine_sala_cultural",
    "estadio_abierto",
    "otro",
    "parque_aire_libre",
    "teatro"
]

# Lista por defecto de variables estructurales numéricas ex-ante (sin tasa de ocupación)
DEFAULT_NUMERIC_FEATURES = [
    "ratio_precio_max",
    "percentil_precio_evento",
    "peso_aforo",
    "percentil_precio_absoluto_dentro_tipo"
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

VARIABLES_MONITOREO_DRIFT = [
    "ratio_precio_max",
    "percentil_precio_evento",
    "peso_aforo",
    "percentil_precio_absoluto_dentro_tipo",
    "tag_palco",
    "tag_vip",
    "tag_platea",
    "tag_preferencial",
    "tag_general",
    "tag_balcon",
    "tag_piso_alto"
]

DISTRIBUCION_ESPERADA_ARQUETIPOS = {
    "Admisión Única / Tarifa Plana": 15375 / 33775,
    "Popular / Balcón / Visibilidad Parcial": 5861 / 33775,
    "VIP / Palcos / Premium": 5070 / 33775,
    "Platea General / Intermedia": 3270 / 33775,
    "Preferencial / Platea Frontal": 3229 / 33775,
    "Grada General / Masiva": 970 / 33775,
}




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
    
    # 1. Metricas agrupadas a nivel evento
    if "peso_aforo" in df_work.columns:
        max_peso_aforo_evento = df_work.groupby("t_performance_id")["peso_aforo"].transform("max")
        es_monozona = (df_work["peso_aforo"] >= 0.99) | (max_peso_aforo_evento >= 0.99)
    else:
        n_localidades_evento = df_work.groupby("t_performance_id")["dn_quota"].transform("count")
        es_monozona = (n_localidades_evento == 1)
    
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
    peso_type_site: float = 0.5,
    categorias_type_site: Optional[List[str]] = None,
    scaler_type: str = "robust",
    scaler: Optional[Any] = None,
    tfidf_vectorizer: Optional[Any] = None
) -> Tuple[np.ndarray, Any, Any, List[str]]:
    """
    Construye el espacio vectorial mixto de 35 dimensiones (Modelo v2.3) combinando:
    1. Metricas numericas relativas ex-ante escaladas (4 variables: ratio_precio_max, percentil_precio_evento,
       peso_aforo, percentil_precio_absoluto_dentro_tipo).
    2. Tags estructurales densos (7 variables: 5 de jerarquia comercial + 2 de nivel vertical).
    3. One-hot encoding de type_site ponderado (9 categorias canonicas ponderadas por peso_type_site,
       excluyendo 'desconocido' del one-hot pero preservando su flag).
    4. Embeddings/TF-IDF del texto limpio de la localidad (15 n-gramas) ponderados por peso_nlp.
    
    Total de dimensiones por defecto: 4 + 7 + 9 + 15 = 35 dimensiones.
    """
    df_work = df.copy()
    if "type_site" not in df_work.columns:
        from src.feature_engineering import enriquecer_type_site
        df_work = enriquecer_type_site(df_work)
    if "percentil_precio_absoluto_dentro_tipo" in columnas_numericas and "percentil_precio_absoluto_dentro_tipo" not in df_work.columns:
        from src.feature_engineering import calcular_percentil_precio_absoluto_dentro_tipo
        df_work = calcular_percentil_precio_absoluto_dentro_tipo(df_work)

    # 1. Variables numericas continuas
    for c in columnas_numericas:
        if c not in df_work.columns:
            df_work[c] = 0.50 if "percentil" in c else 0.0
    cols_num_presentes = list(columnas_numericas)
    X_num = df_work[cols_num_presentes].values
    
    if len(cols_num_presentes) > 0:
        if scaler is None:
            scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
            X_num_scaled = scaler.fit_transform(X_num)
        else:
            X_num_scaled = scaler.transform(X_num)
    else:
        X_num_scaled = np.empty((len(df_work), 0))
        
    feature_names = list(cols_num_presentes)
    
    # 2. Variables binarias (Tags) en su escala natural [0, 1]
    for c in columnas_tags:
        if c not in df_work.columns:
            df_work[c] = 0.0
    cols_tags_presentes = list(columnas_tags)
    X_tags = df_work[cols_tags_presentes].values.astype(float)
    feature_names.extend(cols_tags_presentes)

    # 2b. One-hot de type_site ponderado por peso_type_site en el bloque de tags
    if categorias_type_site is None:
        cats_to_use = [c for c in CANONICAL_TYPE_SITE_CATEGORIES if c != "desconocido"]
    else:
        cats_to_use = [c for c in categorias_type_site if c != "desconocido"]

    if len(cats_to_use) > 0:
        s_type = df_work["type_site"].astype(str) if "type_site" in df_work.columns else pd.Series(["desconocido"] * len(df_work), index=df_work.index)
        if peso_type_site > 0.0:
            X_type_site = np.column_stack([
                (s_type == cat).values.astype(float) * peso_type_site for cat in cats_to_use
            ])
        else:
            X_type_site = np.zeros((len(df_work), len(cats_to_use)), dtype=float)
        cols_type_names = [f"type_site_{cat}" for cat in cats_to_use]
        feature_names.extend(cols_type_names)
    else:
        X_type_site = np.empty((len(df_work), 0))
        
    # 3. Concatenacion de bloques estructurados
    bloques = []
    if X_num_scaled.shape[1] > 0:
        bloques.append(X_num_scaled)
    if X_tags.shape[1] > 0:
        bloques.append(X_tags)
    if X_type_site.shape[1] > 0:
        bloques.append(X_type_site)
    
    # 4. TF-IDF sobre texto limpio
    if usar_tfidf_texto and "texto_limpio" in df_work.columns:
        X_tfidf, tfidf_vectorizer = vectorizar_texto_limpio(
            df_work["texto_limpio"],
            max_features=max_tfidf_features,
            vectorizer=tfidf_vectorizer
        )
        X_tfidf_weighted = X_tfidf * peso_nlp
        bloques.append(X_tfidf_weighted)
        tfidf_vocab = [f"tfidf_{w}" for w in tfidf_vectorizer.get_feature_names_out()]
        feature_names.extend(tfidf_vocab)
        
    X_mixto = np.hstack(bloques) if len(bloques) > 0 else np.empty((len(df_work), 0))
        
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
                "num": {"ratio_precio_max": 0.90, "percentil_precio_evento": 0.85, "peso_aforo": 0.08, "percentil_precio_absoluto_dentro_tipo": 0.90},
                "tags": {"tag_palco": 0.6, "tag_vip": 0.4},
                "words": {"tfidf_palco": 0.5, "tfidf_mesa": 0.3}
            },
            "Preferencial / Platea Frontal": {
                "num": {"ratio_precio_max": 0.80, "percentil_precio_evento": 0.75, "peso_aforo": 0.20, "percentil_precio_absoluto_dentro_tipo": 0.80},
                "tags": {"tag_platea": 0.8, "tag_preferencial": 0.3},
                "words": {"tfidf_platea": 0.5, "tfidf_central": 0.3}
            },
            "Popular / Visibilidad Parcial / Balcón": {
                "num": {"ratio_precio_max": 0.35, "percentil_precio_evento": 0.30, "peso_aforo": 0.15, "percentil_precio_absoluto_dentro_tipo": 0.30},
                "tags": {"tag_balcon": 0.4, "tag_piso_alto": 0.4},
                "words": {"tfidf_balcon": 0.4, "tfidf_piso": 0.3, "tfidf_posterior": 0.3}
            },
            "Grada General / Masiva": {
                "num": {"ratio_precio_max": 0.60, "percentil_precio_evento": 0.50, "peso_aforo": 0.60, "percentil_precio_absoluto_dentro_tipo": 0.50},
                "tags": {"tag_general": 0.6},
                "words": {"tfidf_general": 0.5}
            }
        }
    else:
        # Configuración para k=5 (óptimo de codo y mínimo Davies-Bouldin en multi-zona)
        perfiles_config = {
            "VIP / Palcos / Premium": {
                "num": {"ratio_precio_max": 0.90, "percentil_precio_evento": 0.85, "peso_aforo": 0.08, "percentil_precio_absoluto_dentro_tipo": 0.90},
                "tags": {"tag_palco": 0.6, "tag_vip": 0.4},
                "words": {"tfidf_palco": 0.5, "tfidf_mesa": 0.3}
            },
            "Preferencial / Platea Frontal": {
                "num": {"ratio_precio_max": 0.85, "percentil_precio_evento": 0.82, "peso_aforo": 0.10, "percentil_precio_absoluto_dentro_tipo": 0.82},
                "tags": {"tag_platea": 0.8, "tag_preferencial": 0.3},
                "words": {"tfidf_platea": 0.5, "tfidf_central": 0.3}
            },
            "Platea General / Intermedia": {
                "num": {"ratio_precio_max": 0.80, "percentil_precio_evento": 0.70, "peso_aforo": 0.35, "percentil_precio_absoluto_dentro_tipo": 0.65},
                "tags": {"tag_platea": 0.4},
                "words": {"tfidf_platea": 0.3}
            },
            "Grada General / Masiva": {
                "num": {"ratio_precio_max": 0.75, "percentil_precio_evento": 0.60, "peso_aforo": 0.75, "percentil_precio_absoluto_dentro_tipo": 0.50},
                "tags": {"tag_general": 0.6},
                "words": {"tfidf_general": 0.5}
            },
            "Popular / Balcón / Visibilidad Parcial": {
                "num": {"ratio_precio_max": 0.35, "percentil_precio_evento": 0.30, "peso_aforo": 0.12, "percentil_precio_absoluto_dentro_tipo": 0.30},
                "tags": {"tag_balcon": 0.4, "tag_piso_alto": 0.4},
                "words": {"tfidf_balcon": 0.4, "tfidf_piso": 0.3, "tfidf_posterior": 0.3}
            }
        }
    
    # Identificar nombres y orden de columnas numéricas que maneja el scaler
    cols_num_scaler = []
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        cols_num_scaler = list(scaler.feature_names_in_)
    else:
        cols_num_scaler = [
            f for f in feature_names 
            if not f.startswith("tag_") and not f.startswith("tfidf_") and not f.startswith("type_site_")
        ]
        
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
            elif feat.startswith("type_site_"):
                vec.append(0.0)
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
    peso_type_site: float = 0.5,
    categorias_type_site: Optional[List[str]] = None,
    max_tfidf_features: int = 15,
    scaler_type: str = "robust"
) -> Tuple[pd.DataFrame, KMeans, Any, Any, List[str], Dict[str, Any]]:
    """
    Ejecuta el pipeline de clustering en dos etapas (Modelo v2.3 optimizado):
    1. Etapa 1 (Determinística): Aísla funciones monozona / tarifa plana (~45.5%).
       Se asignan directamente a 'Admisión Única / Tarifa Plana' con cluster = -1.
    2. Etapa 2 (Machine Learning): Construye el espacio vectorial mixto de 35D sobre multi-zona (~54.5%)
       con peso_nlp calibrado en 0.2 y peso_type_site en 0.5 para enriquecimiento por tipo de venue.
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
        peso_type_site=peso_type_site,
        categorias_type_site=categorias_type_site,
        scaler_type=scaler_type
    )

    
    # Selección automática de k si se solicita 'auto' mediante Score Compuesto Codo-DB
    # Combina la distancia ortogonal a la cuerda en la curva de inercia (codo) y la minimización de Davies-Bouldin
    if isinstance(n_clusters_multizona, str) and n_clusters_multizona.lower() == "auto":
        cand_ks = [3, 4, 5, 6, 7]
        inertias = []
        db_cands = []
        for cand_k in cand_ks:
            km_cand = KMeans(n_clusters=cand_k, random_state=random_state, n_init=10)
            lbls = km_cand.fit_predict(X_multizona)
            inertias.append(km_cand.inertia_)
            db_cands.append(davies_bouldin_score(X_multizona, lbls))
            
        # Distancia ortogonal a la secante (codo)
        P1 = np.array([cand_ks[0], inertias[0], 0.0])
        P2 = np.array([cand_ks[-1], inertias[-1], 0.0])
        vec_secante = P2 - P1
        norm_secante = np.linalg.norm(vec_secante)
        
        codo_dists = []
        for i, k_val in enumerate(cand_ks):
            P0 = np.array([k_val, inertias[i], 0.0])
            d = np.linalg.norm(np.cross(vec_secante, P1 - P0)) / norm_secante
            codo_dists.append(d)
            
        codo_arr = np.array(codo_dists)
        d_arr = np.array(db_cands)
        
        norm_codo = (codo_arr - codo_arr.min()) / (codo_arr.max() - codo_arr.min() + 1e-8)
        norm_db = (d_arr.max() - d_arr) / (d_arr.max() - d_arr.min() + 1e-8)
        
        score_compuesto = norm_codo + norm_db
        best_idx = int(np.argmax(score_compuesto))
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

    # 4b. Ajuste de GMM para probabilidades posteriores alineadas a los centroides
    gmm = GaussianMixture(
        n_components=n_clusters_multizona,
        random_state=random_state,
        covariance_type="diag",
        means_init=kmeans.cluster_centers_
    )
    gmm.fit(X_multizona)
    metricas["gmm"] = gmm
    metricas["mapa_arquetipos"] = mapa_arquetipos

    # Distancias a centroides para margen geometrico, frontera y segundo arquetipo
    distancias = kmeans.transform(X_multizona)
    orden_dist = np.argsort(distancias, axis=1)
    d1 = distancias[np.arange(len(distancias)), orden_dist[:, 0]]
    d2 = distancias[np.arange(len(distancias)), orden_dist[:, 1]]
    margen = (d2 - d1) / (d2 + 1e-9)

    df_multizona["score_confianza"] = margen
    df_multizona["es_frontera"] = margen < 0.15
    df_multizona["segundo_arquetipo"] = [mapa_arquetipos.get(c, str(c)) for c in orden_dist[:, 1]]

    # Cobertura de vocabulario TF-IDF (tokens en minusculas contra vocabulario congelado)
    vocab = set(tfidf_vec.get_feature_names_out()) if hasattr(tfidf_vec, "get_feature_names_out") else set()
    def _calc_cobertura(texto):
        tokens = str(texto).lower().split()
        if not tokens:
            return 0.0
        return sum(t in vocab for t in tokens) / len(tokens)

    if "texto_limpio" in df_multizona.columns:
        df_multizona["cobertura_texto"] = df_multizona["texto_limpio"].apply(_calc_cobertura)
    else:
        df_multizona["cobertura_texto"] = 0.0
    df_multizona["texto_casi_vacio"] = df_multizona["cobertura_texto"] < 0.20

    probs_gmm = gmm.predict_proba(X_multizona)
    df_multizona["probabilidad_gmm"] = probs_gmm.max(axis=1)

    # Variables de observabilidad para monozona (determinísticas por diseño de evento)
    df_monozona["score_confianza"] = 1.0
    df_monozona["es_frontera"] = False
    df_monozona["segundo_arquetipo"] = None
    if "texto_limpio" in df_monozona.columns:
        df_monozona["cobertura_texto"] = df_monozona["texto_limpio"].apply(_calc_cobertura)
    else:
        df_monozona["cobertura_texto"] = 0.0
    df_monozona["texto_casi_vacio"] = df_monozona["cobertura_texto"] < 0.20
    df_monozona["probabilidad_gmm"] = 1.0

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


# ==============================================================================
# PERSISTENCIA E INFERENCIA DE PRODUCCIÓN (JOB_LIB + ETAPA 1 DETERMINÍSTICA)
# ==============================================================================

def generar_referencia_drift(
    df_referencia: pd.DataFrame,
    variables: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Genera los histogramas de referencia (bin_edges y esperado_pct) para monitoreo de drift (PSI).
    """
    vars_to_monitor = variables or VARIABLES_MONITOREO_DRIFT
    referencia = {
        "variables": {},
        "distribucion_arquetipos": DISTRIBUCION_ESPERADA_ARQUETIPOS.copy()
    }
    for v in vars_to_monitor:
        if v in df_referencia.columns:
            valores = df_referencia[v].dropna().values
            if len(valores) > 0:
                v_max = float(valores.max())
                rango = (0.0, 1.0) if v_max <= 1.05 else (0.0, float(np.percentile(valores, 99.5)))
                counts, bin_edges = np.histogram(valores, bins=10, range=rango)
                total = counts.sum()
                esperado_pct = counts / total if total > 0 else np.ones(10) / 10.0
                referencia["variables"][v] = {
                    "esperado_pct": esperado_pct.tolist(),
                    "bin_edges": bin_edges.tolist()
                }
    return referencia


def calcular_psi(actual: np.ndarray, esperado: np.ndarray, eps: float = 1e-4) -> float:
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones (frecuencias o porcentajes).
    Formula: sum((actual% - esperado%) * ln(actual% / esperado%))
    """
    act = np.asarray(actual, dtype=float)
    esp = np.asarray(esperado, dtype=float)
    
    total_act = act.sum()
    total_esp = esp.sum()
    
    if total_act == 0 or total_esp == 0:
        return 0.0
        
    a = act / total_act
    e = esp / total_esp
    
    a = np.clip(a, eps, None)
    a = a / a.sum()
    
    e = np.clip(e, eps, None)
    e = e / e.sum()
    
    psi_val = np.sum((a - e) * np.log(a / e))
    return float(psi_val)


def guardar_modelo_clustering(
    filepath: str,
    kmeans: KMeans,
    scaler: Any,
    tfidf_vectorizer: Any,
    feature_names: List[str],
    mapa_arquetipos: Dict[int, str],
    metricas: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    gmm: Optional[GaussianMixture] = None,
    referencia_drift: Optional[Dict[str, Any]] = None,
    df_referencia: Optional[pd.DataFrame] = None,
    distribucion_percentil_tipo: Optional[Dict[str, Any]] = None,
    categorias_type_site: Optional[List[str]] = None
) -> None:
    """
    Persiste el pipeline de clusterizacion entrenado en un archivo .joblib.
    Incluye transformadores, modelo K-Means, modelo GMM, referencia drift PSI,
    distribucion de referencia de percentil por tipo, categorias type_site,
    vocabulario de features, mapa de arquetipos y metadatos de version.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    metricas_dict = metricas or {}
    
    if gmm is None:
        gmm = metricas_dict.get("gmm", None)

    if referencia_drift is None:
        if df_referencia is not None:
            referencia_drift = generar_referencia_drift(df_referencia)
        elif "referencia_drift" in metricas_dict:
            referencia_drift = metricas_dict["referencia_drift"]
        else:
            referencia_drift = {
                "variables": {},
                "distribucion_arquetipos": DISTRIBUCION_ESPERADA_ARQUETIPOS.copy()
            }

    if distribucion_percentil_tipo is None:
        if df_referencia is not None:
            from src.feature_engineering import generar_referencia_percentil_tipo
            distribucion_percentil_tipo = generar_referencia_percentil_tipo(df_referencia)
        elif "distribucion_percentil_tipo" in metricas_dict:
            distribucion_percentil_tipo = metricas_dict["distribucion_percentil_tipo"]
        else:
            distribucion_percentil_tipo = {}

    if categorias_type_site is None:
        categorias_type_site = (
            metricas_dict.get("categorias_type_site")
            or [f.replace("type_site_", "") for f in feature_names if f.startswith("type_site_")]
            or CANONICAL_TYPE_SITE_CATEGORIES
        )

    payload = {
        "version": MODEL_VERSION,
        "kmeans": kmeans,
        "scaler": scaler,
        "tfidf_vectorizer": tfidf_vectorizer,
        "feature_names": feature_names,
        "mapa_arquetipos": mapa_arquetipos,
        "metricas": metricas_dict,
        "metadata": metadata or {},
        "gmm": gmm,
        "referencia_drift": referencia_drift,
        "distribucion_percentil_tipo": distribucion_percentil_tipo,
        "categorias_type_site": categorias_type_site
    }
    joblib.dump(payload, filepath)


def cargar_modelo_clustering(filepath: str) -> Dict[str, Any]:
    """
    Carga un pipeline de clusterización previamente persistido con joblib.
    """
    return joblib.load(filepath)


def predecir_arquetipos_demanda(
    df: pd.DataFrame,
    modelo: Union[str, Dict[str, Any]],
    peso_nlp: float = 0.2
) -> pd.DataFrame:
    """
    Funcion de inferencia bietapica con observabilidad completa para produccion:
    1. Etapa 1 (Deterministica): Evalua funciones a nivel evento (t_performance_id).
       Si es monozona (1 sola localidad o aforo >= 99%), asigna 'Admisión Única / Tarifa Plana' (cluster = -1).
       Score de confianza = 1.0, es_frontera = False, segundo_arquetipo = None, probabilidad_gmm = 1.0.
    2. Etapa 2 (Machine Learning): Transforma las localidades multi-zona usando scaler,
       vectorizador TF-IDF, percentil referenciado por type_site y feature_names persistidos,
       predice con K-Means y GMM. Calcula score_confianza, es_frontera, segundo_arquetipo,
       cobertura_texto y texto_casi_vacio.
    3. Reensamblaje: Retorna el DataFrame unificado preservando exactamente el indice original.
    """
    if isinstance(modelo, str):
        modelo_dict = cargar_modelo_clustering(modelo)
    else:
        modelo_dict = modelo

    kmeans: KMeans = modelo_dict["kmeans"]
    scaler = modelo_dict["scaler"]
    tfidf_vec = modelo_dict["tfidf_vectorizer"]
    mapa_arquetipos = modelo_dict["mapa_arquetipos"]
    gmm: Optional[GaussianMixture] = modelo_dict.get("gmm", None)
    dist_perc = modelo_dict.get("distribucion_percentil_tipo", {})
    cats_type = modelo_dict.get("categorias_type_site", CANONICAL_TYPE_SITE_CATEGORIES)
    peso_type_site = modelo_dict.get("metadata", {}).get("peso_type_site", 0.5)

    df_input = df.copy()
    if "type_site" not in df_input.columns:
        from src.feature_engineering import enriquecer_type_site
        df_input = enriquecer_type_site(df_input)

    if "texto_limpio" not in df_input.columns:
        col_nlp = next(
            (c for c in ["logical_seat_category", "product", "translation_name", "cd_name", "nombre_localidad"] if c in df_input.columns),
            None
        )
        if col_nlp:
            from src.nlp_utils import pipeline_procesamiento_nlp
            df_input = pipeline_procesamiento_nlp(df_input, col_nombre=col_nlp)
        else:
            df_input["texto_limpio"] = ""

    # Asegurar presencia de tags estructurales requeridos por el modelo
    for tag_col in DEFAULT_TAG_FEATURES:
        if tag_col not in df_input.columns:
            df_input[tag_col] = 0

    # Asegurar percentil_precio_absoluto_dentro_tipo contra distribucion de referencia persistida
    if "percentil_precio_absoluto_dentro_tipo" not in df_input.columns or bool(dist_perc):
        from src.feature_engineering import calcular_percentil_precio_absoluto_dentro_tipo
        df_input = calcular_percentil_precio_absoluto_dentro_tipo(
            df_input, referencia_distribucion=dist_perc if dist_perc else None
        )

    # Asegurar presencia de variables numericas continuas
    if any(c not in df_input.columns for c in DEFAULT_NUMERIC_FEATURES):
        if "t_performance_id" in df_input.columns and any(
            c in df_input.columns for c in ["ave_unit_amt_itx", "med_unit_amt_itx", "net_sold_p_qty"]
        ):
            from src.feature_engineering import calcular_metricas_relativas
            df_input = calcular_metricas_relativas(df_input)
        for num_col in DEFAULT_NUMERIC_FEATURES:
            if num_col not in df_input.columns:
                df_input[num_col] = 0.50 if "percentil" in num_col else 0.0

    vocab = set(tfidf_vec.get_feature_names_out()) if hasattr(tfidf_vec, "get_feature_names_out") else set()
    def _calc_cobertura(texto):
        tokens = str(texto).lower().split()
        if not tokens:
            return 0.0
        return sum(t in vocab for t in tokens) / len(tokens)

    # 1. Separacion a nivel evento
    df_mono, df_multi = separar_admision_unica_multizona(df_input)
    df_mono = df_mono.copy()
    df_mono["cluster"] = -1
    df_mono["arquetipo_demanda"] = "Admisión Única / Tarifa Plana"
    df_mono["score_confianza"] = 1.0
    df_mono["es_frontera"] = False
    df_mono["segundo_arquetipo"] = None
    df_mono["cobertura_texto"] = df_mono["texto_limpio"].apply(_calc_cobertura)
    df_mono["texto_casi_vacio"] = df_mono["cobertura_texto"] < 0.20
    df_mono["probabilidad_gmm"] = 1.0

    # 2. Inferencia en multi-zona si existen registros
    if len(df_multi) > 0:
        df_multi = df_multi.copy()
        X_multi, _, _, feat_multi = construir_espacio_vectorial_mixto(
            df_multi,
            scaler=scaler,
            tfidf_vectorizer=tfidf_vec,
            peso_nlp=peso_nlp,
            peso_type_site=peso_type_site,
            categorias_type_site=cats_type
        )
        
        target_features = modelo_dict.get("feature_names", feat_multi)
        if feat_multi != target_features:
            feat_idx_map = {f: i for i, f in enumerate(feat_multi)}
            cols_aligned = []
            for tf in target_features:
                if tf in feat_idx_map:
                    cols_aligned.append(X_multi[:, feat_idx_map[tf]])
                else:
                    cols_aligned.append(np.zeros(len(df_multi)))
            X_multi = np.column_stack(cols_aligned)

        
        distancias = kmeans.transform(X_multi)
        orden = np.argsort(distancias, axis=1)
        d1 = distancias[np.arange(len(distancias)), orden[:, 0]]
        d2 = distancias[np.arange(len(distancias)), orden[:, 1]]
        margen = (d2 - d1) / (d2 + 1e-9)

        labels_multi = orden[:, 0]
        df_multi["cluster"] = labels_multi
        df_multi["arquetipo_demanda"] = df_multi["cluster"].map(mapa_arquetipos)
        df_multi["score_confianza"] = margen
        df_multi["es_frontera"] = margen < 0.15
        df_multi["segundo_arquetipo"] = [mapa_arquetipos.get(c, str(c)) for c in orden[:, 1]]
        df_multi["cobertura_texto"] = df_multi["texto_limpio"].apply(_calc_cobertura)
        df_multi["texto_casi_vacio"] = df_multi["cobertura_texto"] < 0.20

        if gmm is not None:
            probs = gmm.predict_proba(X_multi)
            df_multi["probabilidad_gmm"] = probs.max(axis=1)
        else:
            df_multi["probabilidad_gmm"] = margen

    # 3. Reensamblaje preservando indice original
    if len(df_multi) == 0:
        return df_mono.loc[df.index]
    if len(df_mono) == 0:
        return df_multi.loc[df.index]

    df_res = pd.concat([df_mono, df_multi], axis=0).loc[df.index]
    return df_res


def evaluar_drift_lote(
    df_nuevo: pd.DataFrame,
    modelo: Union[str, Dict[str, Any]],
    eps: float = 1e-4,
    umbral_psi_alerta: float = 0.25
) -> Dict[str, Any]:
    """
    Evalua el Population Stability Index (PSI) de variables y la distribucion de arquetipos
    sobre un nuevo lote de localidades para detectar drift silencioso.
    """
    if isinstance(modelo, str):
        payload = cargar_modelo_clustering(modelo)
    else:
        payload = modelo

    df_pred = predecir_arquetipos_demanda(df_nuevo, payload)
    
    referencia_drift = payload.get("referencia_drift", {})
    vars_ref = referencia_drift.get("variables", {})
    arq_esperado = referencia_drift.get("distribucion_arquetipos", DISTRIBUCION_ESPERADA_ARQUETIPOS)

    # Evaluar PSI de variables estructurales sobre el segmento multi-zona
    df_eval_vars = df_pred[df_pred["cluster"] != -1] if (df_pred["cluster"] != -1).any() else df_pred

    psi_por_variable = {}
    alertas = []
    
    for v, info in vars_ref.items():
        if v in df_eval_vars.columns:
            bin_edges = np.array(info["bin_edges"])
            esperado_pct = np.array(info["esperado_pct"])
            valores = df_eval_vars[v].dropna().values
            counts_nuevo, _ = np.histogram(valores, bins=bin_edges)
            psi_val = calcular_psi(counts_nuevo, esperado_pct, eps=eps)
            
            if psi_val < 0.10:
                estado = "ESTABLE"
            elif psi_val <= umbral_psi_alerta:
                estado = "REVISAR"
            else:
                estado = "DRIFT_CRITICO"
                alertas.append(f"Drift critico en variable {v} (PSI = {psi_val:.4f})")
                
            psi_por_variable[v] = {
                "psi": round(psi_val, 4),
                "estado": estado
            }

    total_filas = len(df_pred)
    conteo_arquetipos = df_pred["arquetipo_demanda"].value_counts().to_dict()
    dist_observada = {k: conteo_arquetipos.get(k, 0) / max(total_filas, 1) for k in arq_esperado.keys()}
    
    desviaciones_arquetipos = {}
    for arq, p_esp in arq_esperado.items():
        p_obs = dist_observada.get(arq, 0.0)
        diff = p_obs - p_esp
        desviaciones_arquetipos[arq] = {
            "esperado": round(p_esp, 4),
            "observado": round(p_obs, 4),
            "diferencia": round(diff, 4)
        }
        if abs(diff) > 0.05:
            alertas.append(f"Desviacion de arquetipo {arq}: {diff:+.1%}")

    pct_frontera = float(df_pred["es_frontera"].mean()) if "es_frontera" in df_pred.columns else 0.0
    pct_texto_vacio = float(df_pred["texto_casi_vacio"].mean()) if "texto_casi_vacio" in df_pred.columns else 0.0
    score_confianza_promedio = float(df_pred["score_confianza"].mean()) if "score_confianza" in df_pred.columns else 1.0

    psi_maximo = max((info["psi"] for info in psi_por_variable.values()), default=0.0)
    estado_general = "ESTABLE"
    if psi_maximo > umbral_psi_alerta or len(alertas) > 0:
        estado_general = "DRIFT_CRITICO" if psi_maximo > umbral_psi_alerta else "REVISAR"
    elif psi_maximo >= 0.10:
        estado_general = "REVISAR"

    return {
        "estado_general": estado_general,
        "total_registros": total_filas,
        "psi_maximo": psi_maximo,
        "psi_por_variable": psi_por_variable,
        "distribucion_arquetipos": desviaciones_arquetipos,
        "pct_frontera": round(pct_frontera, 4),
        "pct_texto_casi_vacio": round(pct_texto_vacio, 4),
        "score_confianza_promedio": round(score_confianza_promedio, 4),
        "alertas": alertas
    }

