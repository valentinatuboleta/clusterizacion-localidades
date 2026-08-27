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
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)
from src.nlp_utils import vectorizar_texto_limpio


# Lista por defecto de variables estructurales numéricas y booleanas
DEFAULT_NUMERIC_FEATURES = [
    "ratio_precio_max",
    "percentil_precio_evento",
    "peso_aforo",
    "tasa_ocupacion",
    "tasa_venta_paga"
]

DEFAULT_TAG_FEATURES = [
    "tag_palco",
    "tag_vip",
    "tag_platea",
    "tag_preferencial",
    "tag_general",
    "tag_balcon",
    "tag_piso_alto",
    "tag_piso_bajo",
    "tag_occidental",
    "tag_oriental",
    "tag_norte",
    "tag_sur",
    "tag_lateral",
    "tag_vista_parcial",
    "tag_familiar",
    "tag_menores",
    "tag_movilidad_reducida"
]


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
    Construye el espacio vectorial mixto combinando:
    1. Métricas numéricas relativas normalizadas (precio relativo, aforo, ocupación).
    2. Tags estructurales/espaciales y restricciones.
    3. Embeddings/TF-IDF del texto limpio de la localidad.
    """
    cols_presentes = [c for c in columnas_numericas + columnas_tags if c in df.columns]
    X_num = df[cols_presentes].fillna(0).values
    
    if scaler is None:
        scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
    else:
        X_num_scaled = scaler.transform(X_num)
        
    feature_names = list(cols_presentes)
    
    if usar_tfidf_texto and "texto_limpio" in df.columns:
        X_tfidf, tfidf_vectorizer = vectorizar_texto_limpio(
            df["texto_limpio"],
            max_features=max_tfidf_features,
            vectorizer=tfidf_vectorizer
        )
        X_tfidf_weighted = X_tfidf * peso_nlp
        X_mixto = np.hstack([X_num_scaled, X_tfidf_weighted])
        tfidf_vocab = [f"tfidf_{w}" for w in tfidf_vectorizer.get_feature_names_out()]
        feature_names.extend(tfidf_vocab)
    else:
        X_mixto = X_num_scaled
        
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


def asignar_arquetipos_demanda(df_clustered: pd.DataFrame, col_cluster: str = "cluster") -> pd.DataFrame:
    """
    Interpreta los centroides de cada cluster en términos de precio relativo, peso de aforo y semántica,
    asignando nombres de arquetipos estandarizados de negocio.
    """
    df_res = df_clustered.copy()
    
    # Calcular resumen por cluster
    clusters_info = []
    for c_id in sorted(df_res[col_cluster].unique()):
        sub = df_res[df_res[col_cluster] == c_id]
        clusters_info.append({
            "cluster": c_id,
            "precio_prom": sub["ratio_precio_max"].mean(),
            "aforo_prom": sub["peso_aforo"].mean(),
            "ocupacion_prom": sub["tasa_ocupacion"].mean(),
            "general_share": sub["tag_general"].mean() if "tag_general" in sub.columns else 0.0,
            "palco_vip_share": (sub["tag_palco"].mean() + sub["tag_vip"].mean()) if "tag_palco" in sub.columns else 0.0,
            "platea_share": sub["tag_platea"].mean() if "tag_platea" in sub.columns else 0.0
        })
        
    df_info = pd.DataFrame(clusters_info)
    mapa_arquetipos = {}
    
    # 1. Identificar cluster Grada General (mayor aforo relativo y mayor share de 'general')
    gen_c = df_info.sort_values(by=["aforo_prom", "general_share"], ascending=False).iloc[0]["cluster"]
    mapa_arquetipos[int(gen_c)] = "Grada General / Masiva"
    
    # 2. Identificar cluster VIP / Palcos (mayor concentración de palco/vip o mayor ocupación + alto precio)
    restantes = df_info[df_info["cluster"] != gen_c].copy()
    vip_c = restantes.sort_values(by=["palco_vip_share", "ocupacion_prom"], ascending=False).iloc[0]["cluster"]
    mapa_arquetipos[int(vip_c)] = "VIP / Palcos / Premium"
    
    # 3. Entre los restantes, separar Preferencial (mayor precio relativo) de Popular (menor precio relativo)
    restantes_2 = restantes[restantes["cluster"] != vip_c].sort_values(by="precio_prom", ascending=False)
    if len(restantes_2) > 0:
        pref_c = restantes_2.iloc[0]["cluster"]
        mapa_arquetipos[int(pref_c)] = "Preferencial / Platea Frontal"
    if len(restantes_2) > 1:
        pop_c = restantes_2.iloc[1]["cluster"]
        mapa_arquetipos[int(pop_c)] = "Popular / Visibilidad Parcial / Balcón"
        
    # Asignar fallback para cualquier otro cluster si k > 4
    for _, row in df_info.iterrows():
        c = int(row["cluster"])
        if c not in mapa_arquetipos:
            mapa_arquetipos[c] = f"Segmento #{c}"
            
    df_res["arquetipo_demanda"] = df_res[col_cluster].map(mapa_arquetipos)
    return df_res
