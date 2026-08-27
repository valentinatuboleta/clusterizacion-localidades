import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def preprocesar_datos(df: pd.DataFrame, columnas_features: list) -> np.ndarray:
    """
    Selecciona y estandariza las variables numéricas para la clusterización.
    """
    df_sub = df[columnas_features].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_sub)
    return data_scaled, scaler

def ejecutar_kmeans(data_scaled: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> tuple:
    """
    Ejecuta el algoritmo K-Means en los datos estandarizados.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels) if len(np.unique(labels)) > 1 else -1
    return kmeans, labels, score
