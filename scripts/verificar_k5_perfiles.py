"""
Verificación de los 5 clusters de k=5 en multi-zona con peso_nlp=0.2
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import filtrar_consistencia_localidades, calcular_metricas_relativas
from src.nlp_utils import pipeline_procesamiento_nlp
from src.clustering import separar_admision_unica_multizona, construir_espacio_vectorial_mixto

def chequear_k5():
    df_raw = pd.read_parquet("data/raw/localidades_eda.parquet")
    df_clean = filtrar_consistencia_localidades(df_raw)
    df_rel = calcular_metricas_relativas(df_clean)
    df_enriquecido = pipeline_procesamiento_nlp(df_rel)
    df_mono, df_multi = separar_admision_unica_multizona(df_enriquecido)

    X_multi, scaler, tfidf, feats = construir_espacio_vectorial_mixto(
        df_multi, peso_nlp=0.2, scaler_type="robust"
    )

    km = KMeans(n_clusters=5, random_state=42, n_init=15)
    labels = km.fit_predict(X_multi)
    df_multi_eval = df_multi.copy()
    df_multi_eval["cluster"] = labels

    print("==================================================================")
    print("PERFILES DE LOS 5 CLUSTERS DE MULTI-ZONA (k=5, peso_nlp=0.2)")
    print("==================================================================")
    for c in range(5):
        df_c = df_multi_eval[df_multi_eval["cluster"] == c]
        n = len(df_c)
        pct = n / len(df_multi_eval) * 100
        p_ratio = df_c["ratio_precio_max"].mean()
        p_perc = df_c["percentil_precio_evento"].mean()
        aforo = df_c["peso_aforo"].mean()
        
        tags_activos = []
        for t in ["tag_vip", "tag_palco", "tag_platea", "tag_balcon", "tag_general", "tag_preferencial", "tag_piso_alto"]:
            if t in df_c.columns and df_c[t].mean() > 0.15:
                tags_activos.append(f"{t}: {df_c[t].mean()*100:.1f}%")
        
        col_texto = "logical_seat_category" if "logical_seat_category" in df_c.columns else "texto_limpio"
        ejemplos = df_c[col_texto].value_counts().head(3).index.tolist()
        print(f"\nCluster {c} (N={n}, {pct:.1f}%):")
        print(f"  Ratio Precio Max: {p_ratio:.2f} | Percentil Precio: {p_perc:.2f} | Peso Aforo: {aforo*100:.1f}%")
        print(f"  Tags Activos (>15%): {', '.join(tags_activos) if tags_activos else 'Ninguno predominante (General)'}")
        print(f"  Ejemplos nombres: {', '.join(ejemplos)}")

if __name__ == "__main__":
    chequear_k5()
