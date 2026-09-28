#!/usr/bin/env python3
"""
Script de Entrenamiento de la Jerarquía de Micro-Clusters (v3.0 Candidata).

Arquitectura en Dos Niveles:
- Nivel 1 (Producción v2.3 congelado): Pipeline bietápico de 6 arquetipos de demanda.
  Admisión Única / Tarifa Plana es un micro-cluster terminal único por definición (AU-0).
- Nivel 2 (Sub-clustering por Arquetipo Macro):
  Para cada uno de los 5 arquetipos multi-zona, entrena un sub-modelo K-Means sobre un
  sub-espacio especializado de 32 dimensiones:
    * 4 numéricas (RobustScaler propio del sub-espacio)
    * 13 tags estructurales expandidos (7 v2.3 + lateral, occidental, oriental, norte, sur, mesa)
    * 15 TF-IDF léxicos locales (peso_nlp=0.2)
    * Sin one-hot de type_site (dentro de un arquetipo macro, el venue es ruido, no señal).

Criterios de Evaluación y Compuertas de Calidad por Sub-Espacio:
- Pureza de naming ponderada >= 0.85
- Estabilidad bootstrap-ARI >= 0.85 (20 remuestreos al 80%)
- Cero clusters degenerados (< 3% del sub-espacio)
- Selección de k in {2, 3, 4, 5} mediante Score Compuesto Codo-DB.
- Manejo de incertidumbre: localidades con es_frontera=True se sub-clusterizan pero se marcan
  con segmento_incierto=True en la asignación final y en el catálogo (es_frontera_pct).
"""

import os
import sys
import time
import unicodedata
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib

# Asegurar path raíz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.feature_engineering import preparar_dataset_enriquecido
from src.clustering import (
    separar_admision_unica_multizona,
    pipeline_clustering_dos_etapas,
    construir_espacio_vectorial_mixto,
    DEFAULT_NUMERIC_FEATURES
)
from src.nlp_utils import normalizar_texto
from scripts.explorar_microclusters import (
    calcular_pureza_cluster,
    generar_label_auto,
    calcular_estabilidad_bootstrap_ari,
    TAG_LABEL_MAP
)

# 13 Tags estructurales expandidos de la Decisión D
TAGS_EXPANDIDOS_13 = [
    "tag_palco", "tag_vip", "tag_platea", "tag_preferencial", "tag_general",
    "tag_balcon", "tag_piso_alto", "tag_lateral", "tag_occidental", "tag_oriental",
    "tag_norte", "tag_sur", "tag_mesa"
]

# Arquetipos macro multi-zona sometidos a Nivel 2
ARQUETIPOS_MULTIZONA = [
    "VIP / Palcos / Premium",
    "Popular / Balcón / Visibilidad Parcial",
    "Platea General / Intermedia",
    "Preferencial / Platea Frontal",
    "Grada General / Masiva"
]


def seleccionar_sub_k_codo_db(
    X_sub: np.ndarray,
    df_sub: pd.DataFrame,
    vocab_sub: List[str],
    k_range: List[int] = [2, 3, 4, 5],
    min_pct_piso: float = 0.03,
    sample_size: int = 3000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Selecciona el k óptimo en el sub-espacio mediante score compuesto Codo-DB
    descartando soluciones con clusters degenerados (< 3% del sub-espacio).
    """
    n_sub = len(df_sub)
    eval_size = min(sample_size, n_sub)
    rng = np.random.RandomState(random_state)
    idx_eval = rng.choice(n_sub, size=eval_size, replace=False)
    X_eval = X_sub[idx_eval]

    candidatos = []
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_sub)
        inertias.append(km.inertia_)

        labels_eval = labels[idx_eval]
        sil = float(silhouette_score(X_eval, labels_eval))
        db = float(davies_bouldin_score(X_eval, labels_eval))
        ch = float(calinski_harabasz_score(X_eval, labels_eval))

        # Distribución de tamaños
        counts = pd.Series(labels).value_counts()
        min_cluster_size = counts.min()
        min_pct = min_cluster_size / n_sub
        es_degenerado = bool(min_pct < min_pct_piso)

        # Pureza de naming ponderada
        df_temp = df_sub.copy()
        df_temp["_cluster"] = labels
        p_namings = []
        p_tags = []
        p_terms = []
        for c_id in range(k):
            sub_c = df_temp[df_temp["_cluster"] == c_id]
            res_p = calcular_pureza_cluster(sub_c, TAGS_EXPANDIDOS_13, vocab_sub)
            p_namings.append(res_p["purity_naming"])
            p_tags.append(res_p["purity_tag"])
            p_terms.append(res_p["purity_term"])

        weights = np.array([len(df_temp[df_temp["_cluster"] == c_id]) for c_id in range(k)])
        pureza_naming = float(np.sum(weights * np.array(p_namings)) / np.sum(weights))
        pureza_tag = float(np.sum(weights * np.array(p_tags)) / np.sum(weights))
        pureza_term = float(np.sum(weights * np.array(p_terms)) / np.sum(weights))

        # Bootstrap-ARI (20 réplicas al 80%)
        def _estimador(X_s, seed):
            return KMeans(n_clusters=k, random_state=seed, n_init=5).fit_predict(X_s)

        ari_mean, ari_std, _ = calcular_estabilidad_bootstrap_ari(
            X_sub, labels, _estimador, n_iter=20, frac=0.80, random_state=random_state
        )

        candidatos.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "min_cluster_pct": min_pct,
            "es_degenerado": es_degenerado,
            "pureza_naming": pureza_naming,
            "pureza_tag": pureza_tag,
            "pureza_term": pureza_term,
            "bootstrap_ari_mean": ari_mean,
            "bootstrap_ari_std": ari_std,
            "model": km,
            "labels": labels
        })

    # Codo ortogonal
    k_vals = np.array(k_range, dtype=float)
    in_vals = np.array(inertias, dtype=float)
    P1 = np.array([k_vals[0], in_vals[0], 0.0])
    P2 = np.array([k_vals[-1], in_vals[-1], 0.0])
    vec_secante = P2 - P1
    norm_secante = np.linalg.norm(vec_secante)
    distancias_codo = []
    for i, k in enumerate(k_range):
        P0 = np.array([k_vals[i], in_vals[i], 0.0])
        d = np.linalg.norm(np.cross(vec_secante, P1 - P0)) / (norm_secante + 1e-9)
        distancias_codo.append(float(d))

    codo_arr = np.array(distancias_codo)
    db_arr = np.array([c["davies_bouldin"] for c in candidatos])

    norm_codo = (codo_arr - codo_arr.min()) / (codo_arr.max() - codo_arr.min() + 1e-8)
    norm_db = (db_arr.max() - db_arr) / (db_arr.max() - db_arr.min() + 1e-8)
    score_compuesto = norm_codo + norm_db

    for i, cand in enumerate(candidatos):
        cand["distancia_codo"] = distancias_codo[i]
        cand["score_compuesto"] = float(score_compuesto[i])
        cand["pasa_compuertas"] = bool(
            cand["pureza_naming"] >= 0.85 and
            cand["bootstrap_ari_mean"] >= 0.85 and
            not cand["es_degenerado"]
        )

    # Filtrar no degenerados
    no_degenerados = [c for c in candidatos if not c["es_degenerado"]]
    if not no_degenerados:
        no_degenerados = candidatos

    # Priorizar candidatos que pasan compuertas; si varios o ninguno, mayor score compuesto
    candidatos_validos = [c for c in no_degenerados if c["pasa_compuertas"]]
    if candidatos_validos:
        ganador = max(candidatos_validos, key=lambda x: x["score_compuesto"])
    else:
        ganador = max(no_degenerados, key=lambda x: x["score_compuesto"])

    return {
        "k_optimo": ganador["k"],
        "ganador": ganador,
        "candidatos": candidatos
    }


def entrenar_jerarquia(
    data_path: str = "data/raw/localidades_eda.parquet",
    output_modelo: str = "data/processed/modelo_jerarquia_v3.joblib",
    output_catalogo: str = "data/processed/cluster_catalog_v3.csv",
    output_asignacion: str = "data/processed/asignacion_microclusters.csv",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de entrenamiento jerárquico v3.0.
    """
    print("==================================================================")
    print("ENTRENAMIENTO JERÁRQUICO DE MICRO-CLUSTERS (v3.0 CANDIDATA)")
    print("==================================================================")

    # 1. Cargar y enriquecer dataset
    print("\n1. Preparando dataset enriquecido...")
    df_raw = pd.read_parquet(data_path)
    df_enr = preparar_dataset_enriquecido(df_raw)

    # 2. Nivel 1: Pipeline v2.3 congelado
    print("\n2. Ejecutando Nivel 1 (Producción v2.3 congelada)...")
    df_n1, km_n1, scaler_n1, tfidf_n1, fnames_n1, met_n1 = pipeline_clustering_dos_etapas(
        df_enr, n_clusters_multizona=5, random_state=random_state
    )

    # Admisión Única queda como micro-cluster terminal único por definición
    df_au = df_n1[df_n1["es_monozona"]].copy()
    df_au["micro_cluster_id"] = "AU-0"
    df_au["label_auto"] = "Admisión Única"
    df_au["segmento_incierto"] = False
    print(f" Nivel 1 listo: Admisión Única={len(df_au):,}, Multi-zona={len(df_n1) - len(df_au):,}")

    # 3. Nivel 2: Sub-clustering por arquetipo multi-zona
    print("\n3. Entrenando Nivel 2 por arquetipo (Sub-espacios 32D, 13 tags)...")
    sub_modelos = {}
    reporte_subespacios = []
    filas_catalogo = []
    dfs_asignacion = []
    todos_pasan = True

    # Registrar Admisión Única en catálogo
    filas_catalogo.append({
        "micro_cluster_id": "AU-0",
        "arquetipo_macro": "Admisión Única / Tarifa Plana",
        "n": len(df_au),
        "es_frontera_pct": 0.0,
        "purity_tag": 1.0,
        "tag_dominante": "tag_general",
        "purity_term": 1.0,
        "term_dominante": "admision",
        "label_auto": "Admisión Única"
    })
    dfs_asignacion.append(df_au[[
        "logical_seat_category", "micro_cluster_id", "label_auto",
        "arquetipo_demanda", "score_confianza", "segmento_incierto"
    ]])

    # Iniciales cortas de arquetipos para micro_cluster_id
    ARQ_PREFIX = {
        "VIP / Palcos / Premium": "VIP",
        "Popular / Balcón / Visibilidad Parcial": "POP",
        "Platea General / Intermedia": "PGI",
        "Preferencial / Platea Frontal": "PPF",
        "Grada General / Masiva": "GGM"
    }

    for arq in ARQUETIPOS_MULTIZONA:
        prefix = ARQ_PREFIX[arq]
        sub_df = df_n1[df_n1["arquetipo_demanda"] == arq].copy()
        n_sub = len(sub_df)
        print(f"\n--- Sub-espacio: {arq} (N = {n_sub:,}) ---")

        # Construir sub-espacio vectorial propio 32D
        X_sub, sub_scaler, sub_tfidf, fnames_sub = construir_espacio_vectorial_mixto(
            sub_df,
            columnas_tags=TAGS_EXPANDIDOS_13,
            usar_tfidf_texto=True,
            max_tfidf_features=15,
            peso_nlp=0.2,
            peso_type_site=0.0
        )
        vocab_sub = list(sub_tfidf.get_feature_names_out())

        # Sub-selección de k
        res_sel = seleccionar_sub_k_codo_db(
            X_sub=X_sub,
            df_sub=sub_df,
            vocab_sub=vocab_sub,
            k_range=[2, 3, 4, 5],
            min_pct_piso=0.03,
            sample_size=3000,
            random_state=random_state
        )
        ganador = res_sel["ganador"]
        k_opt = res_sel["k_optimo"]
        labels_sub = ganador["labels"]

        if not ganador["pasa_compuertas"]:
            todos_pasan = False
            print(f" [ALERTA] Sub-espacio '{arq}' NO superó todas las compuertas de calidad.")
            print(f"   Pureza={ganador['pureza_naming']:.4f}, ARI={ganador['bootstrap_ari_mean']:.4f}, Degenerado={ganador['es_degenerado']}")
        else:
            print(f" [OK] Sub-espacio '{arq}' pasa compuertas con k={k_opt}.")
            print(f"   Pureza={ganador['pureza_naming']:.4f}, ARI={ganador['bootstrap_ari_mean']:.4f}, Silueta={ganador['silhouette']:.4f}, DB={ganador['davies_bouldin']:.4f}")

        reporte_subespacios.append({
            "arquetipo_macro": arq,
            "k_elegido": k_opt,
            "n": n_sub,
            "pureza_naming": round(ganador["pureza_naming"], 4),
            "pureza_tag": round(ganador["pureza_tag"], 4),
            "pureza_term": round(ganador["pureza_term"], 4),
            "bootstrap_ari": round(ganador["bootstrap_ari_mean"], 4),
            "bootstrap_ari_std": round(ganador["bootstrap_ari_std"], 4),
            "silhouette": round(ganador["silhouette"], 4),
            "davies_bouldin": round(ganador["davies_bouldin"], 4),
            "min_cluster_pct": round(ganador["min_cluster_pct"] * 100, 2),
            "pasa_compuertas": ganador["pasa_compuertas"]
        })

        # Generar micro-cluster IDs, labels y mapeos
        mapa_labels = {}
        sub_df["sub_cluster_num"] = labels_sub
        sub_df["micro_cluster_id"] = [f"{prefix}-{c}" for c in labels_sub]
        # Propagar incertidumbre si es_frontera=True en Nivel 1
        sub_df["segmento_incierto"] = sub_df["es_frontera"].astype(bool)

        for c_id in range(k_opt):
            m_id = f"{prefix}-{c_id}"
            c_mask = (sub_df["sub_cluster_num"] == c_id)
            c_data = sub_df[c_mask]

            p_info = calcular_pureza_cluster(c_data, TAGS_EXPANDIDOS_13, vocab_sub)
            label_c = generar_label_auto(
                cluster_id=c_id,
                tag_shares=p_info["tag_shares"],
                term_dominante=p_info["term_dominante"],
                purity_tag=p_info["purity_tag"],
                purity_term=p_info["purity_term"]
            )
            mapa_labels[c_id] = label_c

            pct_frontera = float(c_data["es_frontera"].mean() * 100) if len(c_data) > 0 else 0.0

            filas_catalogo.append({
                "micro_cluster_id": m_id,
                "arquetipo_macro": arq,
                "n": len(c_data),
                "es_frontera_pct": round(pct_frontera, 2),
                "purity_tag": round(p_info["purity_tag"], 4),
                "tag_dominante": p_info["tag_dominante"],
                "purity_term": round(p_info["purity_term"], 4),
                "term_dominante": p_info["term_dominante"],
                "label_auto": label_c
            })

        sub_df["label_auto"] = sub_df["sub_cluster_num"].map(mapa_labels)
        dfs_asignacion.append(sub_df[[
            "logical_seat_category", "micro_cluster_id", "label_auto",
            "arquetipo_demanda", "score_confianza", "segmento_incierto"
        ]])

        sub_modelos[arq] = {
            "prefix": prefix,
            "k": k_opt,
            "kmeans": ganador["model"],
            "scaler": sub_scaler,
            "tfidf": sub_tfidf,
            "feature_names": fnames_sub,
            "mapa_labels": mapa_labels
        }

    # Consolidar asignación y catálogo
    df_asignacion_total = pd.concat(dfs_asignacion, ignore_index=True)
    df_catalogo_total = pd.DataFrame(filas_catalogo)
    df_reporte = pd.DataFrame(reporte_subespacios)

    total_microclusters = len(df_catalogo_total)
    print("\n==================================================================")
    print(f"RESUMEN GLOBAL JERÁRQUICO: Total Micro-Clusters = {total_microclusters}")
    print("==================================================================")
    print(df_reporte.to_string(index=False))

    # Validación estricta de rollup 1:1
    assert (df_asignacion_total.groupby("micro_cluster_id")["arquetipo_demanda"].nunique() == 1).all(), (
        "Violación de jerarquía: Existe un micro-cluster asignado a múltiples arquetipos macro!"
    )
    print("\n [ASSERT PASSED] Rollup 1:1 estricto validado por construcción.")

    # 4. Persistencia de Artefactos Candidatos v3.0
    print("\n4. Persistiendo artefactos candidatos v3.0 en data/processed/ (producción v2.3 en models/ permanece intacta)...")
    payload_v3 = {
        "version": "3.0-hier",
        "nivel_1": {
            "version_base": "2.3",
            "kmeans": km_n1,
            "scaler": scaler_n1,
            "tfidf": tfidf_n1,
            "feature_names": fnames_n1,
            "metricas": met_n1
        },
        "sub_modelos": sub_modelos,
        "columnas_tags_expandidas": TAGS_EXPANDIDOS_13,
        "total_microclusters": total_microclusters,
        "todos_pasan_compuertas": todos_pasan,
        "metadata": {
            "autor": "Data Science TuBoleta",
            "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
            "todos_pasan_compuertas": todos_pasan
        }
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_modelo)), exist_ok=True)
    joblib.dump(payload_v3, output_modelo)
    print(f" Modelo jerárquico serializado en: {output_modelo}")

    df_catalogo_total.to_csv(output_catalogo, index=False)
    print(f" Catálogo v3 guardado en: {output_catalogo}")

    df_asignacion_total.to_csv(output_asignacion, index=False)
    print(f" Asignación de micro-clusters para negocio guardada en: {output_asignacion}")

    if not todos_pasan:
        print("\n [AUDITORÍA DE COMPUERTAS] Al menos un sub-espacio no superó todas las compuertas (pureza >= 0.85). Los artefactos se conservan como candidatos v3.0 para análisis.")

    return {
        "todos_pasan": todos_pasan,
        "reporte": df_reporte,
        "catalogo": df_catalogo_total,
        "asignacion": df_asignacion_total,
        "total_microclusters": total_microclusters
    }


if __name__ == "__main__":
    entrenar_jerarquia()
