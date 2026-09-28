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

from src.feature_engineering import preparar_dataset_enriquecido
from src.clustering import (
    separar_admision_unica_multizona,
    pipeline_clustering_dos_etapas,
    DEFAULT_NUMERIC_FEATURES
)
from src.jerarquia import (
    HIERARCHY_VERSION,
    TAGS_SUBESPACIO_13,
    ARQ_PREFIX,
    MICRO_CLUSTERS_CANONICAL,
    construir_subespacio_arquetipo,
    seleccionar_sub_k_codo_db,
    generar_label_auto_v3,
    calcular_pureza_cluster,
    calcular_estabilidad_bootstrap_ari
)

# Aliases de compatibilidad
TAGS_EXPANDIDOS_13 = TAGS_SUBESPACIO_13
generar_label_auto = generar_label_auto_v3

# Arquetipos macro multi-zona sometidos a Nivel 2
ARQUETIPOS_MULTIZONA = [
    "VIP / Palcos / Premium",
    "Popular / Balcón / Visibilidad Parcial",
    "Platea General / Intermedia",
    "Preferencial / Platea Frontal",
    "Grada General / Masiva"
]


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
        X_sub, sub_scaler, sub_tfidf, fnames_sub = construir_subespacio_arquetipo(
            sub_df,
            columnas_tags=TAGS_SUBESPACIO_13,
            max_tfidf=15,
            peso_nlp=0.2
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

            p_info = calcular_pureza_cluster(c_data, TAGS_SUBESPACIO_13, vocab_sub)
            label_c = generar_label_auto_v3(
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
            "k_sub": k_opt,
            "kmeans": ganador["model"],
            "kmeans_sub": ganador["model"],
            "scaler": sub_scaler,
            "scaler_sub": sub_scaler,
            "tfidf": sub_tfidf,
            "tfidf_sub": sub_tfidf,
            "columnas_tags": TAGS_SUBESPACIO_13,
            "columnas_tags_sub": TAGS_SUBESPACIO_13,
            "feature_names": fnames_sub,
            "mapa_labels": mapa_labels,
            "mapa_micro_label": mapa_labels
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
    dist_referencia = (
        df_asignacion_total["micro_cluster_id"]
        .value_counts(normalize=True)
        .to_dict()
    )

    payload_v3 = {
        "version": HIERARCHY_VERSION,
        "nivel_1": {
            "version_base": "2.3",
            "kmeans": km_n1,
            "scaler": scaler_n1,
            "tfidf": tfidf_n1,
            "feature_names": fnames_n1,
            "metricas": met_n1
        },
        "sub_modelos": sub_modelos,
        "distribucion_referencia_microclusters": dist_referencia,
        "categorias_vigentes": MICRO_CLUSTERS_CANONICAL,
        "columnas_tags_expandidas": TAGS_SUBESPACIO_13,
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
