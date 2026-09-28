#!/usr/bin/env python3
"""
Script de Diagnostico de Pureza en Sub-espacios y Evaluacion Condicional (v3.0-hier).

Objetivo:
Analizar la separabilidad lexico-estructural de los 4 sub-espacios que fallaron la
compuerta de pureza (VIP, Popular, Platea General, Grada) para determinar si la senal
esta "ahogada" por el peso numerico (recuperable ajustando omega_nlp) o si la senal es
"inexistente" (irreducible por ausencia de variabilidad lexico-estructural en los datos).

Metricas:
1. Tags de identidad: Tags expandidos excluyendo los tags que definen el macro-arquetipo.
2. Cobertura de identidad: % de filas con >=1 tag de identidad activo o termino distintivo.
3. Pureza oracle: Techo teorico alcanzable particionando por combinacion exacta de tags.
4. Veredicto formal:
   - 'senal_ahogada' si cobertura >= 50% y pureza_oracle >= 0.85 y pureza_actual < 0.85.
   - 'senial_inexistente' en cualquier otro caso.
5. Sweep condicional en VIP (solo si senal_ahogada): omega_nlp in {0.2, 0.35, 0.5}.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# Asegurar path raiz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import preparar_dataset_enriquecido
from src.clustering import construir_espacio_vectorial_mixto
from scripts.entrenar_jerarquia_microclusters import (
    TAGS_EXPANDIDOS_13,
    seleccionar_sub_k_codo_db
)
from scripts.explorar_microclusters import calcular_pureza_cluster

# Mapeo explicito y visible: Arquetipo macro -> tags que lo definen (excluidos de identidad)
TAGS_EXCLUIDOS_ARQUETIPO: Dict[str, List[str]] = {
    "VIP / Palcos / Premium": ["tag_palco", "tag_vip"],
    "Popular / Balcón / Visibilidad Parcial": ["tag_balcon", "tag_piso_alto"],
    "Platea General / Intermedia": ["tag_platea", "tag_general"],
    "Grada General / Masiva": ["tag_general"]
}

SUBESPACIOS_A_DIAGNOSTICAR: List[str] = [
    "VIP / Palcos / Premium",
    "Popular / Balcón / Visibilidad Parcial",
    "Platea General / Intermedia",
    "Grada General / Masiva"
]


def determinar_veredicto(cobertura: float, pureza_oracle: float, pureza_actual: float) -> str:
    """
    Regla explicita de veredicto:
    - senal_ahogada si: cobertura >= 50% y pureza_oracle >= 0.85 y pureza_actual < 0.85.
    - senial_inexistente en cualquier otro caso.
    """
    if cobertura >= 50.0 and pureza_oracle >= 0.85 and pureza_actual < 0.85:
        return "senal_ahogada"
    return "senial_inexistente"


def calcular_top_terminos_microcluster(
    df_cluster: pd.DataFrame,
    tfidf_vectorizer: Any,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Calcula los top N terminos TF-IDF a partir del centroide textual del micro-cluster.
    """
    if len(df_cluster) == 0:
        return []
    
    textos = df_cluster["texto_limpio"].fillna("").astype(str).tolist()
    matriz_tfidf = tfidf_vectorizer.transform(textos)
    if matriz_tfidf.shape[0] == 0:
        return []
    
    centroide_textual = np.asarray(matriz_tfidf.mean(axis=0)).flatten()
    vocabulario = np.array(tfidf_vectorizer.get_feature_names_out())
    
    idx_top = np.argsort(centroide_textual)[::-1][:top_n]
    return [(vocabulario[i], float(centroide_textual[i])) for i in idx_top if centroide_textual[i] > 0]


def diagnosticar_subespacios(
    data_path: str = "data/raw/localidades_eda.parquet",
    asignacion_path: str = "data/processed/asignacion_microclusters.csv",
    catalogo_path: str = "data/processed/cluster_catalog_v3.csv",
    output_reporte_csv: str = "reports/diagnostico_subespacios.csv"
) -> Dict[str, Any]:
    """
    Ejecuta el diagnostico completo de pureza sobre los 4 sub-espacios.
    """
    print("==================================================================")
    print("DIAGNOSTICO DE PUREZA EN SUB-ESPACIOS (v3.0-hier)")
    print("==================================================================")

    # 1. Cargar datasets
    print("\n1. Cargando datos enriquecidos y asignacion jerarquica actual...")
    df_asig = pd.read_csv(asignacion_path)
    df_cat = pd.read_csv(catalogo_path)
    df_raw = pd.read_parquet(data_path)
    df_enr = preparar_dataset_enriquecido(df_raw)

    df_merged = df_enr.copy()
    df_merged["micro_cluster_id"] = df_asig["micro_cluster_id"]
    df_merged["label_auto"] = df_asig["label_auto"]
    df_merged["arquetipo_demanda_asig"] = df_asig["arquetipo_demanda"]

    map_term_dom = df_cat.set_index("micro_cluster_id")["term_dominante"].to_dict()
    map_pureza_actual = df_cat.groupby("arquetipo_macro").apply(
        lambda g: float(np.sum(g["n"] * g["purity_naming"]) / g["n"].sum())
        if "purity_naming" in g.columns
        else float(np.sum(g["n"] * g["purity_tag"]) / g["n"].sum())
    ).to_dict()

    filas_reporte = []
    top_terminos_por_subespacio = {}

    print("\n2. Evaluando metricas de identidad, cobertura y pureza oracle...")
    for arq in SUBESPACIOS_A_DIAGNOSTICAR:
        sub_df = df_merged[df_merged["arquetipo_demanda_asig"] == arq].copy()
        n_sub = len(sub_df)
        excl = TAGS_EXCLUIDOS_ARQUETIPO[arq]
        tags_id = [t for t in TAGS_EXPANDIDOS_13 if t not in excl]

        # A. Tags de identidad
        has_tag_id = (sub_df[tags_id].sum(axis=1) > 0)
        pct_tag_id = float(has_tag_id.mean() * 100)

        # B. Termino distintivo del micro-cluster asignado
        def _has_distinctive_term(row):
            m_id = row["micro_cluster_id"]
            term = map_term_dom.get(m_id, "")
            if not term or term == "ninguno":
                return False
            tokens = str(row["texto_limpio"]).lower().split()
            return term.lower() in tokens

        has_term_dist = sub_df.apply(_has_distinctive_term, axis=1)
        pct_term_dist = float(has_term_dist.mean() * 100)

        # Cobertura de identidad (union)
        cobertura = float((has_tag_id | has_term_dist).mean() * 100)

        # C. Pureza alcanzable teorica (Oracle)
        def _get_identity(row):
            active = [t for t in tags_id if row[t] == 1]
            if not active:
                return "sin_identidad"
            return "+".join(sorted(active))

        sub_df["identidad_oracle"] = sub_df.apply(_get_identity, axis=1)

        # Espacio mixto para vectorizador TF-IDF local
        _, _, sub_tfidf, _ = construir_espacio_vectorial_mixto(
            sub_df,
            columnas_tags=TAGS_EXPANDIDOS_13,
            usar_tfidf_texto=True,
            max_tfidf_features=15,
            peso_nlp=0.2,
            peso_type_site=0.0
        )
        vocab_sub = list(sub_tfidf.get_feature_names_out())

        # Medir pureza de la particion oracle
        oracle_clusters = sub_df["identidad_oracle"].unique()
        p_namings_oracle = []
        weights_oracle = []

        for c_val in oracle_clusters:
            c_rows = sub_df[sub_df["identidad_oracle"] == c_val]
            res_p = calcular_pureza_cluster(c_rows, tags_id, vocab_sub)
            p_namings_oracle.append(res_p["purity_naming"])
            weights_oracle.append(len(c_rows))

        pureza_oracle = float(np.sum(np.array(weights_oracle) * np.array(p_namings_oracle)) / n_sub)

        # Pureza actual de v3.0
        pureza_act = map_pureza_actual.get(arq, 0.50)

        # Veredicto
        veredicto = determinar_veredicto(cobertura, pureza_oracle, pureza_act)

        # Top 5 terminos por micro-cluster
        top_terminos_mc = {}
        for m_id in sorted(sub_df["micro_cluster_id"].unique()):
            mc_df = sub_df[sub_df["micro_cluster_id"] == m_id]
            top_terms = calcular_top_terminos_microcluster(mc_df, sub_tfidf, top_n=5)
            top_terminos_mc[m_id] = top_terms

        top_terminos_por_subespacio[arq] = top_terminos_mc

        filas_reporte.append({
            "arquetipo_macro": arq,
            "n": n_sub,
            "tags_excluidos": ",".join(excl),
            "n_tags_identidad": len(tags_id),
            "pct_tag_identidad": round(pct_tag_id, 2),
            "pct_termino_distintivo": round(pct_term_dist, 2),
            "cobertura_identidad": round(cobertura, 2),
            "pureza_actual": round(pureza_act, 4),
            "pureza_oracle": round(pureza_oracle, 4),
            "veredicto": veredicto
        })

    df_rep = pd.DataFrame(filas_reporte)
    os.makedirs(os.path.dirname(os.path.abspath(output_reporte_csv)), exist_ok=True)
    df_rep.to_csv(output_reporte_csv, index=False)
    print(f"\n Reporte guardado en: {output_reporte_csv}")

    print("\n==================================================================")
    print("RESUMEN DE DIAGNOSTICO DE PUREZA")
    print("==================================================================")
    print(df_rep[["arquetipo_macro", "n", "cobertura_identidad", "pureza_actual", "pureza_oracle", "veredicto"]].to_string(index=False))

    print("\n--- TOP TERMINOS TF-IDF POR MICRO-CLUSTER ACTUAL ---")
    for arq, mc_dict in top_terminos_por_subespacio.items():
        print(f"\n[{arq}]:")
        for m_id, terms in mc_dict.items():
            str_terms = ", ".join([f"{w} ({val:.3f})" for w, val in terms]) if terms else "ninguno"
            print(f"  {m_id}: {str_terms}")

    # 3. Fase 2: Intervencion condicional en VIP (solo si senal_ahogada)
    vip_row = df_rep[df_rep["arquetipo_macro"] == "VIP / Palcos / Premium"].iloc[0]
    veredicto_vip = vip_row["veredicto"]
    resultados_sweep_vip = []

    if veredicto_vip == "senal_ahogada":
        print("\n==================================================================")
        print("FASE 2: INTERVENCION CONDICIONAL EN VIP (omega_nlp in {0.2, 0.35, 0.5})")
        print("==================================================================")
        sub_vip = df_merged[df_merged["arquetipo_demanda_asig"] == "VIP / Palcos / Premium"].copy()

        algun_omega_pasa = False
        omega_ganador = None

        for omega in [0.2, 0.35, 0.5]:
            X_vip, _, tfidf_vip, _ = construir_espacio_vectorial_mixto(
                sub_vip,
                columnas_tags=TAGS_EXPANDIDOS_13,
                usar_tfidf_texto=True,
                max_tfidf_features=15,
                peso_nlp=omega,
                peso_type_site=0.0
            )
            vocab_vip = list(tfidf_vip.get_feature_names_out())
            res_vip = seleccionar_sub_k_codo_db(
                X_sub=X_vip,
                df_sub=sub_vip,
                vocab_sub=vocab_vip,
                k_range=[2, 3, 4, 5],
                min_pct_piso=0.03,
                sample_size=3000,
                random_state=42
            )
            cand_ganador = res_vip["ganador"]
            pasa_todos = cand_ganador["pasa_compuertas"]

            resultados_sweep_vip.append({
                "omega_nlp": omega,
                "k_elegido": cand_ganador["k"],
                "pureza_naming": round(cand_ganador["pureza_naming"], 4),
                "pureza_tag": round(cand_ganador["pureza_tag"], 4),
                "pureza_term": round(cand_ganador["pureza_term"], 4),
                "bootstrap_ari": round(cand_ganador["bootstrap_ari_mean"], 4),
                "bootstrap_ari_std": round(cand_ganador["bootstrap_ari_std"], 4),
                "silueta": round(cand_ganador["silhouette"], 4),
                "davies_bouldin": round(cand_ganador["davies_bouldin"], 4),
                "min_cluster_pct": round(cand_ganador["min_cluster_pct"] * 100, 2),
                "es_degenerado": cand_ganador["es_degenerado"],
                "pasa_compuertas": pasa_todos
            })

            print(f"  omega={omega}: k={cand_ganador['k']}, pureza={cand_ganador['pureza_naming']:.4f}, ARI={cand_ganador['bootstrap_ari_mean']:.4f}, degen={cand_ganador['es_degenerado']}, pasa={pasa_todos}")

            if pasa_todos and not algun_omega_pasa:
                algun_omega_pasa = True
                omega_ganador = omega

        df_sweep = pd.DataFrame(resultados_sweep_vip)
        print("\n--- RESULTADOS SWEEP VIP ---")
        print(df_sweep.to_string(index=False))

        if not algun_omega_pasa:
            max_p = df_sweep["pureza_naming"].max()
            print(f"\n [TOPE DURO ALCANZADO] Incluso con omega_nlp = 0.5, la pureza maxima es {max_p:.4f} (< 0.85).")
            print("   Se declara el sub-espacio VIP IRREDUCIBLE bajo la compuerta formal. No se prueban mas combinaciones.")
        else:
            print(f"\n [EXITO] omega_nlp = {omega_ganador} supero todas las compuertas.")

    return {
        "reporte": df_rep,
        "top_terminos": top_terminos_por_subespacio,
        "sweep_vip": pd.DataFrame(resultados_sweep_vip) if resultados_sweep_vip else None
    }


if __name__ == "__main__":
    diagnosticar_subespacios()
