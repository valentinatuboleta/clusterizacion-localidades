"""
Modulo de Arquitectura Jerarquica de Micro-Clusters (v3.0-hier.1).

Este modulo implementa el envoltorio operativo y generador de features de micro-clusters
en dos niveles a partir de arquetipos macro v2.3 congelados y sub-clustering especializado.

Componentes:
1. Constantes de version y catalogo canonico (20 micro-clusters vigentes).
2. Construccion de sub-espacio vectorial propio 32D por arquetipo.
3. Optimizacion Codo-DB por sub-espacio con descalificacion de soluciones degeneradas.
4. Generacion deterministica de etiquetas con deduplicacion insensible a acentos.
5. Inferencia de micro-clusters en produccion (predecir_microclusters) a prueba de fallos.
6. Evaluacion de drift estadistico especifico de micro-clusters (PSI).
"""

import os
import time
import unicodedata
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

from src.feature_engineering import (
    filtrar_consistencia_localidades,
    calcular_metricas_relativas,
    enriquecer_type_site
)
from src.nlp_utils import normalizar_texto, pipeline_procesamiento_nlp
from src.clustering import (
    separar_admision_unica_multizona,
    construir_espacio_vectorial_mixto,
    predecir_arquetipos_demanda,
    cargar_modelo_clustering,
    DEFAULT_NUMERIC_FEATURES
)


HIERARCHY_VERSION = "3.0-hier.1"

# 13 Tags estructurales expandidos del sub-espacio Nivel 2
TAGS_SUBESPACIO_13 = [
    "tag_palco", "tag_vip", "tag_platea", "tag_preferencial", "tag_general",
    "tag_balcon", "tag_piso_alto", "tag_lateral", "tag_occidental", "tag_oriental",
    "tag_norte", "tag_sur", "tag_mesa"
]

# Prefijos estandarizados de backend por arquetipo macro
ARQ_PREFIX = {
    "VIP / Palcos / Premium": "VIP",
    "Popular / Balcón / Visibilidad Parcial": "POP",
    "Platea General / Intermedia": "PGI",
    "Preferencial / Platea Frontal": "PPF",
    "Grada General / Masiva": "GGM"
}

# 20 Micro-clusters canonicos vigentes (1 terminal de admision unica + 19 multi-zona)
MICRO_CLUSTERS_CANONICAL = [
    "AU-0",
    "VIP-0", "VIP-1", "VIP-2", "VIP-3",
    "POP-0", "POP-1", "POP-2", "POP-3",
    "PGI-0", "PGI-1", "PGI-2", "PGI-3",
    "PPF-0", "PPF-1", "PPF-2", "PPF-3",
    "GGM-0", "GGM-1", "GGM-2"
]

TAG_LABEL_MAP = {
    "tag_palco": "Palco",
    "tag_mesa": "Mesa",
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


def _normalizar_token(token: str) -> str:
    """Normaliza un token eliminando acentos y diacriticos para comparaciones de duplicidad."""
    nfkd = unicodedata.normalize("NFD", token)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower().strip()


def generar_label_auto_v3(
    cluster_id: int,
    tag_shares: pd.Series,
    term_dominante: str,
    purity_tag: float,
    purity_term: float,
    threshold_tag_active: float = 0.60,
    min_purity_tag: float = 0.70
) -> str:
    """
    Genera automaticamente el nombre representativo del micro-cluster con deduplicacion insensible a acentos.
    Si la pureza estructural es inferior a min_purity_tag (0.70), retorna 'hibrido_k{cluster_id}'.
    """
    if purity_tag < min_purity_tag or not term_dominante or term_dominante == "ninguno" or purity_term < 0.10:
        return f"hibrido_k{cluster_id}"

    active_tags = []
    if len(tag_shares) > 0:
        for tag_col, val in tag_shares.items():
            if val >= threshold_tag_active:
                clean_name = TAG_LABEL_MAP.get(tag_col, tag_col.replace("tag_", "").capitalize())
                active_tags.append(clean_name)

    if not active_tags and len(tag_shares) > 0 and tag_shares.max() > 0:
        dom_col = tag_shares.idxmax()
        active_tags.append(TAG_LABEL_MAP.get(dom_col, dom_col.replace("tag_", "").capitalize()))

    partes = []
    seen = set()
    for tag_name in active_tags:
        for word in tag_name.split():
            norm_word = _normalizar_token(word)
            if norm_word not in seen:
                seen.add(norm_word)
                partes.append(word)

    norm_term = _normalizar_token(term_dominante)
    if norm_term not in seen and purity_term >= 0.15:
        seen.add(norm_term)
        partes.append(term_dominante.capitalize())

    if not partes:
        return f"hibrido_k{cluster_id}"

    return " ".join(partes)


def calcular_pureza_cluster(
    df_cluster: pd.DataFrame,
    tag_cols: List[str],
    tfidf_vocab: List[str]
) -> Dict[str, Any]:
    """
    Calcula la pureza estructural y lexica de un cluster individual.
    - purity_tag: share del tag estructural dominante (max activacion).
    - purity_term: share del termino TF-IDF dominante en el texto limpio.
    - purity_naming: combinacion de ambas senales max(purity_tag, purity_term).
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

    # 2. Terminos lexicos TF-IDF
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


def calcular_estabilidad_bootstrap_ari(
    X: np.ndarray,
    labels_full: np.ndarray,
    estimador_fn: Any,
    n_iter: int = 20,
    frac: float = 0.80,
    random_state: int = 42
) -> Tuple[float, float, List[float]]:
    """
    Evalua la estabilidad de la particion mediante remuestreos al 80%.
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

        ari = adjusted_rand_score(labels_full[idx_sub], labels_sub)
        ari_scores.append(float(ari))

    return float(np.mean(ari_scores)), float(np.std(ari_scores)), ari_scores


def construir_subespacio_arquetipo(
    df_sub: pd.DataFrame,
    columnas_tags: List[str] = TAGS_SUBESPACIO_13,
    max_tfidf: int = 15,
    peso_nlp: float = 0.2,
    scaler: Optional[Any] = None,
    tfidf_vectorizer: Optional[Any] = None
) -> Tuple[np.ndarray, Any, Any, List[str]]:
    """
    Construye la representacion vectorial 32D propia del sub-espacio:
    4 numericas relativas (RobustScaler) + 13 tags estructurales binarios + 15 componentes TF-IDF (peso 0.2).
    Se excluye one-hot de type_site para evitar sobrefragmentacion por venue.
    """
    return construir_espacio_vectorial_mixto(
        df_sub,
        columnas_numericas=DEFAULT_NUMERIC_FEATURES,
        columnas_tags=columnas_tags,
        usar_tfidf_texto=True,
        max_tfidf_features=max_tfidf,
        peso_nlp=peso_nlp,
        peso_type_site=0.0,
        scaler_type="robust",
        scaler=scaler,
        tfidf_vectorizer=tfidf_vectorizer
    )


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
    Selecciona el k optimo en el sub-espacio mediante score compuesto Codo-DB
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

        counts = pd.Series(labels).value_counts()
        min_cluster_size = counts.min()
        min_pct = min_cluster_size / n_sub
        es_degenerado = bool(min_pct < min_pct_piso)

        # Pureza de naming
        df_temp = df_sub.copy()
        df_temp["_cluster"] = labels
        p_namings, p_tags, p_terms = [], [], []

        for c_id in range(k):
            sub_c = df_temp[df_temp["_cluster"] == c_id]
            res_p = calcular_pureza_cluster(sub_c, TAGS_SUBESPACIO_13, vocab_sub)
            p_namings.append(res_p["purity_naming"])
            p_tags.append(res_p["purity_tag"])
            p_terms.append(res_p["purity_term"])

        weights = np.array([len(df_temp[df_temp["_cluster"] == c_id]) for c_id in range(k)])
        pureza_naming = float(np.sum(weights * np.array(p_namings)) / np.sum(weights))
        pureza_tag = float(np.sum(weights * np.array(p_tags)) / np.sum(weights))
        pureza_term = float(np.sum(weights * np.array(p_terms)) / np.sum(weights))

        # Bootstrap-ARI
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

    no_degenerados = [c for c in candidatos if not c["es_degenerado"]]
    if not no_degenerados:
        no_degenerados = candidatos

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


def predecir_microclusters(
    df_nuevas: pd.DataFrame,
    payload_jerarquia: Union[str, Dict[str, Any]],
    payload_v23: Optional[Union[str, Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Asigna micro-clusters y arquetipos de demanda para un nuevo DataFrame de localidades.
    
    Flujo:
    1. Ejecuta Nivel 1 v2.3 para determinar arquetipo_demanda, score_confianza y es_frontera.
    2. Monozona / Admision Unica -> 'AU-0' con label 'Admisión Única'.
    3. Multi-zona: enriquece con tags NLP, transforma en el sub-espacio del arquetipo
       y predice con su sub-KMeans local.
    4. Robusto: Ante venues desconocidos o datos faltantes, activa flags seguros sin fallar.
    
    Devuelve DataFrame alineado al indice original con:
    logical_seat_category, micro_cluster_id, label_auto, arquetipo_demanda, score_confianza, es_frontera.
    """
    if isinstance(payload_jerarquia, str):
        payload_hier = joblib.load(payload_jerarquia)
    else:
        payload_hier = payload_jerarquia

    df_input = df_nuevas.copy()

    # Consistencia y preparacion defensiva sin perder registros
    if "logical_seat_category" not in df_input.columns:
        col_alt = next((c for c in ["product", "translation_name", "cd_name", "nombre_localidad"] if c in df_input.columns), None)
        if col_alt:
            df_input["logical_seat_category"] = df_input[col_alt].astype(str)
        else:
            df_input["logical_seat_category"] = "GENERAL"

    # Tipologia de venue
    if "type_site" not in df_input.columns:
        df_input = enriquecer_type_site(df_input)

    # Texto y tags NLP
    if "texto_limpio" not in df_input.columns or any(t not in df_input.columns for t in TAGS_SUBESPACIO_13):
        df_input = pipeline_procesamiento_nlp(df_input, col_nombre="logical_seat_category")

    # Metricas relativas
    if any(c not in df_input.columns for c in DEFAULT_NUMERIC_FEATURES):
        if "t_performance_id" in df_input.columns and "performance_quota" in df_input.columns and "dn_quota" in df_input.columns:
            if "med_unit_amt_itx" not in df_input.columns:
                col_p = next((c for c in ["price_amount", "precio", "unit_amount"] if c in df_input.columns), None)
                if col_p:
                    df_input["med_unit_amt_itx"] = df_input[col_p]
            if "med_unit_amt_itx" in df_input.columns and "net_sold_p_qty" in df_input.columns and "net_sold_c_qty" in df_input.columns:
                try:
                    df_input = calcular_metricas_relativas(df_input)
                except Exception:
                    pass

        if "peso_aforo" not in df_input.columns:
            if "dn_quota" in df_input.columns and "performance_quota" in df_input.columns:
                denom = df_input["performance_quota"].replace(0, np.nan)
                df_input["peso_aforo"] = np.clip(df_input["dn_quota"] / denom, 0.0, 1.0).fillna(0.0)
            elif "t_performance_id" in df_input.columns:
                counts = df_input.groupby("t_performance_id")["t_performance_id"].transform("count")
                df_input["peso_aforo"] = np.where(counts == 1, 1.0, 0.0)

        for num_col in DEFAULT_NUMERIC_FEATURES:
            if num_col not in df_input.columns:
                df_input[num_col] = 0.50 if "percentil" in num_col else 0.0

    # 1. Ejecutar Nivel 1 (v2.3)
    if payload_v23 is not None:
        if isinstance(payload_v23, str):
            p_v23 = cargar_modelo_clustering(payload_v23)
        else:
            p_v23 = payload_v23
        df_n1 = predecir_arquetipos_demanda(df_input, p_v23)
    elif "nivel_1" in payload_hier and "kmeans" in payload_hier["nivel_1"]:
        # Usar nivel_1 persistido en el modelo jerarquico
        n1_info = payload_hier["nivel_1"]
        km_n1 = n1_info["kmeans"]
        sc_n1 = n1_info["scaler"]
        tf_n1 = n1_info["tfidf"]
        mapa_n1 = n1_info.get("metricas", {}).get("mapa_arquetipos") or n1_info.get("mapa_arquetipos", {})

        df_mono, df_multi = separar_admision_unica_multizona(df_input)
        df_mono = df_mono.copy()
        df_mono["arquetipo_demanda"] = "Admisión Única / Tarifa Plana"
        df_mono["score_confianza"] = 1.0
        df_mono["es_frontera"] = False

        if len(df_multi) > 0:
            df_multi = df_multi.copy()
            X_multi, _, _, _ = construir_espacio_vectorial_mixto(
                df_multi,
                columnas_numericas=DEFAULT_NUMERIC_FEATURES,
                scaler=sc_n1,
                tfidf_vectorizer=tf_n1,
                peso_nlp=0.2,
                peso_type_site=0.5
            )
            dist_n1 = km_n1.transform(X_multi)
            orden_dist = np.argsort(dist_n1, axis=1)
            d1 = dist_n1[np.arange(len(dist_n1)), orden_dist[:, 0]]
            d2 = dist_n1[np.arange(len(dist_n1)), orden_dist[:, 1]]
            margen = (d2 - d1) / (d2 + 1e-9)

            labels_n1 = orden_dist[:, 0]
            df_multi["arquetipo_demanda"] = [mapa_n1.get(c, str(c)) for c in labels_n1]
            df_multi["score_confianza"] = margen
            df_multi["es_frontera"] = margen < 0.15

        df_n1 = pd.concat([df_mono, df_multi]).loc[df_input.index]
    else:
        # Fallback de seguridad
        df_n1 = df_input.copy()
        df_n1["arquetipo_demanda"] = "Admisión Única / Tarifa Plana"
        df_n1["score_confianza"] = 1.0
        df_n1["es_frontera"] = False

    # 2. Inicializar salidas de micro-clusters
    df_res = df_n1.copy()
    df_res["micro_cluster_id"] = "AU-0"
    df_res["label_auto"] = "Admisión Única"

    sub_modelos = payload_hier.get("sub_modelos", {})

    # 3. Prediccion en sub-espacios Nivel 2
    for arq, sub_model in sub_modelos.items():
        mask = (df_res["arquetipo_demanda"] == arq)
        if not mask.any():
            continue

        sub_rows = df_res[mask].copy()
        cols_num = sub_model.get("columnas_numericas", DEFAULT_NUMERIC_FEATURES)
        
        # Validar valores numericos no nulos
        vals_num = sub_rows[cols_num].fillna(0.0).values
        scaler_sub = sub_model.get("scaler") or sub_model.get("scaler_sub")
        if hasattr(scaler_sub, "transform"):
            X_num = scaler_sub.transform(vals_num)
        else:
            X_num = vals_num

        tag_cols = sub_model.get("columnas_tags") or sub_model.get("columnas_tags_sub") or TAGS_SUBESPACIO_13
        for t_col in tag_cols:
            if t_col not in sub_rows.columns:
                sub_rows[t_col] = 0
        X_tags = sub_rows[tag_cols].values.astype(float)

        tfidf_sub = sub_model.get("tfidf") or sub_model.get("tfidf_sub")
        if hasattr(tfidf_sub, "transform"):
            X_tfidf = tfidf_sub.transform(sub_rows["texto_limpio"].fillna("")).toarray() * 0.2
        else:
            X_tfidf = np.zeros((len(sub_rows), 15))

        X_sub = np.hstack([X_num, X_tags, X_tfidf])

        km_sub = sub_model.get("kmeans") or sub_model.get("kmeans_sub")
        sub_labels = km_sub.predict(X_sub)

        prefix = sub_model.get("prefix", ARQ_PREFIX.get(arq, "SUB"))
        mapa_lbl = sub_model.get("mapa_labels") or sub_model.get("mapa_micro_label", {})

        mc_ids = [f"{prefix}-{c}" for c in sub_labels]
        lbl_autos = [mapa_lbl.get(c, f"hibrido_k{c}") for c in sub_labels]

        df_res.loc[mask, "micro_cluster_id"] = mc_ids
        df_res.loc[mask, "label_auto"] = lbl_autos

    # Asegurar columnas exactas de retorno preservando orden original
    cols_salida = [
        "logical_seat_category",
        "micro_cluster_id",
        "label_auto",
        "arquetipo_demanda",
        "score_confianza",
        "es_frontera"
    ]
    return df_res[cols_salida].loc[df_nuevas.index]


def evaluar_drift_microclusters(
    df_pred: pd.DataFrame,
    payload_jerarquia: Union[str, Dict[str, Any]],
    umbral_psi: float = 0.10,
    eps: float = 1e-4
) -> Dict[str, Any]:
    """
    Evalua el Population Stability Index (PSI) de la distribucion de los 20 micro-clusters
    frente a la distribucion de referencia persistida en el payload jerarquico.
    """
    if isinstance(payload_jerarquia, str):
        payload_hier = joblib.load(payload_jerarquia)
    else:
        payload_hier = payload_jerarquia

    dist_ref = payload_hier.get("distribucion_referencia_microclusters", {})
    categorias = payload_hier.get("categorias_vigentes", MICRO_CLUSTERS_CANONICAL)

    if not dist_ref:
        # Si no esta persistida, asumir uniforme sobre categorias
        dist_ref = {c: 1.0 / len(categorias) for c in categorias}

    total_filas = max(len(df_pred), 1)
    conteo_obs = df_pred["micro_cluster_id"].value_counts().to_dict()
    dist_obs = {c: conteo_obs.get(c, 0) / total_filas for c in categorias}

    # Calculo formal de PSI discreto
    psi_total = 0.0
    desviaciones = {}
    alertas = []

    for cat in categorias:
        p = dist_ref.get(cat, eps)
        q = dist_obs.get(cat, 0.0)
        p_safe = max(p, eps)
        q_safe = max(q, eps)
        psi_i = (q_safe - p_safe) * np.log(q_safe / p_safe)
        psi_total += psi_i

        diff = q - p
        desviaciones[cat] = {
            "referencia": round(p, 4),
            "observado": round(q, 4),
            "diferencia": round(diff, 4),
            "psi_parcial": round(float(psi_i), 4)
        }
        if abs(diff) > 0.08:
            alertas.append(f"Desviacion en micro-cluster {cat}: {diff:+.1%}")

    psi_total = float(psi_total)

    if psi_total < 0.10:
        estado = "ESTABLE"
    elif psi_total <= 0.25:
        estado = "REVISAR"
    else:
        estado = "DRIFT_CRITICO"
        alertas.append(f"Drift critico global en micro-clusters (PSI = {psi_total:.4f})")

    alerta_activa = bool(psi_total >= umbral_psi or len(alertas) > 0)

    return {
        "psi_global": round(psi_total, 4),
        "estado": estado,
        "umbral_evaluado": umbral_psi,
        "alerta_activa": alerta_activa,
        "distribucion_microclusters": desviaciones,
        "total_registros": total_filas,
        "alertas": alertas
    }
