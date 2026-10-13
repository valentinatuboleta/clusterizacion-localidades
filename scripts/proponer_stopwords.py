"""
Script de minería y propuesta de stopwords de marketing para nombres de localidades/productos.
Fase: Cola de revisión humana navegable (top-100 comprimido mediante subsumción y clases sistemáticas).

Funcionalidades:
1. Clases sistemáticas por regex (calendario, crédito, duración, ciudad, venue_paren) agrupadas
   en una sola fila representativa por clase con conteo agregado y ejemplos reales.
2. Extracción de n-gramas (1-3) candidatos bajo las 4 rejillas de seguridad:
   (a) Fuera del vocabulario de los tags estructurales/espaciales/restricción.
   (b) Fuera del vocabulario TF-IDF vigente.
   (c) Fuera del vocabulario estructural auxiliar (PALCO, VIP, PLATEA, GRADA, etc.).
   (d) Frecuencia >= min_eventos (default 20).
3. Deduplicación por subsumción de n-gramas: colapsa sub-frases en el n-grama padre de mayor
   frecuencia y máxima longitud, registrando la columna 'subsumidos'.
4. Separación de observación: candidatos entre 3 y 19 eventos (o excedente top-100) van a
   data/lookup/stopwords_observacion.csv (ignorado por git).
5. Historial permanente y no re-proposición: preserva filas previamente rechazadas o aceptadas;
   nunca re-propone patrones con decision='rechazar' ni ya integrados en el CSV oficial.
6. Tope de salida: máximo 100 filas en data/lookup/stopwords_revision_humana.csv ordenadas por
   frecuencia descendente.
7. Modo --consolidar: valida decision in {aceptar, rechazar}, migra aceptadas al CSV oficial
   incrementando version_vocab, sincroniza con Azure Blob Storage si existe AZURE_STOPWORDS_URL,
   y preserva el historial permanente en la cola de revisión.
"""

import os
import sys
import re
import argparse
import datetime
import unicodedata
import warnings
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp_utils import normalizar_texto, cargar_stopwords

# Rejilla (a): Vocabulario léxico utilizado en la extracción de tags
VOCABULARIO_TAGS: Set[str] = {
    "PALCO", "PALCOS", "BOX", "BOXES", "SUITE", "SUITES", "MESA", "MESAS",
    "VIP", "PLATINUM", "PLATINO", "PREMIUM", "GOLD", "DIAMANTE", "ORO", "PLATA",
    "PLATEA", "SILLAS", "SILLERIA", "PISTA", "CANCHA",
    "PREFERENCIAL", "PREFERENTE", "CENTRAL", "FRONTAL",
    "GENERAL", "TIQUETE", "ENTRADA", "STANDARD", "NORMAL", "ADMISION",
    "BALCON", "BALCONES", "MEZZANINE", "VOLADIZO",
    "ALTA", "ALTAS", "PISO", "SEGUNDO", "TERCER", "CUARTO", "POSTERIOR", "ALTO",
    "BAJA", "BAJAS", "PRIMER", "PLANTA", "DELANTERA", "FILA", "BAJO",
    "OCCIDENTAL", "OCC", "OESTE",
    "ORIENTAL", "ORI", "ESTE",
    "NORTE", "NTE",
    "SUR",
    "LATERAL", "LATERALES", "COSTADO", "ESQUINA",
    "VISTA", "VISIBILIDAD", "PARCIAL", "RESTRINGIDA", "REDUCIDA", "OBSTRUIDA", "PILARES",
    "FAMILIAR", "FAMILIA",
    "MENORES", "KIDS", "NINOS", "INFANTIL", "LIBRE", "ALCOHOL", "CERO",
    "MOVILIDAD", "DISCAPACIDAD", "PMR", "RUEDAS", "ACCESIBLE"
}

# Rejilla (b): Vocabulario TF-IDF base
VOCABULARIO_TFIDF_BASE: Set[str] = {
    "OCCIDENTAL", "NORTE", "PLATEA", "PALCO", "BALCON", "POSTERIOR",
    "CENTRAL", "GENERAL", "PISO", "LATERAL", "SEGUNDO", "SUR", "MESA", "1", "2"
}

# Rejilla (c): Vocabulario estructural auxiliar que jamás debe suprimirse
VOCABULARIO_ESTRUCTURAL_AUXILIAR: Set[str] = {
    "GENERAL", "PALCO", "PLATEA", "GRADA", "GRADAS", "BALCON", "VIP",
    "PREFERENCIAL", "FILA", "SILLA", "SILLAS", "MESA", "MESAS", "PISO",
    "OCCIDENTAL", "NORTE", "SUR", "ORIENTAL", "TRIBUNA", "SECTOR",
    "LOCALIDAD", "ASIENTO", "BOLETA", "ENTRADA", "NIVEL", "ZONA",
    "SECCION", "BLOQUE", "PUERTA", "ACCESO", "CORREDOR", "UBICACION"
}

# Stopwords gramaticales del español que no deben iniciar o terminar un n-grama
STOPWORDS_GRAMATICALES_ES: Set[str] = {
    "DE", "LA", "EL", "EN", "Y", "A", "DEL", "LOS", "LAS", "POR", "CON",
    "PARA", "UN", "UNA", "AL", "O", "E", "ES", "SE", "SU", "SUS", "MAS",
    "LO", "LE", "LES"
}

# Clases sistemáticas por regex
CLASES_SISTEMATICAS: Dict[str, Dict[str, str]] = {
    "calendario": {
        "patron_detect": r"\b(ENE|FEB|MAR|ABR|MAYO|JUN|JUL|AGO|SEP|OCT|NOV|DIC|ENERO|FEBRERO|MARZO|ABRIL|JUNIO|JULIO|AGOSTO|SEPTIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)\b",
        "patron_salida": r"\b(ENE|FEB|MAR|ABR|MAYO|JUN|JUL|AGO|SEP|OCT|NOV|DIC)\b",
        "tipo": "calendario"
    },
    "credito": {
        "patron_detect": r"\(DIR\.[^\)]*\)|\bDIR\.\s+[A-Z]+|\(DIR\.",
        "patron_salida": r"\(DIR\.[^\)]*\)",
        "tipo": "credito"
    },
    "duracion": {
        "patron_detect": r"\b\d+\s*MIN\b",
        "patron_salida": r"\b\d+\s*MIN\b",
        "tipo": "duracion"
    },
    "ciudad": {
        "patron_detect": r"\b(BOGOTA|MEDELLIN|CALI|BARRANQUILLA|CARTAGENA|BUCARAMANGA|PEREIRA|MANIZALES|SANTA MARTA|IBAGUE|CUCUTA|VILLAVICENCIO|PASTO|POPAYAN|ARMENIA|MONTERIA|VALLEDUPAR)\b",
        "patron_salida": r"\b(BOGOTA|MEDELLIN|CALI|BARRANQUILLA|CARTAGENA|BUCARAMANGA|PEREIRA|MANIZALES|SANTA MARTA|IBAGUE|CUCUTA|VILLAVICENCIO|PASTO|POPAYAN|ARMENIA|MONTERIA|VALLEDUPAR)\b",
        "tipo": "ciudad"
    },
    "venue_paren": {
        "patron_detect": r"\([^\)]*(TEATRO|MUSEO|PARQUE|ARENA|COLISEO)[^\)]*\)",
        "patron_salida": r"\([^\)]*(TEATRO|MUSEO|PARQUE|ARENA|COLISEO)[^\)]*\)",
        "tipo": "venue_paren"
    }
}

# Tokens individuales que componen las clases y que no deben generar n-gramas aislados
TOKENS_CLASES_BLOQUEADOS: Set[str] = (
    set(re.findall(r"\w+", CLASES_SISTEMATICAS["calendario"]["patron_detect"].upper())) |
    set(re.findall(r"\w+", CLASES_SISTEMATICAS["ciudad"]["patron_detect"].upper())) |
    {"DIR"}
)


def obtener_vocabulario_prohibido(ruta_modelo: Optional[str] = None) -> Set[str]:
    """
    Unifica las rejillas (a), (b) y (c) en un conjunto en mayúsculas sin tildes.
    Si se encuentra el artefacto del modelo, extrae dinámicamente el vocabulario TF-IDF vigente.
    """
    vocab_prohibido = set(VOCABULARIO_TAGS) | set(VOCABULARIO_ESTRUCTURAL_AUXILIAR) | set(VOCABULARIO_TFIDF_BASE)

    if ruta_modelo and os.path.exists(ruta_modelo):
        try:
            import joblib
            modelo_dict = joblib.load(ruta_modelo)
            tfidf_vec = modelo_dict.get("tfidf_vectorizer")
            if tfidf_vec and hasattr(tfidf_vec, "vocabulary_"):
                for term in tfidf_vec.vocabulary_.keys():
                    term_norm = normalizar_texto(term)
                    if term_norm:
                        vocab_prohibido.add(term_norm)
        except Exception:
            pass

    return {normalizar_texto(w) for w in vocab_prohibido if normalizar_texto(w)}


def detectar_clases_sistematicas(
    prods: pd.DataFrame,
    col_producto: str = "product",
    col_evento: str = "t_performance_id"
) -> List[Dict[str, Any]]:
    """
    Detecta y agrupa clases sistemáticas por regex (calendario, crédito, duración, ciudad, venue_paren).
    Genera exactamente una fila representativa por clase con conteo agregado y ejemplos reales.
    """
    filas_clases = []

    for nombre_clase, conf in CLASES_SISTEMATICAS.items():
        pat_detect = conf["patron_detect"]
        pat_salida = conf["patron_salida"]
        tipo = conf["tipo"]

        eventos = set()
        frecuencia = 0
        ejemplos = []
        variantes = set()

        for _, row in prods.iterrows():
            p_str = normalizar_texto(str(row[col_producto]))
            perf_id = row[col_evento]
            matches = list(re.finditer(pat_detect, p_str, re.IGNORECASE))
            if matches:
                eventos.add(perf_id)
                frecuencia += len(matches)
                for m in matches:
                    variantes.add(m.group(0).upper())
                if len(ejemplos) < 3 and p_str not in ejemplos:
                    ejemplos.append(p_str)

        if len(eventos) > 0:
            filas_clases.append({
                "patron": pat_salida,
                "tipo": tipo,
                "frecuencia": frecuencia,
                "n_eventos": len(eventos),
                "subsumidos": len(variantes),
                "ejemplo_producto": " | ".join(ejemplos) if ejemplos else "",
                "decision": "",
                "motivo": ""
            })

    return filas_clases


def aplicar_subsumcion(candidatos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicación por subsumción de n-gramas:
    Descarta n-gramas contenidos como sub-frase en otro candidato de mayor o igual frecuencia.
    Regla: Prioriza mayor frecuencia y máxima longitud; contabiliza en 'subsumidos' cuántos candidatos absorbió.
    """
    if not candidatos:
        return []

    # Ordenar por: 1) frecuencia desc, 2) n_eventos desc, 3) n_tokens desc, 4) longitud desc
    cands_ordenados = sorted(
        candidatos,
        key=lambda x: (x["frecuencia"], x["n_eventos"], x.get("n_tokens", 1), len(x.get("raw_text", ""))),
        reverse=True
    )

    sobrevivientes: List[Dict[str, Any]] = []
    subsumidos_map: Dict[str, int] = defaultdict(int)

    for cand in cands_ordenados:
        raw = cand.get("raw_text", cand["patron"].replace(r"\b", ""))
        fue_subsumido = False

        for sob in sobrevivientes:
            sob_raw = sob.get("raw_text", sob["patron"].replace(r"\b", ""))
            # Verificar si raw está contenido como sub-frase en sob_raw con límites de palabra
            if re.search(r"\b" + re.escape(raw) + r"\b", sob_raw):
                subsumidos_map[sob["patron"]] += 1 + cand.get("subsumidos", 0)
                fue_subsumido = True
                break

        if not fue_subsumido:
            cand_copia = cand.copy()
            cand_copia["subsumidos"] = cand.get("subsumidos", 0)
            sobrevivientes.append(cand_copia)

    # Actualizar contador de subsumidos en los sobrevivientes
    for s in sobrevivientes:
        s["subsumidos"] = subsumidos_map[s["patron"]]

    return sobrevivientes


def extraer_candidatos_stopwords(
    df: pd.DataFrame,
    col_producto: str = "product",
    col_evento: str = "t_performance_id",
    min_eventos: int = 20,
    ruta_modelo: Optional[str] = None,
    ruta_stopwords_oficial: str = "data/lookup/stopwords_marketing.csv",
    ruta_revision_existente: str = "data/lookup/stopwords_revision_humana.csv",
    max_salida: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Pipeline completo de extracción, agrupamiento en clases, subsumción y separación por umbral:
    Retorna:
    - df_revision: Top <= max_salida candidatos con n_eventos >= min_eventos.
    - df_observacion: Candidatos con 3 <= n_eventos < min_eventos y excedentes del top 100.
    - metricas_compresion: Dict con conteos de etapas ('brutos', 'tras_subsumcion', 'tras_clases', 'final').
    """
    if col_producto not in df.columns:
        raise ValueError(f"Columna '{col_producto}' no encontrada en el DataFrame.")

    if col_evento not in df.columns:
        col_evento = "event_id_fallback"
        df = df.copy()
        df[col_evento] = range(len(df))

    vocab_prohibido = obtener_vocabulario_prohibido(ruta_modelo)

    # 1. Leer historial existente para no re-proponer rechazados ni aceptados
    patrones_rechazados: Set[str] = set()
    patrones_aceptados: Set[str] = set()
    historial_pendiente: Dict[str, Dict[str, str]] = {}

    if os.path.exists(ruta_revision_existente):
        try:
            df_rev_exist = pd.read_csv(ruta_revision_existente)
            if "patron" in df_rev_exist.columns:
                for _, r in df_rev_exist.iterrows():
                    pat = str(r["patron"]).strip()
                    dec = str(r.get("decision", "")).strip().lower()
                    mot = str(r.get("motivo", "")).strip() if pd.notna(r.get("motivo")) else ""
                    if dec == "rechazar":
                        patrones_rechazados.add(pat)
                    elif dec == "aceptar":
                        patrones_aceptados.add(pat)
                    elif pat:
                        historial_pendiente[pat] = {"decision": r.get("decision", ""), "motivo": mot}
        except Exception:
            pass

    # Stopwords oficiales ya aprobadas
    patrones_oficiales: Set[str] = set()
    patrones_limpios_oficiales: Set[str] = set()
    if os.path.exists(ruta_stopwords_oficial):
        try:
            for p in cargar_stopwords(ruta_stopwords_oficial):
                patrones_oficiales.add(p)
                p_clean = re.sub(r"\\b|[\\^$]", "", p).strip().upper()
                if p_clean:
                    patrones_limpios_oficiales.add(p_clean)
        except Exception:
            pass

    prods = df[[col_producto, col_evento]].dropna().drop_duplicates()

    # 2. Detección previa de Clases Sistemáticas
    clases_detectadas = detectar_clases_sistematicas(prods, col_producto, col_evento)

    filas_clases_rev: List[Dict[str, Any]] = []
    filas_clases_obs: List[Dict[str, Any]] = []

    for c in clases_detectadas:
        pat = c["patron"]
        if pat in patrones_rechazados or pat in patrones_oficiales:
            continue
        if pat in historial_pendiente:
            c["decision"] = historial_pendiente[pat]["decision"]
            c["motivo"] = historial_pendiente[pat]["motivo"]

        if c["n_eventos"] >= min_eventos:
            filas_clases_rev.append(c)
        elif c["n_eventos"] >= 3:
            c["motivo"] = "observacion (eventos < umbral)"
            filas_clases_obs.append(c)

    # 3. Minería de n-gramas individuales (1 a 3)
    eventos_por_ngram = defaultdict(set)
    frecuencia_ngram = defaultdict(int)
    ejemplo_por_ngram = {}

    for _, row in prods.iterrows():
        p_str = str(row[col_producto])
        perf_id = row[col_evento]
        t = normalizar_texto(p_str)
        t = re.sub(r"[^A-Z0-9\s]", " ", t)
        tokens = [w for w in t.split() if len(w) > 1 and not w.isdigit()]

        n_tokens = len(tokens)
        for n in (1, 2, 3):
            for i in range(n_tokens - n + 1):
                ngram_tokens = tokens[i:i + n]

                # Rejillas (a), (b), (c) y exclusión de tokens capturados por clases sistemáticas
                if any(tok in vocab_prohibido or tok in TOKENS_CLASES_BLOQUEADOS for tok in ngram_tokens):
                    continue

                if n == 1 and ngram_tokens[0] in STOPWORDS_GRAMATICALES_ES:
                    continue
                if ngram_tokens[0] in STOPWORDS_GRAMATICALES_ES or ngram_tokens[-1] in STOPWORDS_GRAMATICALES_ES:
                    continue

                ngram = " ".join(ngram_tokens)
                if len(ngram) < 3:
                    continue

                patron_candidato = f"\\b{ngram}\\b"
                if patron_candidato in patrones_rechazados or patron_candidato in patrones_oficiales or ngram in patrones_limpios_oficiales:
                    continue

                eventos_por_ngram[ngram].add(perf_id)
                frecuencia_ngram[ngram] += 1
                if ngram not in ejemplo_por_ngram:
                    ejemplo_por_ngram[ngram] = p_str

    total_brutos = len(eventos_por_ngram)

    # 4. Separación por umbral de eventos (>= min_eventos vs 3-19)
    cands_para_revision: List[Dict[str, Any]] = []
    cands_para_observacion: List[Dict[str, Any]] = []

    for ngram, eventos in eventos_por_ngram.items():
        n_ev = len(eventos)
        freq = frecuencia_ngram[ngram]
        cand = {
            "patron": f"\\b{ngram}\\b",
            "tipo": "tour",
            "frecuencia": freq,
            "n_eventos": n_ev,
            "n_tokens": len(ngram.split()),
            "raw_text": ngram,
            "ejemplo_producto": ejemplo_por_ngram[ngram],
            "decision": historial_pendiente.get(f"\\b{ngram}\\b", {}).get("decision", ""),
            "motivo": historial_pendiente.get(f"\\b{ngram}\\b", {}).get("motivo", "")
        }

        if n_ev >= min_eventos:
            cands_para_revision.append(cand)
        elif n_ev >= 3:
            cand["motivo"] = "observacion (eventos < umbral)"
            cands_para_observacion.append(cand)

    # 5. Aplicar Subsumción sobre los candidatos a revisión
    cands_sobrevivientes = aplicar_subsumcion(cands_para_revision)
    total_tras_subsumcion = len(cands_sobrevivientes)

    # 6. Integración con Clases Sistemáticas
    cola_combinada = filas_clases_rev + cands_sobrevivientes
    total_tras_clases = len(cola_combinada)

    # Ordenar por frecuencia desc y n_eventos desc
    cola_combinada = sorted(cola_combinada, key=lambda x: (x["frecuencia"], x["n_eventos"]), reverse=True)

    # 7. Tope de salida (máximo max_salida = 100)
    top_revision = cola_combinada[:max_salida]
    excedente_revision = cola_combinada[max_salida:]

    for exc in excedente_revision:
        exc_copia = exc.copy()
        exc_copia["motivo"] = "observacion (excedente top-100)"
        cands_para_observacion.append(exc_copia)

    # Formatear DataFrames
    cols_revision = ["patron", "tipo", "frecuencia", "n_eventos", "subsumidos", "ejemplo_producto", "decision", "motivo"]
    cols_obs = ["patron", "tipo", "frecuencia", "n_eventos", "subsumidos", "ejemplo_producto", "motivo"]

    df_rev = pd.DataFrame(top_revision)
    if not df_rev.empty:
        # Asegurar presencia de todas las columnas
        for col in cols_revision:
            if col not in df_rev.columns:
                df_rev[col] = ""
        df_rev = df_rev[cols_revision]
    else:
        df_rev = pd.DataFrame(columns=cols_revision)

    # También aplicar subsumción rápida en observación para higiene de almacenamiento
    obs_combinada = filas_clases_obs + cands_para_observacion
    obs_combinada = sorted(obs_combinada, key=lambda x: (x["frecuencia"], x["n_eventos"]), reverse=True)
    df_obs = pd.DataFrame(obs_combinada)
    if not df_obs.empty:
        for col in cols_obs:
            if col not in df_obs.columns:
                df_obs[col] = ""
        df_obs = df_obs[cols_obs]
    else:
        df_obs = pd.DataFrame(columns=cols_obs)

    metricas_compresion = {
        "brutos": total_brutos,
        "tras_subsumcion": total_tras_subsumcion,
        "tras_clases": total_tras_clases,
        "final": len(df_rev)
    }

    return df_rev, df_obs, metricas_compresion


def consolidar_stopwords(
    ruta_revision: str = "data/lookup/stopwords_revision_humana.csv",
    ruta_oficial: str = "data/lookup/stopwords_marketing.csv"
) -> Tuple[int, str]:
    """
    Modo --consolidar:
    1. Lee stopwords_revision_humana.csv y valida que decision in {'', 'aceptar', 'rechazar'}.
       Cualquier valor inválido emite warning y se omite de la consolidación.
    2. Identifica filas con decision='aceptar' que aún no estén en stopwords_marketing.csv.
    3. Anexa las nuevas entradas a stopwords_marketing.csv (fuente='propuesto', fecha=hoy, version_vocab bump).
    4. Sube a Azure Blob Storage si AZURE_STOPWORDS_URL está configurada.
    5. Marca en el archivo de revisión la fecha de consolidación sin eliminar las filas
       (el archivo es un historial permanente que previene re-litigios).

    Retorna (num_filas_consolidadas, nueva_version).
    """
    if not os.path.exists(ruta_revision):
        print(f"No existe archivo de revision en '{ruta_revision}'.")
        return 0, ""

    df_rev = pd.read_csv(ruta_revision)
    if "decision" not in df_rev.columns:
        print("El archivo de revision no contiene columna 'decision'.")
        return 0, ""

    # 1. Validación estricta de decision
    decisiones_raw = df_rev["decision"].fillna("").astype(str).str.strip().str.lower()
    valores_validos = {"", "aceptar", "rechazar"}
    valores_invalidos = set(decisiones_raw.unique()) - valores_validos

    if valores_invalidos:
        filas_invalidas = df_rev[decisiones_raw.isin(valores_invalidos)]
        warnings.warn(
            f"Se detectaron {len(filas_invalidas)} filas con valores inválidos en 'decision': "
            f"{list(valores_invalidos)}. Solo se admiten ['aceptar', 'rechazar']. "
            f"Patrones afectados: {filas_invalidas['patron'].tolist()}. Estas filas no serán consolidadas.",
            UserWarning,
            stacklevel=2
        )

    mask_aceptar = decisiones_raw == "aceptar"
    df_aceptados = df_rev[mask_aceptar].copy()

    if df_aceptados.empty:
        print("No se encontraron filas válidas con decision='aceptar' para consolidar.")
        return 0, ""

    # 2. Cargar CSV oficial existente o inicializar
    if os.path.exists(ruta_oficial):
        df_oficial = pd.read_csv(ruta_oficial)
    else:
        df_oficial = pd.DataFrame(columns=["patron", "tipo", "fecha_agregado", "fuente", "version_vocab"])

    # 3. Determinar nueva versión
    versiones = df_oficial["version_vocab"].dropna().astype(str).tolist() if "version_vocab" in df_oficial.columns else []
    version_nums = []
    for v in versiones:
        m = re.search(r"v(\d+)", v)
        if m:
            version_nums.append(int(m.group(1)))
    max_ver = max(version_nums) if version_nums else 1
    nueva_version = f"v{max_ver + 1}"
    fecha_hoy = datetime.date.today().isoformat()

    nuevas_filas = []
    patrones_oficiales = set(df_oficial["patron"].dropna().tolist()) if "patron" in df_oficial.columns else set()

    for idx, row in df_aceptados.iterrows():
        patron = str(row["patron"]).strip()
        tipo_fila = str(row.get("tipo", "tour")).strip() or "tour"
        if patron in patrones_oficiales:
            continue
        nuevas_filas.append({
            "patron": patron,
            "tipo": tipo_fila,
            "fecha_agregado": fecha_hoy,
            "fuente": "propuesto",
            "version_vocab": nueva_version
        })
        patrones_oficiales.add(patron)

        # Marcar consolidación en el DataFrame de revisión
        df_rev.loc[idx, "motivo"] = f"consolidado {fecha_hoy} ({nueva_version})"

    if nuevas_filas:
        df_nuevas = pd.DataFrame(nuevas_filas)
        df_oficial = pd.concat([df_oficial, df_nuevas], ignore_index=True)
        os.makedirs(os.path.dirname(os.path.abspath(ruta_oficial)), exist_ok=True)
        df_oficial.to_csv(ruta_oficial, index=False)
        print(f"Consolidadas {len(nuevas_filas)} nuevas stopwords en '{ruta_oficial}' con version '{nueva_version}'.")
    else:
        print("Todas las filas marcadas con 'aceptar' ya se encontraban presentes en el CSV oficial.")

    # 4. Actualizar archivo de revisión preservando el historial
    df_rev.to_csv(ruta_revision, index=False)
    print(f"Archivo de revisión '{ruta_revision}' preservado como registro permanente.")

    # 5. Sincronización con Azure Blob Storage si existe la variable de entorno
    azure_blob_url = os.getenv("AZURE_STOPWORDS_URL")
    if azure_blob_url and nuevas_filas:
        try:
            from azure.storage.blob import BlobClient
            blob_client = BlobClient.from_blob_url(azure_blob_url)
            csv_content = df_oficial.to_csv(index=False).encode("utf-8")
            blob_client.upload_blob(csv_content, overwrite=True)
            print(f"CSV oficial subido exitosamente a Azure Blob ({azure_blob_url}).")
        except Exception as exc:
            print(f"Warning: No se pudo subir CSV oficial a Azure Blob ({azure_blob_url}): {exc}")

    return len(nuevas_filas), nueva_version


def main():
    parser = argparse.ArgumentParser(description="Minería y propuesta de stopwords de marketing con cola navegable.")
    parser.add_argument("--input", default="data/raw/localidades_eda.parquet", help="Ruta al archivo parquet con datos crudos.")
    parser.add_argument("--output", default="data/lookup/stopwords_revision_humana.csv", help="Ruta para el archivo de revisión humana (top-100).")
    parser.add_argument("--observacion", default="data/lookup/stopwords_observacion.csv", help="Ruta para el archivo de observación.")
    parser.add_argument("--oficial", default="data/lookup/stopwords_marketing.csv", help="Ruta al CSV oficial de stopwords.")
    parser.add_argument("--modelo", default="data/processed/modelo_clustering_v2_5.joblib", help="Ruta al modelo para extraer TF-IDF.")
    parser.add_argument("--min-eventos", type=int, default=20, help="Frecuencia mínima en eventos distintos (default: 20).")
    parser.add_argument("--max-salida", type=int, default=100, help="Tope máximo de filas en la cola de revisión (default: 100).")
    parser.add_argument("--consolidar", action="store_true", help="Consolidar filas con decision='aceptar' en el CSV oficial.")

    args = parser.parse_args()

    if args.consolidar:
        consolidar_stopwords(ruta_revision=args.output, ruta_oficial=args.oficial)
        return

    print(f"Leyendo dataset crudo desde '{args.input}'...")
    if not os.path.exists(args.input):
        print(f"Error: No existe el archivo de entrada '{args.input}'.")
        sys.exit(1)

    df_raw = pd.read_parquet(args.input)
    print(f"Dataset cargado: {len(df_raw):,} registros. Ejecutando extracción de candidatos (piso={args.min_eventos} eventos)...")

    df_revision, df_observacion, metricas = extraer_candidatos_stopwords(
        df=df_raw,
        col_producto="product",
        col_evento="t_performance_id",
        min_eventos=args.min_eventos,
        ruta_modelo=args.modelo if os.path.exists(args.modelo) else None,
        ruta_stopwords_oficial=args.oficial,
        ruta_revision_existente=args.output,
        max_salida=args.max_salida
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_revision.to_csv(args.output, index=False)
    print(f"Cola de revision generada: {len(df_revision)} filas guardadas en '{args.output}'.")

    os.makedirs(os.path.dirname(os.path.abspath(args.observacion)), exist_ok=True)
    df_observacion.to_csv(args.observacion, index=False)
    print(f"Cola de observacion guardada: {len(df_observacion):,} filas en '{args.observacion}' (ignorado por git).")

    print(
        f"\nCompresion de candidatos: "
        f"{metricas['brutos']:,} brutos -> "
        f"{metricas['tras_subsumcion']:,} tras subsumcion -> "
        f"{metricas['tras_clases']:,} tras clases -> "
        f"{metricas['final']} final (top-{args.max_salida})"
    )

    if not df_revision.empty:
        print("\nDistribución por tipo en cola de revisión:")
        print(df_revision["tipo"].value_counts().to_string())
        print("\nTop 10 candidatos para revisión humana:")
        print(df_revision.head(10)[["patron", "tipo", "frecuencia", "n_eventos", "subsumidos", "ejemplo_producto"]])


if __name__ == "__main__":
    main()
