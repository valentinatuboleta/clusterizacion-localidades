"""
Script de minería y propuesta de stopwords de marketing para nombres de localidades/productos.
Precedente exacto del flujo de curaduría léxica de venues (clasificar_sites.py).

Funcionalidades:
1. Extraer n-gramas (1-3) frecuentes en el campo 'product' que pasen las 4 rejillas de seguridad:
   (a) Fuera del vocabulario de los tags estructurales/espaciales/restricción.
   (b) Fuera del vocabulario TF-IDF vigente.
   (c) Fuera del vocabulario estructural auxiliar (PALCO, PLATEA, VIP, GENERAL, etc.).
   (d) Frecuencia >= 3 eventos distintos (t_performance_id).
2. Generar data/lookup/stopwords_revision_humana.csv (patron, frecuencia, n_eventos, ejemplo_producto, decision).
3. Modo --consolidar: Mueve las filas aprobadas (decision=aceptar) al CSV oficial
   (data/lookup/stopwords_marketing.csv) incrementando version_vocab y subiendo a Azure Blob
   si AZURE_STOPWORDS_URL está configurada.
"""

import os
import sys
import re
import argparse
import datetime
import unicodedata
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


def extraer_candidatos_stopwords(
    df: pd.DataFrame,
    col_producto: str = "product",
    col_evento: str = "t_performance_id",
    min_eventos: int = 3,
    ruta_modelo: Optional[str] = None,
    ruta_stopwords_oficial: str = "data/lookup/stopwords_marketing.csv"
) -> pd.DataFrame:
    """
    Extrae n-gramas (1-3) candidatos sobre el campo de producto aplicando las 4 rejillas:
    (a) Fuera de los tags.
    (b) Fuera del vocabulario TF-IDF.
    (c) Fuera del vocabulario estructural auxiliar.
    (d) Frecuencia >= min_eventos en eventos distintos.
    """
    if col_producto not in df.columns:
        raise ValueError(f"Columna '{col_producto}' no encontrada en el DataFrame.")

    if col_evento not in df.columns:
        col_evento = "event_id_fallback"
        df = df.copy()
        df[col_evento] = range(len(df))

    vocab_prohibido = obtener_vocabulario_prohibido(ruta_modelo)

    # Obtener stopwords oficiales ya aprobadas
    try:
        stopwords_existentes = set(cargar_stopwords(ruta_stopwords_oficial))
    except Exception:
        stopwords_existentes = set()

    # Pre-calcular patrones normalizados limpios existentes para evitar duplicados
    patrones_limpios_existentes = set()
    for p in stopwords_existentes:
        p_clean = re.sub(r"\\b|[\\^$]", "", p).strip().upper()
        if p_clean:
            patrones_limpios_existentes.add(p_clean)

    prods = df[[col_producto, col_evento]].dropna().drop_duplicates()

    eventos_por_ngram = defaultdict(set)
    frecuencia_ngram = defaultdict(int)
    ejemplo_por_ngram = {}

    for _, row in prods.iterrows():
        p_str = str(row[col_producto])
        perf_id = row[col_evento]
        t = normalizar_texto(p_str)
        # Limpiar signos y caracteres especiales
        t = re.sub(r"[^A-Z0-9\s]", " ", t)
        tokens = [w for w in t.split() if len(w) > 1 and not w.isdigit()]

        n_tokens = len(tokens)
        for n in (1, 2, 3):
            for i in range(n_tokens - n + 1):
                ngram_tokens = tokens[i:i + n]

                # Rejillas (a), (b), (c): Ningún token puede estar en el vocabulario prohibido
                if any(tok in vocab_prohibido for tok in ngram_tokens):
                    continue

                # Evitar stopwords gramaticales en bordes o aisladas
                if n == 1 and ngram_tokens[0] in STOPWORDS_GRAMATICALES_ES:
                    continue
                if ngram_tokens[0] in STOPWORDS_GRAMATICALES_ES or ngram_tokens[-1] in STOPWORDS_GRAMATICALES_ES:
                    continue

                ngram = " ".join(ngram_tokens)
                if len(ngram) < 3:
                    continue

                # Evitar n-gramas ya presentes en el CSV oficial
                patron_candidato = f"\\b{ngram}\\b"
                if patron_candidato in stopwords_existentes or ngram in patrones_limpios_existentes:
                    continue

                eventos_por_ngram[ngram].add(perf_id)
                frecuencia_ngram[ngram] += 1
                if ngram not in ejemplo_por_ngram:
                    ejemplo_por_ngram[ngram] = p_str

    candidatos = []
    for ngram, eventos in eventos_por_ngram.items():
        n_ev = len(eventos)
        # Rejilla (d): Frecuencia mínima de eventos distintos
        if n_ev >= min_eventos:
            candidatos.append({
                "patron": f"\\b{ngram}\\b",
                "frecuencia": frecuencia_ngram[ngram],
                "n_eventos": n_ev,
                "ejemplo_producto": ejemplo_por_ngram[ngram],
                "decision": ""
            })

    if not candidatos:
        return pd.DataFrame(columns=["patron", "frecuencia", "n_eventos", "ejemplo_producto", "decision"])

    df_cand = pd.DataFrame(candidatos)
    df_cand = df_cand.sort_values(by=["n_eventos", "frecuencia"], ascending=False).reset_index(drop=True)
    return df_cand


def consolidar_stopwords(
    ruta_revision: str = "data/lookup/stopwords_revision_humana.csv",
    ruta_oficial: str = "data/lookup/stopwords_marketing.csv"
) -> Tuple[int, str]:
    """
    Modo --consolidar:
    1. Lee stopwords_revision_humana.csv y filtra decision='aceptar'.
    2. Determina la versión del vocabulario siguiente (e.g. v1 -> v2).
    3. Anexa las nuevas entradas a stopwords_marketing.csv (fuente='propuesto', fecha=hoy).
    4. Actualiza stopwords_revision_humana.csv removiendo las filas consolidadas.
    5. Si existe AZURE_STOPWORDS_URL, sube el CSV oficial a Azure Blob.

    Retorna (num_filas_consolidadas, nueva_version).
    """
    if not os.path.exists(ruta_revision):
        print(f"No existe archivo de revision en '{ruta_revision}'.")
        return 0, ""

    df_rev = pd.read_csv(ruta_revision)
    if "decision" not in df_rev.columns:
        print("El archivo de revision no contiene columna 'decision'.")
        return 0, ""

    mask_aceptar = df_rev["decision"].fillna("").astype(str).str.strip().str.lower() == "aceptar"
    df_aceptados = df_rev[mask_aceptar].copy()

    if df_aceptados.empty:
        print("No se encontraron filas con decision='aceptar' en el archivo de revision.")
        return 0, ""

    # Cargar CSV oficial existente o crear estructura base
    if os.path.exists(ruta_oficial):
        df_oficial = pd.read_csv(ruta_oficial)
    else:
        df_oficial = pd.DataFrame(columns=["patron", "tipo", "fecha_agregado", "fuente", "version_vocab"])

    # Calcular versión incrementada
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
    patrones_existentes = set(df_oficial["patron"].dropna().tolist()) if "patron" in df_oficial.columns else set()

    for _, row in df_aceptados.iterrows():
        patron = str(row["patron"]).strip()
        if patron in patrones_existentes:
            continue
        nuevas_filas.append({
            "patron": patron,
            "tipo": "tour",  # Por defecto 'tour' al ser propuesto desde products
            "fecha_agregado": fecha_hoy,
            "fuente": "propuesto",
            "version_vocab": nueva_version
        })
        patrones_existentes.add(patron)

    if nuevas_filas:
        df_nuevas = pd.DataFrame(nuevas_filas)
        df_oficial = pd.concat([df_oficial, df_nuevas], ignore_index=True)
        os.makedirs(os.path.dirname(os.path.abspath(ruta_oficial)), exist_ok=True)
        df_oficial.to_csv(ruta_oficial, index=False)
        print(f"Consolidadas {len(nuevas_filas)} nuevas stopwords en '{ruta_oficial}' con version '{nueva_version}'.")

    # Actualizar archivo de revisión: remover filas consolidadas
    df_restante = df_rev[~mask_aceptar].copy()
    df_restante.to_csv(ruta_revision, index=False)
    print(f"Archivo de revision '{ruta_revision}' actualizado (restan {len(df_restante)} filas).")

    # Subida a Azure Blob si AZURE_STOPWORDS_URL está definida
    azure_blob_url = os.getenv("AZURE_STOPWORDS_URL")
    if azure_blob_url:
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
    parser = argparse.ArgumentParser(description="Minería y propuesta de stopwords de marketing.")
    parser.add_argument("--input", default="data/raw/localidades_eda.parquet", help="Ruta al archivo parquet con datos crudos.")
    parser.add_argument("--output", default="data/lookup/stopwords_revision_humana.csv", help="Ruta para el archivo de revisión humana.")
    parser.add_argument("--oficial", default="data/lookup/stopwords_marketing.csv", help="Ruta al CSV oficial de stopwords.")
    parser.add_argument("--modelo", default="data/processed/modelo_clustering_v2_5.joblib", help="Ruta al modelo para extraer TF-IDF.")
    parser.add_argument("--min-eventos", type=int, default=3, help="Frecuencia mínima en eventos distintos (Rejilla d).")
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
    print(f"Dataset cargado: {len(df_raw):,} registros. Ejecutando extracción de candidatos con 4 rejillas...")

    df_candidatos = extraer_candidatos_stopwords(
        df=df_raw,
        col_producto="product",
        col_evento="t_performance_id",
        min_eventos=args.min_eventos,
        ruta_modelo=args.modelo if os.path.exists(args.modelo) else None,
        ruta_stopwords_oficial=args.oficial
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_candidatos.to_csv(args.output, index=False)
    print(f"Propuesta completada. {len(df_candidatos):,} candidatos exportados a '{args.output}'.")
    if not df_candidatos.empty:
        print("\nTop 10 candidatos detectados:")
        print(df_candidatos.head(10)[["patron", "frecuencia", "n_eventos", "ejemplo_producto"]])


if __name__ == "__main__":
    main()
