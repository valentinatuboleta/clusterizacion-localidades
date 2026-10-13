"""
Módulo de Procesamiento de Lenguaje Natural (NLP) para Nombres de Localidades.

Este módulo se encarga de:
1. Limpiar el ruido publicitario y de marketing (nombres de giras, patrocinios, promociones).
2. Extraer atributos estructurales de nivel (VIP, Palco, Platea, General, Balcón).
3. Extraer orientación espacial y ubicación física dentro del venue (Occidental, Oriental, Norte, Sur, etc.).
4. Detectar restricciones de acceso (Familiar, Menores, Sin Alcohol, Movilidad Reducida).
5. Vectorizar semánticamente el texto limpio para enriquecer el espacio de clustering.
"""

import os
import io
import re
import logging
import unicodedata
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Lista de términos publicitarios, nombres de gira, patrocinios y ruido comercial común en TuBoleta (red de seguridad embebida)
DEFAULT_STOPWORDS_MARKETING = [
    r"\bSENDE\b",
    r"\bBOMBASTIK\b",
    r"\bAY MI PAPA\b",
    r"\bSANKA\b",
    r"\bMODO LEYENDA\b",
    r"\bLLEGO EL PODER\b",
    r"\bCANTINERO\b",
    r"\bEL REENCUENTRO\b",
    r"\bTA MALO\b",
    r"\bSIGO INVICTO\b",
    r"\bENTRE GRANDES\b",
    r"\bNEGRA PULOY\b",
    r"\bPASEO DE LA AURORA\b",
    r"\bEXPERIENCIA\b",
    r"\bTOUR\b",
    r"\bFESTIVAL\b",
    r"\bCONCIERTO\b",
    r"\bPATROCINADO\b",
    r"\bPRESENTADO POR\b",
    r"\bETAPA\s*\d+\b",
    r"\bPREVENTA\b",
    r"\bEARLY BIRD\b",
    r"\bPROMO\b"
]


def _resolver_ruta_local(ruta: str) -> str:
    """Resuelve la ruta del archivo verificando existencia en CWD o relativa a la raíz del repositorio."""
    if not ruta:
        return ""
    if os.path.isabs(ruta) or os.path.exists(ruta):
        return ruta
    ruta_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ruta))
    if os.path.exists(ruta_repo):
        return ruta_repo
    return ruta


def cargar_stopwords(ruta_local: str = "data/lookup/stopwords_marketing.csv") -> List[str]:
    """
    Carga la lista de patrones de stopwords de marketing mediante una cadena de resolución:
    - Nivel 1: Descarga desde Azure Blob Storage si existe la variable de entorno AZURE_STOPWORDS_URL.
    - Nivel 2: CSV local en ruta_local.
    - Nivel 3: Fallback a DEFAULT_STOPWORDS_MARKETING (red de seguridad embebida).

    Cada caída loguea un warning y nunca lanza excepción.
    """
    # Nivel 1: Azure Blob Storage
    blob_url = os.getenv("AZURE_STOPWORDS_URL")
    if blob_url:
        try:
            from azure.storage.blob import BlobClient
            blob_client = BlobClient.from_blob_url(blob_url)
            data = blob_client.download_blob().readall()
            df_blob = pd.read_csv(io.BytesIO(data))
            if "patron" in df_blob.columns:
                patrones = df_blob["patron"].dropna().astype(str).tolist()
                if patrones:
                    return patrones
            logger.warning(
                "El CSV en AZURE_STOPWORDS_URL no contiene columna 'patron' válida. Cayendo a Nivel 2."
            )
        except Exception as exc:
            logger.warning(
                "Fallo al descargar stopwords desde Azure Blob (Nivel 1: %s): %s. Cayendo a Nivel 2.",
                blob_url, exc
            )

    # Nivel 2: CSV local
    ruta_resuelta = _resolver_ruta_local(ruta_local)
    if ruta_resuelta and os.path.exists(ruta_resuelta):
        try:
            df_local = pd.read_csv(ruta_resuelta)
            if "patron" in df_local.columns:
                patrones = df_local["patron"].dropna().astype(str).tolist()
                if patrones:
                    return patrones
            logger.warning(
                "El CSV local '%s' no contiene columna 'patron' válida. Cayendo a Nivel 3.",
                ruta_resuelta
            )
        except Exception as exc:
            logger.warning(
                "Fallo al leer stopwords locales en '%s' (Nivel 2): %s. Cayendo a Nivel 3.",
                ruta_resuelta, exc
            )
    else:
        logger.warning(
            "Archivo local de stopwords no encontrado en '%s' (Nivel 2). Cayendo a Nivel 3.",
            ruta_local
        )

    # Nivel 3: Default embebido
    logger.warning("Usando stopwords por defecto embebidas (Nivel 3: red de seguridad).")
    return DEFAULT_STOPWORDS_MARKETING.copy()


# Carga única al importar/inicializar el módulo
STOPWORDS_MARKETING = cargar_stopwords()


def normalizar_texto(texto: str) -> str:
    """
    Normaliza el texto convirtiéndolo a mayúsculas, estandarizando espacios y tildes.
    """
    if pd.isna(texto) or not str(texto).strip():
        return ""
    
    t = str(texto).upper().strip()
    # Remover tildes para uniformizar tokens
    t = "".join(
        c for c in unicodedata.normalize("NFD", t) 
        if unicodedata.category(c) != "Mn"
    )
    # Estandarizar espacios
    t = re.sub(r"\s+", " ", t)
    return t


def normalizar_venue(texto: str) -> str:
    """
    Normaliza el nombre de un venue eliminando tildes, caracteres especiales
    y colapsando espacios continuos para matching canonico y robusto.
    """
    if pd.isna(texto) or not isinstance(texto, str):
        return ""
    texto = texto.upper().strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto



def limpiar_ruido_marketing(texto: str, stopwords: Optional[List[str]] = None) -> str:
    """
    Elimina sufijos publicitarios, marcas comerciales, nombres de gira y números de silletería/rango.
    Conserva únicamente los descriptores estructurales y espaciales de la localidad.
    """
    t = normalizar_texto(texto)
    if not t:
        return ""
    
    patrones = stopwords if stopwords is not None else STOPWORDS_MARKETING
    
    # 1. Remover stopwords y patrones de marketing conocidos
    for pattern in patrones:
        # Remover patrón insensible a tildes
        norm_pattern = "".join(
            c for c in unicodedata.normalize("NFD", pattern) 
            if unicodedata.category(c) != "Mn"
        )
        t = re.sub(norm_pattern, "", t, flags=re.IGNORECASE)
    
    # 2. Remover rangos de numeración irrelevantes para la categoría (ej. "302 - 306 & 314 - 318", "201 AL 219")
    t = re.sub(r"\b\d{2,4}\s*(?:[-–&/]|AL|A|TO|Y)\s*\d{2,4}\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d{3,4}\b", "", t)  # Números específicos de 3 o 4 dígitos (asientos/filas internas)
    
    # 3. Remover caracteres especiales sobrantes y paréntesis vacíos
    t = re.sub(r"[#\-_/&,.:;+*]", " ", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    
    return t


def extraer_atributos_estructurales(texto: str) -> Dict[str, int]:
    """
    Extrae las 18 variables binarias estructurales, espaciales y de restricción
    organizadas en 4 dimensiones ortogonales independientes (sin solapamiento léxico entre tags).
    """
    t = normalizar_texto(texto)
    
    return {
        # --- Dimensión 1: Jerarquía Comercial / Tipo de Asiento (6 tags) ---
        "tag_palco": int(bool(re.search(r"\b(PALCO|PALCOS|BOX|BOXES|SUITE|SUITES|MESA|MESAS)\b", t))),
        "tag_mesa": int(bool(re.search(r"\b(MESAS?)\b", t))),
        "tag_vip": int(bool(re.search(r"\b(VIP|PLATINUM|PLATINO|PREMIUM|GOLD|DIAMANTE|ORO|PLATA)\b", t))),
        "tag_platea": int(bool(re.search(r"\b(PLATEA|SILLAS|SILLERIA|PISTA|CANCHA)\b", t))),
        "tag_preferencial": int(bool(re.search(r"\b(PREFERENCIAL|PREFERENTE|CENTRAL|FRONTAL)\b", t))),
        "tag_general": int(bool(re.search(r"\b(GENERAL|TIQUETE|ENTRADA|STANDARD|NORMAL|ADMISION)\b", t))),
        
        # --- Dimensión 2: Nivel Vertical y Arquitectura del Venue (3 tags) ---
        "tag_balcon": int(bool(re.search(r"\b(BALCON|BALCONES|MEZZANINE|VOLADIZO)\b", t))),
        "tag_piso_alto": int(bool(re.search(r"\b(ALTA|ALTAS|PISO 2|PISO 3|PISO 4|PISO 5|SEGUNDO PISO|TERCER PISO|CUARTO PISO|POSTERIOR|ALTO)\b", t))),
        "tag_piso_bajo": int(bool(re.search(r"\b(BAJA|BAJAS|PISO 1|PRIMER PISO|PLANTA BAJA|DELANTERA|PRIMERA FILA|BAJO)\b", t))),
        
        # --- Dimensión 3: Orientación Espacial y Geografía en el Venue (6 tags) ---
        "tag_occidental": int(bool(re.search(r"\b(OCCIDENTAL|OCC|OESTE)\b", t))),
        "tag_oriental": int(bool(re.search(r"\b(ORIENTAL|ORI|ESTE)\b", t))),
        "tag_norte": int(bool(re.search(r"\b(NORTE|NTE)\b", t))),
        "tag_sur": int(bool(re.search(r"\b(SUR)\b", t))),
        "tag_lateral": int(bool(re.search(r"\b(LATERAL|LATERALES|COSTADO|ESQUINA)\b", t))),
        "tag_vista_parcial": int(bool(re.search(r"\b(VISTA PARCIAL|VISIBILIDAD PARCIAL|RESTRINGIDA|REDUCIDA|OBSTRUIDA|PILARES)\b", t))),
        
        # --- Dimensión 4: Restricciones de Acceso y Audiencia (3 tags) ---
        "tag_familiar": int(bool(re.search(r"\b(FAMILIAR|FAMILIA)\b", t))),
        "tag_menores": int(bool(re.search(r"\b(MENORES|KIDS|NINOS|INFANTIL|LIBRE DE ALCOHOL|CERO ALCOHOL)\b", t))),
        "tag_movilidad_reducida": int(bool(re.search(r"\b(MOVILIDAD REDUCIDA|DISCAPACIDAD|PMR|SILLA DE RUEDAS|ACCESIBLE)\b", t)))
    }


def pipeline_procesamiento_nlp(df: pd.DataFrame, col_nombre: str = "logical_seat_category") -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de NLP sobre el DataFrame de localidades:
    1. Crea la columna 'texto_limpio' sin ruido publicitario.
    2. Genera y acopla las columnas de tags estructurales a partir del 'texto_limpio'.
    """
    df_res = df.copy()
    
    # 1. Limpieza de texto
    df_res["texto_limpio"] = df_res[col_nombre].apply(limpiar_ruido_marketing)
    
    # 2. Extracción de tags sobre el texto limpio
    tags_df = df_res["texto_limpio"].apply(extraer_atributos_estructurales).apply(pd.Series)
    
    for col in tags_df.columns:
        df_res[col] = tags_df[col]
        
    return df_res


def vectorizar_texto_limpio(
    textos: pd.Series, 
    max_features: int = 25, 
    vectorizer: TfidfVectorizer = None
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Genera una representación vectorial TF-IDF de n-gramas estructurados sobre el texto limpio.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            token_pattern=r"\b[A-Za-z0-9]+\b"
        )
        tfidf_matrix = vectorizer.fit_transform(textos.fillna("")).toarray()
    else:
        tfidf_matrix = vectorizer.transform(textos.fillna("")).toarray()
        
    return tfidf_matrix, vectorizer
