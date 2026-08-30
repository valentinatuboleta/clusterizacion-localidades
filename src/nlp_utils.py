"""
Módulo de Procesamiento de Lenguaje Natural (NLP) para Nombres de Localidades.

Este módulo se encarga de:
1. Limpiar el ruido publicitario y de marketing (nombres de giras, patrocinios, promociones).
2. Extraer atributos estructurales de nivel (VIP, Palco, Platea, General, Balcón).
3. Extraer orientación espacial y ubicación física dentro del recinto (Occidental, Oriental, Norte, Sur, etc.).
4. Detectar restricciones de acceso (Familiar, Menores, Sin Alcohol, Movilidad Reducida).
5. Vectorizar semánticamente el texto limpio para enriquecer el espacio de clustering.
"""

import re
import unicodedata
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer

# Lista de términos publicitarios, nombres de gira, patrocinios y ruido comercial común en TuBoleta
STOPWORDS_MARKETING = [
    r"\bSEND[EÉ]\b",
    r"\bBOMBASTIK\b",
    r"\bAY MI PAP[AÁ]\b",
    r"\bSANKA\b",
    r"\bMODO LEYENDA\b",
    r"\bLLEG[OÓ] EL PODER\b",
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


def limpiar_ruido_marketing(texto: str) -> str:
    """
    Elimina sufijos publicitarios, marcas comerciales, nombres de gira y números de silletería/rango.
    Conserva únicamente los descriptores estructurales y espaciales de la localidad.
    """
    t = normalizar_texto(texto)
    if not t:
        return ""
    
    # 1. Remover stopwords y patrones de marketing conocidos
    for pattern in STOPWORDS_MARKETING:
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
    Extrae variables binarias que identifican la jerarquía de nivel, orientación y restricciones
    organizadas en 4 dimensiones ortogonales independientes (sin solapamiento léxico entre tags).
    """
    t = normalizar_texto(texto)
    
    return {
        # --- Dimensión 1: Jerarquía Comercial / Tipo de Asiento ---
        "tag_palco": int(bool(re.search(r"\b(PALCO|PALCOS|BOX|BOXES|SUITE|SUITES|MESA|MESAS)\b", t))),
        "tag_vip": int(bool(re.search(r"\b(VIP|PLATINUM|PLATINO|PREMIUM|GOLD|DIAMANTE|ORO|PLATA)\b", t))),
        "tag_platea": int(bool(re.search(r"\b(PLATEA|SILLAS|SILLERIA|PISTA|CANCHA)\b", t))),
        "tag_preferencial": int(bool(re.search(r"\b(PREFERENCIAL|PREFERENTE|CENTRAL|FRONTAL)\b", t))),
        "tag_general": int(bool(re.search(r"\b(GENERAL|TIQUETE|ENTRADA|STANDARD|NORMAL|ADMISION)\b", t))),
        
        # --- Dimensión 2: Nivel Vertical y Arquitectura del Venue ---
        "tag_balcon": int(bool(re.search(r"\b(BALCON|BALCONES|MEZZANINE|VOLADIZO)\b", t))),
        "tag_piso_alto": int(bool(re.search(r"\b(ALTA|ALTAS|PISO 2|PISO 3|PISO 4|PISO 5|SEGUNDO PISO|TERCER PISO|CUARTO PISO|POSTERIOR|ALTO)\b", t))),
        "tag_piso_bajo": int(bool(re.search(r"\b(BAJA|BAJAS|PISO 1|PRIMER PISO|PLANTA BAJA|DELANTERA|PRIMERA FILA|BAJO)\b", t))),
        
        # --- Dimensión 3: Orientación Espacial y Geografía en el Venue ---
        "tag_occidental": int(bool(re.search(r"\b(OCCIDENTAL|OCC|OESTE)\b", t))),
        "tag_oriental": int(bool(re.search(r"\b(ORIENTAL|ORI|ESTE)\b", t))),
        "tag_norte": int(bool(re.search(r"\b(NORTE|NTE)\b", t))),
        "tag_sur": int(bool(re.search(r"\b(SUR)\b", t))),
        "tag_lateral": int(bool(re.search(r"\b(LATERAL|LATERALES|COSTADO|ESQUINA)\b", t))),
        "tag_vista_parcial": int(bool(re.search(r"\b(VISTA PARCIAL|VISIBILIDAD PARCIAL|RESTRINGIDA|REDUCIDA|OBSTRUIDA|PILARES)\b", t))),
        
        # --- Dimensión 4: Restricciones de Acceso y Audiencia ---
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
