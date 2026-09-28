"""
Script de clasificación léxica determinística de venues (reglas estructuradas + curaduría experta).
Aplica doble pasada ortogonal léxica sobre la lista de venues únicos.
Separa consensos en site_type_lookup.csv (fuente='reglas_heuristicas') y discrepancias en site_type_revision_humana.csv.
"""

import os
import re
import datetime
import unicodedata
from typing import Optional, Tuple
import pandas as pd

CATEGORIAS_VALIDAS = [
    "ARENA",
    "CINEMATECA",
    "ESTADIO",
    "COLISEO",
    "MUSEO",
    "OTROS_RECINTOS",
    "PARQUE",
    "PARQUEADERO",
    "RESTAURANTE",
    "TEATRO",
    "desconocido"
]


from src.nlp_utils import normalizar_venue
normalizar_recinto = normalizar_venue

# Diccionario maestro de venues emblemáticos de Colombia con asignación certificada
DICCIONARIO_EMBLEMATICO = {
    # Teatros
    "TEATRO MAYOR JULIO MARIO SANTO DOMINGO": "TEATRO",
    "CENTRO NACIONAL DE LAS ARTES - SALA TEATRO COLON CLL10 #5-32": "TEATRO",
    "TEATRO COLON": "TEATRO",
    "TEATRO JORGE ELIECER GAITAN": "TEATRO",
    "TEATRO SANTANDER": "TEATRO",
    "TEATRO COLSUBSIDIO": "TEATRO",
    "TEATRO ASTOR PLAZA": "TEATRO",
    "TEATRO METROPOLITANO DE MEDELLIN": "TEATRO",
    "TEATRO PABLO TOBON URIBE": "TEATRO",
    "TEATRO MUNICIPAL ENRIQUE BUENAVENTURA": "TEATRO",
    "TEATRO JORGE ISAACS - CRA 3 #12-28 CALI": "TEATRO",
    "TEATRO ADOLFO MEJIA": "TEATRO",
    "TEATRO CAFAM - AV CAR 68 NO 90 - 88": "TEATRO",
    "TEATRO PETRA": "TEATRO",
    "TEATRO FUNDADORES": "TEATRO",
    "TEATRO EL ENSUEO - TV 70 D # 60 - 90 SUR, BOGOTA": "TEATRO",
    "TEATRO EL TESORO": "TEATRO",
    "ROYAL CENTER": "TEATRO",
    
    # Arenas
    "MOVISTAR ARENA": "ARENA",
    "ARENA CAAVERALEJO": "ARENA",
    "ARENA CAAVERALEJO - CALI (AJUSTE)": "ARENA",
    "NECTAR ARENA CENTRO DE EVENTOS": "ARENA",
    "INDIGO - MOVISTAR ARENA": "ARENA",
    "DAVIARENA": "ARENA",
    "ARENA LAS MEJORES PRODUCCIONES": "ARENA",

    # Coliseos
    "COLISEO MEDPLUS": "COLISEO",
    "COLISEO ELIAS CHEGWIN": "COLISEO",
    "COLISEO MAYOR JORGE ARANGO URIBE": "COLISEO",
    "PALACIO DE LOS DEPORTES": "COLISEO",
    "COLISEO BICENTENARIO - BUCARAMANGA": "COLISEO",
    "COLISEO MAYOR DE IBAGUE": "COLISEO",
    "COLISEO BERNARDO CARABALLO": "COLISEO",
    "COLISEO DEL COLEGIO DE LA PRESENTACION": "COLISEO",
    "COLISEO LAS BETHLEMITAS - CARRERA 13 #24-12": "COLISEO",
    "COLISEO ARENA DE SAL": "COLISEO",

    # Estadios
    "ESTADIO EL CAMPIN": "ESTADIO",
    "ESTADIO ATANASIO GIRARDOT": "ESTADIO",
    "ESTADIO PASCUAL GUERRERO": "ESTADIO",
    "ESTADIO METROPOLITANO": "ESTADIO",
    "ESTADIO PALOGRANDE - MANIZALES": "ESTADIO",
    "ESTADIO MANUEL MURILLO TORO": "ESTADIO",
    "ESTADIO METROPOLITANO DE TECHO": "ESTADIO",
    "ESTADIO ROMELIO MARTINEZ": "ESTADIO",
    "ESTADIO BELLO HORIZONTE - REY PELE (VILLAVICENCIO)": "ESTADIO",
    "ESTADIO EDGAR RENTERIA": "ESTADIO",
    "ESTADIO GENERAL SANTANDER": "ESTADIO",
    "CUCUTA ESTADIO GENERAL SANTANDER": "ESTADIO",
    "ESTADIO GUILLERMO PLAZAS ALCID": "ESTADIO",
    "ESTADIO LA INDEPENDENCIA": "ESTADIO",
    "ESTADIO JAIME MORON": "ESTADIO",
    "ESTADIO JOSE AMERICO MONTANINI": "ESTADIO",
    "ESTADIO HERNAN RAMIREZ VILLEGAS": "ESTADIO",
    "ESTADIO SIERRA NEVADA": "ESTADIO",
    "ESTADIO JARAGUAY": "ESTADIO",
    "ESTADIO FRANCISCO RIVERA ESCOBAR - PALMIRA": "ESTADIO",
    "COPA AMERICA - NRG STADIUM, HOUSTON, TEXAS": "ESTADIO",

    # Cinematecas
    "SALA 3 CINEMATECA": "CINEMATECA",
    "SALA CAPITAL CINEMATECA": "CINEMATECA",
    "SALA 2 CINEMATECA": "CINEMATECA",
    "CINE COLOMBO, MEDELLIN": "CINEMATECA",
    "CINEMATECA DE BOGOTA - CENTRO GRAL": "CINEMATECA",

    # Museos
    "MALOKA": "MUSEO",
    "PLANETARIO DE BOGOTA": "MUSEO",
    "YAWA, CENTRO DE CIENCIA, ARTE Y TECNOLOGIA - CALI": "MUSEO",
    "MUSEO LA TERTULIA": "MUSEO",
    "SALA DE CONCIERTOS DE LA BIBLIOTECA LUIS ANGEL ARANGO": "MUSEO",
    "BIBLIOTECA NACIONAL DE COLOMBIA": "MUSEO",
    "PARQUE MUSEO EL CHICO": "MUSEO",
    "TEATRO MUSEO DEL ARTE - PEREIRA": "MUSEO",
    "CIUDAD DEL RIO - DETRAS DEL MUSEO DE ARTE MODERNO": "MUSEO",

    # Otros Recintos (Auditorios, Carpas, Centros de Convenciones)
    "CARPA DELIRIO": "OTROS_RECINTOS",
    "COMPLEJO CULTURAL DELIRIO": "OTROS_RECINTOS",
    "CORFERIAS": "OTROS_RECINTOS",
    "CARPA AMERICAS CORFERIAS": "OTROS_RECINTOS",
    "CHAMORRO CITY HALL - AUTO NTE #153-81": "OTROS_RECINTOS",
    "CENTRO DE CONVENCIONES CARTAGENA": "OTROS_RECINTOS",
    "CENTRO DE EVENTOS CENFER": "OTROS_RECINTOS",
    "PUERTA DE ORO BARRANQUILLA": "OTROS_RECINTOS",
    "PUERTA DE ORO BARRANQUILLA - LA EXPLANADA": "OTROS_RECINTOS",
    "EXPLANADA- PUERTA DE ORO": "OTROS_RECINTOS",
    "EXPOFUTURO - PEREIRA": "OTROS_RECINTOS",
    "CENTRO DE CONVENCIONES G12": "OTROS_RECINTOS",
    "PLAZA MAYOR": "OTROS_RECINTOS",
    "CENTRO DE EVENTOS AUTOPISTA NORTE": "OTROS_RECINTOS",
    "PABELLON DE CRISTAL - GRAN MALECON": "OTROS_RECINTOS",

    # Salas CNA y Teatro Satélites
    "CENTRO NACIONAL DE LAS ARTES - SALA DELIA ZAPATA": "TEATRO",
    "CENTRO NACIONAL DE LAS ARTES - SALA FANNY MIKEY": "TEATRO",
    "CENTRO NACIONAL DE LAS ARTES - SALA TERESITA GOMEZ": "TEATRO",
    "CENTRO NACIONAL DE LAS ARTES - SALA FOYER CLL10 NO 5-32": "TEATRO",
    "CENTRO NACIONAL DE LAS ARTES - SALA TEATRO": "TEATRO",
    "SALA GAITAN": "TEATRO",
    "SALON ESPEJOS TEATRO JORGE ELIECER GAITAN": "TEATRO",
    "TEATRO ESTUDIO - JULIO MARIO SANTO DOMINGO": "TEATRO",
    "CENTRO CULTURAL DEL GIMNASIO MODERNO": "TEATRO",
    "CENTRO CULTURAL GIMNASIO MODERNO": "TEATRO",
    "COLECTIVO TEATRAL INFINITO - CALI": "TEATRO",
    "MBS THEATER": "TEATRO",

    # Escenarios Deportivos Masivos Adicionales
    "DIAMANTE DE BEISBOL - MEDELLIN": "ESTADIO",
    "DIAMANTE DE SOFTBOL": "ESTADIO",
    "ESTADIO DE BEISBOL LA ESPERANZA": "ESTADIO",
    "ESTADIO DITAIRES": "ESTADIO",
    "ESTADIO HERMIDES PADILLA": "ESTADIO",
    "ESTADIO MUNICIPAL DE VILLETA": "ESTADIO",
    "ESTADIO DE FUTBOL - INMACULADA CONCEPCION": "ESTADIO",

    # Restaurantes, Gastrobares, Bares y Comedy Clubs
    "BOOM STAND UP BAR - BOGOTA": "RESTAURANTE",
    "BOOM STAND UP BAR - CL 26 #43G-30 BARRIO COLOMBIA": "RESTAURANTE",
    "WOW RESTAURANTE BAR": "RESTAURANTE",
    "LOURDES MUSIC HALL  - BOGOTA": "RESTAURANTE",
    "CANTINA LA 70 - CRA 70 #44B - 76 (MEDELLIN)": "RESTAURANTE",
    "SAFARI DISCO CLUB, AV SANTANDER #63 - 122, MANIZALES": "RESTAURANTE",
    "440 MUSIC HALL": "RESTAURANTE",
    "MONASTERY CLUB": "RESTAURANTE",
    "FROGG CLUB": "RESTAURANTE",
    "DISCO MOVISTAR ARENA": "RESTAURANTE",
    "RANCHO MX": "RESTAURANTE",
    "MOYS RESTAURANTE BAR": "RESTAURANTE",
    "CINARUCO BAR - CRA 14 NO 24A -15 - YOPAL": "RESTAURANTE",
    "KABALA BAR - MANIZALES": "RESTAURANTE",
    "RESTAURANTE LA ZIMA": "RESTAURANTE",
    "RESTAURANTE EL PORTICO - EL PORTICO KM 19 AUTOPISTA NORTE": "RESTAURANTE",
    "RESTAURANTE BAR AMAZONICA VILLAVICENCIO": "RESTAURANTE",
    "ITO RESTOBAR": "RESTAURANTE",
    "JUANKA PUNTA DE ANCA - SOGAMOSO": "RESTAURANTE",
    "KIMERA FOOD AND DRINKS": "RESTAURANTE",
    "LOS CAPACHOS - KM 4 VIA ACACIAS, 472 (VILLAVICENCIO)": "RESTAURANTE",
    "MEZCAL MEXICAN CANTINA - BOGOTA": "RESTAURANTE",
    "MR BEEF - FUSAGASUGA": "RESTAURANTE",
    "EL TEMPLO DEL ROCK - CRA 44 #74-05 B/QUILLA": "RESTAURANTE",

    # Parques / Aire Libre
    "PARQUE NORTE": "PARQUE",
    "PARQUE METROPOLITANO SIMON BOLIVAR": "PARQUE",
    "GRAN MALECON BARRANQUILLA": "PARQUE",
    "PARQUE DE LA LEYENDA VALLENATA": "PARQUE",
    "AUTODROMO DE TOCANCIPA": "PARQUE",
    "PARQUE DE LA 93": "PARQUE",
    "PARQUE DE EVENTOS - LA INDEPENDENCIA": "PARQUE",
    "JARDIN BOTANICO - ORQUIDEORAMA": "PARQUE",
    "JARDIN BOTANICO ORQUIDEORAMA - ANOTR": "PARQUE",
    "SALITRE MAGICO": "PARQUE",
    "MUNDO AVENTURA": "PARQUE",
    "AEROPARQUE JUAN PABLO SEGUNDO": "PARQUE",
    "CARRERA 50 BARRANQUILLA": "PARQUE",
    "PLAZA DE BOLIVAR": "PARQUE",
    "PLAZA DE LA PAZ": "PARQUE",
    "LA MEDIA TORTA": "PARQUE",

    # Instituciones Académicas y Auditorios
    "UNIVERSIDAD DE LA SABANA": "OTROS_RECINTOS",
    "COLEGIO LA ENSEANZA - CL 9 SUR #37-345, MEDELLIN": "OTROS_RECINTOS",
    "AUDITORIO UNIVERSIDAD NACIONAL MANIZALES - CRA 27 #62-56": "OTROS_RECINTOS",
    "UNIVERSIDAD EAN - CARRERA 11 # 78-47": "OTROS_RECINTOS",
    "UNIVERSIDAD DE IBAGUE": "OTROS_RECINTOS",
    "UNIVERSIDAD INDUSTRIAL DE SANTANDER": "OTROS_RECINTOS",

    # Parqueaderos emblemáticos
    "PARQUEADERO CORFERIAS": "PARQUEADERO",
    "PARQUEADERO MOVISTAR ARENA": "PARQUEADERO",
    "PARQUEADERO CENTRAL": "PARQUEADERO",
    "PARKING NORTE": "PARQUEADERO",
    "ESTACIONAMIENTO EL CAMPIN": "PARQUEADERO",

    # Otros / Servicios complementarios
    "TREN TURISTICO": "OTROS_RECINTOS",
    "TRANSPORTE": "OTROS_RECINTOS",
    "TRANSPORTE LEGACY TOUR": "OTROS_RECINTOS",
    "FINAL COPA": "OTROS_RECINTOS",
    "BOGOTA (DIRECCION EXACTA SE COMPARTE TRAS INSCRIPCION)": "OTROS_RECINTOS",
    "LABORARTORIO 1 Y 2": "OTROS_RECINTOS",
    "ESTACION METRO LA ESTRELLA": "OTROS_RECINTOS",
    "CAFE INTERNET - SAN FELIPE - CALLE 76 # 20B - 65": "RESTAURANTE",
    "CAFE INTERNET - BOGOTA": "RESTAURANTE",
    "BIBLOS CAR WASH": "OTROS_RECINTOS",
    "CABARET ROSA": "OTROS_RECINTOS",
    "AEROPUERTO INTERNACIONAL JOSE MARIA CORDOVA": "OTROS_RECINTOS",
}

# Diccionario pre-normalizado para busquedas exactas y delimitadas
DICCIONARIO_NORMALIZADO = {normalizar_recinto(k): v for k, v in DICCIONARIO_EMBLEMATICO.items()}


def _buscar_en_diccionario_emblematico(rec_norm: str) -> Optional[Tuple[str, float]]:
    """
    Busqueda segura en diccionario maestro:
    1. Coincidencia exacta total.
    2. Coincidencia por subfrase completa del venue con limites de palabra.
    NUNCA evalua rec_norm in k_norm para evitar que tokens genericos (ej: 'SALA 2')
    activen erradamente venues compuestos (ej: 'SALA 2 CINEMATECA').
    """
    if rec_norm in DICCIONARIO_NORMALIZADO:
        return DICCIONARIO_NORMALIZADO[rec_norm], 0.98

    for k_norm, v in DICCIONARIO_NORMALIZADO.items():
        tokens_k = k_norm.split()
        if len(tokens_k) >= 2 and len(k_norm) >= 8:
            if re.search(r"\b" + re.escape(k_norm) + r"\b", rec_norm):
                return v, 0.98
    return None


def pasaje_a_clasificar(recinto: str, aforo_max: int) -> tuple[str, float]:
    """
    Pasada A: Clasificador Léxico Primario (Toponímico + Arquitectónico).
    Usa límites de palabra (\b) para evitar colisiones (ej. BAR en BARRANQUILLA).
    """
    rec_norm = normalizar_recinto(recinto)

    # 1. Chequeo en diccionario maestro validado
    match_dicc = _buscar_en_diccionario_emblematico(rec_norm)
    if match_dicc:
        return match_dicc

    # 2. Casos especiales prioritarios: Parqueaderos, Car Wash, Vias publicas
    if re.search(r"\b(PARQUEADERO|PARKING|ESTACIONAMIENTO)\b", rec_norm):
        return "PARQUEADERO", 0.95

    if re.search(r"\b(CAR\s*WASH|LAVADERO)\b", rec_norm):
        return "OTROS_RECINTOS", 0.90

    if re.search(r"\b(VIA\s*40|CARRERA\s*50)\b", rec_norm):
        return "PARQUE", 0.90

    # 3. Reglas estructurales léxicas primarias con límites de palabra
    if re.search(r"\b(ESTADIO|CAMPIN|ATANASIO|PASCUAL\s*GUERRERO|PALMASECA|PALOGRANDE|MURILLO\s*TORO|GIRARDOT|STADIUM)\b", rec_norm):
        return "ESTADIO", 0.95

    if re.search(r"\b(MOVISTAR\s*ARENA|ARENA\s+CA[NÑ]AVERALEJO|ARENA\s+BOGOTA|DAVIARENA)\b", rec_norm):
        return "ARENA", 0.95

    if re.search(r"\b(COLISEO|PALACIO\s+DE\s+LOS\s+DEPORTES)\b", rec_norm):
        return "COLISEO", 0.95

    if re.search(r"\b(TEATRO|TEATRINO|SALA\s+TEATRO|SALA\s+TEATRAL)\b", rec_norm):
        return "TEATRO", 0.95

    if re.search(r"\b(AUDITORIO|AULA\s+MAXIMA)\b", rec_norm):
        return "OTROS_RECINTOS", 0.92

    if re.search(r"\b(CINEMATECA|CINE\s+COLOMBO)\b", rec_norm):
        return "CINEMATECA", 0.94

    if re.search(r"\b(PLANETARIO|MALOKA|MUSEO|BIBLIOTECA)\b", rec_norm):
        return "MUSEO", 0.94

    if re.search(r"\b(CARPA|CORFERIAS|CHAMORRO|CENTRO\s+DE\s+EVENTOS|PABELLON|CONVENCIONES|EXPOFUTURO|CENFER|PUERTA\s+DE\s+ORO|CITY\s+HALL)\b", rec_norm):
        return "OTROS_RECINTOS", 0.93

    if re.search(r"\b(BAR|CLUB|RESTAURANTE|DISCOTECA|PUB|GASTROBAR|CANTA\s*BAR|FONDA|STAND\s*UP|RESTOBAR)\b", rec_norm):
        return "RESTAURANTE", 0.91

    if re.search(r"\b(PARQUE|MALECON|BOTANICO|AUTODROMO|PLAZA\s+DE\s+TOROS|CANCHA|DIAMANTE\s+DE\s+BEISBOL|POLIDEPORTIVO)\b", rec_norm):
        return "PARQUE", 0.90

    if re.search(r"\b(HOTEL)\b", rec_norm):
        return "OTROS_RECINTOS", 0.80

    if re.search(r"\b(TREN|TRANSPORTE|AEROPUERTO|FINAL\s+COPA|LABORATORIO)\b", rec_norm):
        return "OTROS_RECINTOS", 0.90

    # Salas genéricas sin cualificador de cinemateca o teatro van a OTROS_RECINTOS
    if re.search(r"\bSALA\b", rec_norm):
        return "OTROS_RECINTOS", 0.60

    # Fallbacks de baja confianza por aforo
    if aforo_max >= 15000:
        return "ESTADIO", 0.75

    if aforo_max >= 4000:
        return "OTROS_RECINTOS", 0.70

    if aforo_max < 300:
        return ("RESTAURANTE" if re.search(r"\b(CAFE|CASA)\b", rec_norm) else "OTROS_RECINTOS"), 0.72

    return "OTROS_RECINTOS", 0.60


def pasaje_b_clasificar(recinto: str, aforo_max: int) -> tuple[str, float]:
    """
    Pasada B: Clasificador Secundario Ortogonal (Basado en Funcionalidad y Aforo).
    """
    rec_norm = normalizar_recinto(recinto)

    # 1. Coincidencia segura en diccionario maestro
    match_dicc = _buscar_en_diccionario_emblematico(rec_norm)
    if match_dicc:
        return match_dicc[0], 0.99

    # 2. Casos prioritarios: Parqueaderos, Car Wash, Vias públicas
    if re.search(r"\b(PARQUEADERO|PARKING|ESTACIONAMIENTO)\b", rec_norm):
        return "PARQUEADERO", 0.95

    if re.search(r"\b(CAR\s*WASH|LAVADERO)\b", rec_norm):
        return "OTROS_RECINTOS", 0.90

    if re.search(r"\b(VIA\s*40|CARRERA\s*50)\b", rec_norm):
        return "PARQUE", 0.90

    # 3. Análisis por patrones de texto alternativos con límites estrictos de palabra
    if re.search(r"\b(ESTADIO|STADIUM|BEISBOL|DIAMANTE)\b", rec_norm):
        return "ESTADIO", 0.95

    if re.search(r"\b(MOVISTAR\s*ARENA|ARENA\s+CA[NÑ]AVERALEJO|DAVIARENA)\b", rec_norm):
        return "ARENA", 0.95

    if re.search(r"\b(COLISEO|POLIDEPORTIVO)\b", rec_norm):
        return ("PARQUE" if "POLIDEPORTIVO" in rec_norm else "COLISEO"), 0.90

    if re.search(r"\b(TEATRO|TEATRINO|SALA\s+TEATRAL)\b", rec_norm):
        return "TEATRO", 0.95

    if re.search(r"\b(AUDITORIO|AULA)\b", rec_norm):
        return "OTROS_RECINTOS", 0.94

    if re.search(r"\b(CINEMATECA|CINE)\b", rec_norm):
        return "CINEMATECA", 0.93

    if re.search(r"\b(MUSEO|PLANETARIO|BIBLIOTECA)\b", rec_norm):
        return "MUSEO", 0.93

    if re.search(r"\b(CARPA|EXPO|FERIA|CONVENCION|PABELLON|CENTRO\s+DE\s+EVENTOS|CITY\s+HALL)\b", rec_norm):
        return "OTROS_RECINTOS", 0.92

    if re.search(r"\b(BAR|CLUB|DISCO|LOUNGE|RESTAURANTE|PUB|GASTRO|TASCA|BARRIL|BEER|RESTOBAR)\b", rec_norm):
        return "RESTAURANTE", 0.92

    if re.search(r"\b(PARQUE|PLAZA|JARDIN|BOULEVARD|MALECON|PLAYA|BEACH|AVENIDA|CARRERA|AUTOPISTA|CALLE)\b", rec_norm):
        return "PARQUE", 0.89

    if re.search(r"\b(HOTEL|RESORT)\b", rec_norm):
        return "OTROS_RECINTOS", 0.78

    if re.search(r"\b(TREN|BUS|TRANSPORTE|AEROPUERTO|VIAJE)\b", rec_norm):
        return "OTROS_RECINTOS", 0.95

    if re.search(r"\bSALA\b", rec_norm):
        if re.search(r"\b(CINEMATECA|CINE)\b", rec_norm):
            return "CINEMATECA", 0.88
        if re.search(r"\b(TEATRO|TEATRAL)\b", rec_norm):
            return "TEATRO", 0.85
        return "OTROS_RECINTOS", 0.60

    if re.search(r"\b(CAPILLA|IGLESIA|CATEDRAL)\b", rec_norm):
        return "OTROS_RECINTOS", 0.85

    if re.search(r"\b(COLEGIO|UNIVERSIDAD|CAMPUS)\b", rec_norm):
        return "OTROS_RECINTOS", 0.83

    # Fallback contextual por aforo
    if aforo_max >= 20000:
        return "ESTADIO", 0.72
    elif aforo_max >= 5000:
        return "OTROS_RECINTOS", 0.68
    elif aforo_max <= 200:
        return "OTROS_RECINTOS", 0.65

    return "OTROS_RECINTOS", 0.55



def consolidar_revision_humana(ruta_lookup: str = "data/lookup/site_type_lookup.csv",
                               ruta_revision: str = "data/lookup/site_type_revision_humana.csv",
                               ruta_unicos: str = "data/lookup/recintos_unicos.csv"):
    """
    Consolida los registros revisados manualmente en la tabla maestra definitiva (site_type_lookup.csv).
    Valida categorias, asigna confianza=1.0 y fuente='revision_humana'.
    """
    print("\n=== CONSOLIDACIÓN DE REVISIÓN HUMANA EN TABLA MAESTRA ===")
    if not os.path.exists(ruta_revision):
        raise FileNotFoundError(f"No se encontro el archivo de revision humana: {ruta_revision}")

    df_rev = pd.read_csv(ruta_revision)
    print(f"Total registros en archivo de revision humana: {len(df_rev)}")

    # Validar categorias
    invalidos = df_rev[~df_rev["type_site_sugerido"].isin(CATEGORIAS_VALIDAS)]
    if len(invalidos) > 0:
        print("\n [ERROR] Existen registros con categorias no validas:")
        print(invalidos[["site", "type_site_sugerido"]])
        raise ValueError("Corrige las categorias invalidas antes de consolidar.")

    fecha_hoy = datetime.date.today().isoformat()
    df_nuevos_humanos = pd.DataFrame({
        "site": df_rev["site"].str.strip(),
        "type_site": df_rev["type_site_sugerido"].str.strip(),
        "confianza": 1.0,
        "aforo_max": df_rev["aforo_max"].fillna(0).astype(int),
        "funciones": df_rev["funciones"].fillna(0).astype(int),
        "fecha_clasificacion": fecha_hoy,
        "fuente": "revision_humana",
        "modelo_llm": None,
        "taxonomia_version": "v2",
        "fecha_reclasificacion": fecha_hoy
    })

    # Cargar o crear site_type_lookup.csv
    if os.path.exists(ruta_lookup):
        df_existente = pd.read_csv(ruta_lookup)
        if "modelo_llm" not in df_existente.columns:
            df_existente["modelo_llm"] = None
        # Concatenar y dar prioridad a la revision humana sobre clasificaciones previas
        df_consolidado = pd.concat([df_existente, df_nuevos_humanos], ignore_index=True)
        df_consolidado = df_consolidado.drop_duplicates(subset=["site"], keep="last")
    else:
        df_consolidado = df_nuevos_humanos

    df_consolidado.to_csv(ruta_lookup, index=False)
    print(f"\n [EXITO] Tabla maestra '{ruta_lookup}' consolidada con exito.")
    print(f"Total recintos en tabla maestra: {len(df_consolidado):,}")
    print(f"  - Por fuente: {dict(df_consolidado['fuente'].value_counts())}")

    if os.path.exists(ruta_unicos):
        df_unicos = pd.read_csv(ruta_unicos)
        cobertura = (len(df_consolidado) / len(df_unicos)) * 100
        print(f"  - Cobertura sobre universo de recintos únicos: {cobertura:.1f}% ({len(df_consolidado)}/{len(df_unicos)})")

    print("\n=== DISTRIBUCIÓN FINAL EN TABLA DE VERDAD (site_type_lookup.csv) ===")
    print(df_consolidado["type_site"].value_counts())
    return df_consolidado


def main():
    import sys
    if "--aplicar-revision" in sys.argv:
        consolidar_revision_humana()
        return

    usar_llm = "--llm" in sys.argv or "--use-llm" in sys.argv
    clasificador_llm = None
    if usar_llm:
        from src.llm_classifier import GeminiVenueClassifier
        candidato_llm = GeminiVenueClassifier()
        if candidato_llm.esta_disponible():
            clasificador_llm = candidato_llm
            print(f"=== MÓDULO LLM ACTIVADO (Modelo: {clasificador_llm.model_name}, temp={clasificador_llm.temperature}) ===")
        else:
            print(" [ALERTA] LLM no disponible: clasificando con reglas (sin API key o SDK google-genai)")
            clasificador_llm = None
    else:
        print("=== PASO 3 & 4: CLASIFICACIÓN LÉXICA DETERMINÍSTICA DE VENUES ===")

    
    ruta_unicos = "data/lookup/recintos_unicos.csv"
    if not os.path.exists(ruta_unicos):
        raise FileNotFoundError(f"No existe {ruta_unicos}. Ejecuta Paso 1 primero.")

    df_unicos = pd.read_csv(ruta_unicos)
    total_recintos = len(df_unicos)
    print(f"Total de recintos cargados para clasificación: {total_recintos}")

    ruta_lookup = "data/lookup/site_type_lookup.csv"
    ruta_revision = "data/lookup/site_type_revision_humana.csv"

    force_reprocess = "--force" in sys.argv

    # Revisar si ya existen registros en lookup
    ya_clasificados = set()
    if os.path.exists(ruta_lookup) and not force_reprocess:
        df_previo = pd.read_csv(ruta_lookup)
        if "site" in df_previo.columns:
            ya_clasificados = set(df_previo["site"].dropna())
            print(f"Recintos ya clasificados en {ruta_lookup}: {len(ya_clasificados)}")
    elif force_reprocess:
        print(f" Reprocesando todos los recintos desde cero (--force activado)...")

    lista_consenso = []
    lista_revision = []
    fecha_hoy = datetime.date.today().isoformat()

    for _, row in df_unicos.iterrows():
        recinto = str(row["recinto"]).strip()
        aforo = int(row["aforo_max"]) if pd.notnull(row["aforo_max"]) else 0
        funcs = int(row["funciones"]) if pd.notnull(row["funciones"]) else 0

        if recinto in ya_clasificados:
            continue

        rec_norm = normalizar_recinto(recinto)
        dicc_match = _buscar_en_diccionario_emblematico(rec_norm)

        # Precedencia 1: DICCIONARIO_EMBLEMATICO con verificación LLM
        if dicc_match:
            tipo_dicc, conf_dicc = dicc_match
            if clasificador_llm:
                llm_res = clasificador_llm.clasificar_venue(recinto, aforo)
                if llm_res and llm_res.get("type_site"):
                    tipo_llm = llm_res["type_site"]
                    if tipo_llm != tipo_dicc:
                        lista_revision.append({
                            "site": recinto,
                            "propuesta_a": tipo_dicc,
                            "propuesta_b": tipo_llm,
                            "confianza_media": round((conf_dicc + llm_res.get("confianza", 0.85)) / 2.0, 3),
                            "aforo_max": aforo,
                            "funciones": funcs,
                            "motivo_revision": f"Discrepancia Diccionario ({tipo_dicc}) vs LLM ({tipo_llm})",
                            "type_site_sugerido": tipo_dicc
                        })
                        continue

            lista_consenso.append({
                "site": recinto,
                "type_site": tipo_dicc,
                "confianza": conf_dicc,
                "aforo_max": aforo,
                "funciones": funcs,
                "fecha_clasificacion": fecha_hoy,
                "fuente": "revision_humana",
                "modelo_llm": clasificador_llm.model_name if clasificador_llm else None,
                "taxonomia_version": "v2",
                "fecha_reclasificacion": fecha_hoy
            })
            continue

        # Precedencia 2: Agente LLM para recintos fuera de curaduría
        if clasificador_llm:
            llm_res = clasificador_llm.clasificar_venue(recinto, aforo)
            if llm_res and llm_res.get("type_site") and llm_res.get("confianza", 0) >= 0.80:
                lista_consenso.append({
                    "site": recinto,
                    "type_site": llm_res["type_site"],
                    "confianza": llm_res["confianza"],
                    "aforo_max": aforo,
                    "funciones": funcs,
                    "fecha_clasificacion": fecha_hoy,
                    "fuente": "llm",
                    "modelo_llm": clasificador_llm.model_name,
                    "taxonomia_version": "v2",
                    "fecha_reclasificacion": fecha_hoy
                })
                continue

        # Precedencia 3: Fallback a doble pasada léxica determinística
        tipo_a, conf_a = pasaje_a_clasificar(recinto, aforo)
        tipo_b, conf_b = pasaje_b_clasificar(recinto, aforo)

        conf_media = round((conf_a + conf_b) / 2.0, 3)
        coinciden = (tipo_a == tipo_b)
        es_alta_confianza = (conf_media >= 0.85)

        if coinciden and es_alta_confianza:
            lista_consenso.append({
                "site": recinto,
                "type_site": tipo_a,
                "confianza": conf_media,
                "aforo_max": aforo,
                "funciones": funcs,
                "fecha_clasificacion": fecha_hoy,
                "fuente": "reglas_heuristicas",
                "modelo_llm": None,
                "taxonomia_version": "v2",
                "fecha_reclasificacion": fecha_hoy
            })
        else:
            motivo = "Desacuerdo entre pasadas" if not coinciden else "Baja confianza (< 0.85)"
            lista_revision.append({
                "site": recinto,
                "propuesta_a": tipo_a,
                "propuesta_b": tipo_b,
                "confianza_media": conf_media,
                "aforo_max": aforo,
                "funciones": funcs,
                "motivo_revision": motivo,
                "type_site_sugerido": tipo_a if conf_a >= conf_b else tipo_b
            })

    # Guardar / Actualizar site_type_lookup.csv
    df_nuevos_consenso = pd.DataFrame(lista_consenso)
    if os.path.exists(ruta_lookup) and not force_reprocess:
        df_existente = pd.read_csv(ruta_lookup)
        if "modelo_llm" not in df_existente.columns:
            df_existente["modelo_llm"] = None
        df_final_lookup = pd.concat([df_existente, df_nuevos_consenso], ignore_index=True).drop_duplicates(subset=["site"])
    else:
        df_final_lookup = df_nuevos_consenso

    df_final_lookup.to_csv(ruta_lookup, index=False)
    print(f"\n [OK] Guardados {len(df_final_lookup):,} recintos en '{ruta_lookup}'")

    # Guardar site_type_revision_humana.csv
    df_revision = pd.DataFrame(lista_revision)
    df_revision.to_csv(ruta_revision, index=False)
    print(f" [ALERTA] Guardados {len(df_revision):,} recintos en '{ruta_revision}' para revisión humana")

    # Resumen de distribución
    print("\n=== DISTRIBUCIÓN EN TABLA DE VERDAD (site_type_lookup.csv) ===")
    print(df_final_lookup["type_site"].value_counts())

    if len(df_revision) > 0:
        print(f"\n=== MUESTRA DE CASOS PARA REVISIÓN HUMANA ({len(df_revision)} en total) ===")
        print(df_revision[["site", "propuesta_a", "propuesta_b", "confianza_media", "motivo_revision"]].head(10))


if __name__ == "__main__":
    main()

