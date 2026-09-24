"""
Script de clasificación léxica determinística de venues (reglas estructuradas + curaduría experta).
Aplica doble pasada ortogonal léxica sobre la lista de venues únicos.
Separa consensos en site_type_lookup.csv (fuente='reglas_heuristicas') y discrepancias en site_type_revision_humana.csv.
"""

import os
import re
import datetime
import unicodedata
import pandas as pd

CATEGORIAS_VALIDAS = [
    "teatro",
    "estadio_abierto",
    "arena_cubierta",
    "centro_eventos_carpa",
    "cine_sala_cultural",
    "auditorio",
    "bar_club",
    "parque_aire_libre",
    "otro"
]


def normalizar_recinto(texto: str) -> str:
    """
    Normalización estricta: mayúsculas, sin tildes (Unicode NFD) y espacios colapsados.
    """
    if not texto or not str(texto).strip():
        return ""
    t = str(texto).upper().strip()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = re.sub(r"\s+", " ", t)
    return t

# Diccionario maestro de recintos emblemáticos de Colombia con asignación certificada
DICCIONARIO_EMBLEMATICO = {
    # Teatros
    "TEATRO MAYOR JULIO MARIO SANTO DOMINGO": "teatro",
    "CENTRO NACIONAL DE LAS ARTES - SALA TEATRO COLON CLL10 #5-32": "teatro",
    "TEATRO COLON": "teatro",
    "TEATRO JORGE ELIECER GAITAN": "teatro",
    "TEATRO SANTANDER": "teatro",
    "TEATRO COLSUBSIDIO": "teatro",
    "TEATRO ASTOR PLAZA": "teatro",
    "TEATRO METROPOLITANO DE MEDELLIN": "teatro",
    "TEATRO PABLO TOBON URIBE": "teatro",
    "TEATRO MUNICIPAL ENRIQUE BUENAVENTURA": "teatro",
    "TEATRO JORGE ISAACS - CRA 3 #12-28 CALI": "teatro",
    "TEATRO ADOLFO MEJIA": "teatro",
    "TEATRO CAFAM - AV CAR 68 NO 90 - 88": "teatro",
    "TEATRO PETRA": "teatro",
    "TEATRO FUNDADORES": "teatro",
    "TEATRO EL ENSUEO - TV 70 D # 60 - 90 SUR, BOGOTA": "teatro",
    "TEATRO EL TESORO": "teatro",
    "ROYAL CENTER": "teatro",  # Tradicional teatro/sala de conciertos acústica
    
    # Arenas y Coliseos
    "MOVISTAR ARENA": "arena_cubierta",
    "COLISEO MEDPLUS": "arena_cubierta",
    "COLISEO ELIAS CHEGWIN": "arena_cubierta",
    "COLISEO MAYOR JORGE ARANGO URIBE": "arena_cubierta",
    "PALACIO DE LOS DEPORTES": "arena_cubierta",
    "ARENA CAAVERALEJO": "arena_cubierta",
    "ARENA CAAVERALEJO - CALI (AJUSTE)": "arena_cubierta",
    "NECTAR ARENA CENTRO DE EVENTOS": "arena_cubierta",
    "COLISEO BICENTENARIO - BUCARAMANGA": "arena_cubierta",
    "COLISEO MAYOR DE IBAGUE": "arena_cubierta",

    # Estadios
    "ESTADIO EL CAMPIN": "estadio_abierto",
    "ESTADIO ATANASIO GIRARDOT": "estadio_abierto",
    "ESTADIO PASCUAL GUERRERO": "estadio_abierto",
    "ESTADIO METROPOLITANO": "estadio_abierto",
    "ESTADIO PALOGRANDE - MANIZALES": "estadio_abierto",
    "ESTADIO MANUEL MURILLO TORO": "estadio_abierto",
    "ESTADIO METROPOLITANO DE TECHO": "estadio_abierto",
    "ESTADIO ROMELIO MARTINEZ": "estadio_abierto",
    "ESTADIO BELLO HORIZONTE - REY PELE (VILLAVICENCIO)": "estadio_abierto",
    "ESTADIO EDGAR RENTERIA": "estadio_abierto",
    "ESTADIO GENERAL SANTANDER": "estadio_abierto",
    "CUCUTA ESTADIO GENERAL SANTANDER": "estadio_abierto",
    "ESTADIO GUILLERMO PLAZAS ALCID": "estadio_abierto",
    "ESTADIO LA INDEPENDENCIA": "estadio_abierto",
    "ESTADIO JAIME MORON": "estadio_abierto",
    "ESTADIO JOSE AMERICO MONTANINI": "estadio_abierto",
    "ESTADIO HERNAN RAMIREZ VILLEGAS": "estadio_abierto",
    "ESTADIO SIERRA NEVADA": "estadio_abierto",
    "ESTADIO JARAGUAY": "estadio_abierto",
    "ESTADIO FRANCISCO RIVERA ESCOBAR - PALMIRA": "estadio_abierto",
    "COPA AMERICA - NRG STADIUM, HOUSTON, TEXAS": "estadio_abierto",

    # Salas de Cine, Museos, Culturales
    "SALA 3 CINEMATECA": "cine_sala_cultural",
    "SALA CAPITAL CINEMATECA": "cine_sala_cultural",
    "SALA 2 CINEMATECA": "cine_sala_cultural",
    "MALOKA": "cine_sala_cultural",
    "PLANETARIO DE BOGOTA": "cine_sala_cultural",
    "YAWA, CENTRO DE CIENCIA, ARTE Y TECNOLOGIA - CALI": "cine_sala_cultural",
    "MUSEO LA TERTULIA": "cine_sala_cultural",
    "SALA DE CONCIERTOS DE LA BIBLIOTECA LUIS ANGEL ARANGO": "cine_sala_cultural",
    "BIBLIOTECA NACIONAL DE COLOMBIA": "cine_sala_cultural",
    "CINE COLOMBO, MEDELLIN": "cine_sala_cultural",

    # Centros de Eventos y Carpas
    "CARPA DELIRIO": "centro_eventos_carpa",
    "COMPLEJO CULTURAL DELIRIO": "centro_eventos_carpa",
    "CORFERIAS": "centro_eventos_carpa",
    "CARPA AMERICAS CORFERIAS": "centro_eventos_carpa",
    "CHAMORRO CITY HALL - AUTO NTE #153-81": "centro_eventos_carpa",
    "CENTRO DE CONVENCIONES CARTAGENA": "centro_eventos_carpa",
    "CENTRO DE EVENTOS CENFER": "centro_eventos_carpa",
    "PUERTA DE ORO BARRANQUILLA": "centro_eventos_carpa",
    "PUERTA DE ORO BARRANQUILLA - LA EXPLANADA": "centro_eventos_carpa",
    "EXPLANADA- PUERTA DE ORO": "centro_eventos_carpa",
    "EXPOFUTURO - PEREIRA": "centro_eventos_carpa",
    "CENTRO DE CONVENCIONES G12": "centro_eventos_carpa",
    "PLAZA MAYOR": "centro_eventos_carpa",
    "CENTRO DE EVENTOS AUTOPISTA NORTE": "centro_eventos_carpa",
    "PABELLON DE CRISTAL - GRAN MALECON": "centro_eventos_carpa",

    # Salas CNA y Teatro Satélites
    "CENTRO NACIONAL DE LAS ARTES - SALA DELIA ZAPATA": "teatro",
    "CENTRO NACIONAL DE LAS ARTES - SALA FANNY MIKEY": "teatro",
    "CENTRO NACIONAL DE LAS ARTES - SALA TERESITA GOMEZ": "teatro",
    "CENTRO NACIONAL DE LAS ARTES - SALA FOYER CLL10 NO 5-32": "teatro",
    "CENTRO NACIONAL DE LAS ARTES - SALA TEATRO": "teatro",
    "SALA GAITAN": "teatro",
    "SALON ESPEJOS TEATRO JORGE ELIECER GAITAN": "teatro",
    "TEATRO ESTUDIO - JULIO MARIO SANTO DOMINGO": "teatro",
    "CENTRO CULTURAL DEL GIMNASIO MODERNO": "teatro",
    "CENTRO CULTURAL GIMNASIO MODERNO": "teatro",

    # Escenarios Deportivos Masivos Adicionales
    "DIAMANTE DE BEISBOL - MEDELLIN": "estadio_abierto",
    "DIAMANTE DE SOFTBOL": "estadio_abierto",
    "ESTADIO DE BEISBOL LA ESPERANZA": "estadio_abierto",
    "ESTADIO DITAIRES": "estadio_abierto",
    "ESTADIO HERMIDES PADILLA": "estadio_abierto",
    "ESTADIO MUNICIPAL DE VILLETA": "estadio_abierto",
    "ESTADIO DE FUTBOL - INMACULADA CONCEPCION": "estadio_abierto",

    # Bares, Comedy Clubs, Discotecas
    "BOOM STAND UP BAR - BOGOTA": "bar_club",
    "BOOM STAND UP BAR - CL 26 #43G-30 BARRIO COLOMBIA": "bar_club",
    "WOW RESTAURANTE BAR": "bar_club",
    "LOURDES MUSIC HALL  - BOGOTA": "bar_club",
    "CANTINA LA 70 - CRA 70 #44B - 76 (MEDELLIN)": "bar_club",
    "SAFARI DISCO CLUB, AV SANTANDER #63 - 122, MANIZALES": "bar_club",
    "440 MUSIC HALL": "bar_club",
    "MONASTERY CLUB": "bar_club",
    "FROGG CLUB": "bar_club",
    "DISCO MOVISTAR ARENA": "bar_club",
    "RANCHO MX": "bar_club",
    "MOYS RESTAURANTE BAR": "bar_club",
    "CINARUCO BAR - CRA 14 NO 24A -15 - YOPAL": "bar_club",
    "KABALA BAR - MANIZALES": "bar_club",

    # Parques / Aire Libre
    "PARQUE NORTE": "parque_aire_libre",
    "PARQUE METROPOLITANO SIMON BOLIVAR": "parque_aire_libre",
    "GRAN MALECON BARRANQUILLA": "parque_aire_libre",
    "PARQUE DE LA LEYENDA VALLENATA": "parque_aire_libre",
    "AUTODROMO DE TOCANCIPA": "parque_aire_libre",
    "PARQUE DE LA 93": "parque_aire_libre",
    "PARQUE MUSEO EL CHICO": "parque_aire_libre",
    "PARQUE DE EVENTOS - LA INDEPENDENCIA": "parque_aire_libre",
    "JARDIN BOTANICO - ORQUIDEORAMA": "parque_aire_libre",
    "JARDIN BOTANICO ORQUIDEORAMA - ANOTR": "parque_aire_libre",
    "SALITRE MAGICO": "parque_aire_libre",
    "MUNDO AVENTURA": "parque_aire_libre",
    "AEROPARQUE JUAN PABLO SEGUNDO": "parque_aire_libre",
    "CARRERA 50 BARRANQUILLA": "parque_aire_libre",
    "BIBLOS CAR WASH": "otro",
    "PLAZA DE BOLIVAR": "parque_aire_libre",
    "PLAZA DE LA PAZ": "parque_aire_libre",
    "LA MEDIA TORTA": "parque_aire_libre",

    # Instituciones Académicas (Auditorios)
    "UNIVERSIDAD DE LA SABANA": "auditorio",
    "COLEGIO LA ENSEANZA - CL 9 SUR #37-345, MEDELLIN": "auditorio",
    "AUDITORIO UNIVERSIDAD NACIONAL MANIZALES - CRA 27 #62-56": "auditorio",
    "UNIVERSIDAD EAN - CARRERA 11 # 78-47": "auditorio",
    "UNIVERSIDAD DE IBAGUE": "auditorio",
    "UNIVERSIDAD INDUSTRIAL DE SANTANDER": "auditorio",

    # Otros / No recintos fijos
    "TREN TURISTICO": "otro",
    "TRANSPORTE": "otro",
    "TRANSPORTE LEGACY TOUR": "otro",
    "FINAL COPA": "otro",
    "BOGOTA (DIRECCION EXACTA SE COMPARTE TRAS INSCRIPCION)": "otro",
    "LABORARTORIO 1 Y 2": "otro",
    "ESTACION METRO LA ESTRELLA": "otro",
    "CAFE INTERNET - SAN FELIPE - CALLE 76 # 20B - 65": "otro",
    "CAFE INTERNET - BOGOTA": "otro",
}

# Diccionario pre-normalizado para busquedas exactas y delimitadas
DICCIONARIO_NORMALIZADO = {normalizar_recinto(k): v for k, v in DICCIONARIO_EMBLEMATICO.items()}


def _buscar_en_diccionario_emblematico(rec_norm: str) -> tuple[str, float] | None:
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
        return "otro", 0.90

    if re.search(r"\b(CAR\s*WASH|LAVADERO)\b", rec_norm):
        return "otro", 0.90

    if re.search(r"\b(VIA\s*40|CARRERA\s*50)\b", rec_norm):
        return "parque_aire_libre", 0.90

    # 3. Reglas estructurales léxicas primarias con límites de palabra
    if re.search(r"\b(ESTADIO|CAMPIN|ATANASIO|PASCUAL\s*GUERRERO|PALMASECA|PALOGRANDE|MURILLO\s*TORO|GIRARDOT|STADIUM)\b", rec_norm):
        return "estadio_abierto", 0.95

    if re.search(r"\b(MOVISTAR\s*ARENA|COLISEO|ARENA\s+CA[NÑ]AVERALEJO|ARENA\s+BOGOTA|PALACIO\s+DE\s+LOS\s+DEPORTES)\b", rec_norm):
        return "arena_cubierta", 0.92

    if re.search(r"\b(TEATRO|TEATRINO|SALA\s+TEATRO|SALA\s+TEATRAL)\b", rec_norm):
        return "teatro", 0.95

    if re.search(r"\b(AUDITORIO|AULA\s+MAXIMA)\b", rec_norm):
        return "auditorio", 0.92

    if re.search(r"\b(CINEMATECA|PLANETARIO|MALOKA|MUSEO|BIBLIOTECA|SALA\s+DE\s+CINE|CINE\s+COLOMBO)\b", rec_norm):
        return "cine_sala_cultural", 0.94

    if re.search(r"\b(CARPA|CORFERIAS|CHAMORRO|CENTRO\s+DE\s+EVENTOS|PABELLON|CONVENCIONES|EXPOFUTURO|CENFER|PUERTA\s+DE\s+ORO|CITY\s+HALL)\b", rec_norm):
        return "centro_eventos_carpa", 0.93

    if re.search(r"\b(BAR|CLUB|RESTAURANTE|DISCOTECA|PUB|GASTROBAR|CANTA\s*BAR|FONDA|STAND\s*UP)\b", rec_norm):
        return "bar_club", 0.91

    if re.search(r"\b(PARQUE|MALECON|BOTANICO|AUTODROMO|PLAZA\s+DE\s+TOROS|CANCHA|DIAMANTE\s+DE\s+BEISBOL|POLIDEPORTIVO)\b", rec_norm):
        return "parque_aire_libre", 0.90

    if re.search(r"\b(HOTEL)\b", rec_norm):
        return ("centro_eventos_carpa" if aforo_max > 400 else "otro"), 0.80

    if re.search(r"\b(TREN|TRANSPORTE|AEROPUERTO|FINAL\s+COPA|LABORATORIO)\b", rec_norm):
        return "otro", 0.92

    # Salas genéricas sin cualificador de cinemateca o teatro van a bajo score
    if re.search(r"\bSALA\b", rec_norm):
        return "otro", 0.60

    # Fallbacks de baja confianza por aforo
    if aforo_max >= 15000:
        return "estadio_abierto", 0.75

    if aforo_max >= 4000:
        return "centro_eventos_carpa", 0.70

    if aforo_max < 300:
        return ("bar_club" if re.search(r"\b(CAFE|CASA)\b", rec_norm) else "otro"), 0.72

    return "otro", 0.60


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
        return "otro", 0.90

    if re.search(r"\b(CAR\s*WASH|LAVADERO)\b", rec_norm):
        return "otro", 0.90

    if re.search(r"\b(VIA\s*40|CARRERA\s*50)\b", rec_norm):
        return "parque_aire_libre", 0.90

    # 3. Análisis por patrones de texto alternativos con límites estrictos de palabra
    if re.search(r"\b(ESTADIO|STADIUM|BEISBOL|DIAMANTE)\b", rec_norm):
        return "estadio_abierto", 0.95

    if re.search(r"\b(ARENA|COLISEO|POLIDEPORTIVO)\b", rec_norm):
        return ("parque_aire_libre" if "POLIDEPORTIVO" in rec_norm else "arena_cubierta"), 0.90

    if re.search(r"\b(TEATRO|TEATRINO|SALA\s+TEATRAL)\b", rec_norm):
        return "teatro", 0.95

    if re.search(r"\b(AUDITORIO|AULA)\b", rec_norm):
        return "auditorio", 0.94

    if re.search(r"\b(CINE|CINEMATECA|MUSEO|PLANETARIO|BIBLIOTECA)\b", rec_norm):
        return "cine_sala_cultural", 0.93

    if re.search(r"\b(CARPA|EXPO|FERIA|CONVENCION|PABELLON|CENTRO\s+DE\s+EVENTOS|CITY\s+HALL)\b", rec_norm):
        return "centro_eventos_carpa", 0.92

    if re.search(r"\b(BAR|CLUB|DISCO|LOUNGE|RESTAURANTE|PUB|GASTRO|TASCA|BARRIL|BEER)\b", rec_norm):
        return "bar_club", 0.92

    if re.search(r"\b(PARQUE|PLAZA|JARDIN|BOULEVARD|MALECON|PLAYA|BEACH|AVENIDA|CARRERA|AUTOPISTA|CALLE)\b", rec_norm):
        return "parque_aire_libre", 0.89

    if re.search(r"\b(HOTEL|RESORT)\b", rec_norm):
        return ("centro_eventos_carpa" if aforo_max >= 500 else "otro"), 0.78

    if re.search(r"\b(TREN|BUS|TRANSPORTE|AEROPUERTO|VIAJE)\b", rec_norm):
        return "otro", 0.95

    if re.search(r"\bSALA\b", rec_norm):
        if re.search(r"\b(CINEMATECA|CINE)\b", rec_norm):
            return "cine_sala_cultural", 0.88
        if re.search(r"\b(TEATRO|TEATRAL)\b", rec_norm):
            return "teatro", 0.85
        return "otro", 0.60

    if re.search(r"\b(CAPILLA|IGLESIA|CATEDRAL)\b", rec_norm):
        return "otro", 0.85

    if re.search(r"\b(COLEGIO|UNIVERSIDAD|CAMPUS)\b", rec_norm):
        return "auditorio", 0.83

    # Fallback contextual por aforo
    if aforo_max >= 20000:
        return "estadio_abierto", 0.72
    elif aforo_max >= 5000:
        return "centro_eventos_carpa", 0.68
    elif aforo_max <= 200:
        return "otro", 0.65

    return "otro", 0.55


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
        "modelo_llm": None
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
        clasificador_llm = GeminiVenueClassifier()
        print(f"=== MÓDULO LLM ACTIVADO (Modelo: {clasificador_llm.model_name}, temp={clasificador_llm.temperature}) ===")
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
                "modelo_llm": clasificador_llm.model_name if clasificador_llm else None
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
                    "modelo_llm": clasificador_llm.model_name
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
                "modelo_llm": None
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

