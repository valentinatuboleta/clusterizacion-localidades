"""
Punto de entrada de la API FastAPI para clustering y micro-clusters de localidades (Bloque 1).
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import joblib
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.clustering import cargar_modelo_clustering
from src.nlp_utils import cargar_stopwords
from src.api.schemas import EventoInput, PrediccionOutput, HealthOutput
from src.api.servicio import predecir_evento

logger = logging.getLogger("api.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def asegurar_artefactos_cargados(target_app: FastAPI):
    """
    Garantiza que ambos payloads de modelos y las stopwords esten cargados en target_app.state.
    Si ya estan cargados, no realiza trabajo redundante.
    """
    if getattr(target_app.state, "payload_v25", None) is None:
        path_v25 = os.getenv("MODELO_V25_PATH", "data/processed/modelo_clustering_v2_5.joblib")
        if not os.path.exists(path_v25):
            raise FileNotFoundError(f"No se encontro el modelo macro v2.5 en: {path_v25}")
        target_app.state.payload_v25 = cargar_modelo_clustering(path_v25)

    if getattr(target_app.state, "payload_jer", None) is None:
        path_jer = os.getenv("MODELO_JER_PATH", "data/processed/modelo_jerarquia_v3.joblib")
        if not os.path.exists(path_jer):
            raise FileNotFoundError(f"No se encontro el modelo jerarquico v3.0 en: {path_jer}")
        target_app.state.payload_jer = joblib.load(path_jer)

    if getattr(target_app.state, "stopwords", None) is None:
        ruta_stopwords = os.getenv("STOPWORDS_LOCAL_PATH", "data/lookup/stopwords_marketing.csv")
        target_app.state.stopwords = cargar_stopwords(ruta_local=ruta_stopwords)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ciclo de vida de la aplicación:
    Carga ambos payloads de modelos y la tabla de lookup de stopwords una sola vez al arrancar.
    """
    logger.info("Iniciando servicio de inferencia de localidades...")
    asegurar_artefactos_cargados(app)

    v25_ver = app.state.payload_v25.get("version", "2.5")
    jer_ver = app.state.payload_jer.get("version", "3.0-hier.2")
    logger.info(
        "Modelos cargados: Macro v%s, Jerarquia %s. Stopwords activas: %d",
        v25_ver, jer_ver, len(app.state.stopwords)
    )

    yield

    logger.info("Cerrando servicio...")


app = FastAPI(
    title="TuBoleta Localidades Clustering API",
    description="Microservicio de inferencia para segmentacion y arquetipos de demanda de localidades",
    version="1.0.0",
    lifespan=lifespan
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Captura errores de validación y retorna HTTP 400 con mensaje legible.
    """
    mensajes = []
    for err in exc.errors():
        campo = " -> ".join(str(loc) for loc in err.get("loc", []) if loc != "body")
        msg = err.get("msg", "Valor invalido")
        mensajes.append(f"Campo '{campo}': {msg}" if campo else msg)
    detalle = "Error de validacion en la solicitud: " + "; ".join(mensajes)
    logger.warning("Solicitud con datos invalidos (400): %s", detalle)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": detalle}
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Captura excepciones imprevistas del pipeline y responde HTTP 500 con detalle genérico,
    sin filtrar nunca trazas internas ni detalles del entorno al cliente.
    """
    logger.error("Excepcion no controlada durante inferencia: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor al procesar la prediccion del evento."}
    )


@app.get("/health", response_model=HealthOutput, tags=["Monitoreo"])
async def health_check(request: Request):
    """
    Verifica la salud del microservicio y reporta las versiones de los modelos cargados en memoria.
    """
    asegurar_artefactos_cargados(request.app)
    p_v25 = getattr(request.app.state, "payload_v25", {})
    p_jer = getattr(request.app.state, "payload_jer", {})

    v25_ver = p_v25.get("version", "2.5") if isinstance(p_v25, dict) else "2.5"
    jer_ver = p_jer.get("version", "3.0-hier.2") if isinstance(p_jer, dict) else "3.0-hier.2"

    return {
        "status": "ok",
        "modelo_clustering_version": str(v25_ver),
        "modelo_jerarquia_version": str(jer_ver),
        "modelo_version": f"v{v25_ver} / {jer_ver}"
    }


@app.post("/v1/predecir-localidades", response_model=PrediccionOutput, tags=["Inferencia"])
async def predecir_localidades_endpoint(evento: EventoInput, request: Request):
    """
    Ejecuta el pipeline completo de inferencia bietapica para un evento con sus localidades.
    """
    asegurar_artefactos_cargados(request.app)
    payload_v25 = request.app.state.payload_v25
    payload_jer = request.app.state.payload_jer

    evento_dict = evento.model_dump()
    resultado = predecir_evento(evento_dict, payload_v25, payload_jer)
    return resultado
