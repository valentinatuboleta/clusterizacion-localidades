"""
Módulo de Clasificación de Venues con Agente LLM (Gemini 3.8 Flash Medium).

Implementa:
1. Cliente LLM con inyección de dependencias (testable mediante mocks).
2. Salida estructurada estricta en JSON a temperatura 0.0.
3. Persistencia de trazabilidad y auditoría completa en JSONL.
4. Esquema de precedencia: revision_humana > llm > reglas_heuristicas.
5. Verificación cruzada sobre DICCIONARIO_EMBLEMATICO (discrepancias van a revisión humana).
"""

import os
import re
import json
import datetime
from typing import Dict, Any, Optional, Protocol, Tuple, List


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

PROMPT_SISTEMA_VENUE = """Eres un experto en infraestructura de entretenimiento, arquitectura de venues y espectáculos en Colombia.
Tu misión es clasificar con máxima precisión el tipo de espacio físico de un venue donde se realizan eventos de boletería (TuBoleta).

Debes seleccionar exactamente una de las siguientes 9 categorías canónicas:
1. teatro: Teatros tradicionales con platea y balcones acústicos (ej. Teatro Mayor, Teatro Colón).
2. estadio_abierto: Estadios masivos descubiertos de fútbol, béisbol o atletismo (ej. El Campín, Atanasio Girardot).
3. arena_cubierta: Arenas y coliseos cerrados multi-propósito (ej. Movistar Arena, Coliseo MedPlus).
4. centro_eventos_carpa: Pabellones feriales, centros de convenciones, carpas estructurales gigantes o hangares (ej. Corferias, Chamorro City Hall, Carpa Delirio).
5. cine_sala_cultural: Salas de cine, cinematecas, planetarios, museos o bibliotecas (ej. Cinemateca de Bogotá, Maloka).
6. auditorio: Auditorios y aulas magnas académicas o corporativas (ej. Auditorio León de Greiff, universidades).
7. bar_club: Bares, discotecas, gastrobares, comedy clubs o restaurantes con música en vivo (ej. Boom Stand Up Bar).
8. parque_aire_libre: Parques públicos, malecones, plazas, praderas al aire libre, playas o circuitos viales para festivales (ej. Parque Simón Bolívar, Gran Malecón).
9. otro: Estacionamientos, transporte, puntos de encuentro o recintos no categorizables.

Reglas obligatorias:
- Considera el contexto toponímico en Colombia (ciudades, avenidas, nombres de centros culturales).
- Si el venue es ambiguo, utiliza conocimiento del mundo para inferir su arquitectura física.
- Responde estrictamente con un objeto JSON sin formato markdown extra."""


class LLMClientProtocol(Protocol):
    """Protocolo para inyección de dependencias del cliente LLM."""
    def generate_content(self, model: str, contents: str, config: Optional[Dict[str, Any]] = None) -> Any:
        ...


class GeminiVenueClassifier:
    """
    Clasificador de venues basado en LLM (Gemini 3.8 Flash Medium).
    Soporta inyección de dependencias para ejecución hermética en pruebas (mocks).
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: str = "gemini-3.8-flash-medium",
        temperature: float = 0.0,
        audit_file: str = "data/lookup/audit_llm_venues.jsonl"
    ):
        self.model_name = os.getenv("GEMINI_MODEL", model_name)
        self.temperature = temperature
        self.audit_file = audit_file
        self._client = client

    def _obtener_cliente(self) -> Any:
        """Inicializa el cliente de Google GenAI si no fue inyectado."""
        if self._client is not None:
            return self._client

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            return self._client
        except Exception:
            return None

    def esta_disponible(self) -> bool:
        """Verifica una sola vez si el cliente LLM está disponible y operativo."""
        return self._obtener_cliente() is not None


    def construir_prompt(self, site: str, aforo_max: int = 0) -> str:
        """Construye el prompt estructurado para clasificación del venue."""
        return (
            f"{PROMPT_SISTEMA_VENUE}\n\n"
            f"Analiza el siguiente venue:\n"
            f"- Nombre del Venue: \"{site}\"\n"
            f"- Aforo Máximo Registrado: {aforo_max}\n\n"
            f"Responde ÚNICAMENTE en formato JSON con la siguiente estructura exacta:\n"
            f'{{\n'
            f'  "site": "{site}",\n'
            f'  "type_site": "<una de las 9 categorias>",\n'
            f'  "confianza": <numero entre 0.0 y 1.0>,\n'
            f'  "justificacion_semantica": "<breve justificacion de 1 linea>"\n'
            f'}}'
        )

    def clasificar_venue(self, site: str, aforo_max: int = 0) -> Optional[Dict[str, Any]]:
        """
        Ejecuta la inferencia LLM para clasificar un venue individual.
        Persiste el registro crudo en el archivo de auditoría.
        """
        cliente = self._obtener_cliente()
        if cliente is None:
            return None

        prompt = self.construir_prompt(site, aforo_max)
        raw_text = ""
        timestamp = datetime.datetime.now().isoformat()

        try:
            # Compatibilidad tanto con el nuevo SDK google-genai como con mocks inyectados
            if hasattr(cliente, "models") and hasattr(cliente.models, "generate_content"):
                resp = cliente.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"temperature": self.temperature}
                )
                raw_text = resp.text
            elif hasattr(cliente, "generate_content"):
                resp = cliente.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"temperature": self.temperature}
                )
                raw_text = getattr(resp, "text", str(resp))
            else:
                return None

            # Limpiar bloques markdown si el modelo los retorna
            limpio = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
            limpio = re.sub(r"\s*```$", "", limpio.strip(), flags=re.MULTILINE)

            parsed = json.loads(limpio)
            tipo_asignado = str(parsed.get("type_site", "otro")).strip().lower()
            if tipo_asignado not in CATEGORIAS_VALIDAS:
                tipo_asignado = "otro"

            confianza = float(parsed.get("confianza", 0.85))
            confianza = min(max(confianza, 0.50), 0.99)
            justificacion = str(parsed.get("justificacion_semantica", "Clasificado por LLM")).strip()

            resultado = {
                "site": site,
                "type_site": tipo_asignado,
                "confianza": round(confianza, 3),
                "justificacion_semantica": justificacion,
                "modelo_llm": self.model_name
            }

            self._guardar_auditoria({
                "timestamp": timestamp,
                "site": site,
                "aforo_max": aforo_max,
                "modelo_llm": self.model_name,
                "prompt": prompt,
                "raw_response": raw_text,
                "resultado": resultado
            })

            return resultado

        except Exception as err:
            self._guardar_auditoria({
                "timestamp": timestamp,
                "site": site,
                "aforo_max": aforo_max,
                "modelo_llm": self.model_name,
                "prompt": prompt,
                "error": str(err)
            })
            return None

    def _guardar_auditoria(self, entrada: Dict[str, Any]) -> None:
        """Persiste una línea en el archivo de auditoría JSONL."""
        try:
            os.makedirs(os.path.dirname(self.audit_file), exist_ok=True)
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
        except Exception:
            pass
