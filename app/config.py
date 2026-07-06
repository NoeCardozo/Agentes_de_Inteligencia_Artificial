"""
=============================================================================
CONFIGURACIÓN GLOBAL DEL PROYECTO
=============================================================================

Aquí se cargan las variables de entorno desde el archivo .env y se definen
las rutas a los datos del proyecto.

Para la defensa:
  - La API key NO va en el código, va en .env (buena práctica de seguridad).
  - El SYSTEM_PROMPT le dice a Gemini cómo comportarse y qué herramientas usar.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Carga variables del archivo .env (GOOGLE_API_KEY, GEMINI_MODEL, etc.)
load_dotenv()

# --- Rutas de carpetas del proyecto ---
BASE_DIR = Path(__file__).resolve().parent.parent  # Raíz: tp-agent/
DATA_DIR = BASE_DIR / "data" / "guias"              # Guías turísticas (.txt)
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"        # Base vectorial del RAG

# --- Credenciales y modelo de IA ---
# Se leen del .env. Si no existen, quedan vacías o con valor por defecto.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# text-embedding-004 fue deprecado ene 2026 → usar gemini-embedding-001
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")


def validate_api_key() -> None:
    """
    Verifica que la API key de Google esté configurada y tenga formato válido.

    Google emite dos formatos de clave:
      - AIza...  → formato antiguo (legacy)
      - AQ....   → formato nuevo (auth key de AI Studio)
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY no configurada. "
            "Copia .env.example a .env y agrega tu API key de Google AI Studio."
        )

    if not (GOOGLE_API_KEY.startswith("AIza") or GOOGLE_API_KEY.startswith("AQ.")):
        raise ValueError(
            "GOOGLE_API_KEY no parece válida. "
            "Debe empezar con 'AIza' (legacy) o 'AQ.' (nuevo formato). "
            "Obtén una en https://aistudio.google.com/apikey"
        )


# Instrucciones que recibe Gemini al iniciar cada conversación.
# Define el rol del agente y cuándo debe usar cada herramienta.
SYSTEM_PROMPT = """Eres un asistente experto en turismo de ARGENTINA.
Tu objetivo es ayudar a planificar viajes dentro del país y dar información sobre ciudades argentinas.

Tienes acceso a las siguientes herramientas:
- consultar_ciudad_argentina: datos oficiales (API Georef) + resumen de Wikipedia de una ciudad.
- listar_provincias_argentina: lista las 24 provincias argentinas (API oficial del gobierno).
- buscar_vuelos: opciones de vuelo entre ciudades (datos de demostración).
- crear_itinerario: plan día a día de un viaje.
- consultar_guia_turistica: busca en guías locales (RAG) sobre Buenos Aires, Mendoza, Córdoba, Bariloche, Salta.

Reglas:
- Responde siempre en español.
- Prioriza destinos y rutas dentro de Argentina.
- Para datos oficiales de ubicación/provincia/coordenadas, usa consultar_ciudad_argentina.
- Para atracciones, comida y consejos detallados, usa consultar_guia_turistica.
- Si preguntan por provincias, usa listar_provincias_argentina.
- Sé conciso pero completo. Si falta información, pregúntala.
"""
