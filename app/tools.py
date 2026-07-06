"""
=============================================================================
HERRAMIENTAS (TOOLS) DEL AGENTE
=============================================================================

Las herramientas son ACCIONES que el agente puede ejecutar.
LangChain las registra con el decorador @tool para que Gemini sepa
cuándo y cómo llamarlas.

Herramientas disponibles:
  1. consultar_ciudad_argentina → API Georef + Wikipedia (datos en tiempo real).
  2. listar_provincias_argentina → Las 24 provincias (API oficial).
  3. buscar_vuelos               → Vuelos simulados (demo).
  4. crear_itinerario            → Plan día a día.
  5. consultar_guia_turistica    → RAG con guías locales en data/guias/.

Para la defensa:
  - En producción, buscar_vuelos se conectaría a una API real (Amadeus, etc.).
  - consultar_guia_turistica es la implementación del RAG en el agente.
"""

import random

from langchain_core.tools import tool

from app.argentina_api import buscar_localidad, consultar_ciudad, listar_provincias
from app.rag import consultar_guia

# ---------------------------------------------------------------------------
# Datos simulados de vuelos (solo para demostración del TP)
# Clave: (ciudad_origen, ciudad_destino) en minúsculas
# ---------------------------------------------------------------------------
# Vuelos de demostración — rutas populares dentro y hacia/desde Argentina
VUELOS_DEMO = {
    ("buenos aires", "mendoza"): [
        {"aerolinea": "Aerolíneas Argentinas", "precio_usd": 95, "escalas": 0, "duracion": "1h 50m"},
        {"aerolinea": "Flybondi", "precio_usd": 65, "escalas": 0, "duracion": "1h 55m"},
        {"aerolinea": "JetSMART", "precio_usd": 58, "escalas": 0, "duracion": "2h 00m"},
    ],
    ("buenos aires", "bariloche"): [
        {"aerolinea": "Aerolíneas Argentinas", "precio_usd": 120, "escalas": 0, "duracion": "2h 15m"},
        {"aerolinea": "Flybondi", "precio_usd": 85, "escalas": 0, "duracion": "2h 20m"},
    ],
    ("buenos aires", "cordoba"): [
        {"aerolinea": "Aerolíneas Argentinas", "precio_usd": 75, "escalas": 0, "duracion": "1h 20m"},
        {"aerolinea": "Flybondi", "precio_usd": 55, "escalas": 0, "duracion": "1h 25m"},
    ],
    ("buenos aires", "salta"): [
        {"aerolinea": "Aerolíneas Argentinas", "precio_usd": 110, "escalas": 0, "duracion": "2h 05m"},
        {"aerolinea": "JetSMART", "precio_usd": 78, "escalas": 0, "duracion": "2h 10m"},
    ],
    ("buenos aires", "ushuaia"): [
        {"aerolinea": "Aerolíneas Argentinas", "precio_usd": 145, "escalas": 0, "duracion": "3h 25m"},
        {"aerolinea": "Flybondi", "precio_usd": 105, "escalas": 1, "duracion": "5h 10m"},
    ],
    ("buenos aires", "madrid"): [
        {"aerolinea": "Iberia", "precio_usd": 890, "escalas": 0, "duracion": "12h 30m"},
        {"aerolinea": "Air Europa", "precio_usd": 750, "escalas": 1, "duracion": "15h 10m"},
    ],
}

# Plantillas de actividades según el interés del viajero
ACTIVIDADES_POR_INTERES = {
    "museos": ["Visita a museo principal", "Barrio histórico a pie", "Galería de arte"],
    "gastronomía": ["Mercado local por la mañana", "Tour gastronómico", "Cena en restaurante típico"],
    "playa": ["Mañana en la playa", "Paseo costero", "Atardecer en el malecón"],
    "naturaleza": ["Parque nacional o reserva", "Senderismo ligero", "Mirador panorámico"],
    "general": ["Atracción icónica", "Barrio emblemático", "Experiencia local"],
}


def _normalizar_ciudad(ciudad: str) -> str:
    """Convierte el nombre de ciudad a minúsculas y sin espacios extra."""
    return ciudad.strip().lower()


def _formatear_vuelos(origen: str, destino: str, fecha: str, vuelos: list[dict]) -> str:
    """Arma el texto de respuesta con la lista de vuelos encontrados."""
    lineas = [f"Vuelos de {origen} → {destino} para {fecha}:", ""]

    for i, vuelo in enumerate(vuelos, 1):
        escalas = "directo" if vuelo["escalas"] == 0 else f"{vuelo['escalas']} escala(s)"
        lineas.append(
            f"{i}. {vuelo['aerolinea']} | ${vuelo['precio_usd']} USD | "
            f"{vuelo['duracion']} | {escalas}"
        )

    lineas.extend(["", "(Datos simulados para demostración. Integra una API real en producción.)"])
    return "\n".join(lineas)


def _vuelos_aleatorios() -> list[dict]:
    """Genera vuelos ficticios cuando la ruta no está en la base de demo."""
    aerolineas_ar = ["Aerolíneas Argentinas", "Flybondi", "JetSMART", "LATAM"]
    return [
        {
            "aerolinea": random.choice(aerolineas_ar),
            "precio_usd": random.randint(50, 350),
            "escalas": random.choice([0, 0, 1]),
            "duracion": f"{random.randint(1, 4)}h {random.randint(0, 55):02d}m",
        }
        for _ in range(3)
    ]


@tool
def consultar_ciudad_argentina(ciudad: str, provincia: str = "") -> str:
    """
    Consulta datos oficiales y turísticos de una ciudad argentina.

    Usa la API Georef (gobierno argentino) para ubicación, provincia y coordenadas,
    y Wikipedia en español para contexto histórico y turístico.

    Args:
        ciudad: Nombre de la ciudad (ej: Mendoza, Bariloche, Salta).
        provincia: Opcional, para desambiguar (ej: "Neuquén" si buscás Bariloche).
    """
    return consultar_ciudad(ciudad, provincia)


@tool
def listar_provincias_argentina() -> str:
    """
    Lista las 24 provincias de Argentina con datos oficiales de Georef (datos.gob.ar).

    Usar cuando el usuario pregunte por provincias, regiones o quiera elegir destino.
    """
    return listar_provincias()


@tool
def buscar_localidades_argentina(nombre: str, provincia: str = "") -> str:
    """
    Busca localidades/ciudades argentinas por nombre en la API Georef.

    Args:
        nombre: Nombre a buscar (ej: "Rosario", "Mar del Plata").
        provincia: Opcional, filtra por provincia (ej: "Santa Fe").
    """
    return buscar_localidad(nombre, provincia=provincia)


@tool
def buscar_vuelos(origen: str, destino: str, fecha: str) -> str:
    """
    Busca vuelos entre dos ciudades para una fecha.

    Gemini llama esta función cuando el usuario pregunta por vuelos.
    """
    ruta = (_normalizar_ciudad(origen), _normalizar_ciudad(destino))
    ruta_inversa = (ruta[1], ruta[0])

    vuelos = VUELOS_DEMO.get(ruta) or VUELOS_DEMO.get(ruta_inversa) or _vuelos_aleatorios()
    return _formatear_vuelos(origen, destino, fecha, vuelos)


@tool
def crear_itinerario(destino: str, dias: int, intereses: str = "general") -> str:
    """
    Crea un itinerario día a día para un destino.

    Gemini la usa cuando el usuario pide un plan de viaje.
  """
    dias = max(1, min(dias, 14))  # Entre 1 y 14 días
    actividades = ACTIVIDADES_POR_INTERES.get(intereses.lower(), ACTIVIDADES_POR_INTERES["general"])

    lineas = [
        f"Itinerario de {dias} día(s) en {destino}",
        f"Intereses: {intereses}",
        "",
    ]

    for dia in range(1, dias + 1):
        # Primer y último día tienen horarios especiales
        if dia == 1:
            horario = "Tarde (llegada y check-in)"
        elif dia == dias:
            horario = "Mañana (check-out y despedida)"
        else:
            horario = "Día completo"

        actividad = actividades[(dia - 1) % len(actividades)]
        lineas.append(f"Día {dia} ({horario}):")
        lineas.append(f"  - {actividad}")
        lineas.append(f"  - Tiempo libre para explorar {destino}")
        lineas.append("")

    lineas.append("Tip: Usa consultar_guia_turistica para detalles de atracciones y restaurantes.")
    return "\n".join(lineas)


@tool
def consultar_guia_turistica(consulta: str) -> str:
    """
    Busca información en las guías turísticas usando RAG.

    Es el puente entre el agente y el módulo rag.py.
    Gemini la usa para preguntas sobre destinos, comida, atracciones, etc.
    """
    return consultar_guia(consulta)


def get_tools() -> list:
    """Devuelve la lista de herramientas que el agente puede usar."""
    return [
        consultar_ciudad_argentina,
        listar_provincias_argentina,
        buscar_localidades_argentina,
        buscar_vuelos,
        crear_itinerario,
        consultar_guia_turistica,
    ]
