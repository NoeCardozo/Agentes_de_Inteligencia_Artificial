"""
Cliente MCP del Agente de Turismo Argentina.
Expone las tools del servidor MCP como funciones Python invocables por LangChain.
"""
import httpx
from langchain_core.tools import tool
from config.settings import MCP_HOST


def _post(endpoint: str, payload: dict) -> dict:
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(f"{MCP_HOST}{endpoint}", json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        return {
            "error": True,
            "mensaje": (
                f"No se pudo consultar el servidor MCP ({MCP_HOST}{endpoint}): {exc}. "
                "Asegurate de tenerlo levantado con: uvicorn mcp.mcp_server:app --port 8001"
            ),
        }


# ── Tools LangChain ────────────────────────────────────────────────────────────

@tool
def tool_clima(zona: str, fecha_inicio: str, fecha_fin: str) -> str:
    """
    Consulta el pronóstico climático extendido para una zona de Argentina.
    Parámetros: zona (ej: 'Bariloche'), fecha_inicio y fecha_fin en formato YYYY-MM-DD.
    Retorna temperatura, precipitaciones esperadas y si hay alerta climática.
    """
    result = _post("/tools/weather", {
        "zona": zona,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
    })
    if result.get("error"):
        return result["mensaje"]
    alerta = "ALERTA: precipitaciones importantes esperadas." if result.get("alerta") else ""
    fuente = result.get("fuente", "desconocida")
    return (
        f"Clima en {result.get('zona', zona)} ({fecha_inicio} al {fecha_fin}):\n"
        f"- {result.get('descripcion', 'Sin descripción')}\n"
        f"- Temperatura: {result.get('temp_min', '?')}°C a {result.get('temp_max', '?')}°C\n"
        f"- Precipitaciones: {result.get('lluvia_mm', '?')} mm/día\n"
        f"- Fuente de datos: {fuente}\n"
        f"{alerta}"
    )


@tool
def tool_alojamiento(destino: str, fecha_inicio: str, fecha_fin: str, presupuesto_max: float = 0.0) -> str:
    """
    Busca alojamientos disponibles en un destino argentino para las fechas indicadas.
    Parámetros: destino, fecha_inicio, fecha_fin (YYYY-MM-DD), presupuesto_max por noche (0 = sin límite).
    Retorna listado con nombre, tipo, precio por noche y disponibilidad.
    """
    result = _post("/tools/accommodations", {
        "destino": destino,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
        "presupuesto_max": presupuesto_max,
    })
    if result.get("error"):
        return result["mensaje"]
    if not result.get("opciones"):
        return f"No hay alojamientos disponibles en {destino} para esas fechas con el presupuesto indicado."

    lines = [f"Alojamientos disponibles en {destino}:"]
    for opt in result["opciones"]:
        estrellas = f" ({opt['estrellas']}★)" if opt.get('estrellas') else ""
        lines.append(f"  - {opt['nombre']}{estrellas}: ${opt['precio_noche']:,}/noche ({opt['tipo']})")
    return "\n".join(lines)


@tool
def tool_actividades(destino: str, categoria: str = "") -> str:
    """
    Lista atracciones y actividades disponibles en un destino argentino.
    Parámetros: destino, categoria opcional (aventura, cultura, gastronomia, naturaleza).
    Retorna nombre, precio de entrada y horarios.
    """
    result = _post("/tools/attractions", {
        "destino": destino,
        "categoria": categoria,
    })
    if result.get("error"):
        return result["mensaje"]
    if not result.get("actividades"):
        return f"No se encontraron actividades en {destino} para la categoría '{categoria}'."

    fuente = result.get("fuente", "desconocida")
    lines = [f"Actividades en {destino} (fuente: {fuente}):"]
    for act in result["actividades"]:
        precio_val = act.get("precio", 0)
        if precio_val and precio_val > 0:
            precio = f"${precio_val:,}"
        elif fuente.startswith("openstreetmap"):
            precio = "consultar / variable"
        else:
            precio = "Gratuito"
        lines.append(
            f"  - {act['nombre']} [{act['categoria']}]: {precio}, "
            f"horario {act.get('horario', 'consultar')}, "
            f"duración ~{act.get('duracion_hs', 2)}hs"
        )
    return "\n".join(lines)


@tool
def tool_traslados(origen: str, destino: str, fecha: str, tipo: str = "bus") -> str:
    """
    Consulta tarifas y disponibilidad de traslados entre ciudades argentinas.
    Parámetros: origen, destino, fecha (YYYY-MM-DD), tipo ('bus' o 'vuelo').
    Retorna precio estimado y duración del viaje.
    """
    result = _post("/tools/transfers", {
        "origen": origen,
        "destino": destino,
        "fecha": fecha,
        "tipo": tipo,
    })
    if result.get("error"):
        return result["mensaje"]
    if not result.get("disponible"):
        return f"No hay {tipo} disponible de {origen} a {destino} para el {fecha}."

    moneda = result.get("moneda", "ARS")
    fuente = result.get("fuente", "mock-local")
    lines = [
        f"Traslado {result.get('tipo', tipo)} de {origen} a {destino} ({fecha}):",
        f"  Precio: ${result['precio']:,} {moneda}",
        f"  Duración estimada: {result['duracion_estimada_hs']}hs",
    ]
    if result.get("vuelo"):
        lines.append(f"  Vuelo: {result['vuelo']}")
    if result.get("salida"):
        lines.append(f"  Salida: {result['salida']}")
    if result.get("origen_iata") and result.get("destino_iata"):
        lines.append(f"  Ruta IATA: {result['origen_iata']} → {result['destino_iata']}")
    lines.append(f"  Fuente: {fuente}")
    return "\n".join(lines)


@tool
def tool_guardar_itinerario(user_id: str, destination: str, dates: str, budget: float, days_json: str) -> str:
    """
    Guarda el itinerario final generado en la base de datos.
    Parámetros: user_id, destination, dates (descripción), budget (monto), days_json (JSON string con el plan).
    """
    import json as _json
    try:
        days = _json.loads(days_json)
    except Exception:
        days = []

    result = _post("/tools/save_itinerary", {
        "user_id": user_id,
        "destination": destination,
        "dates": dates,
        "budget": budget,
        "days": days,
    })
    if result.get("error"):
        return result["mensaje"]
    if result.get("status") == "ok":
        return f"Itinerario guardado correctamente con ID #{result['id']}."
    return "Error al guardar el itinerario."


@tool
def tool_consulta_gestion(query_type: str, filters_json: str = "{}") -> str:
    """
    Ejecuta consultas de gestión y agregaciones sobre los datos de destinos.
    query_type puede ser: 'count_destinations', 'avg_price', 'list_destinations'.
    filters_json: JSON opcional con filtros (ej: '{\"destino\": \"mendoza\"}').
    """
    import json as _json
    try:
        filters = _json.loads(filters_json)
    except Exception:
        filters = {}

    result = _post("/tools/query", {
        "query_type": query_type,
        "filters": filters,
    })
    if result.get("error"):
        return result["mensaje"]
    return str(result)


# Lista de todas las tools disponibles para el agente
ALL_MCP_TOOLS = [
    tool_clima,
    tool_alojamiento,
    tool_actividades,
    tool_traslados,
    tool_guardar_itinerario,
    tool_consulta_gestion,
]
