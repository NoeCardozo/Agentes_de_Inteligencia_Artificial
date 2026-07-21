"""
Servidor MCP (Model Context Protocol) del Agente de Turismo Argentina.
Expone herramientas como endpoints REST que el agente consume vía MCP client.
Ejecutar con: uvicorn mcp.mcp_server:app --port 8001

Clima y atracciones: Open-Meteo + Overpass (gratuitas).
Vuelos: Travelpayouts/Aviasales (token). Hoteles: mock (Hotellook API 404).
Fallback a datos simulados si la red falla.
"""
import os
import random
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from mcp.real_apis import (
    fetch_attractions_overpass,
    fetch_flights_travelpayouts,
    fetch_weather_open_meteo,
)

load_dotenv()

app = FastAPI(title="MCP Turismo Argentina", version="1.2.0")

USE_REAL_WEATHER = os.getenv("USE_REAL_WEATHER", "true").lower() in {"1", "true", "yes"}
USE_REAL_ATTRACTIONS = os.getenv("USE_REAL_ATTRACTIONS", "true").lower() in {"1", "true", "yes"}
USE_REAL_FLIGHTS = os.getenv("USE_REAL_FLIGHTS", "true").lower() in {"1", "true", "yes"}
TRAVELPAYOUTS_TOKEN = os.getenv("TRAVELPAYOUTS_TOKEN", "")
RAIN_ALERT_MM = float(os.getenv("RAIN_ALERT_MM", "20.0"))


# ── Modelos de entrada ────────────────────────────────────────────────────────

class WeatherRequest(BaseModel):
    zona: str
    fecha_inicio: str  # YYYY-MM-DD
    fecha_fin: str     # YYYY-MM-DD

class AccommodationRequest(BaseModel):
    destino: str
    fecha_inicio: str
    fecha_fin: str
    presupuesto_max: float = 0.0  # 0 = sin límite

class AttractionRequest(BaseModel):
    destino: str
    categoria: str = ""  # aventura, cultura, gastronomia, naturaleza, etc.

class TransferRequest(BaseModel):
    origen: str
    destino: str
    fecha: str
    tipo: str = "bus"  # bus, vuelo, transfer

class ItineraryRequest(BaseModel):
    user_id: str
    destination: str
    dates: str
    budget: float
    days: list[dict]

class SQLRequest(BaseModel):
    query_type: str   # count_destinations, avg_price, etc.
    filters: dict = {}


# ── Datos simulados (reemplazar con DB real en producción) ───────────────────

WEATHER_DATA = {
    "patagonia":    {"lluvia_mm": 5.0,  "temp_max": 12, "temp_min": 4,  "descripcion": "Frío y ventoso con cielos parcialmente nublados"},
    "mendoza":      {"lluvia_mm": 2.0,  "temp_max": 22, "temp_min": 10, "descripcion": "Soleado y seco, ideal para visitar bodegas"},
    "bariloche":    {"lluvia_mm": 8.0,  "temp_max": 15, "temp_min": 5,  "descripcion": "Nublado con posibles lluvias por la tarde"},
    "salta":        {"lluvia_mm": 0.5,  "temp_max": 28, "temp_min": 14, "descripcion": "Soleado y cálido, excelente para excursiones"},
    "buenos aires": {"lluvia_mm": 15.0, "temp_max": 20, "temp_min": 12, "descripcion": "Días variables con posibles lluvias"},
    "iguazu":       {"lluvia_mm": 25.0, "temp_max": 30, "temp_min": 22, "descripcion": "Caluroso y húmedo, lluvias frecuentes"},
    "default":      {"lluvia_mm": 10.0, "temp_max": 18, "temp_min": 8,  "descripcion": "Clima moderado"},
}

ACCOMMODATIONS = {
    "bariloche": [
        {"nombre": "Hostería del Lago",  "tipo": "hosterías", "precio_noche": 45000, "estrellas": 3, "disponible": True},
        {"nombre": "Hotel Cacique",      "tipo": "hotel",     "precio_noche": 80000, "estrellas": 4, "disponible": True},
        {"nombre": "Cabañas del Río",    "tipo": "cabaña",    "precio_noche": 60000, "estrellas": 0, "disponible": False},
    ],
    "mendoza": [
        {"nombre": "Apart Vendimia",     "tipo": "apart",  "precio_noche": 35000, "estrellas": 3, "disponible": True},
        {"nombre": "Hotel Aconcagua",    "tipo": "hotel",  "precio_noche": 95000, "estrellas": 5, "disponible": True},
        {"nombre": "Hostel El Viñedo",   "tipo": "hostel", "precio_noche": 12000, "estrellas": 0, "disponible": True},
    ],
    "salta": [
        {"nombre": "Posada de la Viña",  "tipo": "posada", "precio_noche": 30000, "estrellas": 3, "disponible": True},
        {"nombre": "Hotel del Norte",    "tipo": "hotel",  "precio_noche": 55000, "estrellas": 4, "disponible": True},
    ],
}

ATTRACTIONS = {
    "bariloche": [
        {"nombre": "Circuito Chico",         "categoria": "naturaleza", "precio": 0,     "horario": "09:00-18:00", "duracion_hs": 4},
        {"nombre": "Cerro Catedral",         "categoria": "aventura",   "precio": 28000, "horario": "09:00-17:00", "duracion_hs": 6},
        {"nombre": "Isla Victoria",          "categoria": "naturaleza", "precio": 15000, "horario": "10:00-16:00", "duracion_hs": 5},
        {"nombre": "Centro Cívico",          "categoria": "cultura",    "precio": 0,     "horario": "00:00-23:59", "duracion_hs": 2},
    ],
    "mendoza": [
        {"nombre": "Ruta del Vino",          "categoria": "gastronomia","precio": 20000, "horario": "10:00-18:00", "duracion_hs": 5},
        {"nombre": "Aconcagua (base)",       "categoria": "aventura",   "precio": 5000,  "horario": "08:00-16:00", "duracion_hs": 8},
        {"nombre": "Parque Gral. San Martín","categoria": "naturaleza", "precio": 0,     "horario": "07:00-21:00", "duracion_hs": 3},
    ],
    "salta": [
        {"nombre": "Tren a las Nubes",       "categoria": "cultura",    "precio": 45000, "horario": "07:00-22:00", "duracion_hs": 15},
        {"nombre": "Quebrada de Humahuaca",  "categoria": "cultura",    "precio": 3000,  "horario": "08:00-18:00", "duracion_hs": 6},
        {"nombre": "Cachi y Los Cardones",   "categoria": "naturaleza", "precio": 4000,  "horario": "08:00-17:00", "duracion_hs": 7},
    ],
}

TRANSFER_PRICES = {
    ("buenos aires", "bariloche", "vuelo"): 120000,
    ("buenos aires", "mendoza",   "vuelo"):  85000,
    ("buenos aires", "salta",     "vuelo"):  95000,
    ("buenos aires", "bariloche", "bus"):    45000,
    ("buenos aires", "mendoza",   "bus"):    38000,
}

_saved_itineraries: list[dict] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/tools/weather")
def get_weather(req: WeatherRequest) -> dict:
    """Pronóstico climático: Open-Meteo real, con fallback a mock."""
    if USE_REAL_WEATHER:
        real = fetch_weather_open_meteo(
            req.zona, req.fecha_inicio, req.fecha_fin, rain_alert_mm=RAIN_ALERT_MM,
        )
        if real:
            return real

    zona_key = req.zona.lower()
    data = WEATHER_DATA.get(zona_key, WEATHER_DATA["default"])
    # match parcial (ej. "San Carlos de Bariloche")
    if data is WEATHER_DATA["default"]:
        for key, val in WEATHER_DATA.items():
            if key != "default" and key in zona_key:
                data = val
                break
    return {
        "zona": req.zona,
        "fecha_inicio": req.fecha_inicio,
        "fecha_fin": req.fecha_fin,
        **data,
        "alerta": data["lluvia_mm"] >= RAIN_ALERT_MM,
        "fuente": "mock-local",
    }


@app.post("/tools/accommodations")
def get_accommodations(req: AccommodationRequest) -> dict:
    """Lista alojamientos disponibles con precios (datos de ejemplo)."""
    destino_key = req.destino.lower()
    options = ACCOMMODATIONS.get(destino_key, [])
    if not options:
        for key, val in ACCOMMODATIONS.items():
            if key in destino_key:
                options = val
                break

    if req.presupuesto_max > 0:
        options = [o for o in options if o["precio_noche"] <= req.presupuesto_max]

    available = [o for o in options if o["disponible"]]
    return {
        "destino": req.destino,
        "fecha_inicio": req.fecha_inicio,
        "fecha_fin": req.fecha_fin,
        "opciones": available,
        "total_opciones": len(available),
        "fuente": "mock-local",
    }


@app.post("/tools/attractions")
def get_attractions(req: AttractionRequest) -> dict:
    """Atracciones: OpenStreetMap/Overpass real, con fallback a mock."""
    if USE_REAL_ATTRACTIONS:
        real = fetch_attractions_overpass(req.destino, req.categoria)
        if real:
            return {
                "destino": req.destino,
                "categoria_filtro": req.categoria or "todas",
                "actividades": real,
                "fuente": "openstreetmap-overpass",
            }

    destino_key = req.destino.lower()
    attractions = ATTRACTIONS.get(destino_key, [])
    if not attractions:
        for key, val in ATTRACTIONS.items():
            if key in destino_key:
                attractions = val
                break

    if req.categoria:
        attractions = [a for a in attractions if a["categoria"] == req.categoria.lower()]

    return {
        "destino": req.destino,
        "categoria_filtro": req.categoria or "todas",
        "actividades": attractions,
        "fuente": "mock-local",
    }


@app.post("/tools/transfers")
def get_transfers(req: TransferRequest) -> dict:
    """Retorna opciones y precios de traslados (vuelos reales vía Travelpayouts)."""
    tipo = (req.tipo or "bus").lower().strip()
    tipo_mock = "vuelo" if tipo in {"vuelo", "avion", "avión", "flight"} else tipo

    if tipo_mock == "vuelo" and USE_REAL_FLIGHTS and TRAVELPAYOUTS_TOKEN:
        real = fetch_flights_travelpayouts(
            req.origen, req.destino, req.fecha, token=TRAVELPAYOUTS_TOKEN,
        )
        if real:
            return real

    key = (req.origen.lower(), req.destino.lower(), tipo_mock)
    precio = TRANSFER_PRICES.get(key)

    if precio is None:
        precio = random.randint(20000, 150000)

    return {
        "origen": req.origen,
        "destino": req.destino,
        "fecha": req.fecha,
        "tipo": tipo_mock,
        "precio": precio,
        "disponible": True,
        "duracion_estimada_hs": 2 if tipo_mock == "vuelo" else 14,
        "fuente": "mock-local",
    }


@app.post("/tools/save_itinerary")
def save_itinerary(req: ItineraryRequest) -> dict:
    """Persiste un itinerario generado."""
    entry = {
        "saved_at": datetime.now().isoformat(),
        "user_id": req.user_id,
        "destination": req.destination,
        "dates": req.dates,
        "budget": req.budget,
        "days": req.days,
    }
    _saved_itineraries.append(entry)
    return {"status": "ok", "id": len(_saved_itineraries) - 1}


@app.post("/tools/query")
def run_query(req: SQLRequest) -> dict:
    """Consultas de gestión y agregaciones."""
    if req.query_type == "count_destinations":
        return {"total": len(ACCOMMODATIONS)}
    if req.query_type == "avg_price":
        destino = req.filters.get("destino", "").lower()
        options = ACCOMMODATIONS.get(destino, [])
        if options:
            avg = sum(o["precio_noche"] for o in options) / len(options)
            return {"destino": destino, "precio_promedio_noche": avg}
        return {"error": "Destino no encontrado"}
    if req.query_type == "list_destinations":
        return {"destinos": list(ACCOMMODATIONS.keys())}
    return {"error": f"query_type '{req.query_type}' no reconocido"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "MCP Turismo Argentina",
        "use_real_weather": USE_REAL_WEATHER,
        "use_real_attractions": USE_REAL_ATTRACTIONS,
        "use_real_flights": USE_REAL_FLIGHTS and bool(TRAVELPAYOUTS_TOKEN),
    }
