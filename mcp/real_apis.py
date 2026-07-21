"""
Integraciones con APIs reales:
  - Open-Meteo Geocoding + Forecast / Archive (clima, sin key)
  - Overpass API / OpenStreetMap (atracciones, sin key)
  - Travelpayouts / Aviasales (vuelos baratos; requiere token)

Si falla la red o no hay resultados, el caller usa el mock local.
Nota: la API pública de Hotellook (hoteles) responde 404; hoteles siguen en mock.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import httpx

USER_AGENT = "AgenteTurismoArgentina/1.0 (proyecto-educativo)"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TRAVELPAYOUTS_CHEAP_URL = "https://api.travelpayouts.com/v1/prices/cheap"

# Códigos IATA frecuentes en Argentina (ciudad → aeropuerto/metro)
_IATA_AR: dict[str, str] = {
    "buenos aires": "BUE",
    "caba": "BUE",
    "capital federal": "BUE",
    "ezeiza": "EZE",
    "aeroparque": "AEP",
    "bariloche": "BRC",
    "san carlos de bariloche": "BRC",
    "mendoza": "MDZ",
    "salta": "SLA",
    "cordoba": "COR",
    "córdoba": "COR",
    "iguazu": "IGR",
    "iguazú": "IGR",
    "puerto iguazu": "IGR",
    "puerto iguazú": "IGR",
    "ushuaia": "USH",
    "el calafate": "FTE",
    "calafate": "FTE",
    "trelew": "REL",
    "neuquen": "NQN",
    "neuquén": "NQN",
    "mar del plata": "MDQ",
    "rosario": "ROS",
    "tucuman": "TUC",
    "tucumán": "TUC",
    "san miguel de tucuman": "TUC",
    "posadas": "PSS",
    "jujuy": "JUJ",
    "san salvador de jujuy": "JUJ",
    "comodoro rivadavia": "CRD",
    "rio gallegos": "RGL",
    "río gallegos": "RGL",
    "san martin de los andes": "CPC",
    "san martín de los andes": "CPC",
    "chapelco": "CPC",
    "esquel": "EQS",
    "resistencia": "RES",
    "corrientes": "CNQ",
    "bahia blanca": "BHI",
    "bahía blanca": "BHI",
    "la rioja": "IRJ",
    "catamarca": "CTC",
    "santiago del estero": "SDE",
    "formosa": "FMA",
    "rio cuarto": "RCU",
    "río cuarto": "RCU",
}

# WMO Weather interpretation codes → texto en español
_WMO_ES = {
    0: "Despejado",
    1: "Mayormente despejado",
    2: "Parcialmente nublado",
    3: "Nublado",
    45: "Niebla",
    48: "Niebla con escarcha",
    51: "Llovizna ligera",
    53: "Llovizna moderada",
    55: "Llovizna intensa",
    61: "Lluvia ligera",
    63: "Lluvia moderada",
    65: "Lluvia intensa",
    71: "Nevada ligera",
    73: "Nevada moderada",
    75: "Nevada intensa",
    80: "Chubascos ligeros",
    81: "Chubascos moderados",
    82: "Chubascos intensos",
    85: "Chubascos de nieve ligeros",
    86: "Chubascos de nieve intensos",
    95: "Tormenta",
    96: "Tormenta con granizo",
    99: "Tormenta fuerte con granizo",
}

_CATEGORY_TAGS = {
    "naturaleza": '["tourism"="viewpoint"];["leisure"~"park|nature_reserve"];["natural"~"peak|beach|water"]',
    "cultura": '["tourism"~"museum|gallery|artwork"];["historic"]',
    "aventura": '["tourism"~"attraction|theme_park|ski"];["sport"]',
    "gastronomia": '["amenity"~"restaurant|cafe|winery"];["craft"="winery"]',
}


def geocode_argentina(lugar: str) -> dict[str, Any] | None:
    """Resuelve un lugar de Argentina a lat/lon vía Open-Meteo Geocoding."""
    query = lugar.strip()
    if not query:
        return None
    # Ayuda a desambiguar (ej. Mendoza, Córdoba)
    if "argentina" not in query.lower():
        query = f"{query}, Argentina"

    try:
        with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
            r = client.get(
                GEOCODE_URL,
                params={
                    "name": query,
                    "count": 5,
                    "language": "es",
                    "format": "json",
                },
            )
            r.raise_for_status()
            results = r.json().get("results") or []
    except httpx.HTTPError:
        return None

    # Preferir resultados en AR
    for item in results:
        if (item.get("country_code") or "").upper() == "AR":
            return {
                "nombre": item.get("name", lugar),
                "lat": item["latitude"],
                "lon": item["longitude"],
                "admin1": item.get("admin1", ""),
            }
    if results:
        item = results[0]
        return {
            "nombre": item.get("name", lugar),
            "lat": item["latitude"],
            "lon": item["longitude"],
            "admin1": item.get("admin1", ""),
        }
    return None


def _describe_codes(codes: list[int]) -> str:
    if not codes:
        return "Sin datos"
    # Código más frecuente
    freq: dict[int, int] = {}
    for c in codes:
        freq[c] = freq.get(c, 0) + 1
    top = max(freq, key=freq.get)
    return _WMO_ES.get(top, f"Código climático {top}")


def _fetch_daily_weather(lat: float, lon: float, start: str, end: str, url: str) -> dict | None:
    try:
        with httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}) as client:
            r = client.get(
                url,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start,
                    "end_date": end,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                    "timezone": "America/Argentina/Buenos_Aires",
                },
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError:
        return None


def fetch_weather_open_meteo(
    zona: str,
    fecha_inicio: str,
    fecha_fin: str,
    rain_alert_mm: float = 20.0,
) -> dict[str, Any] | None:
    """
    Pronóstico real (si está dentro de ~16 días) o estimación estacional
    con datos del archivo histórico del mismo período del año anterior.
    """
    geo = geocode_argentina(zona)
    if not geo:
        return None

    try:
        start = date.fromisoformat(fecha_inicio)
        end = date.fromisoformat(fecha_fin)
    except ValueError:
        return None
    if end < start:
        end = start

    today = date.today()
    horizon = today + timedelta(days=15)
    fuente = "open-meteo-forecast"

    if start <= horizon:
        # Acotar fin al horizonte de forecast
        end_q = min(end, horizon)
        start_q = max(start, today)
        raw = _fetch_daily_weather(geo["lat"], geo["lon"], start_q.isoformat(), end_q.isoformat(), FORECAST_URL)
    else:
        # Fechas lejanas: usar mismo rango del año anterior (estimación de temporada)
        try:
            start_h = start.replace(year=today.year - 1)
            end_h = end.replace(year=today.year - 1 if end.year == start.year else today.year)
        except ValueError:
            # 29 feb etc.
            start_h = start.replace(year=today.year - 1, day=28)
            end_h = end.replace(year=today.year - 1, day=28)
        raw = _fetch_daily_weather(geo["lat"], geo["lon"], start_h.isoformat(), end_h.isoformat(), ARCHIVE_URL)
        fuente = "open-meteo-archive-estacional"

    if not raw or "daily" not in raw:
        return None

    daily = raw["daily"]
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    precip = daily.get("precipitation_sum") or []
    codes = daily.get("weathercode") or []

    if not tmax:
        return None

    lluvia_avg = round(sum(precip) / len(precip), 1) if precip else 0.0
    return {
        "zona": geo["nombre"],
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
        "temp_max": round(max(tmax)),
        "temp_min": round(min(tmin)),
        "lluvia_mm": lluvia_avg,
        "descripcion": _describe_codes([int(c) for c in codes if c is not None]),
        "alerta": lluvia_avg >= rain_alert_mm,
        "fuente": fuente,
        "coordenadas": {"lat": geo["lat"], "lon": geo["lon"]},
    }


def _tag_to_categoria(tags: dict) -> str:
    tourism = (tags.get("tourism") or "").lower()
    leisure = (tags.get("leisure") or "").lower()
    amenity = (tags.get("amenity") or "").lower()
    if tourism in {"museum", "gallery", "artwork"} or tags.get("historic"):
        return "cultura"
    if amenity in {"restaurant", "cafe", "bar", "fast_food"} or tourism == "winery":
        return "gastronomia"
    if tourism in {"attraction", "theme_park", "ski"} or tags.get("sport"):
        return "aventura"
    if leisure in {"park", "nature_reserve"} or tourism in {"viewpoint", "zoo"}:
        return "naturaleza"
    return "naturaleza"


def fetch_attractions_overpass(
    destino: str,
    categoria: str = "",
    radio_m: int = 12000,
    limit: int = 12,
) -> list[dict[str, Any]] | None:
    """Busca atracciones reales cerca del destino vía Overpass (OpenStreetMap)."""
    geo = geocode_argentina(destino)
    if not geo:
        return None

    lat, lon = geo["lat"], geo["lon"]
    cat = (categoria or "").lower().strip()

    if cat == "cultura":
        filters = (
            f'node["tourism"~"museum|gallery|artwork"](around:{radio_m},{lat},{lon});'
            f'node["historic"](around:{radio_m},{lat},{lon});'
            f'way["tourism"~"museum|gallery"](around:{radio_m},{lat},{lon});'
        )
    elif cat == "gastronomia":
        filters = (
            f'node["amenity"~"restaurant|cafe|winery"](around:{radio_m},{lat},{lon});'
            f'node["tourism"="winery"](around:{radio_m},{lat},{lon});'
        )
    elif cat == "aventura":
        filters = (
            f'node["tourism"~"attraction|theme_park"](around:{radio_m},{lat},{lon});'
            f'node["sport"](around:{radio_m},{lat},{lon});'
            f'way["tourism"="attraction"](around:{radio_m},{lat},{lon});'
        )
    else:
        # naturaleza / todas
        filters = (
            f'node["tourism"~"attraction|viewpoint|museum|zoo|theme_park"](around:{radio_m},{lat},{lon});'
            f'node["leisure"~"park|nature_reserve"](around:{radio_m},{lat},{lon});'
            f'way["tourism"~"attraction|museum"](around:{radio_m},{lat},{lon});'
            f'way["leisure"="park"](around:{radio_m},{lat},{lon});'
        )

    query = f"[out:json][timeout:25];({filters});out center tags {limit};"

    try:
        with httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
            r = client.post(OVERPASS_URL, data={"data": query})
            r.raise_for_status()
            elements = r.json().get("elements") or []
    except (httpx.HTTPError, ValueError):
        return None

    activities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for el in elements:
        tags = el.get("tags") or {}
        nombre = tags.get("name") or tags.get("name:es")
        if not nombre or nombre.lower() in seen:
            continue
        seen.add(nombre.lower())
        cat_item = _tag_to_categoria(tags)
        if cat and cat not in {"todas", "all"} and cat_item != cat and cat != "naturaleza":
            # Si filtramos naturaleza, permitir viewpoint/park ya incluidos
            if cat == "naturaleza" and cat_item != "naturaleza":
                continue
            if cat not in ("",) and cat_item != cat:
                continue
        activities.append({
            "nombre": nombre,
            "categoria": cat_item,
            "precio": 0,  # OSM no trae precio; gratuito/desconocido
            "horario": tags.get("opening_hours", "consultar"),
            "duracion_hs": 2,
            "fuente": "openstreetmap-overpass",
        })
        if len(activities) >= limit:
            break

    return activities if activities else None


def resolve_iata(lugar: str) -> str | None:
    """Mapea un nombre de ciudad argentina a código IATA (si está en la tabla)."""
    key = (lugar or "").strip().lower()
    if not key:
        return None
    if len(key) == 3 and key.isalpha():
        return key.upper()
    if key in _IATA_AR:
        return _IATA_AR[key]
    for nombre, code in _IATA_AR.items():
        if nombre in key or key in nombre:
            return code
    return None


def fetch_flights_travelpayouts(
    origen: str,
    destino: str,
    fecha: str,
    token: str,
    currency: str = "ars",
) -> dict[str, Any] | None:
    """
    Precio aproximado de vuelo vía Travelpayouts (Aviasales) /v1/prices/cheap.
    `fecha` puede ser YYYY-MM-DD o YYYY-MM; se usa el mes de salida.
    """
    if not token:
        return None
    origin = resolve_iata(origen)
    dest = resolve_iata(destino)
    if not origin or not dest:
        return None

    fecha = (fecha or "").strip()
    depart_date = fecha[:7] if len(fecha) >= 7 else fecha  # YYYY-MM

    try:
        with httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}) as client:
            r = client.get(
                TRAVELPAYOUTS_CHEAP_URL,
                params={
                    "origin": origin,
                    "destination": dest,
                    "depart_date": depart_date,
                    "currency": currency,
                    "token": token,
                },
            )
            r.raise_for_status()
            payload = r.json()
    except (httpx.HTTPError, ValueError):
        return None

    if not payload.get("success"):
        return None

    data = payload.get("data") or {}
    # data: { "BRC": { "0": {...}, "1": {...} } } o similar
    best: dict[str, Any] | None = None
    for _dest_key, flights in data.items():
        if not isinstance(flights, dict):
            continue
        for _k, flight in flights.items():
            if not isinstance(flight, dict) or "price" not in flight:
                continue
            if best is None or flight["price"] < best["price"]:
                best = flight

    if not best:
        return None

    duration_min = best.get("duration_to") or best.get("duration") or 0
    try:
        duration_hs = round(float(duration_min) / 60.0, 1)
    except (TypeError, ValueError):
        duration_hs = 2.0

    airline = best.get("airline") or ""
    flight_number = best.get("flight_number")
    vuelo_label = f"{airline}{flight_number}" if airline and flight_number else (airline or "consultar")

    return {
        "origen": origen,
        "destino": destino,
        "origen_iata": origin,
        "destino_iata": dest,
        "fecha": fecha,
        "tipo": "vuelo",
        "precio": int(best["price"]),
        "disponible": True,
        "duracion_estimada_hs": duration_hs,
        "aerolinea": airline,
        "vuelo": vuelo_label,
        "salida": best.get("departure_at", ""),
        "moneda": (payload.get("currency") or currency).upper(),
        "fuente": "travelpayouts-aviasales",
    }
