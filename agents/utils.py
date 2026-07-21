"""Utilidades compartidas para normalizar salidas de LLM (Gemini, etc.)."""
from __future__ import annotations

import re
from datetime import date, timedelta


KNOWN_DESTINATIONS = [
    "Bariloche",
    "Mendoza",
    "Salta",
    "Iguazú",
    "Ushuaia",
    "El Calafate",
    "Córdoba",
    "Mar del Plata",
    "Puerto Madryn",
    "Cafayate",
    "Tilcara",
    "Purmamarca",
    "San Martín de los Andes",
    "Villa La Angostura",
    "El Chaltén",
    "Rosario",
    "Buenos Aires",
]


def to_text(content) -> str:
    """Convierte content de LLM (str | list[dict|str]) a texto plano."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def extract_destination(text: str, fallback: str = "") -> str:
    """Detecta un destino argentino mencionado en el texto del usuario."""
    if not text:
        return fallback
    lower = text.lower()
    for dest in KNOWN_DESTINATIONS:
        if dest.lower() in lower:
            return dest
    aliases = {
        "iguazu": "Iguazú",
        "iguazú": "Iguazú",
        "calafate": "El Calafate",
        "chalten": "El Chaltén",
        "chaltén": "El Chaltén",
        "cordoba": "Córdoba",
        "córdoba": "Córdoba",
    }
    for alias, dest in aliases.items():
        if alias in lower:
            return dest
    return fallback


def looks_like_iso_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", (value or "").strip()))


def asks_for_period_recommendation(text: str) -> bool:
    """True si el usuario pide consejo de fechas/período, no un itinerario completo."""
    lower = (text or "").lower()
    markers = [
        "qué período", "que periodo", "qué periodo", "que período",
        "qué fechas", "que fechas", "cuándo conviene", "cuando conviene",
        "qué semana", "que semana", "mejor época", "mejor epoca",
        "me recomendás", "me recomendas", "qué me recomend", "que me recomend",
        "período me recomend", "periodo me recomend",
    ]
    return any(m in lower for m in markers)


def wants_full_itinerary(text: str) -> bool:
    """True si pide armar el viaje / itinerario completo."""
    lower = (text or "").lower()
    markers = [
        "armame", "armá", "arma un", "itinerario", "plan día", "plan dia",
        "planific", "organizame", "organizá", "reserva", "presupuesto cerrado",
        "vamos con la opción", "vamos con la opcion", "eligo la opción",
        "eligo la opcion", "opción 1", "opción 2", "opción 3",
        "opcion 1", "opcion 2", "opcion 3",
    ]
    return any(m in lower for m in markers)


def is_conversation_followup(text: str) -> bool:
    """True para mensajes cortos de seguimiento (elegir opción, confirmar, etc.)."""
    lower = (text or "").lower().strip()
    markers = [
        "opción", "opcion", "vamos con", "me quedo con", "eligo", "elegí", "elegi",
        "la propuesta", "esas fechas", "ese período", "ese periodo",
        "dale", "perfecto", "ok ", "ok,", "sí,", "si,", "de acuerdo",
        "reserva", "reservar", "podés", "podes", "puedes", "ajustar", "modificar",
        "cambiar el hotel", "otro hotel", "el itinerario", "agregar", "sacar",
        "cuánto", "cuanto", "y el clima", "gracias",
    ]
    return any(m in lower for m in markers)


def has_active_trip_session(session: dict | None) -> bool:
    if not session:
        return False
    return bool(
        session.get("destino")
        or session.get("proposed_periods")
        or session.get("chosen_period")
        or session.get("last_intent")
    )


def is_new_trip_request(text: str, session: dict | None = None) -> bool:
    """True si el usuario arranca un viaje distinto / reinicia."""
    lower = (text or "").lower()
    if any(
        kw in lower
        for kw in (
            "empezar de nuevo", "otra consulta", "otro destino", "otra ciudad",
            "cambiar de destino", "olvidate", "olvídate", "nuevo viaje a",
        )
    ):
        return True
    if not session:
        return False
    dest_msg = extract_destination(text)
    dest_session = session.get("destino") or ""
    # Menciona un destino distinto y pide ir / planificar
    if dest_msg and dest_session and dest_msg.lower() != dest_session.lower():
        if any(kw in lower for kw in ("quiero ir", "viajar a", "viaje a", "planificar")):
            return True
    return False


_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def _spanish_date(day: str, month_name: str, year: str) -> str | None:
    month = _MONTHS.get(month_name.lower())
    if not month:
        return None
    try:
        return date(int(year), month, int(day)).isoformat()
    except ValueError:
        return None


def extract_period_options(text: str) -> list[dict]:
    """
    Extrae ventanas de fechas de una respuesta de recomendación.
    Soporta ISO y frases 'del 10 al 17 de diciembre de 2026'
    o 'del 28 de diciembre de 2026 al 7 de enero de 2027'.
    """
    if not text:
        return []

    # Bloque estructurado opcional
    json_match = re.search(r"PERIODOS_JSON\s*:\s*(\[[\s\S]*?\])", text)
    if json_match:
        try:
            import json as _json
            data = _json.loads(json_match.group(1))
            out = []
            for i, item in enumerate(data, start=1):
                out.append({
                    "n": int(item.get("n", i)),
                    "inicio": item.get("inicio", ""),
                    "fin": item.get("fin", ""),
                    "label": item.get("label", f"Opción {i}"),
                })
            if out:
                return out
        except Exception:
            pass

    options: list[dict] = []

    # del 28 de diciembre de 2026 al 7 de enero de 2027
    cross = re.finditer(
        r"del\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\s+al\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
        text,
        re.IGNORECASE,
    )
    for m in cross:
        inicio = _spanish_date(m.group(1), m.group(2), m.group(3))
        fin = _spanish_date(m.group(4), m.group(5), m.group(6))
        if inicio and fin:
            options.append({
                "n": len(options) + 1,
                "inicio": inicio,
                "fin": fin,
                "label": f"{inicio} a {fin}",
            })

    # del 10 al 17 de diciembre de 2026
    same_month = re.finditer(
        r"del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
        text,
        re.IGNORECASE,
    )
    for m in same_month:
        inicio = _spanish_date(m.group(1), m.group(3), m.group(4))
        fin = _spanish_date(m.group(2), m.group(3), m.group(4))
        if inicio and fin and not any(o["inicio"] == inicio and o["fin"] == fin for o in options):
            options.append({
                "n": len(options) + 1,
                "inicio": inicio,
                "fin": fin,
                "label": f"{inicio} a {fin}",
            })

    # Pares ISO consecutivos
    isos = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    for i in range(0, len(isos) - 1, 2):
        inicio, fin = isos[i], isos[i + 1]
        if not any(o["inicio"] == inicio and o["fin"] == fin for o in options):
            options.append({
                "n": len(options) + 1,
                "inicio": inicio,
                "fin": fin,
                "label": f"{inicio} a {fin}",
            })

    # Renumerar
    for i, opt in enumerate(options, start=1):
        opt["n"] = i
    return options[:5]


def resolve_option_choice(message: str, session: dict) -> tuple[str, str, str] | None:
    """
    Si el usuario elige 'opción N', retorna (mensaje_enriquecido, fecha_inicio, fecha_fin).
    """
    if not session:
        return None
    periods = session.get("proposed_periods") or []
    if not periods:
        return None

    lower = (message or "").lower()
    # Tolera typos: opción / opcion / opcióm
    m = re.search(r"opci\w*\s*(\d+)", lower)
    if not m:
        # "vamos con la 2" / "la 2"
        m = re.search(r"\b(?:la|el)\s*(\d+)\b", lower)
    if not m:
        return None

    idx = int(m.group(1)) - 1
    if idx < 0 or idx >= len(periods):
        return None

    chosen = periods[idx]
    destino = session.get("destino") or "el destino acordado"
    inicio = chosen.get("inicio", "")
    fin = chosen.get("fin", "")
    enriched = (
        f"Quiero planificar el viaje completo a {destino} "
        f"del {inicio} al {fin} (opción {idx + 1} que me recomendaste: "
        f"{chosen.get('label', '')}). "
        f"Mensaje del usuario: {message}"
    )
    return enriched, inicio, fin


def _parse_iso(value: str) -> date | None:
    if not looks_like_iso_date(value):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def resolve_trip_dates(
    fecha_inicio: str = "",
    fecha_fin: str = "",
    mensaje_usuario: str = "",
    dias: int = 7,
    today: date | None = None,
) -> tuple[str, str]:
    """
    Resuelve fechas de viaje a YYYY-MM-DD futuras.
    Interpreta frases como 'fin de año' y descarta fechas pasadas / inválidas.
    """
    today = today or date.today()
    text = " ".join([mensaje_usuario or "", fecha_inicio or "", fecha_fin or ""]).lower()

    start = _parse_iso(fecha_inicio)
    end = _parse_iso(fecha_fin)

    # Descartar fechas en el pasado (p.ej. alucinaciones 2024)
    if start and start < today:
        start = None
    if end and end < today:
        end = None

    if start is None and any(
        kw in text
        for kw in ("fin de año", "fin de ano", "año nuevo", "ano nuevo", "nochebuena", "navidad")
    ):
        year = today.year if today < date(today.year, 12, 28) else today.year + 1
        start = date(year, 12, 28)
        end = date(year + 1, 1, 4)

    if start is None and any(
        kw in text for kw in ("próxima semana", "proxima semana", "la semana que viene")
    ):
        start = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
        end = start + timedelta(days=dias - 1)

    if start is None:
        start = today + timedelta(days=14)
    if end is None:
        end = start + timedelta(days=max(dias - 1, 1))
    if end < start:
        end = start + timedelta(days=max(dias - 1, 1))

    return start.isoformat(), end.isoformat()
