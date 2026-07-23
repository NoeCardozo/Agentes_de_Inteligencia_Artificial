"""Tools LangChain para Gmail y Google Calendar."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from config.settings import USER_EMAIL
from integrations.email_format import itinerary_to_email_html, polish_agent_reply
from integrations.google_workspace import (
    create_calendar_events,
    oauth_configured,
    send_email,
)


@tool
def tool_enviar_email(to: str, subject: str, body: str) -> str:
    """
    Envía un correo electrónico con el itinerario u otra información del viaje.
    Parámetros: to (email destino), subject (asunto), body (itinerario en texto/markdown).
    Si to está vacío, usa USER_EMAIL configurado en el entorno.
    """
    if not oauth_configured():
        return (
            "Google OAuth no configurado. Colocá credentials/credentials.json "
            "y ejecutá: python scripts/google_oauth_setup.py"
        )
    dest = (to or "").strip() or (USER_EMAIL or "").strip()
    if not dest:
        return (
            "No hay destinatario. Pedile al usuario su email o configurá USER_EMAIL en .env."
        )
    try:
        polished = polish_agent_reply(body)
        html_body = itinerary_to_email_html(polished)
        result = send_email(
            dest,
            subject.strip() or "Tu itinerario de viaje — Agente Turismo Argentina",
            html_body,
            html=True,
        )
        return f"Email enviado a {result['to']} (id={result['id']})."
    except Exception as exc:
        return f"No se pudo enviar el email: {exc}"


@tool
def tool_crear_eventos_calendario(events_json: str) -> str:
    """
    Crea eventos en Google Calendar a partir de un JSON array.
    Cada evento: title, start, end, location (opcional), description (opcional),
    all_day (opcional bool). start/end: YYYY-MM-DDTHH:MM:SS o YYYY-MM-DD.
    Ejemplo: [{"title":"Trekking Cerro Campanario","start":"2026-09-15T09:00:00",
    "end":"2026-09-15T12:00:00","location":"Bariloche"}]
    """
    if not oauth_configured():
        return (
            "Google OAuth no configurado. Colocá credentials/credentials.json "
            "y ejecutá: python scripts/google_oauth_setup.py"
        )
    try:
        events = json.loads(events_json)
    except json.JSONDecodeError as exc:
        return f"JSON inválido en events_json: {exc}"
    if not isinstance(events, list) or not events:
        return "events_json debe ser un array JSON no vacío de eventos."
    try:
        result = create_calendar_events(events)
    except Exception as exc:
        return f"No se pudieron crear eventos: {exc}"
    lines = [
        f"Calendario: {result['created']} creados, {result['failed']} fallidos "
        f"(status={result['status']})."
    ]
    for ev in result.get("events") or []:
        link = ev.get("htmlLink") or ""
        lines.append(f"  - {ev.get('summary')}: {link}")
    for err in result.get("errors") or []:
        lines.append(f"  ! error[{err.get('index')}]: {err.get('error')}")
    return "\n".join(lines)


DELIVERY_TOOLS = [tool_enviar_email, tool_crear_eventos_calendario]
