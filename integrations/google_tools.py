"""Tools LangChain para envío de email (Resend) y Google Calendar."""
from __future__ import annotations

import json
import re

from langchain_core.tools import tool

from integrations.email_format import itinerary_to_email_html, polish_agent_reply
from integrations.google_workspace import (
    create_calendar_events,
    oauth_configured,
)
from integrations.resend_email import (
    is_valid_email,
    resend_configured,
    send_email,
)

_DAY1_IN_BODY = re.compile(r"(?im)(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*\*)?d[ií]a\s*1\b")


@tool
def tool_enviar_email(to: str, subject: str, body: str) -> str:
    """
    Envía por email el itinerario de turismo a la dirección que indicó el usuario.
    Parámetros: to (email destino OBLIGATORIO indicado por el usuario),
    subject (asunto), body (itinerario COMPLETO en texto/markdown, con Día 1…).
    No uses un email inventado ni uno por defecto: si el usuario no dio email, pedíselo.
    """
    if not resend_configured():
        return (
            "Envío de email no configurado. Definí RESEND_API_KEY y RESEND_FROM en .env "
            "(ver https://resend.com)."
        )
    dest = (to or "").strip()
    if not dest:
        return (
            "No hay destinatario. Preguntá al usuario: "
            "'¿A qué email querés que te envíe el itinerario?'"
        )
    if not is_valid_email(dest):
        return f"El email '{dest}' no es válido. Pedile al usuario una dirección correcta."
    polished = polish_agent_reply(body or "")
    if not _DAY1_IN_BODY.search(polished):
        return (
            "No envié el email: el body no parece un itinerario día a día "
            "(falta 'Día 1'). Generá primero el plan completo y reintentá."
        )
    try:
        html_body = itinerary_to_email_html(polished)
        result = send_email(
            dest,
            subject.strip() or "Tu itinerario de viaje — Agente Turismo Argentina",
            html_body,
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
