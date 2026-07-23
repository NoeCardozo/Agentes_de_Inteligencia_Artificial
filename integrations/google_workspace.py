"""
Integración Google Workspace: Gmail (enviar) + Calendar (crear eventos).
Autenticación OAuth2 con credentials.json + token.json.
"""
from __future__ import annotations

import base64
import json
import re
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.settings import (
    GOOGLE_CALENDAR_ID,
    GOOGLE_CALENDAR_TZ,
    GOOGLE_OAUTH_CLIENT_SECRETS,
    GOOGLE_OAUTH_TOKEN,
)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events",
]

EVENTS_JSON_RE = re.compile(
    r"EVENTS_JSON\s*:\s*(\[[\s\S]*?\])\s*(?=\n[A-Z_]+\s*:|$)",
    re.IGNORECASE,
)


def oauth_configured() -> bool:
    return Path(GOOGLE_OAUTH_CLIENT_SECRETS).exists()


def get_credentials(*, interactive: bool = False) -> Credentials:
    """
    Carga o refresca credenciales OAuth.
    Si no hay token y interactive=True, abre el navegador (solo setup).
    """
    secrets = Path(GOOGLE_OAUTH_CLIENT_SECRETS)
    token_path = Path(GOOGLE_OAUTH_TOKEN)
    if not secrets.exists():
        raise FileNotFoundError(
            f"No está {secrets}. Descargá credentials.json desde Google Cloud Console "
            "(OAuth cliente Desktop) y guardalo en esa ruta."
        )

    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")
        elif interactive:
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
            creds = flow.run_local_server(port=0)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")
        else:
            raise RuntimeError(
                "Falta autorización Google. Corré una vez: "
                "python scripts/google_oauth_setup.py"
            )
    return creds


def gmail_service():
    return build("gmail", "v1", credentials=get_credentials(), cache_discovery=False)


def calendar_service():
    return build("calendar", "v3", credentials=get_credentials(), cache_discovery=False)


def send_email(
    to: str,
    subject: str,
    body: str,
    *,
    html: bool = False,
) -> dict[str, Any]:
    """Envía un correo con la cuenta OAuth autorizada (Gmail API)."""
    if not to or "@" not in to:
        raise ValueError("Destinatario de email inválido.")
    mime = MIMEText(body, "html" if html else "plain", "utf-8")
    mime["to"] = to
    mime["subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("utf-8")
    try:
        sent = (
            gmail_service()
            .users()
            .messages()
            .send(userId="me", body={"raw": raw})
            .execute()
        )
    except HttpError as exc:
        raise RuntimeError(f"Error Gmail API: {exc}") from exc
    return {"status": "ok", "id": sent.get("id"), "to": to, "subject": subject}


def create_calendar_event(
    title: str,
    start: str,
    end: str,
    *,
    description: str = "",
    location: str = "",
    timezone: str | None = None,
    all_day: bool = False,
) -> dict[str, Any]:
    """
    Crea un evento en Google Calendar.
    start/end: ISO date (YYYY-MM-DD) si all_day, o datetime ISO (YYYY-MM-DDTHH:MM:SS).
    """
    tz = timezone or GOOGLE_CALENDAR_TZ
    if all_day or (len(start) == 10 and "T" not in start):
        body = {
            "summary": title,
            "description": description,
            "location": location,
            "start": {"date": start[:10]},
            "end": {"date": end[:10]},
        }
    else:
        body = {
            "summary": title,
            "description": description,
            "location": location,
            "start": {"dateTime": start, "timeZone": tz},
            "end": {"dateTime": end, "timeZone": tz},
        }
    try:
        created = (
            calendar_service()
            .events()
            .insert(calendarId=GOOGLE_CALENDAR_ID, body=body)
            .execute()
        )
    except HttpError as exc:
        raise RuntimeError(f"Error Calendar API: {exc}") from exc
    return {
        "status": "ok",
        "id": created.get("id"),
        "htmlLink": created.get("htmlLink"),
        "summary": created.get("summary"),
    }


def create_calendar_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Crea varios eventos. Cada ítem: title, start, end, description?, location?, all_day?"""
    results = []
    errors = []
    for i, ev in enumerate(events):
        try:
            title = (ev.get("title") or ev.get("summary") or "").strip()
            start = (ev.get("start") or "").strip()
            end = (ev.get("end") or start).strip()
            if not title or not start:
                errors.append({"index": i, "error": "Faltan title/start"})
                continue
            created = create_calendar_event(
                title=title,
                start=start,
                end=end,
                description=str(ev.get("description") or ""),
                location=str(ev.get("location") or ""),
                all_day=bool(ev.get("all_day", False)),
            )
            results.append(created)
        except Exception as exc:
            errors.append({"index": i, "error": str(exc)})
    return {
        "status": "ok" if results and not errors else ("partial" if results else "error"),
        "created": len(results),
        "failed": len(errors),
        "events": results,
        "errors": errors,
    }


def extract_events_json(text: str) -> list[dict[str, Any]]:
    """Extrae EVENTS_JSON:[...] del texto del itinerario."""
    if not text:
        return []
    m = EVENTS_JSON_RE.search(text)
    if not m:
        # fallback: buscar primer array JSON con claves típicas
        m2 = re.search(r"(\[\s*\{[\s\S]*?\}\s*\])", text)
        if not m2:
            return []
        raw = m2.group(1)
    else:
        raw = m.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def strip_events_json(text: str) -> str:
    """Oculta el bloque técnico EVENTS_JSON al usuario."""
    return re.sub(
        r"\n?EVENTS_JSON\s*:\s*\[[\s\S]*?\]\s*",
        "\n",
        text or "",
        flags=re.IGNORECASE,
    ).strip()


def events_from_itinerary_via_llm(itinerary_text: str) -> list[dict[str, Any]]:
    """
    Si no hay EVENTS_JSON, pide al LLM un listado estructurado de eventos.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    from config.settings import GOOGLE_API_KEY, ORCHESTRATOR_MODEL

    if not itinerary_text.strip():
        return []
    llm = ChatGoogleGenerativeAI(
        model=ORCHESTRATOR_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    system = SystemMessage(content=(
        "Extraé eventos de calendario de un itinerario de viaje argentino. "
        "Respondé SOLO un JSON array (sin markdown) con objetos: "
        '{"title","start","end","location","description","all_day"}. '
        "start/end en ISO local Argentina: YYYY-MM-DDTHH:MM:SS o YYYY-MM-DD si all_day=true. "
        "Si no hay fechas concretas, inventá horarios razonables dentro de los días mencionados. "
        "Máximo 20 eventos."
    ))
    human = HumanMessage(content=itinerary_text[:8000])
    raw = llm.invoke([system, human]).content
    if isinstance(raw, list):
        raw = "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in raw
        )
    raw = str(raw).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return extract_events_json(raw)
    return data if isinstance(data, list) else []
