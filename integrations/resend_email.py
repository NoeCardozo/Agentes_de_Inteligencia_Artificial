"""
Envío de emails con Resend (https://resend.com).
Usa la API HTTP directamente vía httpx (sin SDK extra).
Permite enviar el itinerario a cualquier dirección de correo.
"""
from __future__ import annotations

import re
from typing import Any

import httpx

from config.settings import RESEND_API_KEY, RESEND_FROM

RESEND_ENDPOINT = "https://api.resend.com/emails"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def resend_configured() -> bool:
    """True si hay API key y remitente configurados."""
    return bool(RESEND_API_KEY and RESEND_FROM)


def is_valid_email(value: str) -> bool:
    return bool(value and _EMAIL_RE.fullmatch(value.strip()))


def find_email(text: str) -> str:
    """Devuelve el primer email válido encontrado en el texto, o ''."""
    if not text:
        return ""
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else ""


def send_email(
    to: str,
    subject: str,
    html: str,
    *,
    reply_to: str | None = None,
) -> dict[str, Any]:
    """
    Envía un email HTML con Resend.
    Lanza RuntimeError/ValueError ante configuración o datos inválidos.
    """
    if not resend_configured():
        raise RuntimeError(
            "Resend no está configurado. Definí RESEND_API_KEY y RESEND_FROM en .env."
        )
    dest = (to or "").strip()
    if not is_valid_email(dest):
        raise ValueError(f"Destinatario de email inválido: '{to}'")

    payload: dict[str, Any] = {
        "from": RESEND_FROM,
        "to": [dest],
        "subject": subject or "Tu itinerario de viaje",
        "html": html,
    }
    if reply_to:
        payload["reply_to"] = reply_to

    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                RESEND_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Error de red al llamar a Resend: {exc}") from exc

    if resp.status_code >= 400:
        detail = resp.text
        try:
            data = resp.json()
            detail = data.get("message") or data.get("error") or detail
        except Exception:
            pass
        raise RuntimeError(f"Resend respondió {resp.status_code}: {detail}")

    data = resp.json()
    return {"status": "ok", "id": data.get("id"), "to": dest, "subject": subject}
