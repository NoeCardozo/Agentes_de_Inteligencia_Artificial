"""Formato de emails de itinerario y limpieza de respuestas del agente."""
from __future__ import annotations

import re

import markdown as md_lib


_THOUGHT_MARKERS = re.compile(
    r"(?is)("
    r"he recopilado|ahora,? voy a|voy a (leer|diseñar|consultar|buscar|armar|planificar|actualizar)|"
    r"primero,? necesito|updated todo list|write_todos|tool_clima|tool_alojamiento|"
    r"successfully (replaced|wrote|updated)|"
    r"\*\*aventura:\*\*|\*\*cultura:\*\*|\*\*gastronom[ií]a:\*\*|\*\*naturaleza:\*\*"
    r")"
)

_DAY1_RE = re.compile(
    r"(?im)^(?:#{1,3}\s*)?(?:\*\*)?d[ií]a\s*1\b"
)


def polish_agent_reply(text: str) -> str:
    """
    Quita cadenas de pensamiento / dumps de tools y deja el mensaje útil.
    Si hay itinerario día a día, prioriza desde Día 1 con un saludo corto.
    """
    if not text or not text.strip():
        return text or ""

    text = text.strip()
    day1 = _DAY1_RE.search(text)
    if day1:
        head = text[: day1.start()].strip()
        body = text[day1.start() :].strip()
        greeting = ""
        if head and not _THOUGHT_MARKERS.search(head):
            # Primer párrafo corto como saludo
            para = re.split(r"\n\s*\n", head)[0].strip()
            if 20 <= len(para) <= 350 and "?" not in para[:40]:
                greeting = para
        elif head:
            # Intentar recuperar solo la primera oración amable
            first = re.split(r"(?<=[.!?])\s+", head)[0].strip()
            if (
                first
                and len(first) <= 220
                and not _THOUGHT_MARKERS.search(first)
                and not first.lower().startswith(("ahora", "primero", "voy a"))
            ):
                greeting = first
        polished = f"{greeting}\n\n{body}".strip() if greeting else body
    else:
        polished = text

    # Cortar coletillas meta de proceso interno
    polished = re.sub(
        r"(?is)\n*EVENTS_JSON\s*:.*$",
        "",
        polished,
    ).strip()
    return polished


def itinerary_to_email_html(itinerary_md: str, *, destination: str = "") -> str:
    """Convierte el itinerario (markdown) a un HTML amigable para Gmail."""
    clean = polish_agent_reply(itinerary_md)
    body_html = md_lib.markdown(
        clean,
        extensions=["nl2br", "sane_lists", "fenced_code"],
    )
    title = "Tu itinerario de viaje"
    if destination:
        title = f"Tu itinerario — {destination}"

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
</head>
<body style="margin:0;padding:0;background:#f4f1ea;font-family:Georgia,'Times New Roman',serif;color:#1c1917;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f4f1ea;padding:24px 12px;">
    <tr>
      <td align="center">
        <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="max-width:600px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e7e5e4;">
          <tr>
            <td style="background:#0f766e;color:#ffffff;padding:28px 32px;">
              <div style="font-size:13px;letter-spacing:0.08em;text-transform:uppercase;opacity:0.85;">Agente Turismo Argentina</div>
              <div style="font-size:26px;line-height:1.25;margin-top:8px;font-weight:700;">{title}</div>
              <div style="font-size:14px;margin-top:10px;opacity:0.9;">Plan listo para tu próximo viaje</div>
            </td>
          </tr>
          <tr>
            <td style="padding:28px 32px;font-size:16px;line-height:1.6;color:#292524;">
              <div style="margin:0;">
                {body_html}
              </div>
            </td>
          </tr>
          <tr>
            <td style="padding:18px 32px 28px;border-top:1px solid #e7e5e4;font-size:12px;line-height:1.5;color:#78716c;">
              Enviado automáticamente por tu Agente de Turismo Argentina.<br/>
              Los precios son estimativos y pueden variar según disponibilidad.
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""
