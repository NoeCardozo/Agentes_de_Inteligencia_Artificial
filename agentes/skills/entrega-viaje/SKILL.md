---
name: entrega-viaje
description: Usá esta skill cuando el usuario apruebe un itinerario o pida enviarlo por email y/o cargarlo en Google Calendar.
---

# Entrega de itinerario (email + calendario)

## Objetivo

Enviar el itinerario completo por correo (Resend, a la dirección que indique el usuario)
y crear eventos en Google Calendar (cuenta OAuth del servidor).

## Instrucciones

1. **Preguntá siempre** a qué email enviar el itinerario si el usuario aún no lo dijo.
   Ejemplo: "¿A qué email querés que te envíe el itinerario?"
2. Solo enviá cuando el itinerario esté **completo día a día** (incluye al menos Día 1 y Día 2,
   actividades, alojamiento y costos). Nunca envíes mensajes de "estoy planificando" o
   "te aviso cuando esté listo".
3. Cuando el itinerario esté listo:
   - Incluí al final (sin destacarlo) un bloque:
     `EVENTS_JSON:[{"title":"...","start":"YYYY-MM-DDTHH:MM:SS","end":"...","location":"...","description":"..."},...]`
   - Usá timezone Argentina (horarios locales, sin Z).
4. Llamá `tool_enviar_email` con:
   - `to` = el email que escribió el usuario (obligatorio)
   - `body` = el itinerario completo desde Día 1 (sin el bloque JSON técnico)
5. Llamá `tool_crear_eventos_calendario` con el mismo array JSON como string
   (calendario de la cuenta Google OAuth configurada; no usa el email de entrega).
6. Confirmá al usuario: email enviado a X + cantidad de eventos creados.

## Reglas

- No inventes el email ni uses uno por defecto: pedilo.
- No envíes un body sin estructura Día 1 / Día 2…
- No crees eventos sin fechas razonables.
- Máximo ~20 eventos por itinerario.
