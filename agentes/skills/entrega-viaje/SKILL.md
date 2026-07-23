---
name: entrega-viaje
description: Usá esta skill cuando el usuario apruebe un itinerario o pida enviarlo por email y/o cargarlo en Google Calendar.
---

# Entrega de itinerario (email + calendario)

## Objetivo

Enviar el itinerario por correo y crear eventos en Google Calendar del usuario.

## Instrucciones

1. Pedí el email del usuario si no está en memoria / perfil.
2. Cuando el itinerario esté listo (día a día con horarios):
   - Incluí al final del texto (sin mostrarlo de forma destacada) un bloque:
     `EVENTS_JSON:[{"title":"...","start":"YYYY-MM-DDTHH:MM:SS","end":"...","location":"...","description":"..."},...]`
   - Usá timezone Argentina (horarios locales, sin Z).
3. Llamá `tool_enviar_email` con el itinerario completo (sin el bloque JSON técnico si preferís texto limpio).
4. Llamá `tool_crear_eventos_calendario` pasando el mismo array JSON como string.
5. Confirmá al usuario: email enviado + cantidad de eventos creados + links si hay.

## Reglas

- No inventes el email: pedilo o usá el configurado.
- No crees eventos sin fechas razonables.
- Máximo ~20 eventos por itinerario.
