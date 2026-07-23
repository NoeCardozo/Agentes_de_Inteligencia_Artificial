---
name: planificacion
description: Usá esta skill para planificar itinerarios de viaje en Argentina — selección de destino, distribución de días, actividades, alojamiento, traslados y chequeo de presupuesto.
---

# Planificación de itinerario

## Objetivo

Construir un itinerario de viaje completo, coherente y dentro del presupuesto.

## Instrucciones

### 1. Selección de destino

- Cruzar intereses del usuario con la temporada y el clima proyectado.
- Consultar `tool_clima` para las fechas indicadas.
- Verificar que el destino no esté en la lista de "ya visitados" del usuario.

### 2. Distribución de días

- Calcular días disponibles descontando traslados de llegada y partida.
- Asignar actividades priorizando las de mayor relevancia al perfil del usuario.
- No superar 2-3 actividades por día para itinerarios confortables.

### 3. Asignación de actividades

- Consultar `tool_actividades` para cada día.
- Verificar horarios: ninguna actividad puede comenzar antes de su apertura ni terminar después de su cierre.
- Respetar restricciones de movilidad y alimentarias del perfil del usuario.

### 4. Alojamiento

- Consultar `tool_alojamiento` con el presupuesto disponible para hospedaje.
- Preferir disponibilidad confirmada sobre precio más bajo sin disponibilidad.

### 5. Traslados

- Consultar `tool_traslados` para vuelos o buses de llegada y partida.
- Incluir tiempo de traslado en el cómputo del primer y último día.

### 6. Self-check de presupuesto (obligatorio antes de presentar el plan)

- Sumar: traslados + alojamiento (noches × precio) + actividades pagas.
- Si el total supera el presupuesto: ajustar actividades (sustituir pagas por gratuitas) o tipo de alojamiento antes de presentar al usuario.
- Si no es posible ajustar: notificar al usuario con el exceso exacto.

### 7. Contrato de salida

- El itinerario debe tener, para cada día: lista de actividades con horario, nombre del alojamiento y precio, y costo acumulado hasta ese día.
- Idioma de salida: español.
- Tono: amigable, inclusivo y políticamente correcto.
