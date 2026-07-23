---
name: monitoreo
description: Usá esta skill para monitorear condiciones de un viaje activo en Argentina — clima, cierres de atracciones, cancelaciones de traslados y propuestas de alternativas.
---

# Monitoreo de condiciones en tiempo real

## Objetivo

Detectar cambios que afecten el itinerario y proponer alternativas.

## Instrucciones

### 1. Consulta periódica

- Invocar `tool_clima` para el destino y las fechas del itinerario activo.
- Frecuencia sugerida: una vez por día mientras el viaje está activo.

### 2. Criterios de alerta (deterministas)

- Lluvia ≥ 20 mm/día → alerta por clima adverso.
- Actividad marcada como "cerrada" en `tool_actividades` → alerta por cierre.
- Traslado no disponible → alerta por cancelación.

### 3. Propuesta de alternativa

- Ante alerta climática: sustituir actividades al aire libre por opciones cubiertas (museos, bodegas, gastronomía) consultando `tool_actividades` con categoría `cultura`.
- Ante cierre de atracción: buscar actividad alternativa de la misma categoría y duración similar.
- Ante cancelación de traslado: consultar `tool_traslados` con tipo alternativo.

### 4. Escalado a human-in-the-loop

- Siempre presentar la alternativa propuesta al usuario para aprobación.
- No modificar el itinerario sin confirmación explícita.
- Si el cambio implica costo adicional > 10% del presupuesto original, escalar obligatoriamente al humano.
