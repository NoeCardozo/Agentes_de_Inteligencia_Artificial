"""
=============================================================================
MENSAJES DE ERROR AMIGABLES
=============================================================================

Traduce errores técnicos de la API de Gemini a mensajes que el usuario
pueda entender en la interfaz gráfica.

Sin este módulo, el usuario vería JSON crudo con códigos como 429 o
RESOURCE_EXHAUSTED.
"""

import re


def format_agent_error(error: Exception) -> str:
    """
    Convierte una excepción en un mensaje legible para mostrar en el chat.

    Detecta los errores más comunes y devuelve instrucciones claras.
    """
    msg = str(error)

    # Error 429: cuota de la API agotada o modelo sin cupo gratuito
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        retry_secs = _extraer_segundos_reintento(msg)
        retry_hint = f" Reintenta en ~{retry_secs}s." if retry_secs else ""
        return (
            "Cuota de la API de Gemini agotada o no disponible para este modelo."
            f"{retry_hint}\n\n"
            "Qué puedes hacer:\n"
            "1. En AI Studio revisa 'Uso' y 'Límite de frecuencia'.\n"
            "2. Cambia GEMINI_MODEL en .env (prueba gemini-2.5-flash).\n"
            "3. Espera unos minutos si superaste el límite por minuto.\n"
            "4. Si tu cuenta es educativa, puede tener cupos distintos."
        )

    # API key inválida o mal configurada
    if "API key not valid" in msg or "API_KEY_INVALID" in msg:
        return (
            "API key inválida. Obtén una nueva en https://aistudio.google.com/apikey "
            "y configúrala en el archivo .env como GOOGLE_API_KEY."
        )

    # Otros errores: se muestran tal cual (ya suelen ser legibles)
    return msg


def _extraer_segundos_reintento(msg: str) -> int | None:
    """Extrae del mensaje de error cuántos segundos sugiere Google esperar."""
    match = re.search(r"retry in (\d+)", msg, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r'"retryDelay":\s*"(\d+)s"', msg)
    if match:
        return int(match.group(1))

    return None
