"""
Guardarraíles del Agente de Turismo Argentina.
Todos los chequeos son deterministas (no dependen del LLM) salvo la
clasificación de dominio, que usa un LLM con temperatura 0.
"""
import re
from typing import Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import ORCHESTRATOR_MODEL, GOOGLE_API_KEY, DOMAIN_DESCRIPTION
from agents.utils import to_text, is_conversation_followup

# ── 1. Palabras inapropiadas ──────────────────────────────────────────────────
# Lista representativa; ampliar según necesidad.
BAD_WORDS = {
    "mierda", "puta", "hijo de puta", "pelotudo", "boludo", "concha",
    "carajo", "fuck", "shit", "ass", "bastard", "cunt", "damn",
}

def check_bad_words(text: str) -> Tuple[bool, str]:
    """
    Retorna (True, "") si el texto es limpio.
    Retorna (False, motivo) si contiene lenguaje inapropiado.
    """
    lower = text.lower()
    for word in BAD_WORDS:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, lower):
            return False, f"El texto contiene lenguaje inapropiado: '{word}'"
    return True, ""


# ── 2. Idioma ─────────────────────────────────────────────────────────────────
SPANISH_RESPONSE = (
    "Lo siento, solo puedo comunicarme en español. "
    "Por favor, escribime en español y con gusto te ayudo con tu viaje por Argentina."
)

def detect_non_spanish(text: str) -> bool:
    """
    Heurística simple: si más del 40% de las palabras son ASCII puras y
    no aparecen artículos/preposiciones típicos del español, asume otro idioma.
    Para producción, reemplazar por langdetect o similar.
    """
    spanish_markers = {
        "el", "la", "los", "las", "un", "una", "de", "en", "que", "y",
        "a", "es", "se", "no", "te", "me", "mi", "tu", "su", "por",
        "con", "para", "del", "al", "como", "pero", "más", "lo", "le",
    }
    words = text.lower().split()
    if not words:
        return False
    spanish_count = sum(1 for w in words if w in spanish_markers)
    # Si hay muy pocos marcadores españoles en un texto largo, probablemente no es español
    ratio = spanish_count / len(words)
    return len(words) > 4 and ratio < 0.05


# ── 3. Dominio ────────────────────────────────────────────────────────────────
_domain_llm = None

def _get_domain_llm():
    global _domain_llm
    if _domain_llm is None:
        _domain_llm = ChatGoogleGenerativeAI(
            model=ORCHESTRATOR_MODEL,
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
        )
    return _domain_llm


def check_domain(user_message: str, has_active_session: bool = False) -> Tuple[bool, str]:
    """
    Verifica si la consulta está dentro del dominio de turismo argentino.
    Retorna (True, "") si está en dominio, (False, motivo) si no.
    """
    # Seguimientos cortos de una conversación de viaje ya abierta
    if has_active_session and is_conversation_followup(user_message):
        return True, ""

    llm = _get_domain_llm()
    system = SystemMessage(content=(
        "Eres un clasificador de consultas. Tu única tarea es determinar si "
        f"la consulta del usuario está relacionada con {DOMAIN_DESCRIPTION}, "
        "incluyendo seguimientos de una planificación en curso "
        "(elegir una opción de fechas, confirmar un período, pedir itinerario). "
        "Responde ÚNICAMENTE con 'SI' o 'NO', sin ninguna otra palabra."
    ))
    human = HumanMessage(content=user_message)
    result = llm.invoke([system, human])
    answer = to_text(result.content).strip().upper()
    if "SI" in answer[:10]:
        return True, ""
    return False, (
        "Esa consulta está fuera de mi área de especialización. "
        "Solo puedo ayudarte con viajes y turismo dentro de Argentina. "
        "¿Te puedo ayudar a planificar tu próximo destino argentino?"
    )


# ── 4. Presupuesto ────────────────────────────────────────────────────────────
def check_budget(total_cost: float, budget: float, threshold: float = 1.0) -> Tuple[bool, str]:
    """
    Retorna (True, "") si el costo está dentro del presupuesto.
    Retorna (False, mensaje) si lo supera.
    threshold=1.0 significa 100% del presupuesto.
    """
    if total_cost <= budget * threshold:
        return True, ""
    exceso = total_cost - budget
    return False, (
        f"El costo proyectado (${total_cost:,.0f}) supera tu presupuesto "
        f"de ${budget:,.0f} en ${exceso:,.0f}. "
        "¿Querés que ajuste el plan o autorizás el exceso?"
    )


# ── 5. Coherencia horaria ─────────────────────────────────────────────────────
def check_schedule_coherence(activities: list[dict]) -> Tuple[bool, str]:
    """
    Verifica que no haya superposición de actividades en el mismo día.
    Cada actividad debe tener: { 'nombre', 'hora_inicio', 'hora_fin', 'dia' }
    Horas en formato HH:MM (24hs).
    """
    from datetime import datetime

    by_day: dict[str, list] = {}
    for act in activities:
        day = act.get("dia", "1")
        by_day.setdefault(day, []).append(act)

    for day, acts in by_day.items():
        sorted_acts = sorted(acts, key=lambda x: x.get("hora_inicio", "00:00"))
        for i in range(len(sorted_acts) - 1):
            fin_actual  = sorted_acts[i].get("hora_fin",   "00:00")
            ini_next    = sorted_acts[i + 1].get("hora_inicio", "00:00")
            try:
                t_fin  = datetime.strptime(fin_actual, "%H:%M")
                t_ini  = datetime.strptime(ini_next,  "%H:%M")
                if t_ini < t_fin:
                    return False, (
                        f"Superposición horaria en el día {day}: "
                        f"'{sorted_acts[i]['nombre']}' termina a las {fin_actual} "
                        f"pero '{sorted_acts[i+1]['nombre']}' empieza a las {ini_next}."
                    )
            except ValueError:
                pass  # si el formato no es parseble, se deja pasar

    return True, ""


# ── 6. Clima adverso ──────────────────────────────────────────────────────────
def check_weather_alert(rain_mm: float, threshold_mm: float) -> Tuple[bool, str]:
    """
    Retorna (True, alerta) si lluvia supera umbral. Lógica 100% determinista.
    """
    if rain_mm >= threshold_mm:
        return True, (
            f"Se esperan {rain_mm:.0f} mm de lluvia, superando el umbral de "
            f"{threshold_mm:.0f} mm/día. Se recomienda revisar el plan de actividades "
            "al aire libre."
        )
    return False, ""


# ── Aplicar todos los guardarraíles de entrada ────────────────────────────────
def apply_input_guardrails(
    user_message: str,
    has_active_session: bool = False,
) -> Tuple[bool, str]:
    """
    Corre todos los chequeos de entrada en orden.
    Retorna (True, "") si todo está bien, (False, mensaje_al_usuario) si falla alguno.
    """
    # 1. Malas palabras en input
    ok, msg = check_bad_words(user_message)
    if not ok:
        return False, "Por favor, utilizá un lenguaje respetuoso. Estoy aquí para ayudarte con tu viaje por Argentina."

    # 2. Idioma (omitir en seguimientos cortos: "opción 2", "dale", etc.)
    if not (has_active_session and is_conversation_followup(user_message)):
        if detect_non_spanish(user_message):
            return False, SPANISH_RESPONSE

    # 3. Dominio
    ok, msg = check_domain(user_message, has_active_session=has_active_session)
    if not ok:
        return False, msg

    return True, ""


def apply_output_guardrails(agent_response: str) -> Tuple[bool, str]:
    """
    Corre guardarraíles sobre la respuesta generada.
    Retorna (True, respuesta_limpia) o (False, mensaje_de_error).
    """
    agent_response = to_text(agent_response)
    ok, _ = check_bad_words(agent_response)
    if not ok:
        return False, "[Respuesta bloqueada por guardarraíl de lenguaje. Regenerando...]"
    return True, agent_response
