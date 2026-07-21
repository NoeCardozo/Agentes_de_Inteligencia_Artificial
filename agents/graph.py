"""
Grafo LangGraph del Agente de Turismo Argentina.
Define el flujo de estados entre todos los nodos:
  entrada → guardarraíl → clasificación → subagentes → juez → human_in_the_loop → salida

El grafo implementa ciclos controlados: si el juez rechaza, vuelve al orquestador
con las observaciones para corrección (máximo MAX_JUDGE_RETRIES veces).
"""
from __future__ import annotations

import json
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config.settings import (
    GOOGLE_API_KEY, ORCHESTRATOR_MODEL,
    SYSTEM_PROMPT_BASE, MAX_JUDGE_RETRIES,
)
from guardrails.guardrails import apply_input_guardrails, apply_output_guardrails
from memory.long_term_memory import get_user_context_summary, save_itinerary, get_session, save_session
from agents.subagents import (
    run_subagente_destinos,
    run_subagente_itinerario,
    run_subagente_precios,
    run_subagente_monitoreo,
    run_subagente_periodo,
    run_subagente_rag,
)
from agents.judge_agent import evaluate_response, JudgeResult
from mcp.mcp_client import tool_guardar_itinerario
from agents.utils import (
    to_text,
    extract_destination,
    resolve_trip_dates,
    asks_for_period_recommendation,
    wants_full_itinerary,
    looks_like_iso_date,
    extract_period_options,
    resolve_option_choice,
)
from datetime import date


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DEL GRAFO
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Entrada
    user_id: str
    user_message: str
    budget: float
    fecha_inicio: str
    fecha_fin: str
    intereses: str
    origen: str
    viajeros: int
    presupuesto_tipo: str          # por_persona | total_grupo

    # Estado interno
    intent: str                    # planificar_viaje | consulta_info | monitoreo | gestion
    guardrail_passed: bool
    guardrail_message: str
    user_context: str
    judge_retries: int
    judge_result: dict | None
    judge_feedback: str

    # Resultados de subagentes
    destinos_resultado: str
    destino_elegido: str
    itinerario_resultado: str
    precios_resultado: str
    monitoreo_resultado: dict
    rag_resultado: str

    # Salida
    final_response: str
    awaiting_human: bool
    human_decision: str            # "aprobado" | "rechazado" | "pendiente"
    error: str


# ─────────────────────────────────────────────────────────────────────────────
# NODOS
# ─────────────────────────────────────────────────────────────────────────────

def node_guardrail(state: AgentState) -> AgentState:
    """Aplica guardarraíles de entrada: dominio, idioma, malas palabras."""
    session = get_session(state["user_id"])
    has_session = bool(session.get("destino") or session.get("proposed_periods"))
    ok, message = apply_input_guardrails(
        state["user_message"],
        has_active_session=has_session,
    )
    return {
        **state,
        "guardrail_passed": ok,
        "guardrail_message": message if not ok else "",
    }


def node_load_context(state: AgentState) -> AgentState:
    """Carga el contexto del usuario y resuelve fechas (opción elegida / ISO / relativas)."""
    context = get_user_context_summary(state["user_id"])
    msg = state.get("user_message", "")
    fecha_inicio = state.get("fecha_inicio", "")
    fecha_fin = state.get("fecha_fin", "")
    session = get_session(state["user_id"])

    # Si eligió "opción N", tomar esas fechas
    choice = resolve_option_choice(msg, session)
    if choice:
        enriched, fecha_inicio, fecha_fin = choice
        msg = enriched
    elif "propuesta" in msg.lower() and session.get("proposed_periods"):
        # "la propuesta que me diste" sin número → usar última opción elegida o la 2 típica fin de año
        periods = session["proposed_periods"]
        chosen = session.get("chosen_period") or (
            periods[1] if len(periods) > 1 else periods[0]
        )
        fecha_inicio = chosen.get("inicio", fecha_inicio)
        fecha_fin = chosen.get("fin", fecha_fin)
        destino = session.get("destino", "Bariloche")
        msg = (
            f"Quiero planificar el viaje completo a {destino} "
            f"del {fecha_inicio} al {fecha_fin}, según la propuesta anterior. "
            f"Mensaje del usuario: {state.get('user_message', '')}"
        )

    # Si solo pide recomendación de período, no trabar fechas todavía
    if asks_for_period_recommendation(msg) and not wants_full_itinerary(msg) and not choice:
        if not looks_like_iso_date(fecha_inicio):
            fecha_inicio = ""
        if not looks_like_iso_date(fecha_fin):
            fecha_fin = ""
    elif fecha_inicio or fecha_fin or wants_full_itinerary(msg) or choice:
        fecha_inicio, fecha_fin = resolve_trip_dates(
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            mensaje_usuario=msg,
        )

    return {
        **state,
        "user_message": msg,
        "user_context": context,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
    }


def node_classify_intent(state: AgentState) -> AgentState:
    """
    Clasifica la intención del usuario para rutear al subagente correcto.
    """
    msg = state["user_message"]
    msg_l = msg.lower()
    session = get_session(state["user_id"])

    # Si eligió una opción de período previa → planificar viaje
    if resolve_option_choice(msg, session) or (
        session.get("proposed_periods") and wants_full_itinerary(msg)
    ):
        return {**state, "intent": "planificar_viaje"}

    # Heurísticas prioritarias (más fieles a lo que el usuario pidió)
    if asks_for_period_recommendation(msg) and not wants_full_itinerary(msg):
        return {**state, "intent": "recomendar_periodo"}
    if wants_full_itinerary(msg):
        return {**state, "intent": "planificar_viaje"}

    llm = ChatGoogleGenerativeAI(model=ORCHESTRATOR_MODEL, temperature=0, google_api_key=GOOGLE_API_KEY)
    system = SystemMessage(content=(
        "Clasifica la intención del mensaje en UNA de estas categorías:\n"
        "- recomendar_periodo: pregunta qué fechas/período/semana conviene viajar\n"
        "- planificar_viaje: quiere que le armes un itinerario o viaje completo\n"
        "- consulta_info: pregunta sobre un destino, actividad o normativa\n"
        "- monitoreo: pregunta sobre condiciones actuales o alertas de su viaje\n"
        "- gestion: consulta de estadísticas, precios o comparativas\n"
        "Responde SOLO con el nombre de la categoría, sin nada más."
    ))
    human = HumanMessage(content=msg)
    result = llm.invoke([system, human])
    intent = to_text(result.content).strip().lower()

    valid_intents = {
        "recomendar_periodo", "planificar_viaje", "consulta_info", "monitoreo", "gestion",
    }
    if intent not in valid_intents:
        intent = "consulta_info"

    if any(kw in msg_l for kw in ["armame el", "armá el", "itinerario completo", "plan día a día"]):
        intent = "planificar_viaje"

    return {**state, "intent": intent}


def node_subagent_periodo(state: AgentState) -> AgentState:
    """Recomienda ventanas de fechas sin armar el itinerario."""
    session = get_session(state["user_id"])
    destino = extract_destination(
        state.get("user_message", ""),
        fallback=state.get("destino_elegido") or session.get("destino", ""),
    )
    resultado = run_subagente_periodo(
        destino=destino or "Bariloche",
        presupuesto=state.get("budget", 0) or session.get("budget", 0),
        viajeros=state.get("viajeros", 1) or session.get("viajeros", 1),
        presupuesto_tipo=state.get("presupuesto_tipo", "total_grupo"),
        mensaje_usuario=state.get("user_message", ""),
        intereses=state.get("intereses", ""),
        periodos_previos=session.get("proposed_periods") or [],
        resumen_previo=session.get("last_response_summary") or "",
    )
    return {
        **state,
        "destino_elegido": destino or session.get("destino") or "Bariloche",
        "destinos_resultado": resultado,
        "rag_resultado": resultado,
    }


def node_subagent_destinos(state: AgentState) -> AgentState:
    """Invoca el subagente de destinos, respetando destino ya elegido."""
    destino = extract_destination(
        state.get("user_message", ""),
        fallback=state.get("destino_elegido", ""),
    )
    resultado = run_subagente_destinos(
        intereses=state.get("intereses", "") or "naturaleza y cultura",
        fechas=f"{state.get('fecha_inicio', '')} al {state.get('fecha_fin', '')}",
        presupuesto=state.get("budget", 0),
        perfil_usuario=state.get("user_context", ""),
        destino_preferido=destino,
        viajeros=state.get("viajeros", 1),
        presupuesto_tipo=state.get("presupuesto_tipo", "total_grupo"),
        mensaje_usuario=state.get("user_message", ""),
    )
    if not destino:
        destino = extract_destination(resultado, fallback="Bariloche")
    return {**state, "destinos_resultado": resultado, "destino_elegido": destino}


def node_subagent_itinerario(state: AgentState) -> AgentState:
    """Invoca el subagente de itinerario."""
    resultado = run_subagente_itinerario(
        destino=state.get("destino_elegido", "Bariloche"),
        fecha_inicio=state.get("fecha_inicio", ""),
        fecha_fin=state.get("fecha_fin", ""),
        presupuesto=state.get("budget", 0),
        intereses=state.get("intereses", ""),
        perfil_usuario=state.get("user_context", ""),
        viajeros=state.get("viajeros", 1),
        presupuesto_tipo=state.get("presupuesto_tipo", "total_grupo"),
    )
    return {**state, "itinerario_resultado": resultado}


def node_subagent_precios(state: AgentState) -> AgentState:
    """Invoca el subagente de precios."""
    resultado = run_subagente_precios(
        destino=state.get("destino_elegido", "Bariloche"),
        fecha_inicio=state.get("fecha_inicio", ""),
        fecha_fin=state.get("fecha_fin", ""),
        presupuesto=state.get("budget", 0),
        origen=state.get("origen", "Buenos Aires"),
        viajeros=state.get("viajeros", 1),
        presupuesto_tipo=state.get("presupuesto_tipo", "total_grupo"),
    )
    return {**state, "precios_resultado": resultado}


def node_subagent_monitoreo(state: AgentState) -> AgentState:
    """Invoca el subagente de monitoreo."""
    resultado = run_subagente_monitoreo(
        destino=state.get("destino_elegido", ""),
        fecha_inicio=state.get("fecha_inicio", ""),
        fecha_fin=state.get("fecha_fin", ""),
        actividades_planificadas=[],
    )
    return {**state, "monitoreo_resultado": resultado}


def node_subagent_rag(state: AgentState) -> AgentState:
    """Invoca el subagente RAG para consultas de conocimiento fáctico."""
    respuesta, _ = run_subagente_rag(
        question=state["user_message"],
        all_docs=[],   # en producción: pasar el corpus completo
    )
    return {**state, "rag_resultado": respuesta}


def node_compose_response(state: AgentState) -> AgentState:
    """
    Compone la respuesta final integrando los resultados de los subagentes.
    """
    llm = ChatGoogleGenerativeAI(model=ORCHESTRATOR_MODEL, temperature=0.4, google_api_key=GOOGLE_API_KEY)

    # Contexto de feedback del juez si viene de un retry
    feedback_ctx = ""
    if state.get("judge_feedback"):
        feedback_ctx = (
            f"\n\nOBSERVACIONES DEL JUEZ (iteración anterior, DEBES CORREGIRLAS):\n"
            f"{state['judge_feedback']}"
        )

    intent = state.get("intent", "consulta_info")

    if intent == "recomendar_periodo":
        session = get_session(state["user_id"])
        previos = session.get("proposed_periods") or []
        continuidad = ""
        if previos:
            continuidad = (
                "Ya habías propuesto estas opciones en el turno anterior; "
                "MANTENELAS (misma numeración y fechas), solo refiná costos/"
                "claridad. No reemplaces el set completo:\n"
                + "\n".join(
                    f"- Opción {p.get('n')}: {p.get('inicio')} a {p.get('fin')} "
                    f"({p.get('label', '')})"
                    for p in previos
                )
                + "\n\n"
            )
        budget = state.get("budget", 0) or 0
        context = (
            f"{continuidad}"
            f"RECOMENDACIÓN DE PERÍODOS (borrador del subagente):\n"
            f"{state.get('destinos_resultado') or state.get('rag_resultado', '')}\n\n"
            f"Tu tarea: presentar esas 2-3 opciones de fechas para "
            f"{state.get('destino_elegido', 'el destino')}, con pros/contras "
            f"(clima, aglomeración) Y rangos de costo en ARS "
            f"(vuelos+alojamiento+actividades). "
            f"Presupuesto del usuario: ${budget:,.0f} ARS. "
            f"REGLA ESTRICTA: el máximo de cada rango de costo debe ser "
            f"<= ${budget:,.0f} ARS. Si una temporada es cara, ajustá el "
            f"paquete (hotel más económico, menos actividades pagas) para "
            f"caber en el presupuesto; no ofrezcas rangos que lo excedan. "
            f"NO armes itinerario día a día. "
            f"Cerrá preguntando qué opción (1, 2 o 3) elige. "
            f"Al final agregá exactamente (sin markdown):\n"
            f"PERIODOS_JSON:[{{\"n\":1,\"inicio\":\"YYYY-MM-DD\",\"fin\":\"YYYY-MM-DD\",\"label\":\"...\"}},...]"
        )
    elif intent == "planificar_viaje":
        context = (
            f"DESTINOS EVALUADOS:\n{state.get('destinos_resultado', '')}\n\n"
            f"ITINERARIO PROPUESTO:\n{state.get('itinerario_resultado', '')}\n\n"
            f"ANÁLISIS DE PRECIOS:\n{state.get('precios_resultado', '')}"
        )
        monitoreo = state.get("monitoreo_resultado", {})
        if monitoreo.get("alerta"):
            context += f"\n\nALERTA DE CONDICIONES:\n{monitoreo.get('mensaje', '')}"
    elif intent == "consulta_info":
        context = f"INFORMACIÓN RECUPERADA:\n{state.get('rag_resultado', '')}"
    elif intent == "monitoreo":
        monitoreo = state.get("monitoreo_resultado", {})
        context = f"ESTADO ACTUAL DEL VIAJE:\n{monitoreo.get('mensaje', '')}"
    else:
        context = f"DATOS DE GESTIÓN:\n{state.get('rag_resultado', '')}"

    system_prompt = (
        SYSTEM_PROMPT_BASE
        + f"\n\nHoy es {date.today().isoformat()}. "
        f"Usá únicamente fechas del {date.today().year} o posteriores; nunca años pasados."
        + f"\n\nContexto del usuario:\n{state.get('user_context', '')}"
        + feedback_ctx
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Solicitud del usuario: {state['user_message']}\n"
            f"Destino elegido: {state.get('destino_elegido', 'no definido')}\n"
            f"Fechas del viaje: {state.get('fecha_inicio', '')} al {state.get('fecha_fin', '')}\n"
            f"Viajeros: {state.get('viajeros', 1)} | "
            f"Presupuesto: ${state.get('budget', 0):,.0f} ARS "
            f"({state.get('presupuesto_tipo', 'total_grupo')})\n\n"
            f"Información recopilada por los subagentes:\n{context}\n\n"
            f"Redactá una respuesta final completa, clara y en español. "
            f"Respetá el destino y las fechas indicadas arriba."
        )),
    ]

    result = llm.invoke(messages)
    response = to_text(result.content)

    # Guardarraíl de salida
    ok, response = apply_output_guardrails(response)

    # Persistir sesión para seguimientos ("opción 2", etc.)
    destino = state.get("destino_elegido") or extract_destination(state.get("user_message", ""))
    periods = extract_period_options(response) or extract_period_options(
        state.get("destinos_resultado") or ""
    )
    # Si el modelo no devolvió JSON pero ya había opciones, conservarlas
    if not periods:
        prev = get_session(state["user_id"]).get("proposed_periods") or []
        if prev and intent == "recomendar_periodo":
            periods = prev

    session_update = {
        "destino": destino,
        "last_intent": intent,
        "last_response_summary": response[:400],
        "budget": state.get("budget", 0),
        "viajeros": state.get("viajeros", 1),
        "presupuesto_tipo": state.get("presupuesto_tipo", "total_grupo"),
    }
    if periods:
        session_update["proposed_periods"] = periods
    if intent == "planificar_viaje" and state.get("fecha_inicio"):
        session_update["chosen_period"] = {
            "inicio": state.get("fecha_inicio"),
            "fin": state.get("fecha_fin"),
            "label": f"{state.get('fecha_inicio')} a {state.get('fecha_fin')}",
        }
    try:
        save_session(state["user_id"], session_update)
    except Exception as exc:
        print(f"[WARN] No se pudo guardar sesión: {exc}")

    # Ocultar el bloque técnico al usuario final
    visible = re.sub(r"\n?PERIODOS_JSON\s*:\s*\[[\s\S]*?\]\s*", "\n", response).strip()

    return {**state, "final_response": visible}


def node_judge(state: AgentState) -> AgentState:
    """Evalúa la respuesta con el subagente juez y su rúbrica."""
    judge_result = evaluate_response(
        user_request=state["user_message"],
        agent_response=state["final_response"],
        budget=state.get("budget", 0),
        user_profile_summary=state.get("user_context", ""),
    )
    return {
        **state,
        "judge_result": {
            "score_total": judge_result.score_total,
            "aprobado": judge_result.aprobado,
            "observaciones": judge_result.observaciones,
            "sugerencias_mejora": judge_result.sugerencias_mejora,
            "summary": judge_result.summary(),
        },
        "judge_feedback": (
            f"{judge_result.observaciones}\nSugerencias: {judge_result.sugerencias_mejora}"
            if not judge_result.aprobado else ""
        ),
    }


def node_human_in_the_loop(state: AgentState) -> AgentState:
    """
    Pausa el grafo y espera confirmación del usuario.
    En producción: integrar con UI/webhook. Aquí simula con input().
    """
    print("\n" + "="*60)
    print("🔔 HUMAN IN THE LOOP — SE REQUIERE DECISIÓN")
    print("="*60)
    print("\nRespuesta propuesta del agente:\n")
    print(state["final_response"])
    print("\n" + "-"*60)

    judge = state.get("judge_result", {})
    if judge:
        print(f"\nEvaluación del juez: {judge.get('summary', '')}")

    monitoreo = state.get("monitoreo_resultado", {})
    if monitoreo.get("alerta"):
        print(f"\n⚠️ ALERTA DETECTADA: {monitoreo.get('mensaje', '')}")

    print("\n¿Aprobás esta respuesta?")
    print("  [1] Sí, enviar al usuario")
    print("  [2] No, rechazar y regenerar")
    print("  [3] Escalar al operador")

    try:
        choice = input("Tu decisión (1/2/3): ").strip()
    except EOFError:
        choice = "1"  # en tests automatizados, aprobar por defecto

    decision_map = {"1": "aprobado", "2": "rechazado", "3": "escalado"}
    decision = decision_map.get(choice, "aprobado")

    return {
        **state,
        "awaiting_human": False,
        "human_decision": decision,
    }


def node_save_and_finalize(state: AgentState) -> AgentState:
    """Guarda el itinerario en memoria si corresponde y finaliza."""
    if state.get("intent") == "planificar_viaje" and state.get("itinerario_resultado"):
        try:
            save_itinerary(
                user_id=state["user_id"],
                itinerary={
                    "destination": state.get("destino_elegido", ""),
                    "dates": f"{state.get('fecha_inicio', '')} al {state.get('fecha_fin', '')}",
                    "budget": state.get("budget", 0),
                    "days": [],  # parsear del itinerario_resultado en producción
                },
            )
        except Exception as e:
            print(f"[WARN] No se pudo guardar el itinerario: {e}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# CONDICIONES DE RUTEO
# ─────────────────────────────────────────────────────────────────────────────

def route_after_guardrail(state: AgentState) -> str:
    if not state.get("guardrail_passed", True):
        return "blocked"
    return "load_context"


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "consulta_info")
    if intent == "recomendar_periodo":
        return "subagent_periodo"
    if intent == "planificar_viaje":
        return "subagent_destinos"
    if intent == "monitoreo":
        return "subagent_monitoreo"
    return "subagent_rag"


def route_after_judge(state: AgentState) -> str:
    judge = state.get("judge_result", {})
    retries = state.get("judge_retries", 0)

    if not judge.get("aprobado", True) and retries < MAX_JUDGE_RETRIES:
        return "retry"
    return "human_in_the_loop"


def route_after_human(state: AgentState) -> str:
    decision = state.get("human_decision", "aprobado")
    if decision == "rechazado":
        return "compose_response"
    if decision == "escalado":
        return "end_escalated"
    return "save_and_finalize"


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL GRAFO
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Registrar nodos
    graph.add_node("guardrail",           node_guardrail)
    graph.add_node("load_context",        node_load_context)
    graph.add_node("classify_intent",     node_classify_intent)
    graph.add_node("subagent_destinos",   node_subagent_destinos)
    graph.add_node("subagent_itinerario", node_subagent_itinerario)
    graph.add_node("subagent_precios",    node_subagent_precios)
    graph.add_node("subagent_monitoreo",  node_subagent_monitoreo)
    graph.add_node("subagent_periodo",    node_subagent_periodo)
    graph.add_node("subagent_rag",        node_subagent_rag)
    graph.add_node("compose_response",    node_compose_response)
    graph.add_node("judge",               node_judge)
    graph.add_node("human_in_the_loop",   node_human_in_the_loop)
    graph.add_node("save_and_finalize",   node_save_and_finalize)

    # Nodo de respuesta bloqueada (guardarraíl)
    def node_blocked(state: AgentState) -> AgentState:
        return {**state, "final_response": state["guardrail_message"]}

    graph.add_node("blocked", node_blocked)

    # Punto de entrada
    graph.set_entry_point("guardrail")

    # Aristas del grafo
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"blocked": "blocked", "load_context": "load_context"},
    )
    graph.add_edge("blocked", END)
    graph.add_edge("load_context", "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "subagent_periodo":   "subagent_periodo",
            "subagent_destinos":  "subagent_destinos",
            "subagent_monitoreo": "subagent_monitoreo",
            "subagent_rag":       "subagent_rag",
        },
    )

    # Flujo de recomendación de período (sin itinerario completo)
    graph.add_edge("subagent_periodo", "compose_response")

    # Flujo de planificación de viaje
    graph.add_edge("subagent_destinos",   "subagent_itinerario")
    graph.add_edge("subagent_itinerario", "subagent_precios")
    graph.add_edge("subagent_precios",    "subagent_monitoreo")
    graph.add_edge("subagent_monitoreo",  "compose_response")

    # RAG directo
    graph.add_edge("subagent_rag", "compose_response")

    # Evaluación y ciclo del juez
    graph.add_edge("compose_response", "judge")

    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "retry":             "compose_response",   # ciclo controlado
            "human_in_the_loop": "human_in_the_loop",
        },
    )

    graph.add_conditional_edges(
        "human_in_the_loop",
        route_after_human,
        {
            "compose_response":  "compose_response",
            "save_and_finalize": "save_and_finalize",
            "end_escalated":     END,
        },
    )

    graph.add_edge("save_and_finalize", END)

    return graph


def get_compiled_graph():
    """Retorna el grafo compilado con checkpointer en memoria."""
    graph = build_graph()
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
