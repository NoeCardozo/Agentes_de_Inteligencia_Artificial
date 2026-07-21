"""
Subagentes especializados del Agente de Turismo Argentina.
Cada subagente usa tool-calling con un conjunto acotado de tools.
El orquestador (LangGraph) los invoca como nodos del grafo.
"""
from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import ORCHESTRATOR_MODEL, GOOGLE_API_KEY
from mcp.mcp_client import (
    tool_clima, tool_alojamiento, tool_actividades,
    tool_traslados, tool_guardar_itinerario, tool_consulta_gestion,
)
from skills.skills import get_skill
from rag.rag_pipeline import run_rag_pipeline, load_vectorstore
from agents.utils import to_text
from datetime import date


def _base_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=ORCHESTRATOR_MODEL,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
    )


def _make_agent_executor(role: str, skill_name: str, tools: list) -> AgentExecutor:
    llm = _base_llm()
    skill = get_skill(skill_name)
    hoy = date.today().isoformat()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres {role}.\n"
            "Responde siempre en español. Sé políticamente correcto e inclusivo.\n\n"
            "{skill}\n\n"
            f"Hoy es {hoy}. NUNCA uses años anteriores a {date.today().year}. "
            "Usá únicamente las fechas que te pasen en la consulta "
            "(formato YYYY-MM-DD). Si necesitás mencionar fechas, usá esas. "
            "Usá las herramientas disponibles cuando necesites datos concretos "
            "(clima, alojamiento, actividades, traslados). "
            "Si el usuario ya indicó un destino, NO propongas otros destinos salvo "
            "que pida alternativas. "
            "Al finalizar, respondé SOLO con texto claro y útil en español "
            "(sin JSON ni bloques técnicos).",
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]).partial(role=role, skill=skill)

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        return_intermediate_steps=False,
    )


def _agent_output(result: dict) -> str:
    return to_text(result.get("output", "")).strip()


# ─────────────────────────────────────────────────────────────────────────────
# SUBAGENTE 1: Destinos
# ─────────────────────────────────────────────────────────────────────────────
def run_subagente_destinos(
    intereses: str,
    fechas: str,
    presupuesto: float,
    perfil_usuario: str,
    destino_preferido: str = "",
    viajeros: int = 1,
    presupuesto_tipo: str = "total_grupo",
    mensaje_usuario: str = "",
) -> str:
    """
    Si el usuario ya eligió destino, lo valida y enriquece.
    Si no, recomienda hasta 3 destinos argentinos.
    """
    executor = _make_agent_executor(
        role="un experto en destinos turísticos argentinos",
        skill_name="planificacion",
        tools=[tool_clima, tool_actividades, tool_consulta_gestion],
    )

    presupuesto_ctx = (
        f"Presupuesto: ${presupuesto:,.0f} ARS "
        f"({'por persona' if presupuesto_tipo == 'por_persona' else 'total del grupo'}) "
        f"para {viajeros} viajero(s)."
    )

    if destino_preferido:
        query = (
            f"El usuario YA eligió el destino: {destino_preferido}. "
            f"NO recomiendes otros destinos. "
            f"Mensaje original: {mensaje_usuario}\n"
            f"Intereses: {intereses or 'no especificados'}. "
            f"Fechas del viaje (OBLIGATORIAS, usalas tal cual): {fechas}. "
            f"{presupuesto_ctx}\n"
            f"Contexto del usuario:\n{perfil_usuario}\n\n"
            f"Confirmá por qué {destino_preferido} encaja, verificá el clima "
            f"para esas fechas exactas y resumí qué tipo de viaje conviene. "
            f"No inventes otras fechas ni años."
        )
    else:
        query = (
            f"Recomendá hasta 3 destinos argentinos. "
            f"Mensaje del usuario: {mensaje_usuario}\n"
            f"Intereses: {intereses or 'variados'}. "
            f"Fechas del viaje (OBLIGATORIAS): {fechas}. {presupuesto_ctx}\n"
            f"Contexto del usuario:\n{perfil_usuario}\n"
            f"Verificá clima para esas fechas y justificá el ranking. "
            f"No inventes otras fechas ni años."
        )

    result = executor.invoke({"input": query})
    return _agent_output(result)


# ─────────────────────────────────────────────────────────────────────────────
# SUBAGENTE 2: Itinerario
# ─────────────────────────────────────────────────────────────────────────────
def run_subagente_itinerario(
    destino: str,
    fecha_inicio: str,
    fecha_fin: str,
    presupuesto: float,
    intereses: str,
    perfil_usuario: str,
    viajeros: int = 1,
    presupuesto_tipo: str = "total_grupo",
) -> str:
    """
    Construye el plan día a día para el destino elegido.
    """
    executor = _make_agent_executor(
        role="un planificador de itinerarios de viaje argentino",
        skill_name="planificacion",
        tools=[tool_actividades, tool_alojamiento, tool_traslados, tool_clima],
    )
    presupuesto_ctx = (
        f"${presupuesto:,.0f} ARS "
        f"({'por persona' if presupuesto_tipo == 'por_persona' else 'total del grupo'}) "
        f"para {viajeros} viajero(s)"
    )
    fechas = (
        f"del {fecha_inicio} al {fecha_fin}"
        if fecha_inicio or fecha_fin
        else "en fechas flexibles (proponé una ventana concreta de ~7 días)"
    )
    query = (
        f"Crea un itinerario detallado día a día para {destino}, {fechas}. "
        f"Presupuesto: {presupuesto_ctx}. "
        f"Intereses del viajero: {intereses or 'generales'}. "
        f"Contexto del usuario:\n{perfil_usuario}\n"
        f"Seguí el skill de planificación. Incluí actividades con horarios, "
        f"alojamiento recomendado y traslados. "
        f"Dimensioná costos y alojamiento para {viajeros} persona(s)."
    )
    result = executor.invoke({"input": query})
    return _agent_output(result)


# ─────────────────────────────────────────────────────────────────────────────
# SUBAGENTE 3: Precios
# ─────────────────────────────────────────────────────────────────────────────
def run_subagente_precios(
    destino: str,
    fecha_inicio: str,
    fecha_fin: str,
    presupuesto: float,
    origen: str = "Buenos Aires",
    viajeros: int = 1,
    presupuesto_tipo: str = "total_grupo",
) -> str:
    """
    Compara opciones de alojamiento y traslados para optimizar el presupuesto.
    """
    executor = _make_agent_executor(
        role="un comparador de precios turísticos especializado en Argentina",
        skill_name="comparacion_precios",
        tools=[tool_alojamiento, tool_traslados, tool_consulta_gestion],
    )
    presupuesto_ctx = (
        f"${presupuesto:,.0f} ARS "
        f"({'por persona' if presupuesto_tipo == 'por_persona' else 'total del grupo'}) "
        f"para {viajeros} viajero(s)"
    )
    query = (
        f"Compará opciones de alojamiento y traslados para un viaje "
        f"a {destino} del {fecha_inicio or 'fecha flexible'} al {fecha_fin or 'fecha flexible'}, "
        f"partiendo desde {origen}, para {viajeros} persona(s). "
        f"Presupuesto: {presupuesto_ctx}. "
        f"Presentá desglose de costos (total y por persona) y recomendá la combinación óptima."
    )
    result = executor.invoke({"input": query})
    return _agent_output(result)


# ─────────────────────────────────────────────────────────────────────────────
# SUBAGENTE 4: Monitoreo
# ─────────────────────────────────────────────────────────────────────────────
def run_subagente_monitoreo(
    destino: str,
    fecha_inicio: str,
    fecha_fin: str,
    actividades_planificadas: list[str],
) -> dict:
    """
    Verifica condiciones actuales y retorna { 'alerta': bool, 'mensaje': str, 'alternativa': str }.
    """
    executor = _make_agent_executor(
        role="un monitor de condiciones turísticas en tiempo real para Argentina",
        skill_name="monitoreo",
        tools=[tool_clima, tool_actividades, tool_traslados],
    )
    actividades_str = ", ".join(actividades_planificadas) if actividades_planificadas else "no especificadas"
    query = (
        f"Verificá las condiciones actuales para un viaje a {destino} "
        f"del {fecha_inicio or 'próximos días'} al {fecha_fin or 'próximos días'}. "
        f"Actividades planificadas: {actividades_str}. "
        f"Identificá si hay alertas climáticas o cierres de atracciones y "
        f"proponé alternativas concretas si es necesario."
    )
    result = executor.invoke({"input": query})
    output = _agent_output(result)

    lower = output.lower()
    # Evitar falsos positivos: el texto puede decir "no se activa una alerta"
    if any(
        neg in lower
        for neg in (
            "no se activa",
            "sin alerta",
            "no hay alerta",
            "no se detecta alerta",
            "no corresponde alerta",
        )
    ):
        alerta = False
    else:
        alerta = any(
            kw in lower
            for kw in (
                "alerta climática",
                "alerta climatica",
                "alerta:",
                "cerrado",
                "cancelado",
            )
        )
    return {
        "alerta": alerta,
        "mensaje": output,
        "alternativa": output if alerta else "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUBAGENTE 5: Recomendación de período / fechas
# ─────────────────────────────────────────────────────────────────────────────
def run_subagente_periodo(
    destino: str,
    presupuesto: float = 0,
    viajeros: int = 1,
    presupuesto_tipo: str = "total_grupo",
    mensaje_usuario: str = "",
    intereses: str = "",
    periodos_previos: list | None = None,
    resumen_previo: str = "",
) -> str:
    """
    Responde qué ventana de fechas conviene, sin armar el itinerario completo.
    Si ya había opciones en la sesión, las retoma en lugar de reinventarlas.
    """
    from datetime import date as _date

    executor = _make_agent_executor(
        role="un asesor de temporadas turísticas en Argentina",
        skill_name="planificacion",
        tools=[tool_clima, tool_consulta_gestion, tool_alojamiento, tool_traslados],
    )
    year = _date.today().year
    tope = int(presupuesto * 1.0) if presupuesto else 0
    presupuesto_ctx = (
        f"${presupuesto:,.0f} ARS "
        f"({'por persona' if presupuesto_tipo == 'por_persona' else 'total del grupo'}) "
        f"para {viajeros} viajero(s). "
        f"TOPE MÁXIMO del rango estimado por opción: ${tope:,.0f} ARS "
        f"(no superar el presupuesto del usuario)."
        if presupuesto
        else "presupuesto no indicado"
    )

    continuidad = ""
    if periodos_previos:
        lineas = []
        for p in periodos_previos:
            lineas.append(
                f"- Opción {p.get('n')}: {p.get('label', '')} "
                f"({p.get('inicio')} a {p.get('fin')})"
            )
        continuidad = (
            "IMPORTANTE — ya hay una conversación en curso. "
            "NO inventes un set nuevo de períodos desde cero. "
            "Retomá y refiná ESTAS opciones (podés ajustar costos/clarificar, "
            "pero mantené la misma numeración y fechas base):\n"
            + "\n".join(lineas)
            + "\n"
        )
        if resumen_previo:
            continuidad += f"Resumen de lo ya dicho:\n{resumen_previo[:500]}\n"

    query = (
        f"El usuario pregunta qué PERÍODO le conviene, no pidió aún el itinerario completo.\n"
        f"Mensaje: {mensaje_usuario}\n"
        f"Destino: {destino or 'Argentina'}. Intereses: {intereses or 'generales'}. "
        f"Presupuesto: {presupuesto_ctx}.\n"
        f"{continuidad}\n"
        f"Hoy estamos en {_date.today().isoformat()}. "
        f"Presentá 2 o 3 ventanas concretas en {year}/{year + 1} "
        f"(YYYY-MM-DD), con pros/contras (clima, aglomeración, eventos) "
        f"y un rango de costo estimado (vuelos+alojamiento+actividades) "
        f"que NUNCA supere el presupuesto declarado. "
        f"Si una temporada es más cara, bajá categoría de hotel/actividades "
        f"para que el máximo del rango quede dentro del presupuesto, "
        f"y acláralo. "
        f"Cerrá preguntando cuál opción (1, 2 o 3) prefiere. "
        f"NO armes plan día a día. "
        f"Al final incluí JSON válido exactamente así: "
        f"PERIODOS_JSON:[{{\"n\":1,\"inicio\":\"YYYY-MM-DD\",\"fin\":\"YYYY-MM-DD\",\"label\":\"...\"}}]"
    )
    result = executor.invoke({"input": query})
    return _agent_output(result)


# ─────────────────────────────────────────────────────────────────────────────
# SUBAGENTE 6: RAG (consultas de conocimiento fáctico)
# ─────────────────────────────────────────────────────────────────────────────
def run_subagente_rag(question: str, all_docs: list) -> tuple[str, list]:
    """
    Responde preguntas sobre destinos usando el pipeline RAG.
    Retorna (respuesta, chunks_recuperados).
    """
    try:
        vectorstore = load_vectorstore()
    except Exception:
        return "Base de conocimiento no disponible en este momento.", []

    chunks, context = run_rag_pipeline(question, vectorstore, all_docs)

    llm = _base_llm(temperature=0.2)
    system = SystemMessage(content=(
        "Eres un experto en turismo argentino. Responde en español usando ÚNICAMENTE "
        "la información del contexto proporcionado. Si la información no está en el "
        "contexto, dilo claramente. No inventes datos ni precios."
    ))
    human = HumanMessage(content=f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}")
    result = llm.invoke([system, human])
    return to_text(result.content), chunks
