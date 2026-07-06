"""
=============================================================================
AGENTE DE IA (LangGraph + Gemini)
=============================================================================

¿Qué es un agente?
  No es solo un chatbot: puede DECIDIR usar herramientas según la pregunta.
  Ejemplo: si preguntás "vuelos Madrid-París", llama a buscar_vuelos().

¿Qué es LangGraph?
  Framework que implementa el patrón ReAct:
    Reason (razonar) → Act (actuar con herramientas) → responder.

¿Qué es la memoria (MemorySaver)?
  Guarda el historial de la conversación para que el agente recuerde
  lo que se habló antes en la misma sesión.
"""

import logging

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.config import GEMINI_MODEL, GOOGLE_API_KEY, SYSTEM_PROMPT, validate_api_key
from app.tools import get_tools

logger = logging.getLogger(__name__)


def create_tourism_agent():
    """
    Crea y devuelve el agente listo para usar.

    Pasos:
      1. Valida la API key.
      2. Conecta el modelo Gemini.
      3. Registra las herramientas (tools.py).
      4. Activa memoria de conversación.
      5. Arma el grafo ReAct con LangGraph.
    """
    validate_api_key()
    logger.info("Inicializando agente con modelo %s", GEMINI_MODEL)

    # Modelo de lenguaje: Gemini de Google
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,   # 0 = muy preciso, 1 = más creativo
        max_retries=0,     # Sin reintentos silenciosos (falla rápido si hay error)
        timeout=30,        # Máximo 30 segundos por llamada a la API
    )

    tools = get_tools()
    memory = MemorySaver()  # Historial en memoria RAM (se pierde al cerrar la app)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )

    return agent


def _extract_content(content) -> str:
    """
    Convierte la respuesta de Gemini a texto plano.

    A veces Gemini devuelve un string, otras veces una lista de bloques.
    Esta función unifica ambos formatos.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        partes = []
        for bloque in content:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict) and bloque.get("type") == "text":
                partes.append(bloque.get("text", ""))
        return "\n".join(p for p in partes if p)

    return str(content) if content else "(Sin respuesta del modelo)"


def invoke_agent(agent, message: str, thread_id: str = "default") -> str:
    """
    Envía un mensaje del usuario al agente y devuelve la respuesta final.

    Parámetros:
      agent      → El agente creado con create_tourism_agent()
      message    → Texto que escribió el usuario
      thread_id  → ID de la conversación (para separar sesiones)

  Para la defensa: thread_id permite que cada sesión de chat tenga
  su propio historial sin mezclarse con otras.
    """
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("Enviando mensaje al agente (thread=%s): %s", thread_id, message[:80])

    # El agente procesa el mensaje (puede llamar herramientas internamente)
    result = agent.invoke(
        {"messages": [("user", message)]},
        config=config,
    )

    # La última entrada de la lista de mensajes es la respuesta final
    ultimo_mensaje = result["messages"][-1]
    respuesta = _extract_content(ultimo_mensaje.content)
    logger.info("Respuesta recibida (%d caracteres)", len(respuesta))
    return respuesta
