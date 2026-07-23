"""
Frontend Flask del Agente de Turismo Argentina.
Orquestador: Deep Agent + tools MCP, skills, subagentes, guardrails y juez.
HITL: botones Aprobar / Regenerar / Escalar en el chat.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request
from markupsafe import Markup
import markdown as md_lib
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Proyecto raíz en sys.path para importar config / mcp / agents / ...
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

from agents.judge_agent import evaluate_response
from agents.utils import to_text
from config.settings import (
    AUTO_DELIVER_ON_APPROVE,
    MAX_JUDGE_RETRIES,
    SYSTEM_PROMPT_BASE,
    USER_EMAIL,
)
from guardrails.guardrails import apply_input_guardrails, apply_output_guardrails
from integrations.google_tools import DELIVERY_TOOLS
from integrations.email_format import itinerary_to_email_html, polish_agent_reply
from integrations.google_workspace import (
    create_calendar_events,
    events_from_itinerary_via_llm,
    extract_events_json,
    oauth_configured,
    send_email,
    strip_events_json,
)
from mcp.mcp_client import (
    ALL_MCP_TOOLS,
    tool_actividades,
    tool_alojamiento,
    tool_clima,
    tool_consulta_gestion,
    tool_traslados,
)
from rag.rag_pipeline import load_vectorstore, run_rag_pipeline

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.template_filter("markdown")
def markdown_filter(text: str) -> Markup:
    """Renderiza Markdown seguro para burbujas del chat."""
    if not text:
        return Markup("")
    # Contenido que ya viene con <br> desde turns anteriores
    raw = str(text).replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    html = md_lib.markdown(
        raw,
        extensions=["nl2br", "sane_lists", "fenced_code"],
    )
    return Markup(html)

MESSAGES_PATH = Path(app.root_path) / "messages" / "messages.json"
PROFILE_PATH = Path(app.root_path) / "memories" / "profile.md"
SKILLS_DIR = Path(app.root_path) / "skills"
HITL_META_PATH = Path(app.root_path) / "messages" / "hitl_meta.json"

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + """
Sos el orquestador de un asistente de turismo argentino con memoria persistente
en /memories/profile.md y skills en /skills/.

Cuándo actualizar memoria:
- El usuario comparte nombre, preferencias, restricciones o feedback.
- Usá write_file / edit_file sobre /memories/profile.md (bullet points, sin credenciales).

Cómo trabajar:
1. Si faltan presupuesto (ARS), fechas, cantidad de viajeros o destino, pedilos
   antes de armar un plan completo.
2. Delegá trabajo especializado a subagentes (destinos, itinerario, precios, monitoreo)
   cuando la tarea lo justifique.
3. Usá las tools de clima, alojamiento, actividades, traslados y consulta_rag
   para datos concretos; no inventes precios ni horarios.
4. Cargá las skills de planificación, monitoreo, comparación de precios o
   entrega-viaje cuando corresponda al pedido.
5. Cuando armes un itinerario completo día a día, al final agregá exactamente:
   EVENTS_JSON:[{{"title":"...","start":"YYYY-MM-DDTHH:MM:SS","end":"...","location":"...","description":"..."}}]
   (horarios locales Argentina). Si el usuario pide enviar el viaje o ya aprobó,
   usá tool_enviar_email y tool_crear_eventos_calendario.
6. Si falta el email del usuario, pedilo. Email por defecto del entorno: {user_email}.
7. Respondé siempre en español, claro y accionable.
8. NUNCA narres tu proceso interno: no digas que vas a leer skills, consultar tools,
   actualizar todos, ni pegues dumps de actividades crudas antes del itinerario.
   Al usuario solo mostrá la respuesta final útil (preguntas claras o el itinerario).
9. Si ya tenés destino, fechas, viajeros y presupuesto, COMPLETÁ el itinerario en este
   mismo turno (consultá tools y respondé con el plan día a día). Nunca digas
   "te aviso cuando esté listo" ni dejes al usuario esperando.
""".format(user_email=USER_EMAIL or "(no configurado — pedilo)")


@tool
def tool_consulta_rag(question: str) -> str:
    """
    Consulta la base de conocimiento RAG sobre turismo argentino
    (destinos, normativas, tips). Usala para preguntas fácticas.
    """
    try:
        vectorstore = load_vectorstore()
    except Exception as exc:
        return f"Base de conocimiento no disponible: {exc}"
    try:
        _chunks, context = run_rag_pipeline(question, vectorstore, [])
    except Exception as exc:
        return f"Error en el pipeline RAG: {exc}"
    if not context or not context.strip():
        return "No encontré información relevante en la base de conocimiento."
    return context


TOURISM_TOOLS = [*ALL_MCP_TOOLS, tool_consulta_rag, *DELIVERY_TOOLS]

SUBAGENTS = [
    {
        "name": "destinos",
        "description": (
            "Recomienda o valida destinos turísticos argentinos según intereses, "
            "clima y presupuesto."
        ),
        "system_prompt": (
            "Sos un experto en destinos turísticos argentinos. Respondé en español. "
            "Si el usuario ya eligió destino, validalo y enriquecé; no propongas otros "
            "salvo que pida alternativas. Usá tool_clima y tool_actividades."
        ),
        "tools": [tool_clima, tool_actividades, tool_consulta_gestion],
    },
    {
        "name": "itinerario",
        "description": (
            "Arma itinerarios día a día con actividades, alojamiento y traslados "
            "dentro del presupuesto."
        ),
        "system_prompt": (
            "Sos un planificador de itinerarios en Argentina. Respondé en español. "
            "Seguí la skill de planificación. Incluí horarios, alojamiento y costos."
        ),
        "tools": [tool_actividades, tool_alojamiento, tool_traslados, tool_clima],
    },
    {
        "name": "precios",
        "description": (
            "Compara precios de alojamiento y traslados y optimiza el presupuesto."
        ),
        "system_prompt": (
            "Sos un comparador de precios turísticos en Argentina. Respondé en español. "
            "Seguí la skill de comparación de precios. Mostrá desglose claro."
        ),
        "tools": [tool_alojamiento, tool_traslados, tool_consulta_gestion],
    },
    {
        "name": "monitoreo",
        "description": (
            "Monitorea clima y condiciones del viaje; propone alternativas ante alertas."
        ),
        "system_prompt": (
            "Sos un monitor de condiciones turísticas en Argentina. Respondé en español. "
            "Seguí la skill de monitoreo. Criterio de lluvia: ≥20 mm/día = alerta."
        ),
        "tools": [tool_clima, tool_actividades, tool_traslados],
    },
]

checkpointer = MemorySaver()
store = InMemoryStore()

agent = create_deep_agent(
    model="google_genai:gemini-2.5-flash",
    system_prompt=SYSTEM_PROMPT,
    tools=TOURISM_TOOLS,
    subagents=SUBAGENTS,
    skills=["/skills/"],
    memory=["/memories/profile.md"],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": FilesystemBackend(
                root_dir=str(Path(app.root_path) / "memories"),
                virtual_mode=True,
            ),
            "/skills/": FilesystemBackend(
                root_dir=str(SKILLS_DIR),
                virtual_mode=True,
            ),
        },
    ),
    checkpointer=checkpointer,
    store=store,
)


# ── Persistencia de chat / HITL ───────────────────────────────────────────────

def _ensure_dirs() -> None:
    MESSAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PROFILE_PATH.exists():
        PROFILE_PATH.write_text(
            "# Perfil del viajero\n\n- (vacío)\n",
            encoding="utf-8",
        )


def load_messages() -> list[dict]:
    _ensure_dirs()
    if not MESSAGES_PATH.exists():
        return []
    try:
        with open(MESSAGES_PATH, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_messages(messages: list[dict]) -> None:
    _ensure_dirs()
    with open(MESSAGES_PATH, "wt", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def load_hitl_meta() -> dict:
    if not HITL_META_PATH.exists():
        return {"judge_retries": 0, "last_user_request": "", "budget": 0.0, "thread_id": ""}
    try:
        with open(HITL_META_PATH, "rt", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("thread_id", "")
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        return {"judge_retries": 0, "last_user_request": "", "budget": 0.0, "thread_id": ""}


def get_or_create_thread_id(incoming: str | None = None) -> str:
    """Persiste el thread_id para que el checkpointer no pierda contexto."""
    meta = load_hitl_meta()
    if incoming and incoming.strip():
        tid = incoming.strip()
    elif meta.get("thread_id"):
        tid = meta["thread_id"]
    else:
        tid = str(uuid7())
    if meta.get("thread_id") != tid:
        meta["thread_id"] = tid
        save_hitl_meta(meta)
    return tid


def save_hitl_meta(meta: dict) -> None:
    _ensure_dirs()
    with open(HITL_META_PATH, "wt", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def pending_hitl_index(messages: list[dict]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("type") == "ai" and m.get("status") == "pending_hitl":
            return i
    return None


def _message_text(content) -> str:
    """Extrae texto visible; ignora bloques de thinking internos del modelo."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str) and c.strip():
                parts.append(c)
                continue
            if not isinstance(c, dict):
                continue
            # Gemini / modelos con thought signatures
            ctype = (c.get("type") or "").lower()
            if ctype in {"thinking", "thought", "reasoning"}:
                continue
            text = c.get("text") or c.get("content") or ""
            if text and str(text).strip():
                parts.append(str(text))
        return "\n".join(parts)
    return to_text(content)


def _msg_type(message) -> str:
    t = getattr(message, "type", None)
    if t:
        return str(t)
    if isinstance(message, dict):
        return str(message.get("type") or "")
    name = type(message).__name__.lower()
    if "human" in name:
        return "human"
    if "ai" in name:
        return "ai"
    if "tool" in name:
        return "tool"
    return ""


_DAY1_IN_TEXT = re.compile(r"(?im)(?:\*\*)?d[ií]a\s*1\b")

_THOUGHT_ONLY = re.compile(
    r"(?is)^(?:\s|"
    r"voy a leer|voy a consultar|updated todo list|write_todos|"
    r"he recopilado la siguiente información|"
    r"ahora,? con esta información,? voy a diseñar"
    r").{0,400}$"
)


def _is_thought_only(text: str) -> bool:
    """True si el mensaje es solo narración interna, sin valor para el usuario."""
    if not text or not text.strip():
        return True
    if _DAY1_IN_TEXT.search(text):
        return False
    if "?" in text:
        return False
    lower = text.lower().strip()
    # Mensajes cortos de planning interno
    thought_starts = (
        "voy a leer",
        "voy a consultar",
        "voy a buscar",
        "voy a actualizar tu perfil",
        "updated todo list",
        "write_todos",
        "he recopilado la siguiente información",
        "ahora, con esta información, voy a diseñar",
        "ahora voy a diseñar",
    )
    if any(lower.startswith(p) or p in lower[:80] for p in thought_starts):
        # Si es largo y tiene contenido de viaje útil, no descartar
        useful = ("presupuesto", "bariloche", "itinerario", "hotel", "diciembre", "fecha")
        if len(text) > 500 and any(u in lower for u in useful):
            return False
        if len(text) < 450:
            return True
    return False


def append_agent_turn(messages: list[dict], res: dict, skip_human: bool = True) -> str:
    """
    Agrega SOLO la respuesta final del turno actual.
    No guarda tools ni cadenas de pensamiento.
    """
    result_msgs = list(res.get("messages") or [])
    if not result_msgs:
        return ""

    start_idx = 0
    for i, message in enumerate(result_msgs):
        if _msg_type(message) == "human":
            start_idx = i

    turn = result_msgs[start_idx:]
    candidates: list[dict] = []

    for message in turn:
        msg_type = _msg_type(message)
        if msg_type != "ai":
            continue
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content", "")
        text = _message_text(content).strip()
        if not text:
            continue
        tokens = None
        usage = getattr(message, "usage_metadata", None)
        if usage:
            tokens = (
                usage.get("total_tokens")
                if isinstance(usage, dict)
                else getattr(usage, "get", lambda *_: None)("total_tokens")
            )
        entry = {"type": "ai", "content": text}
        if tokens is not None:
            entry["total_tokens"] = tokens
        candidates.append(entry)

    if not candidates:
        return ""

    # Preferir la última respuesta que no sea solo pensamiento
    chosen = None
    for entry in reversed(candidates):
        if not _is_thought_only(entry["content"]):
            chosen = entry
            break
    if chosen is None:
        chosen = candidates[-1]

    chosen["content"] = polish_agent_reply(chosen["content"])
    if not (chosen["content"] or "").strip():
        # polish no debería vaciar; fallback al texto original
        chosen["content"] = candidates[-1]["content"]

    messages.append(chosen)
    return chosen["content"]


def _looks_like_itinerary(text: str) -> bool:
    """Solo planes día a día concretos, no menciones sueltas de 'itinerario'."""
    lower = (text or "").lower()
    day_markers = (
        "día 1", "dia 1", "día 2", "dia 2", "día 3", "dia 3",
        "noche 1", "día 4", "dia 4", "día 5", "dia 5", "día 6", "dia 6",
    )
    if any(m in lower for m in day_markers):
        return True
    strong = (
        "plan día a día", "plan dia a dia", "itinerario detallado",
        "costo acumulado", "desglose de costos",
    )
    return sum(1 for m in strong if m in lower) >= 1


def _is_clarifying_reply(text: str) -> bool:
    """
    Respuestas conversacionales que piden datos faltantes:
    no deben pasar por juez ni HITL.
    """
    lower = (text or "").lower()
    # Plan día a día real → no es aclaración
    if any(m in lower for m in ("día 1", "dia 1", "día 2", "dia 2", "día 3", "dia 3")):
        return False

    ask_markers = (
        "necesito", "podrías", "podrias", "puedes", "podés", "podes",
        "indicarme", "confirmarme", "confirmame", "decime", "contame",
        "¿cuáles", "cuales", "¿cuál", "cual ", "¿cuántas", "cuantas",
        "¿cuántos", "cuantos", "fechas", "presupuesto", "viajarán", "viajaran",
        "para poder ayudarte", "faltan", "me falta", "te parece bien",
        "otras fechas", "házmelo saber", "hazmelo saber",
    )
    question_marks = lower.count("?")
    if question_marks >= 1 and any(m in lower for m in ask_markers):
        return True
    # Varias preguntas aunque no matchee marker exacto
    return question_marks >= 2 and not _looks_like_itinerary(text)


def requires_hitl_review(text: str, events: list | None = None) -> bool:
    """HITL + juez solo para entregables de itinerario/plan concreto."""
    if events:
        return True
    if _is_clarifying_reply(text):
        return False
    return _looks_like_itinerary(text)


def _log_agent_message(message) -> None:
    """Imprime en terminal el progreso del agente (tools / respuestas)."""
    msg_type = _msg_type(message)
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content", "")
    text = _message_text(content).strip()
    name = getattr(message, "name", None) or (
        message.get("name") if isinstance(message, dict) else None
    )

    def _out(line: str) -> None:
        print(line, flush=True)

    if msg_type == "human":
        preview = (text[:160] + "…") if len(text) > 160 else text
        _out(f"\n[USER] {preview}")
        return

    if msg_type == "ai":
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                tname = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                args_s = json.dumps(args, ensure_ascii=False)
                if len(args_s) > 220:
                    args_s = args_s[:220] + "…"
                _out(f"[TOOL→] {tname}({args_s})")
        if text:
            preview = (text[:220] + "…") if len(text) > 220 else text
            preview = preview.replace("\n", " ")
            _out(f"[AI] {preview}")
        return

    if msg_type in {"tool", "ToolMessage"} or name:
        label = name or "tool"
        preview = (text[:220] + "…") if len(text) > 220 else text
        preview = preview.replace("\n", " ")
        _out(f"[TOOL←] {label}: {preview}")
        return

    if text:
        preview = (text[:160] + "…") if len(text) > 160 else text
        _out(f"[{msg_type or 'MSG'}] {preview}")


def invoke_agent_with_logs(user_input: str, config: dict) -> dict:
    """
    Ejecuta el Deep Agent mostrando en terminal cada paso
    (llamadas a tools, resultados y borradores de respuesta).
    """
    print("\n" + "=" * 64, flush=True)
    print("[AGENTE] Iniciando turno", flush=True)
    print(f"[AGENTE] thread_id={config.get('configurable', {}).get('thread_id')}", flush=True)
    print(f"[AGENTE] recursion_limit={config.get('recursion_limit')}", flush=True)
    print("=" * 64, flush=True)

    final_state: dict | None = None
    seen = 0
    for state in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="values",
    ):
        final_state = state
        msgs = list((state or {}).get("messages") or [])
        for message in msgs[seen:]:
            _log_agent_message(message)
        seen = len(msgs)

    print("=" * 64, flush=True)
    print(f"[AGENTE] Turno terminado ({seen} mensajes en el hilo)", flush=True)
    print("=" * 64 + "\n", flush=True)
    return final_state or {"messages": []}


def run_agent_and_judge(
    user_input: str,
    thread_id: str,
    messages: list[dict],
    *,
    is_regenerate: bool = False,
) -> tuple[list[dict], str]:
    """Invoca Deep Agent; juez+HITL solo si hay itinerario/plan para revisar."""
    meta = load_hitl_meta()
    if not is_regenerate:
        meta["last_user_request"] = user_input
        meta["judge_retries"] = 0
        save_hitl_meta(meta)

    has_session = any(m.get("type") == "ai" for m in messages)
    # Contexto reciente para que el guardrail de dominio no pierda el destino
    recent = []
    for m in messages[-6:]:
        role = "Usuario" if m.get("type") == "human" else "Agente"
        content = (m.get("content") or "")[:400]
        if content:
            recent.append(f"{role}: {content}")
    conversation_context = "\n".join(recent)
    ok, blocked = apply_input_guardrails(
        user_input,
        has_active_session=has_session,
        conversation_context=conversation_context,
    )
    if not ok:
        messages.append({"type": "human", "content": user_input})
        messages.append({
            "type": "ai",
            "content": blocked,
            "status": "blocked",
        })
        return messages, blocked

    if not is_regenerate:
        messages.append({"type": "human", "content": user_input})

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 80,
    }
    try:
        res = invoke_agent_with_logs(user_input, config)
    except Exception as exc:
        err = (
            f"Hubo un error al procesar tu consulta. "
            f"Detalle técnico: {type(exc).__name__}: {exc}"
        )
        print(f"[ERROR] agent.stream/invoke: {exc}")
        messages.append({"type": "ai", "content": err, "status": "error"})
        return messages, err

    last_ai = append_agent_turn(messages, res, skip_human=True)
    if not last_ai:
        # Diagnóstico: el modelo a veces termina solo con tool calls
        n_msgs = len(res.get("messages") or [])
        err = (
            "No recibí una respuesta de texto del agente (posible corte tras tools). "
            f"Mensajes internos: {n_msgs}. Probá de nuevo con el mismo pedido."
        )
        print(f"[WARN] empty AI reply. keys={list(res.keys()) if isinstance(res, dict) else type(res)} msgs={n_msgs}")
        messages.append({"type": "ai", "content": err, "status": "error"})
        return messages, err

    ok_out, cleaned = apply_output_guardrails(last_ai)
    last_ai = cleaned if ok_out else (cleaned or last_ai)
    events = extract_events_json(last_ai)
    if events:
        meta["pending_events"] = events
        save_hitl_meta(meta)
    visible_ai = polish_agent_reply(strip_events_json(last_ai)) or last_ai

    needs_hitl = is_regenerate or requires_hitl_review(visible_ai, events)

    # Actualizar último mensaje AI
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("type") != "ai":
            continue
        if messages[i].get("status") in {
            "approved", "escalated", "superseded", "blocked", "info", "error",
        }:
            continue
        messages[i]["content"] = visible_ai

        if not needs_hitl:
            # Conversación normal: sin juez ni botones HITL
            messages[i]["status"] = "ok"
            messages[i].pop("judge_summary", None)
            messages[i].pop("judge", None)
            break

        judge = evaluate_response(
            user_request=meta.get("last_user_request") or user_input,
            agent_response=visible_ai,
            budget=float(meta.get("budget") or 0),
        )
        messages[i]["status"] = "pending_hitl"
        messages[i]["judge_summary"] = judge.summary()
        messages[i]["judge"] = {
            "score_total": judge.score_total,
            "aprobado": judge.aprobado,
            "observaciones": judge.observaciones,
            "sugerencias_mejora": judge.sugerencias_mejora,
        }
        break

    return messages, visible_ai


def deliver_itinerary_on_approve(itinerary_text: str) -> str:
    """
    Envía el itinerario por Gmail y crea eventos en Calendar.
    Retorna un resumen para mostrar en el chat.
    """
    if not AUTO_DELIVER_ON_APPROVE:
        return ""
    if not oauth_configured():
        return (
            "Entrega automática omitida: falta OAuth Google "
            "(credentials/credentials.json + python scripts/google_oauth_setup.py)."
        )
    if not USER_EMAIL:
        return (
            "Entrega automática omitida: configurá USER_EMAIL en .env "
            "o pedile al agente que envíe el correo con tu email."
        )
    if not _looks_like_itinerary(itinerary_text):
        return "Respuesta aprobada (no parece un itinerario completo; no se envió email/calendario)."

    lines = []
    meta = load_hitl_meta()
    clean_body = polish_agent_reply(strip_events_json(itinerary_text))
    html_body = itinerary_to_email_html(clean_body, destination="Bariloche" if "bariloche" in clean_body.lower() else "")

    try:
        sent = send_email(
            USER_EMAIL,
            "Tu itinerario de viaje — Agente Turismo Argentina",
            html_body,
            html=True,
        )
        lines.append(f"Email enviado a {sent['to']}.")
    except Exception as exc:
        lines.append(f"No se pudo enviar el email: {exc}")

    events = meta.get("pending_events") or extract_events_json(itinerary_text)
    if not events:
        try:
            events = events_from_itinerary_via_llm(clean_body)
        except Exception as exc:
            lines.append(f"No se pudieron inferir eventos: {exc}")
            events = []

    if events:
        try:
            result = create_calendar_events(events)
            lines.append(
                f"Google Calendar: {result['created']} evento(s) creado(s)"
                + (f", {result['failed']} fallido(s)" if result.get("failed") else "")
                + "."
            )
            for ev in (result.get("events") or [])[:5]:
                if ev.get("htmlLink"):
                    lines.append(f"  · {ev.get('summary')}: {ev['htmlLink']}")
        except Exception as exc:
            lines.append(f"No se pudieron crear eventos en Calendar: {exc}")
    else:
        lines.append("No se encontraron eventos estructurados para el calendario.")

    meta["pending_events"] = []
    save_hitl_meta(meta)
    return "\n".join(lines)


def handle_hitl_approve(messages: list[dict]) -> list[dict]:
    idx = pending_hitl_index(messages)
    itinerary_text = ""
    if idx is not None:
        messages[idx]["status"] = "approved"
        itinerary_text = messages[idx].get("content") or ""
    meta = load_hitl_meta()
    meta["judge_retries"] = 0
    save_hitl_meta(meta)

    delivery = deliver_itinerary_on_approve(itinerary_text)
    if delivery:
        messages.append({
            "type": "ai",
            "content": f"Entrega del viaje:\n{delivery}",
            "status": "info",
        })
    return messages


def handle_hitl_escalate(messages: list[dict]) -> list[dict]:
    idx = pending_hitl_index(messages)
    if idx is not None:
        messages[idx]["status"] = "escalated"
    messages.append({
        "type": "ai",
        "content": (
            "Esta consulta fue escalada a un operador humano. "
            "Un integrante del equipo la revisará y te contactará a la brevedad. "
            "Mientras tanto podés seguir consultando sobre otros destinos o viajes."
        ),
        "status": "info",
    })
    meta = load_hitl_meta()
    meta["judge_retries"] = 0
    save_hitl_meta(meta)
    return messages


def handle_hitl_regenerate(messages: list[dict], thread_id: str) -> list[dict]:
    idx = pending_hitl_index(messages)
    meta = load_hitl_meta()
    retries = int(meta.get("judge_retries") or 0)

    if retries >= MAX_JUDGE_RETRIES:
        if idx is not None:
            messages[idx]["status"] = "escalated"
        messages.append({
            "type": "ai",
            "content": (
                f"Se alcanzó el máximo de regeneraciones ({MAX_JUDGE_RETRIES}). "
                "La consulta se escala a un operador humano."
            ),
            "status": "info",
        })
        meta["judge_retries"] = 0
        save_hitl_meta(meta)
        return messages

    judge_feedback = ""
    if idx is not None:
        judge = messages[idx].get("judge") or {}
        judge_feedback = (
            f"{judge.get('observaciones', '')}\n"
            f"Sugerencias: {judge.get('sugerencias_mejora', '')}"
        ).strip()
        messages[idx]["status"] = "superseded"

    meta["judge_retries"] = retries + 1
    save_hitl_meta(meta)

    last_request = meta.get("last_user_request") or "Reformulá la respuesta anterior."
    regen_prompt = (
        f"OBSERVACIONES DEL JUEZ (debés corregirlas):\n{judge_feedback}\n\n"
        f"Solicitud original del usuario:\n{last_request}\n\n"
        "Regenerá una respuesta mejorada en español, corrigiendo los puntos indicados."
    )
    return run_agent_and_judge(
        regen_prompt,
        thread_id,
        messages,
        is_regenerate=True,
    )[0]


# ── Rutas Flask ───────────────────────────────────────────────────────────────

@app.get("/")
def index_get():
    messages = load_messages()
    thread_id = get_or_create_thread_id()
    return render_template(
        "index.html",
        messages=messages,
        thread_id=thread_id,
        pending_hitl=pending_hitl_index(messages) is not None,
        max_retries=MAX_JUDGE_RETRIES,
        judge_retries=load_hitl_meta().get("judge_retries", 0),
    )


@app.post("/")
def index_post():
    thread_id = get_or_create_thread_id(request.form.get("thread_id"))
    show_tool_messages = bool(request.form.get("show_tool_messages"))

    # Reset
    if request.form.get("reset_button"):
        if MESSAGES_PATH.exists():
            MESSAGES_PATH.unlink()
        if HITL_META_PATH.exists():
            HITL_META_PATH.unlink()
        if PROFILE_PATH.exists():
            PROFILE_PATH.unlink()
        _ensure_dirs()
        new_tid = str(uuid7())
        save_hitl_meta({
            "judge_retries": 0,
            "last_user_request": "",
            "budget": 0.0,
            "thread_id": new_tid,
        })
        return render_template(
            "index.html",
            messages=[],
            show_tool_messages=show_tool_messages,
            thread_id=new_tid,
            pending_hitl=False,
            max_retries=MAX_JUDGE_RETRIES,
            judge_retries=0,
        )

    messages = load_messages()

    # HITL actions
    if request.form.get("hitl_approve"):
        messages = handle_hitl_approve(messages)
        save_messages(messages)
        return render_template(
            "index.html",
            messages=messages,
            show_tool_messages=show_tool_messages,
            thread_id=thread_id,
            pending_hitl=False,
            max_retries=MAX_JUDGE_RETRIES,
            judge_retries=0,
        )

    if request.form.get("hitl_escalate"):
        messages = handle_hitl_escalate(messages)
        save_messages(messages)
        return render_template(
            "index.html",
            messages=messages,
            show_tool_messages=show_tool_messages,
            thread_id=thread_id,
            pending_hitl=False,
            max_retries=MAX_JUDGE_RETRIES,
            judge_retries=0,
        )

    if request.form.get("hitl_regenerate"):
        messages = handle_hitl_regenerate(messages, thread_id)
        save_messages(messages)
        meta = load_hitl_meta()
        return render_template(
            "index.html",
            messages=messages,
            show_tool_messages=show_tool_messages,
            thread_id=thread_id,
            pending_hitl=pending_hitl_index(messages) is not None,
            max_retries=MAX_JUDGE_RETRIES,
            judge_retries=meta.get("judge_retries", 0),
        )

    # Bloquear nuevos mensajes si hay draft pendiente
    if pending_hitl_index(messages) is not None:
        return render_template(
            "index.html",
            messages=messages,
            show_tool_messages=show_tool_messages,
            thread_id=thread_id,
            pending_hitl=True,
            max_retries=MAX_JUDGE_RETRIES,
            judge_retries=load_hitl_meta().get("judge_retries", 0),
            form_error="Primero aprobá, regenerá o escalá la respuesta pendiente.",
        )

    user_input = (request.form.get("user_input") or "").strip()
    if not user_input:
        return render_template(
            "index.html",
            messages=messages,
            show_tool_messages=show_tool_messages,
            thread_id=thread_id,
            pending_hitl=False,
            max_retries=MAX_JUDGE_RETRIES,
            judge_retries=load_hitl_meta().get("judge_retries", 0),
        )

    messages, _ = run_agent_and_judge(user_input, thread_id, messages)
    save_messages(messages)
    meta = load_hitl_meta()

    return render_template(
        "index.html",
        messages=messages,
        show_tool_messages=show_tool_messages,
        thread_id=thread_id,
        pending_hitl=pending_hitl_index(messages) is not None,
        max_retries=MAX_JUDGE_RETRIES,
        judge_retries=meta.get("judge_retries", 0),
    )


if __name__ == "__main__":
    _ensure_dirs()
    app.run(debug=True, port=5000)
