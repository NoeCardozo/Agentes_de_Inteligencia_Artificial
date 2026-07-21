"""
Punto de entrada del Agente de Turismo Argentina.
Uso:
  python main.py                   # modo interactivo
  python main.py --eval            # corre evaluación
  python main.py --demo            # corre un ejemplo predefinido
"""
import argparse
import sys

from agents.graph import get_compiled_graph, AgentState
from evaluation.evaluator import (
    run_guardrail_evaluation,
    print_evaluation_report,
)
from memory.long_term_memory import get_session
from agents.utils import (
    is_conversation_followup,
    resolve_option_choice,
    asks_for_period_recommendation,
    has_active_trip_session,
    is_new_trip_request,
)


def run_agent(
    user_id: str,
    message: str,
    budget: float = 0,
    fecha_inicio: str = "",
    fecha_fin: str = "",
    intereses: str = "",
    origen: str = "Buenos Aires",
    viajeros: int = 1,
    presupuesto_tipo: str = "total_grupo",
    thread_id: str = "default",
) -> str:
    """
    Ejecuta el grafo LangGraph para una consulta del usuario.
    Retorna la respuesta final del agente.
    """
    app = get_compiled_graph()

    initial_state: AgentState = {
        "user_id":             user_id,
        "user_message":        message,
        "budget":              budget,
        "fecha_inicio":        fecha_inicio,
        "fecha_fin":           fecha_fin,
        "intereses":           intereses,
        "origen":              origen,
        "viajeros":            viajeros,
        "presupuesto_tipo":    presupuesto_tipo,
        "intent":              "",
        "guardrail_passed":    True,
        "guardrail_message":   "",
        "user_context":        "",
        "judge_retries":       0,
        "judge_result":        None,
        "judge_feedback":      "",
        "destinos_resultado":  "",
        "destino_elegido":     "",
        "itinerario_resultado": "",
        "precios_resultado":   "",
        "monitoreo_resultado": {},
        "rag_resultado":       "",
        "final_response":      "",
        "awaiting_human":      False,
        "human_decision":      "pendiente",
        "error":               "",
    }

    config = {"configurable": {"thread_id": thread_id}}

    print("\n[AGENTE] Procesando consulta...\n")
    final_state = app.invoke(initial_state, config=config)

    response = final_state.get("final_response", "No se pudo generar una respuesta.")

    # Mostrar evaluación del juez si está disponible
    judge = final_state.get("judge_result")
    if judge:
        print(f"\n{judge.get('summary', '')}\n")

    return response


def parse_budget(raw: str) -> float:
    """
    Acepta formatos como: 1500000, $1.500.000, 1.500.000,50, 1500000.50
    """
    if not raw:
        return 0.0
    cleaned = (
        raw.strip()
        .replace("$", "")
        .replace("ARS", "")
        .replace(" ", "")
    )
    # Formato argentino: puntos de miles y coma decimal
    if "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif cleaned.count(".") > 1:
        # Solo separadores de miles: 1.500.000
        cleaned = cleaned.replace(".", "")
    try:
        return float(cleaned)
    except ValueError:
        print(f"  Aviso: no pude interpretar '{raw}' como presupuesto. Se usará 0.")
        return 0.0


def parse_optional_date(raw: str) -> str:
    """Devuelve YYYY-MM-DD o '' si está vacío / 'no sé' / no es fecha."""
    from agents.utils import looks_like_iso_date

    if not raw or not raw.strip():
        return ""
    text = raw.strip()
    if looks_like_iso_date(text):
        return text
    lower = text.lower()
    if any(
        kw in lower
        for kw in (
            "no sé", "no se", "no tengo", "aún", "aun", "después", "despues",
            "flexible", "cualquiera", "no importa", "sin definir", "ns/nc",
        )
    ):
        return ""
    print(f"  Aviso: '{raw}' no es una fecha YYYY-MM-DD. Se omite.")
    return ""


def interactive_mode():
    """Modo de conversación interactiva."""
    print("="*60)
    print("  AGENTE DE TURISMO ARGENTINA")
    print("  Solo en español · Solo destinos argentinos")
    print("="*60)
    print("Escribí tu consulta de viaje (o 'salir' para terminar).\n")
    print("Ejemplo: Quiero ir a Bariloche a fin de año. ¿Qué período me recomendás?\n")

    user_id = input("Tu nombre de usuario: ").strip() or "usuario_demo"
    last_budget = 0.0
    last_viajeros = 1
    last_presupuesto_tipo = "por_persona"
    last_intereses = ""

    while True:
        print()
        message = input("¿En qué te puedo ayudar con tu viaje? ").strip()
        if message.lower() in {"salir", "exit", "quit"}:
            print("¡Hasta la próxima! Buen viaje.")
            break
        if not message:
            continue

        session = get_session(user_id)
        followup = (
            has_active_trip_session(session)
            and not is_new_trip_request(message, session)
            and (
                is_conversation_followup(message)
                or resolve_option_choice(message, session) is not None
                or bool(session.get("chosen_period") or session.get("last_intent"))
            )
        )

        if followup:
            # Reutilizar todo el contexto previo sin volver a preguntar
            budget = last_budget or float(session.get("budget") or 0)
            viajeros = last_viajeros or int(session.get("viajeros") or 1)
            presupuesto_tipo = (
                "por_persona"
                if viajeros == 1
                else (last_presupuesto_tipo or session.get("presupuesto_tipo") or "total_grupo")
            )
            chosen = session.get("chosen_period") or {}
            fecha_inicio = chosen.get("inicio", "")
            fecha_fin = chosen.get("fin", "")
            intereses = last_intereses or ""
            destino = session.get("destino") or ""
            print(
                f"  Seguimos con tu viaje"
                + (f" a {destino}" if destino else "")
                + f" (presupuesto ${budget:,.0f} ARS, {viajeros} persona/s)."
            )
        else:
            budget_str = input("Presupuesto en ARS (Enter para omitir): ").strip()
            budget = parse_budget(budget_str)

            viajeros_str = input("¿Cuántas personas viajan? (Enter = 1): ").strip()
            try:
                viajeros = max(1, int(viajeros_str)) if viajeros_str else 1
            except ValueError:
                viajeros = 1
                print("  Aviso: no entendí la cantidad de viajeros. Se usará 1.")

            if viajeros == 1:
                # Con una sola persona el presupuesto es necesariamente individual
                presupuesto_tipo = "por_persona"
            else:
                tipo_str = input(
                    "¿El presupuesto es [1] por persona o [2] total del grupo? (Enter=2): "
                ).strip()
                presupuesto_tipo = "por_persona" if tipo_str == "1" else "total_grupo"

            # Si pide recomendación de período, no tiene sentido pedir fechas aún
            if asks_for_period_recommendation(message):
                fecha_inicio = ""
                fecha_fin = ""
                print("  (Las fechas las vemos con las opciones que te proponga.)")
            else:
                fecha_inicio = parse_optional_date(
                    input(
                        "Fecha de inicio (YYYY-MM-DD, o Enter si aún no sabés): "
                    ).strip()
                )
                if fecha_inicio:
                    fecha_fin = parse_optional_date(
                        input(
                            "Fecha de fin (YYYY-MM-DD, o Enter si aún no sabés): "
                        ).strip()
                    )
                else:
                    fecha_fin = ""

            intereses = input(
                "Intereses (aventura, cultura, gastronomía..., Enter para omitir): "
            ).strip()

        last_budget = budget
        last_viajeros = viajeros
        last_presupuesto_tipo = presupuesto_tipo
        if intereses:
            last_intereses = intereses

        response = run_agent(
            user_id=user_id,
            message=message,
            budget=budget,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            intereses=intereses,
            viajeros=viajeros,
            presupuesto_tipo=presupuesto_tipo,
            thread_id=user_id,
        )

        print(f"\nAgente: {response}")


def demo_mode():
    """Corre un ejemplo predefinido sin input del usuario."""
    print("\n[DEMO] Planificación de viaje a Mendoza\n")

    response = run_agent(
        user_id="demo_user",
        message="Quiero planificar un viaje a Mendoza para visitar bodegas y hacer trekking.",
        budget=150_000,
        fecha_inicio="2025-09-15",
        fecha_fin="2025-09-20",
        intereses="gastronomia, aventura, naturaleza",
        origen="Buenos Aires",
        thread_id="demo",
    )

    print("\n" + "="*60)
    print("RESPUESTA DEL AGENTE:")
    print("="*60)
    print(response)

    print("\n[DEMO] Consulta fuera de dominio\n")
    response2 = run_agent(
        user_id="demo_user",
        message="Escribime un poema sobre el mar.",
        thread_id="demo2",
    )
    print(f"Respuesta: {response2}")

    print("\n[DEMO] Consulta en otro idioma\n")
    response3 = run_agent(
        user_id="demo_user",
        message="I want to travel to Patagonia, can you help me?",
        thread_id="demo3",
    )
    print(f"Respuesta: {response3}")


def evaluation_mode():
    """Corre evaluaciones y muestra el reporte."""
    print("\n[EVAL] Corriendo evaluación de guardarraíles...\n")
    guardrail_results = run_guardrail_evaluation()
    print_evaluation_report(guardrail_results=guardrail_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agente de Turismo Argentina")
    parser.add_argument("--eval", action="store_true", help="Correr evaluación")
    parser.add_argument("--demo", action="store_true", help="Correr demo predefinido")
    args = parser.parse_args()

    if args.eval:
        evaluation_mode()
    elif args.demo:
        demo_mode()
    else:
        interactive_mode()
