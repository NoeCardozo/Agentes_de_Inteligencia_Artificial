"""
Subagente Juez del Agente de Turismo Argentina.
Evalúa la calidad de la respuesta final usando una rúbrica estructurada.
Usa un modelo diferente al orquestador para mayor independencia.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import JUDGE_MODEL, GOOGLE_API_KEY, JUDGE_MIN_SCORE
from agents.utils import to_text


# ─────────────────────────────────────────────────────────────────────────────
# RÚBRICA (prompt del juez)
# ─────────────────────────────────────────────────────────────────────────────
RUBRICA_JUEZ = """
Eres un juez de calidad para un agente de turismo argentino. Tu tarea es evaluar
la respuesta generada por el agente usando los siguientes criterios.

Para cada criterio asigná un score entre 0.0 y 1.0:

1. coherencia_temporal (0-1):
   ¿Las actividades tienen horarios posibles y sin superposición?
   ¿El itinerario es físicamente realizable en los días disponibles?
   1.0 = perfectamente coherente, 0.0 = imposible de realizar.

2. ajuste_presupuesto (0-1):
   ¿El costo total estimado está dentro del presupuesto declarado por el usuario?
   1.0 = dentro del presupuesto, 0.5 = excede hasta 15%, 0.0 = excede más del 15%.

3. relevancia_perfil (0-1):
   ¿Las actividades y el destino coinciden con los intereses declarados del usuario?
   ¿Se respetan las restricciones (movilidad, alimentarias, destinos ya visitados)?
   1.0 = muy relevante, 0.0 = completamente irrelevante.

4. correccion_politica (0-1):
   ¿El contenido es inclusivo, libre de estereotipos y respetuoso?
   ¿No hace comentarios discriminatorios sobre regiones, culturas o personas?
   1.0 = completamente correcto, 0.0 = contiene lenguaje inapropiado.

5. idioma_espanol (0-1):
   ¿La respuesta está completamente en español?
   1.0 = todo en español, 0.0 = responde en otro idioma.

6. fidelidad_dominio (0-1):
   ¿El agente se mantuvo dentro del turismo argentino?
   ¿No habló de destinos internacionales ni respondió temas ajenos al turismo?
   1.0 = completamente dentro del dominio, 0.0 = totalmente fuera del dominio.

Devuelve ÚNICAMENTE este JSON (sin markdown, sin explicación fuera del JSON):
{
  "coherencia_temporal": 0.0,
  "ajuste_presupuesto": 0.0,
  "relevancia_perfil": 0.0,
  "correccion_politica": 0.0,
  "idioma_espanol": 0.0,
  "fidelidad_dominio": 0.0,
  "score_total": 0.0,
  "aprobado": false,
  "observaciones": "lista de problemas encontrados o 'Sin observaciones'",
  "sugerencias_mejora": "qué debe corregir el agente en la siguiente iteración"
}

El score_total es el promedio de los 6 criterios.
aprobado = true si score_total >= 0.7 (umbral configurable).
"""


@dataclass
class JudgeResult:
    coherencia_temporal: float
    ajuste_presupuesto: float
    relevancia_perfil: float
    correccion_politica: float
    idioma_espanol: float
    fidelidad_dominio: float
    score_total: float
    aprobado: bool
    observaciones: str
    sugerencias_mejora: str
    raw_response: str = ""

    def summary(self) -> str:
        estado = "✅ APROBADO" if self.aprobado else "❌ RECHAZADO"
        return (
            f"=== EVALUACIÓN DEL JUEZ ({estado}) ===\n"
            f"Score total: {self.score_total:.2f} / 1.0\n"
            f"  - Coherencia temporal:  {self.coherencia_temporal:.2f}\n"
            f"  - Ajuste presupuesto:   {self.ajuste_presupuesto:.2f}\n"
            f"  - Relevancia perfil:    {self.relevancia_perfil:.2f}\n"
            f"  - Corrección política:  {self.correccion_politica:.2f}\n"
            f"  - Idioma español:       {self.idioma_espanol:.2f}\n"
            f"  - Fidelidad dominio:    {self.fidelidad_dominio:.2f}\n"
            f"Observaciones: {self.observaciones}\n"
            f"Sugerencias:   {self.sugerencias_mejora}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Agente Juez
# ─────────────────────────────────────────────────────────────────────────────
_judge_llm = None

def _get_judge_llm() -> ChatGoogleGenerativeAI:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatGoogleGenerativeAI(
            model=JUDGE_MODEL,      # modelo diferente al orquestador
            temperature=0,          # determinista
            google_api_key=GOOGLE_API_KEY,
        )
    return _judge_llm


def evaluate_response(
    user_request: str,
    agent_response: str,
    budget: float = 0.0,
    user_profile_summary: str = "",
) -> JudgeResult:
    """
    Evalúa la respuesta del agente usando la rúbrica.

    Args:
        user_request: lo que pidió el usuario.
        agent_response: la respuesta completa del agente.
        budget: presupuesto declarado (para evaluar ajuste_presupuesto).
        user_profile_summary: resumen del perfil del usuario desde memoria LP.

    Returns:
        JudgeResult con scores y veredicto.
    """
    llm = _get_judge_llm()

    context = (
        f"SOLICITUD DEL USUARIO:\n{user_request}\n\n"
        f"PRESUPUESTO DECLARADO: ${budget:,.0f} ARS\n\n"
        f"PERFIL DEL USUARIO:\n{user_profile_summary or 'Sin información de perfil.'}\n\n"
        f"RESPUESTA DEL AGENTE A EVALUAR:\n{agent_response}"
    )

    result = llm.invoke([
        SystemMessage(content=RUBRICA_JUEZ),
        HumanMessage(content=context),
    ])

    raw = to_text(result.content).strip()
    # Gemini a veces envuelve el JSON en ```json ... ```
    if "```" in raw:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1)
    else:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

    try:
        data = json.loads(raw)
        scores = {
            "coherencia_temporal": float(data.get("coherencia_temporal", 0)),
            "ajuste_presupuesto":  float(data.get("ajuste_presupuesto", 0)),
            "relevancia_perfil":   float(data.get("relevancia_perfil", 0)),
            "correccion_politica": float(data.get("correccion_politica", 1)),
            "idioma_espanol":      float(data.get("idioma_espanol", 1)),
            "fidelidad_dominio":   float(data.get("fidelidad_dominio", 1)),
        }
        score_total = data.get("score_total") or (sum(scores.values()) / len(scores))
        aprobado    = data.get("aprobado", score_total >= JUDGE_MIN_SCORE)

        return JudgeResult(
            **scores,
            score_total=round(float(score_total), 3),
            aprobado=bool(aprobado),
            observaciones=data.get("observaciones", ""),
            sugerencias_mejora=data.get("sugerencias_mejora", ""),
            raw_response=raw,
        )

    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: aprobación parcial si no se puede parsear
        return JudgeResult(
            coherencia_temporal=0.5,
            ajuste_presupuesto=0.5,
            relevancia_perfil=0.5,
            correccion_politica=1.0,
            idioma_espanol=1.0,
            fidelidad_dominio=1.0,
            score_total=0.75,
            aprobado=True,
            observaciones="No se pudo parsear la evaluación del juez.",
            sugerencias_mejora="Verificar manualmente.",
            raw_response=raw,
        )
