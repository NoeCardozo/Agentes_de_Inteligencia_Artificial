"""
Memoria a largo plazo del Agente de Turismo Argentina.
Persiste: perfil del usuario, historial de viajes, itinerarios previos y feedback.
Usa un archivo JSON por usuario (simple y portable); en producción reemplazar por DB.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import MEMORY_PERSIST_DIR, MEMORY_TTL_DAYS


def _user_path(user_id: str) -> Path:
    path = Path(MEMORY_PERSIST_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{user_id}.json"


def _load_profile(user_id: str) -> dict:
    p = _user_path(user_id)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "preferences": {},           # gustos positivos
        "dislikes": [],              # feedback negativo persistente
        "visited_destinations": [],  # destinos ya visitados
        "budget_history": [],        # presupuestos históricos
        "itineraries": [],           # itinerarios anteriores (últimos N)
        "dietary_restrictions": [],  # restricciones alimentarias
        "mobility_restrictions": [], # restricciones de movilidad
    }


def _save_profile(user_id: str, profile: dict) -> None:
    with open(_user_path(user_id), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


# ── API pública ───────────────────────────────────────────────────────────────

def get_user_profile(user_id: str) -> dict:
    """Carga el perfil del usuario desde disco."""
    return _load_profile(user_id)


def update_preferences(user_id: str, new_prefs: dict) -> None:
    """
    Actualiza las preferencias del usuario.
    new_prefs: dict con claves como 'tipo_alojamiento', 'actividades_preferidas', etc.
    """
    profile = _load_profile(user_id)
    profile["preferences"].update(new_prefs)
    profile["updated_at"] = datetime.now().isoformat()
    _save_profile(user_id, profile)


def add_negative_feedback(user_id: str, item: str, reason: str = "") -> None:
    """
    Registra feedback negativo persistente.
    El agente evitará recomendar este ítem en sesiones futuras.
    """
    profile = _load_profile(user_id)
    dislike_entry = {
        "item": item,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    }
    profile["dislikes"].append(dislike_entry)
    _save_profile(user_id, profile)


def add_visited_destination(user_id: str, destination: str) -> None:
    """Registra un destino como visitado para evitar repetirlo sin que el usuario lo pida."""
    profile = _load_profile(user_id)
    if destination not in profile["visited_destinations"]:
        profile["visited_destinations"].append(destination)
    _save_profile(user_id, profile)


def save_itinerary(user_id: str, itinerary: dict) -> None:
    """
    Persiste un itinerario generado. Mantiene los últimos 10.
    itinerary debe incluir: { 'destination', 'dates', 'budget', 'days': [...] }
    """
    profile = _load_profile(user_id)
    entry = {
        "saved_at": datetime.now().isoformat(),
        **itinerary,
    }
    profile["itineraries"].append(entry)
    # Mantener solo los últimos 10
    profile["itineraries"] = profile["itineraries"][-10:]
    _save_profile(user_id, profile)


def get_last_itinerary(user_id: str) -> dict | None:
    """Retorna el último itinerario guardado, o None."""
    profile = _load_profile(user_id)
    itineraries = profile.get("itineraries", [])
    return itineraries[-1] if itineraries else None


def record_budget(user_id: str, amount: float, currency: str = "ARS") -> None:
    """Registra el presupuesto de un viaje para análisis histórico."""
    profile = _load_profile(user_id)
    profile["budget_history"].append({
        "amount": amount,
        "currency": currency,
        "timestamp": datetime.now().isoformat(),
    })
    _save_profile(user_id, profile)


def get_user_context_summary(user_id: str) -> str:
    """
    Genera un resumen del perfil del usuario para incluir en el prompt del agente.
    """
    profile = _load_profile(user_id)
    visited = profile.get("visited_destinations", [])
    dislikes = [d["item"] for d in profile.get("dislikes", [])]
    prefs = profile.get("preferences", {})

    lines = ["=== Contexto del usuario (memoria de largo plazo) ==="]

    if visited:
        lines.append(f"Destinos ya visitados: {', '.join(visited)}")
    if dislikes:
        lines.append(f"Preferencias negativas (no recomendar): {', '.join(dislikes)}")
    if prefs:
        for k, v in prefs.items():
            lines.append(f"{k}: {v}")
    if profile.get("dietary_restrictions"):
        lines.append(f"Restricciones alimentarias: {', '.join(profile['dietary_restrictions'])}")
    if profile.get("mobility_restrictions"):
        lines.append(f"Restricciones de movilidad: {', '.join(profile['mobility_restrictions'])}")

    budget_history = profile.get("budget_history", [])
    if budget_history:
        avg_budget = sum(b["amount"] for b in budget_history) / len(budget_history)
        lines.append(f"Presupuesto promedio histórico: ${avg_budget:,.0f}")

    last_it = get_last_itinerary(user_id)
    if last_it:
        lines.append(
            f"Último viaje planificado: {last_it.get('destination', 'desconocido')} "
            f"({last_it.get('dates', '')})"
        )

    session = profile.get("session", {})
    if session.get("destino"):
        lines.append(f"Conversación en curso — destino: {session['destino']}")
    if session.get("proposed_periods"):
        opts = []
        for p in session["proposed_periods"]:
            opts.append(
                f"opción {p.get('n')}: {p.get('inicio')} a {p.get('fin')} ({p.get('label', '')})"
            )
        lines.append("Períodos propuestos: " + "; ".join(opts))
    if session.get("last_response_summary"):
        lines.append(f"Última respuesta del agente (resumen): {session['last_response_summary']}")

    if len(lines) == 1:
        lines.append("Sin historial previo.")

    return "\n".join(lines)


def get_session(user_id: str) -> dict:
    """Estado corto de la conversación actual (períodos propuestos, destino, etc.)."""
    return _load_profile(user_id).get("session", {})


def save_session(user_id: str, session_data: dict) -> None:
    """Persiste/actualiza el estado de sesión de la conversación."""
    profile = _load_profile(user_id)
    current = profile.get("session", {})
    current.update(session_data)
    current["updated_at"] = datetime.now().isoformat()
    profile["session"] = current
    _save_profile(user_id, profile)


def clear_session(user_id: str) -> None:
    profile = _load_profile(user_id)
    profile["session"] = {}
    _save_profile(user_id, profile)


def purge_old_data(user_id: str) -> None:
    """Elimina entradas más viejas que MEMORY_TTL_DAYS del historial."""
    profile = _load_profile(user_id)
    cutoff = datetime.now() - timedelta(days=MEMORY_TTL_DAYS)

    profile["dislikes"] = [
        d for d in profile.get("dislikes", [])
        if datetime.fromisoformat(d["timestamp"]) > cutoff
    ]
    profile["budget_history"] = [
        b for b in profile.get("budget_history", [])
        if datetime.fromisoformat(b["timestamp"]) > cutoff
    ]
    _save_profile(user_id, profile)
