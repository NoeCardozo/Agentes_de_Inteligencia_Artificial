"""
=============================================================================
PUNTO DE ENTRADA DE LA APLICACIÓN
=============================================================================

Para ejecutar el proyecto:
    python main.py

Este archivo solo hace dos cosas:
  1. Configura los logs en la terminal (para ver qué hace el agente).
  2. Abre la ventana gráfica del chat (app/ui.py).
"""

import logging
import sys

from app.ui import run_app


def setup_logging() -> None:
    """Muestra mensajes informativos en la terminal mientras la app corre."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


if __name__ == "__main__":
    setup_logging()
    run_app()
