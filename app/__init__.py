"""
=============================================================================
AGENTE DE TURISMO - Paquete principal
=============================================================================

Este proyecto es un asistente de viajes con inteligencia artificial.

ARQUITECTURA (cómo se conectan las piezas):
-------------------------------------------
  main.py      → Punto de entrada. Ejecuta la aplicación.
  app/ui.py    → Ventana gráfica (Tkinter) donde el usuario chatea.
  app/agent.py → Cerebro: conecta Gemini + LangGraph.
  app/argentina_api.py → APIs públicas: Georef (gobierno) + Wikipedia.
  app/tools.py         → Acciones: ciudades AR, vuelos, itinerarios, guías.
  app/rag.py   → RAG: busca información en guías turísticas (.txt).
  app/config.py→ Configuración (API key, modelo, instrucciones al agente).
  app/errors.py→ Traduce errores técnicos a mensajes entendibles.
  app/chat_widgets.py → Burbujas de chat, markdown y animaciones.

TECNOLOGÍAS USADAS:
-------------------
  - Gemini (Google): modelo de lenguaje que genera las respuestas.
  - LangGraph: orquesta el ciclo "pensar → usar herramienta → responder".
  - LangChain: integración con Gemini y definición de herramientas.
  - ChromaDB: base de datos vectorial para el RAG.
  - Georef API: datos oficiales de provincias y ciudades de Argentina.
  - Wikipedia API: resúmenes turísticos en español.
  - Tkinter: interfaz gráfica de escritorio.
"""
