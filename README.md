# Agente de Turismo Argentina
### LangChain + LangGraph · RAG avanzado · Subagentes · Human in the Loop

---

## Arquitectura

```
Usuario
  │
  ▼
[Guardarraíl de entrada]  ← dominio, idioma, malas palabras
  │
  ▼
[Orquestador LangGraph]
  ├── Subagente Destinos    → tool_clima, tool_actividades
  ├── Subagente Itinerario  → tool_actividades, tool_alojamiento, tool_traslados
  ├── Subagente Precios     → tool_alojamiento, tool_traslados
  ├── Subagente Monitoreo   → tool_clima, tool_actividades
  └── Subagente RAG         → pipeline RAG completo
        ├── Fragmentación (RecursiveCharacterTextSplitter)
        ├── Query rewriting
        ├── Descomposición en subconsultas
        ├── Retrieval híbrido (semántica + BM25)
        ├── Fusión RRF
        └── Re-ranking cross-encoder
  │
  ▼
[Subagente Juez + Rúbrica]  ← modelo diferente (gpt-4o-mini)
  │  ├── coherencia temporal
  │  ├── ajuste presupuesto
  │  ├── relevancia perfil
  │  ├── corrección política
  │  ├── idioma español
  │  └── fidelidad dominio
  │
  ├── rechaza → vuelve al orquestador (máx. 2 retries)
  │
  ▼
[Human in the Loop]  ← decisión final siempre del usuario
  │
  ▼
[Memoria a largo plazo]  ← guarda itinerario, perfil, feedback
  │
  ▼
Respuesta final
```

---

## Estructura de archivos

```
agente_turismo/
├── config/
│   └── settings.py          # Configuración central y variables de entorno
├── guardrails/
│   └── guardrails.py        # Guardarraíles: dominio, idioma, malas palabras, presupuesto
├── rag/
│   └── rag_pipeline.py      # Pipeline RAG completo (6 etapas)
├── memory/
│   └── long_term_memory.py  # Memoria persistente por usuario
├── mcp/
│   ├── mcp_server.py        # Servidor MCP (FastAPI)
│   └── mcp_client.py        # Tools LangChain que consumen el servidor
├── skills/
│   └── skills.py            # Skills procedurales versionadas
├── agents/
│   ├── subagents.py         # 5 subagentes especializados
│   ├── judge_agent.py       # Subagente juez con rúbrica
│   └── graph.py             # Grafo LangGraph (orquestador principal)
├── evaluation/
│   └── evaluator.py         # Métricas y batería adversarial
├── data/
│   └── sample_docs.py       # Documentos de ejemplo para el RAG
├── main.py                  # Punto de entrada
├── setup.py                 # Inicialización del vector store
├── requirements.txt
└── .env.example
```

---

## Instalación

```bash
# 1. Clonar / copiar el proyecto
cd agente_turismo

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env y agregar tu GOOGLE_API_KEY
# GEMINI_MODEL=gemini-2.5-flash

# 5. Inicializar el vector store RAG
python setup.py

# 6. En una terminal separada, levantar el servidor MCP
uvicorn mcp.mcp_server:app --port 8001

# 7. Correr el agente
python main.py --demo          # ejemplo predefinido
python main.py                 # modo interactivo
python main.py --eval          # evaluación de guardarraíles
```

---

## Componentes clave

### RAG (6 etapas)
1. **Fragmentación**: `RecursiveCharacterTextSplitter` con overlap configurable
2. **Query rewriting**: LLM reformula la consulta para mejor recuperación
3. **Descomposición**: preguntas complejas → subconsultas atómicas en paralelo
4. **Retrieval híbrido**: semántica (ChromaDB) + BM25 ponderados
5. **Fusión RRF**: Reciprocal Rank Fusion para unificar rankings
6. **Re-ranking**: cross-encoder `ms-marco-MiniLM-L-6-v2`

### Subagentes
| Subagente | Tools | Responsabilidad |
|-----------|-------|-----------------|
| Destinos | clima, actividades | Ranking de destinos con justificación |
| Itinerario | actividades, alojamiento, traslados | Plan día a día |
| Precios | alojamiento, traslados | Optimización de presupuesto |
| Monitoreo | clima, actividades | Alertas y planes alternativos |
| Juez | — | Evaluación con rúbrica (modelo diferente) |

### Guardarraíles
- **Dominio**: rechaza consultas fuera del turismo argentino
- **Idioma**: solo responde en español
- **Lenguaje**: filtro de malas palabras (determinista)
- **Presupuesto**: alerta si el costo supera el límite
- **Coherencia horaria**: chequeo duro de superposición de actividades
- **Clima**: umbral determinista de mm de lluvia

### Memoria a largo plazo
- Perfil del usuario (preferencias, restricciones)
- Destinos ya visitados
- Feedback negativo persistente
- Historial de itinerarios (últimos 10)
- TTL configurable (default: 180 días)

---

## Variables de entorno principales

| Variable | Default | Descripción |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | **Requerida** (API key de Google AI / Gemini) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Modelo Gemini por defecto |
| `ORCHESTRATOR_MODEL` | `gemini-2.5-flash` | Modelo del orquestador |
| `JUDGE_MODEL` | `gemini-2.5-flash` | Modelo del juez |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Modelo de embeddings |
| `JUDGE_MIN_SCORE` | `0.7` | Puntaje mínimo de la rúbrica para aprobar |
| `MAX_JUDGE_RETRIES` | `2` | Máximo de reintentos antes de escalar al humano |
| `RAIN_ALERT_MM` | `20.0` | Umbral de lluvia para alerta climática |
| `MCP_HOST` | `http://localhost:8001` | URL del servidor MCP |
