# Agente de Turismo Argentina
### Deep Agent (web) · LangGraph (CLI) · RAG · MCP · Human in the Loop

---

## Arquitectura

### Frontend web (recomendado)

```
Usuario (chat Flask)
  │
  ▼
[Guardarraíl de entrada]
  │
  ▼
[Deep Agent — orquestador]
  ├── Tools MCP (clima, alojamiento, actividades, traslados, gestión)
  ├── Tool RAG
  ├── Tools Gmail + Google Calendar
  ├── Skills (/skills/*.md)
  ├── Subagentes (destinos, itinerario, precios, monitoreo)
  └── Memoria (/memories/profile.md)
  │
  ▼
[Juez + rúbrica]
  │
  ▼
[HITL en UI]  Aprobar · Regenerar · Escalar
  │  (Aprobar → email del itinerario + eventos en Calendar)
  ▼
Respuesta final
```

### CLI (LangGraph)

El grafo en `agents/graph.py` sigue disponible vía `python main.py` (interactivo / demo / eval).

---

## Estructura de archivos

```
tp-agent/
├── agentes/                 # Frontend Flask + Deep Agent
│   ├── app.py               # Orquestador web (tools, skills, HITL)
│   ├── skills/              # SKILL.md (planificacion, monitoreo, precios)
│   ├── memories/            # Perfil persistente del viajero
│   ├── messages/            # Historial del chat
│   ├── templates/           # UI del chat
│   └── static/
├── config/
├── guardrails/
├── rag/
├── memory/
├── mcp/
├── skills/                  # Skills embebidas (CLI / legado)
├── agents/                  # Grafo LangGraph CLI + juez + utils
├── evaluation/
├── integrations/            # Gmail + Google Calendar (OAuth)
├── credentials/             # credentials.json + token.json (no versionar)
├── scripts/
│   └── google_oauth_setup.py
├── data/
├── main.py                  # CLI
├── setup.py
├── requirements.txt
└── .env.example
```

---

## Instalación

```bash
# 1. Clonar / copiar el proyecto
cd tp-agent

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env: GOOGLE_API_KEY, USER_EMAIL, etc.

# 5. Inicializar el vector store RAG
python setup.py

# 6. (Opcional pero recomendado) Gmail + Google Calendar
#    - Google Cloud Console → habilitar Gmail API y Google Calendar API
#    - OAuth consent screen + usuario de prueba
#    - Credenciales → OAuth client ID → Desktop → descargar JSON
#    - Guardar como credentials/credentials.json
#    - Autorizar una vez:
python scripts/google_oauth_setup.py

# 7. En una terminal separada, levantar el servidor MCP
uvicorn mcp.mcp_server:app --port 8001

# 8. Frontend web (Deep Agent + chat)
python agentes/app.py
# Abrir http://127.0.0.1:5000

# Alternativa CLI (LangGraph)
python main.py --demo
python main.py
python main.py --eval
```

### Flujo email + calendario

1. Pedís un itinerario en el chat (destino, fechas, presupuesto).
2. El agente arma el plan y deja un draft con evaluación del juez.
3. Apretás **Aprobar** → si `AUTO_DELIVER_ON_APPROVE=true` y OAuth está listo:
   - envía el itinerario a `USER_EMAIL` por Gmail
   - crea los eventos en Google Calendar (`GOOGLE_CALENDAR_ID`, default `primary`)
4. También podés pedirle en el chat: “enviámelo por mail y cargalo al calendario” (tools `tool_enviar_email` / `tool_crear_eventos_calendario`).

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

En el chat web, estos subagentes se exponen vía Deep Agent (`task` / `subagents=`). El HITL se resuelve con botones **Aprobar / Regenerar / Escalar** en la UI.

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
| `USER_EMAIL` | — | Email destino de itinerarios |
| `GOOGLE_OAUTH_CLIENT_SECRETS` | `./credentials/credentials.json` | Cliente OAuth Desktop |
| `GOOGLE_OAUTH_TOKEN` | `./credentials/token.json` | Token tras `google_oauth_setup.py` |
| `GOOGLE_CALENDAR_ID` | `primary` | Calendario donde crear eventos |
| `AUTO_DELIVER_ON_APPROVE` | `true` | Al aprobar HITL: email + Calendar |
