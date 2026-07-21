"""
Configuración central del Agente de Turismo Argentina.
Carga variables desde .env y expone constantes usadas en todo el proyecto.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Modelos ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY       = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL         = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ORCHESTRATOR_MODEL   = os.getenv("ORCHESTRATOR_MODEL", GEMINI_MODEL)
JUDGE_MODEL          = os.getenv("JUDGE_MODEL", GEMINI_MODEL)
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# ── RAG ───────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR   = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_RETRIEVAL      = int(os.getenv("TOP_K_RETRIEVAL", "6"))
BM25_WEIGHT          = float(os.getenv("BM25_WEIGHT", "0.4"))
SEMANTIC_WEIGHT      = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.5"))

# ── Memoria LP ────────────────────────────────────────────────────────────────
MEMORY_PERSIST_DIR   = os.getenv("MEMORY_PERSIST_DIR", "./data/memory")
MEMORY_TTL_DAYS      = int(os.getenv("MEMORY_TTL_DAYS", "180"))

# ── Guardarraíles ─────────────────────────────────────────────────────────────
BUDGET_THRESHOLD     = float(os.getenv("BUDGET_THRESHOLD", "1.0"))   # fracción máxima del presupuesto
RAIN_ALERT_MM        = float(os.getenv("RAIN_ALERT_MM",    "20.0"))  # mm/día para alerta climática
MAX_JUDGE_RETRIES    = int(os.getenv("MAX_JUDGE_RETRIES",  "2"))
JUDGE_MIN_SCORE      = float(os.getenv("JUDGE_MIN_SCORE",  "0.7"))   # puntaje mínimo rúbrica (0-1)

# ── MCP / APIs reales ─────────────────────────────────────────────────────────
MCP_HOST             = os.getenv("MCP_HOST", "http://localhost:8001")
USE_REAL_WEATHER     = os.getenv("USE_REAL_WEATHER", "true").lower() in {"1", "true", "yes"}
USE_REAL_ATTRACTIONS = os.getenv("USE_REAL_ATTRACTIONS", "true").lower() in {"1", "true", "yes"}
USE_REAL_FLIGHTS     = os.getenv("USE_REAL_FLIGHTS", "true").lower() in {"1", "true", "yes"}
TRAVELPAYOUTS_TOKEN  = os.getenv("TRAVELPAYOUTS_TOKEN", "")

# ── Idioma y dominio ──────────────────────────────────────────────────────────
AGENT_LANGUAGE       = "español"
DOMAIN_DESCRIPTION   = "turismo y viajes dentro de Argentina"

SYSTEM_PROMPT_BASE = """
Eres un agente especializado en turismo argentino. 
REGLAS ESTRICTAS que debes seguir siempre:
1. Respondes ÚNICAMENTE en español, sin excepción.
2. Solo asistes con consultas relacionadas con viajes y turismo dentro de Argentina.
3. Nunca uses malas palabras, lenguaje ofensivo o inapropiado.
4. Eres políticamente correcto, inclusivo y respetuoso de todas las comunidades.
5. Si te piden algo fuera del turismo argentino, educadamente lo rechazas y 
   ofreces ayuda dentro de tu dominio.
"""
