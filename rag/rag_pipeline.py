"""
Pipeline RAG completo para el Agente de Turismo Argentina.
Implementa:
  - Fragmentación con RecursiveCharacterTextSplitter
  - Transformación de consultas (query rewriting)
  - Descomposición en subconsultas
  - Recuperación híbrida (semántica + BM25)
  - Fusión RRF (Reciprocal Rank Fusion)
  - Re-ranking con cross-encoder
  - Evaluación de relevancia con juez LLM
"""
from __future__ import annotations

import json
from typing import Any
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

from config.settings import (
    GOOGLE_API_KEY, ORCHESTRATOR_MODEL, JUDGE_MODEL,
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL,
    BM25_WEIGHT, SEMANTIC_WEIGHT, RAG_RELEVANCE_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FRAGMENTACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        length_function=len,
    )


def ingest_documents(raw_docs: list[Document]) -> Chroma:
    """
    Fragmenta y embebe documentos en el vector store ChromaDB.
    raw_docs: lista de Document con page_content y metadata.
    """
    splitter = build_splitter()
    chunks = splitter.split_documents(raw_docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="turismo_argentina",
    )
    print(f"[RAG] Ingestados {len(chunks)} chunks desde {len(raw_docs)} documentos.")
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="turismo_argentina",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORMACIÓN DE CONSULTAS (Query Rewriting)
# ─────────────────────────────────────────────────────────────────────────────
_rewrite_llm = None

def _get_rewrite_llm():
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = ChatGoogleGenerativeAI(
            model=ORCHESTRATOR_MODEL,
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY,
        )
    return _rewrite_llm


def rewrite_query(original_query: str) -> str:
    """
    Reformula la consulta del usuario para maximizar la recuperación vectorial.
    """
    llm = _get_rewrite_llm()
    system = SystemMessage(content=(
        "Eres un experto en recuperación de información turística argentina. "
        "Tu tarea es reformular la consulta del usuario para que sea más efectiva "
        "al buscar en una base de conocimiento de destinos, actividades y normativa "
        "turística de Argentina. Mantén el idioma español. "
        "Devuelve SOLO la consulta reformulada, sin explicaciones."
    ))
    human = HumanMessage(content=f"Consulta original: {original_query}")
    result = llm.invoke([system, human])
    return result.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. DESCOMPOSICIÓN EN SUBCONSULTAS
# ─────────────────────────────────────────────────────────────────────────────
def decompose_query(complex_query: str) -> list[str]:
    """
    Descompone una pregunta compleja en subconsultas atómicas.
    Retorna lista de strings (subconsultas).
    """
    llm = _get_rewrite_llm()
    system = SystemMessage(content=(
        "Eres un experto en descomposición de consultas turísticas. "
        "Tu tarea es dividir una pregunta compleja en 2 a 4 subconsultas simples "
        "que puedan responderse individualmente con información de destinos argentinos. "
        "Devuelve SOLO un JSON con la clave 'subconsultas' y una lista de strings. "
        "Ejemplo: {\"subconsultas\": [\"actividades en Mendoza en invierno\", "
        "\"alojamientos económicos en Mendoza\"]}"
    ))
    human = HumanMessage(content=complex_query)
    result = llm.invoke([system, human])
    try:
        data = json.loads(result.content.strip())
        return data.get("subconsultas", [complex_query])
    except json.JSONDecodeError:
        return [complex_query]


# ─────────────────────────────────────────────────────────────────────────────
# 4. RECUPERACIÓN HÍBRIDA (Semántica + BM25)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RankedDoc:
    document: Document
    score: float = 0.0


def semantic_retrieve(vectorstore: Chroma, query: str, k: int = TOP_K_RETRIEVAL) -> list[tuple[Document, float]]:
    """Recuperación semántica con scores."""
    return vectorstore.similarity_search_with_score(query, k=k)


def bm25_retrieve(all_docs: list[Document], query: str, k: int = TOP_K_RETRIEVAL) -> list[tuple[Document, float]]:
    """Recuperación BM25 sobre el corpus completo."""
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(all_docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:k]


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUSIÓN RRF (Reciprocal Rank Fusion)
# ─────────────────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[Document]:
    """
    Combina múltiples listas rankeadas usando RRF.
    k: constante de suavizado (típicamente 60).
    Retorna documentos ordenados por score RRF descendente.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _) in enumerate(ranked_list, start=1):
            doc_id = doc.page_content[:100]  # identificador aproximado
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


# ─────────────────────────────────────────────────────────────────────────────
# 6. RE-RANKING con Cross-Encoder (sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            print("[RAG] sentence-transformers no disponible, se omite cross-encoder.")
    return _cross_encoder


def rerank_documents(query: str, docs: list[Document], top_n: int = 4) -> list[Document]:
    """Re-rankea documentos usando un cross-encoder. Fallback: orden original."""
    ce = _get_cross_encoder()
    if ce is None or not docs:
        return docs[:top_n]

    pairs = [(query, doc.page_content) for doc in docs]
    scores = ce.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUACIÓN DE RAG (Juez LLM)
# ─────────────────────────────────────────────────────────────────────────────
_judge_llm = None

def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatGoogleGenerativeAI(
            model=JUDGE_MODEL,
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
        )
    return _judge_llm


@dataclass
class RAGEvalResult:
    relevance: float        # 0-1: ¿el chunk recuperado responde la pregunta?
    faithfulness: float     # 0-1: ¿la respuesta se apoya en los chunks?
    answer_relevance: float # 0-1: ¿la respuesta responde lo que se preguntó?
    passed: bool = field(init=False)

    def __post_init__(self):
        avg = (self.relevance + self.faithfulness + self.answer_relevance) / 3
        self.passed = avg >= RAG_RELEVANCE_THRESHOLD


RUBRICA_RAG = """
Eres un juez de calidad para un sistema RAG de turismo argentino.
Evalúa los siguientes aspectos y devuelve un JSON con scores entre 0.0 y 1.0:

1. relevancia_chunks: ¿Los fragmentos recuperados son relevantes para la pregunta?
2. fidelidad: ¿La respuesta generada se basa fielmente en los fragmentos, sin inventar?
3. relevancia_respuesta: ¿La respuesta realmente contesta lo que el usuario preguntó?

Devuelve SOLO este JSON (sin markdown):
{
  "relevancia_chunks": 0.0,
  "fidelidad": 0.0,
  "relevancia_respuesta": 0.0,
  "justificacion": "..."
}
"""


def evaluate_rag(
    question: str,
    retrieved_chunks: list[Document],
    generated_answer: str,
) -> RAGEvalResult:
    """
    Evalúa la calidad del pipeline RAG para una pregunta dada.
    """
    llm = _get_judge_llm()
    chunks_text = "\n---\n".join(doc.page_content for doc in retrieved_chunks)
    human_content = (
        f"PREGUNTA: {question}\n\n"
        f"FRAGMENTOS RECUPERADOS:\n{chunks_text}\n\n"
        f"RESPUESTA GENERADA:\n{generated_answer}"
    )
    result = llm.invoke([
        SystemMessage(content=RUBRICA_RAG),
        HumanMessage(content=human_content),
    ])
    try:
        data = json.loads(result.content.strip())
        return RAGEvalResult(
            relevance=float(data.get("relevancia_chunks", 0)),
            faithfulness=float(data.get("fidelidad", 0)),
            answer_relevance=float(data.get("relevancia_respuesta", 0)),
        )
    except (json.JSONDecodeError, ValueError):
        return RAGEvalResult(relevance=0.5, faithfulness=0.5, answer_relevance=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────
def run_rag_pipeline(
    question: str,
    vectorstore: Chroma,
    all_docs: list[Document],
) -> tuple[list[Document], str]:
    """
    Ejecuta el pipeline RAG completo y retorna (chunks_finales, contexto_texto).
    Pasos: rewrite → decompose → hybrid retrieve → RRF → rerank.
    """
    # 1. Transformación de consulta
    rewritten = rewrite_query(question)

    # 2. Descomposición en subconsultas
    subqueries = decompose_query(rewritten)

    all_ranked: list[list[tuple[Document, float]]] = []

    for subq in subqueries:
        # 3a. Recuperación semántica
        semantic_results = semantic_retrieve(vectorstore, subq)
        # 3b. Recuperación BM25
        bm25_results = bm25_retrieve(all_docs, subq)
        # Ponderación antes de RRF
        all_ranked.append(semantic_results)
        all_ranked.append(bm25_results)

    # 4. Fusión RRF
    fused_docs = reciprocal_rank_fusion(all_ranked)

    # 5. Re-ranking
    final_docs = rerank_documents(question, fused_docs, top_n=TOP_K_RETRIEVAL)

    # Contexto como texto para el LLM
    context = "\n\n".join(
        f"[Fuente: {doc.metadata.get('source', 'desconocida')}]\n{doc.page_content}"
        for doc in final_docs
    )

    return final_docs, context
