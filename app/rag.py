"""
=============================================================================
RAG - Retrieval Augmented Generation (Generación Aumentada por Recuperación)
=============================================================================

¿Qué es RAG?
  Técnica que combina un modelo de IA con una base de conocimiento propia.
  En vez de que Gemini "invente" datos, busca en documentos reales del proyecto.

¿Cómo funciona aquí?
  1. Carga las guías .txt de data/guias/ (Buenos Aires, Mendoza, Córdoba, etc.).
  2. Las divide en fragmentos (chunks) más pequeños.
  3. Convierte cada fragmento en un vector numérico (embedding) con Gemini.
  4. Guarda los vectores en ChromaDB (base de datos vectorial).
  5. Cuando el usuario pregunta, busca los fragmentos más similares a la consulta.
  6. Devuelve ese texto al agente para que lo use en su respuesta.

Para la defensa:
  - Los embeddings capturan el SIGNIFICADO del texto, no solo palabras exactas.
  - similarity_search encuentra los trozos más relevantes aunque no coincidan literalmente.
"""

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, GOOGLE_API_KEY

# Caché en memoria: evita recargar ChromaDB en cada consulta
_vectorstore: Chroma | None = None
_MANIFEST = CHROMA_DIR / ".guias_manifest"


def _lista_guias_actual() -> list[str]:
    """Nombres de archivos .txt actualmente en data/guias/."""
    return sorted(archivo.name for archivo in DATA_DIR.glob("*.txt"))


def _manifest_content() -> str:
    """Contenido del manifiesto: modelo de embedding + lista de guías."""
    return f"embedding={EMBEDDING_MODEL}\n" + "\n".join(_lista_guias_actual())


def _necesita_reindexar() -> bool:
    """
    Verifica si se agregaron/quitaron guías o cambió el modelo de embedding.

    Si cambió algo, borra ChromaDB y reindexa automáticamente.
    """
    if not _MANIFEST.exists():
        return True
    return _MANIFEST.read_text(encoding="utf-8") != _manifest_content()


def _guardar_manifest() -> None:
    """Guarda el estado actual para detectar cambios futuros."""
    _MANIFEST.write_text(_manifest_content(), encoding="utf-8")


def _limpiar_chroma() -> None:
    """Elimina la base vectorial para forzar una reindexación completa."""
    global _vectorstore
    _vectorstore = None
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def _cargar_guias() -> list[Document]:
    """
    Lee todos los archivos .txt de data/guias/ y los convierte en Documentos.

    Cada Document tiene:
      - page_content: el texto de la guía
      - metadata: nombre de la ciudad (para mostrar en los resultados)
    """
    documentos = []
    for archivo in sorted(DATA_DIR.glob("*.txt")):
        texto = archivo.read_text(encoding="utf-8")
        documentos.append(
            Document(
                page_content=texto,
                metadata={
                    "source": archivo.stem,
                    "ciudad": archivo.stem.replace("_", " ").title(),
                },
            )
        )
    return documentos


def get_vectorstore() -> Chroma:
    """
    Obtiene la base vectorial. Si no existe, la crea indexando las guías.

    Primera ejecución: tarda más (genera embeddings y guarda en chroma_db/).
    Siguientes ejecuciones: carga la base ya creada (más rápido).
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    # Modelo que convierte texto → vector numérico
    # NOTA: text-embedding-004 fue retirado en enero 2026. Usar gemini-embedding-001.
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Si se agregaron guías nuevas, reindexar automáticamente
    if _necesita_reindexar() and any(CHROMA_DIR.iterdir()):
        _limpiar_chroma()

    # Si ya hay datos indexados actualizados, los reutilizamos
    if any(CHROMA_DIR.iterdir()):
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        return _vectorstore

    # Primera vez o reindexación: cargar guías desde cero
    guias = _cargar_guias()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    fragmentos = splitter.split_documents(guias)

    _vectorstore = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    _guardar_manifest()
    return _vectorstore


def consultar_guia(consulta: str, k: int = 4) -> str:
    """
    Busca los k fragmentos más relevantes para la consulta del usuario.

    Parámetros:
      consulta → Pregunta del usuario (ej: "qué comer en Mendoza")
      k        → Cantidad de fragmentos a devolver (por defecto 4)

    Retorna el texto encontrado para que el agente lo use en su respuesta.
    """
    store = get_vectorstore()
    resultados = store.similarity_search(consulta, k=k)

    if not resultados:
        return "No se encontró información relevante en las guías turísticas."

    partes = []
    for i, doc in enumerate(resultados, 1):
        ciudad = doc.metadata.get("ciudad", "Desconocido")
        partes.append(f"[{i}] {ciudad}:\n{doc.page_content}")

    return "\n\n---\n\n".join(partes)
