"""
Setup inicial del Agente de Turismo Argentina.
Corre una sola vez para inicializar el vector store RAG.
Uso: python setup.py
"""
import os
import sys

# Asegurar que el proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.sample_docs import get_sample_documents
from rag.rag_pipeline import ingest_documents


def main():
    print("="*60)
    print("Setup del Agente de Turismo Argentina")
    print("="*60)

    # Verificar API key
    from config.settings import GOOGLE_API_KEY
    if not GOOGLE_API_KEY:
        print("\n❌ ERROR: Falta la variable GOOGLE_API_KEY en el archivo .env")
        print("Creá un archivo .env con: GOOGLE_API_KEY=...")
        print("Y configurá: GEMINI_MODEL=gemini-2.5-flash")
        sys.exit(1)

    print("\n1. Cargando documentos de ejemplo...")
    docs = get_sample_documents()
    print(f"   {len(docs)} documentos encontrados.")

    print("\n2. Inicializando vector store ChromaDB...")
    try:
        vectorstore = ingest_documents(docs)
        print(f"   ✅ Vector store creado en ./data/chroma")
    except Exception as e:
        print(f"   ❌ Error al crear el vector store: {e}")
        sys.exit(1)

    print("\n3. Creando directorios de persistencia...")
    os.makedirs("./data/memory", exist_ok=True)
    print("   ✅ Directorio de memoria creado en ./data/memory")

    print("\n" + "="*60)
    print("✅ Setup completo. Podés iniciar el agente con:")
    print("   python main.py --demo     # Ejemplo predefinido")
    print("   python main.py            # Modo interactivo")
    print("   python main.py --eval     # Evaluación")
    print("\n   Y el servidor MCP con:")
    print("   uvicorn mcp.mcp_server:app --port 8001")
    print("="*60)


if __name__ == "__main__":
    main()
