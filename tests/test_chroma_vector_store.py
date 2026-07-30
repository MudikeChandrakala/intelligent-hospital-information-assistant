"""
test_chroma_vector_store.py

Integration test for the complete RAG preprocessing pipeline.

Pipeline:

Knowledge Base
    ↓
Document Loader
    ↓
Text Chunker
    ↓
Embedding Generator
    ↓
Chroma Vector Store

This test validates that all completed modules work together correctly.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.document_loader import DocumentLoader
from modules.text_chunker import TextChunker
from modules.embedding_generator import EmbeddingGenerator
from modules.chroma_vector_store import ChromaVectorStore

def main() -> None:
    """Run the complete Chroma Vector Store integration test."""

    print("=" * 60)
    print(" Intelligent Hospital Information Assistant")
    print(" Chroma Vector Store Integration Test")
    print("=" * 60)

    # ---------------------------------------------------------
    # Step 1 : Load Documents
    # ---------------------------------------------------------

    print("\n[1/6] Loading knowledge base documents...")

    loader = DocumentLoader(project_root=PROJECT_ROOT)
    documents = loader.load_all_documents()

    print(f"✓ Documents Loaded : {len(documents)}")

    if not documents:
        raise RuntimeError("No documents were loaded.")

    # ---------------------------------------------------------
    # Step 2 : Chunk Documents
    # ---------------------------------------------------------

    print("\n[2/6] Splitting documents into chunks...")

    chunker = TextChunker()
    chunks = chunker.split_documents(documents)

    print(f"✓ Chunks Created   : {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks were created.")

    # ---------------------------------------------------------
    # Step 3 : Load Embedding Model
    # ---------------------------------------------------------

    print("\n[3/6] Loading embedding model...")

    embedding_generator = EmbeddingGenerator()
    embedding_model = embedding_generator.get_embedding_model()

    print("✓ Embedding Model  : Loaded")

    # ---------------------------------------------------------
    # Step 4 : Build Chroma Vector Store
    # ---------------------------------------------------------

    print("\n[4/6] Building Chroma vector database...")

    vector_manager = ChromaVectorStore(
        embedding_model=embedding_model
    )

    vector_store = vector_manager.build_vector_store(
        documents=chunks,
        overwrite=True
    )

    print("✓ Vector Store     : Created")

    # ---------------------------------------------------------
    # Step 5 : Validate Stored Vectors
    # ---------------------------------------------------------

    print("\n[5/6] Validating stored vectors...")

    try:
        collection = vector_store._collection
        vector_count = collection.count()

        print(f"✓ Stored Vectors   : {vector_count}")

        if vector_count == 0:
            raise RuntimeError("No vectors were stored in Chroma.")

    except Exception as exc:
        raise RuntimeError(
            f"Unable to validate vector count: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Step 6 : Reload Vector Store
    # ---------------------------------------------------------

    print("\n[6/6] Reloading vector database...")

    loaded_store = vector_manager.load_vector_store()

    if loaded_store is None:
        raise RuntimeError("Failed to reload vector store.")

    print("✓ Vector Store     : Reloaded")
    # ---------------------------------------------------------
    # Step 7 : Verify Doctor Retrieval
    # ---------------------------------------------------------

    print("\n[7/7] Testing doctor retrieval...")

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 10}
    )

    queries = [
        "Dr. Arvind Kumar",
        "Cardiology doctor",
        "Cardiology doctors"
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print("QUERY:", query)
        print("=" * 80)

        results = retriever.invoke(query)

        if not results:
            print("No results found.")
            continue

        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}")
            print("-" * 60)
            print("Metadata:")
            print(doc.metadata)
            print("\nContent:")
            print(doc.page_content[:500])
    persist_dir = Path("vector_store")

    if not persist_dir.exists():
        raise RuntimeError("Persist directory was not created.")

    print("\n" + "=" * 60)
    print(" CHROMA VECTOR STORE VALIDATION SUCCESSFUL")
    print("=" * 60)

    print(f"Documents Loaded : {len(documents)}")
    print(f"Chunks Created   : {len(chunks)}")
    print(f"Vectors Stored   : {vector_count}")
    print(f"Persist Location : {persist_dir.resolve()}")

    print("\nAll integration tests passed successfully!")

    print("=" * 60)


if __name__ == "__main__":
    main()