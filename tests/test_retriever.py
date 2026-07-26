"""
test_retriever.py

Integration test for the Retriever module.

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
    ↓
Retriever

This test validates that the complete retrieval pipeline works correctly.
"""

from pathlib import Path

from modules.document_loader import DocumentLoader
from modules.text_chunker import TextChunker
from modules.embedding_generator import EmbeddingGenerator
from modules.chroma_vector_store import ChromaVectorStore
from modules.retriever import Retriever


def print_result(title: str, documents) -> None:
    """Pretty-print retrieved documents."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for index, document in enumerate(documents, start=1):
        print(f"\nResult {index}")
        print("-" * 70)
        print(document.page_content[:500])

        if document.metadata:
            print("\nMetadata:")
            for key, value in document.metadata.items():
                print(f"  {key}: {value}")


def main() -> None:
    """Run Retriever integration test."""

    print("=" * 70)
    print(" Intelligent Hospital Information Assistant")
    print(" Retriever Integration Test")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1 : Load Documents
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parent

    print("\n[1/6] Loading knowledge base...")

    loader = DocumentLoader(project_root=project_root)
    documents = loader.load_all_documents()

    print(f"✓ Documents Loaded : {len(documents)}")

    # ---------------------------------------------------------
    # Step 2 : Chunk Documents
    # ---------------------------------------------------------

    print("\n[2/6] Creating chunks...")

    chunker = TextChunker()
    chunks = chunker.split_documents(documents)

    print(f"✓ Chunks Created   : {len(chunks)}")

    # ---------------------------------------------------------
    # Step 3 : Embedding Model
    # ---------------------------------------------------------

    print("\n[3/6] Loading embedding model...")

    embedding_generator = EmbeddingGenerator()
    embedding_model = embedding_generator.get_embedding_model()

    print("✓ Embedding Model  : Loaded")

    # ---------------------------------------------------------
    # Step 4 : Load Vector Store
    # ---------------------------------------------------------

    print("\n[4/6] Loading Chroma Vector Store...")

    vector_manager = ChromaVectorStore(
        embedding_model=embedding_model
    )

    vector_store = vector_manager.load_vector_store()

    print("✓ Vector Store     : Loaded")

    # ---------------------------------------------------------
    # Step 5 : Create Retriever
    # ---------------------------------------------------------

    print("\n[5/6] Creating Retriever...")

    retriever = Retriever(
        vector_store=vector_store,
        k=5,
    )

    print("✓ Retriever        : Created")

    # ---------------------------------------------------------
    # Step 6 : Test Queries
    # ---------------------------------------------------------

    print("\n[6/6] Running retrieval tests...")

    test_queries = [
        "Where is the cardiology department?",
        "Who is the best neurologist?",
        "How can I book an appointment?",
        "What insurance plans are accepted?",
        "What should I do during an emergency?",
    ]

    total_results = 0

    for query in test_queries:

        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)

        results = retriever.retrieve(query)

        if not results:
            raise RuntimeError(
                f"No documents returned for query: {query}"
            )

        total_results += len(results)

        print(f"✓ Retrieved {len(results)} document(s).")

        print_result("Top Results", results)

    print("\n" + "=" * 70)
    print(" RETRIEVER VALIDATION SUCCESSFUL")
    print("=" * 70)

    print(f"Queries Tested    : {len(test_queries)}")
    print(f"Documents Loaded  : {len(documents)}")
    print(f"Chunks Created    : {len(chunks)}")
    print(f"Results Returned  : {total_results}")

    print("\nAll Retriever integration tests passed successfully!")

    print("=" * 70)


if __name__ == "__main__":
    main()