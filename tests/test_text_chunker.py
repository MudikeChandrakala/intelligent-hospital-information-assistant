"""Validate the document loading and text chunking pipeline."""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.document_loader import DocumentLoader
from modules.text_chunker import TextChunker


SEPARATOR = "=" * 60


def main() -> None:
    """Load, chunk, and validate the project knowledge base documents."""

    print(SEPARATOR)
    print("TEXT CHUNKER VALIDATION")
    print(SEPARATOR)

    print("\nLoading documents...")
    document_loader = DocumentLoader(Path("."))
    documents = document_loader.load_all_documents()
    original_document_count = len(documents)
    print(f"Documents Loaded : {original_document_count}")

    print("\nInitializing text chunker...")
    text_chunker = TextChunker()
    print(f"Chunk Size       : {text_chunker.chunk_size}")
    print(f"Chunk Overlap    : {text_chunker.chunk_overlap}")

    print("\nSplitting documents...")
    chunks = text_chunker.split_documents(documents)
    chunk_count = len(chunks)
    print(f"Chunks Generated : {chunk_count}")

    assert chunk_count >= original_document_count, (
        "Chunk count must be greater than or equal to the original document "
        f"count. Received {chunk_count} chunks for {original_document_count} "
        "documents."
    )

    assert chunks, "No chunks were generated; unable to validate the first chunk."

    print("\nFirst Chunk")
    print(SEPARATOR)
    print("Page Content:")
    print(chunks[0].page_content)
    print("\nMetadata:")
    print(chunks[0].metadata)

    for index, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata, (
            f"Chunk at index {index} is missing required metadata: chunk_index."
        )
        assert "total_chunks" in chunk.metadata, (
            f"Chunk at index {index} is missing required metadata: total_chunks."
        )
        assert chunk.page_content.strip(), (
            f"Chunk at index {index} contains empty page_content."
        )

    print("\n" + SEPARATOR)
    print("TEXT CHUNKER VALIDATION")
    print(SEPARATOR)
    print(f"Documents Loaded : {original_document_count}")
    print(f"Chunks Generated : {chunk_count}")
    print("Validation       : PASSED")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
