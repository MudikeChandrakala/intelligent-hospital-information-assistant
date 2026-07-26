"""
test_prompt_builder.py

Integration test for the PromptBuilder module.

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
    ↓
Prompt Builder

This test validates that the complete prompt generation pipeline works
correctly.
"""

from pathlib import Path

from modules.document_loader import DocumentLoader
from modules.text_chunker import TextChunker
from modules.embedding_generator import EmbeddingGenerator
from modules.chroma_vector_store import ChromaVectorStore
from modules.retriever import Retriever
from modules.prompt_builder import PromptBuilder


def print_prompt(prompt: str) -> None:
    """Pretty-print the generated prompt."""

    print("\n" + "=" * 80)
    print("GENERATED PROMPT")
    print("=" * 80)
    print(prompt)
    print("=" * 80)


def validate_prompt(prompt: str, question: str) -> None:
    """
    Validate the generated prompt.

    Raises:
        RuntimeError: If the prompt fails validation.
    """

    if not isinstance(prompt, str):
        raise RuntimeError("Prompt must be a string.")

    if not prompt.strip():
        raise RuntimeError("Prompt cannot be empty.")

    if question not in prompt:
        raise RuntimeError("User question missing from prompt.")

    required_sections = [
        "Hospital Context",
        "User Question",
        "Answer",
    ]

    for section in required_sections:
        if section not in prompt:
            raise RuntimeError(f"Missing section: {section}")

    if "Document 1" not in prompt:
        raise RuntimeError("Retrieved context was not inserted.")

    print("✓ Prompt validation passed.")


def main() -> None:
    """Run PromptBuilder integration test."""

    print("=" * 70)
    print(" Intelligent Hospital Information Assistant")
    print(" Prompt Builder Integration Test")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1 : Load Documents
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parent

    print("\n[1/7] Loading knowledge base...")

    loader = DocumentLoader(project_root=project_root)
    documents = loader.load_all_documents()

    print(f"✓ Documents Loaded : {len(documents)}")

    # ---------------------------------------------------------
    # Step 2 : Chunk Documents
    # ---------------------------------------------------------

    print("\n[2/7] Creating chunks...")

    chunker = TextChunker()
    chunks = chunker.split_documents(documents)

    print(f"✓ Chunks Created   : {len(chunks)}")

    # ---------------------------------------------------------
    # Step 3 : Embedding Model
    # ---------------------------------------------------------

    print("\n[3/7] Loading embedding model...")

    embedding_generator = EmbeddingGenerator()
    embedding_model = embedding_generator.get_embedding_model()

    print("✓ Embedding Model  : Loaded")

    # ---------------------------------------------------------
    # Step 4 : Load Vector Store
    # ---------------------------------------------------------

    print("\n[4/7] Loading Chroma Vector Store...")

    vector_manager = ChromaVectorStore(
        embedding_model=embedding_model
    )

    vector_store = vector_manager.load_vector_store()

    print("✓ Vector Store     : Loaded")

    # ---------------------------------------------------------
    # Step 5 : Create Retriever
    # ---------------------------------------------------------

    print("\n[5/7] Creating Retriever...")

    retriever = Retriever(
        vector_store=vector_store,
        k=5,
    )

    print("✓ Retriever        : Created")

    # ---------------------------------------------------------
    # Step 6 : Create Prompt Builder
    # ---------------------------------------------------------

    print("\n[6/7] Creating Prompt Builder...")

    prompt_builder = PromptBuilder()

    print("✓ Prompt Builder   : Created")

    # ---------------------------------------------------------
    # Step 7 : Generate Prompts
    # ---------------------------------------------------------

    print("\n[7/7] Running prompt generation tests...")

    test_questions = [
        "Where is the cardiology department?",
        "How can I book an appointment?",
        "What insurance plans are accepted?",
        "What should I do during an emergency?",
        "What are the visiting hours?",
    ]

    prompts_generated = 0

    for question in test_questions:

        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print("=" * 70)

        documents = retriever.retrieve(question)

        prompt = prompt_builder.build_prompt(
            question=question,
            documents=documents,
        )

        validate_prompt(prompt, question)

        print_prompt(prompt)

        prompts_generated += 1

    print("\n" + "=" * 70)
    print(" PROMPT BUILDER VALIDATION SUCCESSFUL")
    print("=" * 70)

    print(f"Questions Tested : {len(test_questions)}")
    print(f"Prompts Built    : {prompts_generated}")

    print("\nAll Prompt Builder integration tests passed successfully!")

    print("=" * 70)


if __name__ == "__main__":
    main()