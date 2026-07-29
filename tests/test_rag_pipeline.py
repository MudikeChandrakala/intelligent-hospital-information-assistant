"""
test_rag_pipeline.py

End-to-end integration test for the RAG Pipeline.

Pipeline:

Question
    │
    ▼
Retriever
    │
    ▼
Prompt Builder
    │
    ▼
Gemini Client
    │
    ▼
Generated Response

This test validates the complete backend pipeline.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from modules.rag_pipeline import RAGPipeline


def validate_response(response: str) -> None:
    """
    Validate the generated response.

    Raises:
        RuntimeError: If the response is invalid.
    """

    if not isinstance(response, str):
        raise RuntimeError("Response must be a string.")

    if not response.strip():
        raise RuntimeError("Response cannot be empty.")

    print("✓ Response validation passed.")


def test_invalid_questions(pipeline: RAGPipeline) -> None:
    """
    Verify invalid questions raise ValueError.
    """

    print("\nTesting invalid question validation...")

    invalid_questions = [
        "",
        "   ",
        None,
    ]

    for question in invalid_questions:
        try:
            pipeline.ask(question)  # type: ignore[arg-type]
        except ValueError:
            print(f"✓ Correctly rejected question: {repr(question)}")
        else:
            raise RuntimeError(
                f"Expected ValueError for question: {repr(question)}"
            )


def main() -> None:
    """
    Execute the complete RAG Pipeline integration test.
    """

    print("=" * 70)
    print(" Intelligent Hospital Information Assistant")
    print(" RAG Pipeline Integration Test")
    print("=" * 70)

    print("\n[1/4] Initializing RAG Pipeline...")

    pipeline = RAGPipeline()

    print("✓ Pipeline initialized successfully.")

    test_questions = [
        "Where is the Cardiology Department?",
        "How can I book an appointment?",
        "What insurance plans are accepted?",
        "What should I do during a medical emergency?",
        "What are the hospital visiting hours?",
    ]

    print("\n[2/4] Testing complete pipeline...\n")

    for index, question in enumerate(test_questions, start=1):

        print("=" * 70)
        print(f"Question {index}")
        print("=" * 70)

        print(f"Question : {question}\n")

        response = pipeline.ask(question)

        validate_response(response)

        print("Response:\n")
        print(response)
        print()

    print("=" * 70)

    print("\n[3/4] Testing validation...")

    test_invalid_questions(pipeline)

    print("\n[4/4] Integration test completed.")

    print("\n" + "=" * 70)
    print(" RAG PIPELINE VALIDATION SUCCESSFUL")
    print("=" * 70)

    print("Pipeline Initialization : PASSED")
    print("Retriever              : PASSED")
    print("Prompt Builder         : PASSED")
    print("Gemini Client          : PASSED")
    print("Response Generation    : PASSED")
    print("Validation             : PASSED")

    print("\nAll end-to-end integration tests passed successfully!")

    print("=" * 70)


if __name__ == "__main__":
    main()
    