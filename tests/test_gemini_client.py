"""
test_gemini_client.py

Integration test for the GeminiClient module.

Pipeline:

Sample Prompt
      │
      ▼
Gemini Client
      │
      ▼
Google Gemini
      │
      ▼
Generated Response

This test validates that the Gemini client can successfully connect to
Google Gemini, send prompts, receive responses, and correctly handle
invalid inputs.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.gemini_client import GeminiClient


def validate_response(response: str) -> None:
    """
    Validate the generated Gemini response.

    Raises:
        RuntimeError: If the response is invalid.
    """

    if not isinstance(response, str):
        raise RuntimeError("Response must be a string.")

    if not response.strip():
        raise RuntimeError("Response cannot be empty.")

    print("✓ Response validation passed.")



def validate_invalid_prompt(client: GeminiClient) -> None:
    """
    Verify invalid prompts raise ValueError.
    """

    print("\nTesting invalid prompt validation...")

    invalid_prompts = [
        "",
        "   ",
        None,
    ]

    for prompt in invalid_prompts:
        try:
            client.generate_response(prompt)  # type: ignore[arg-type]
        except ValueError:
            print(f"✓ Correctly rejected prompt: {repr(prompt)}")
        else:
            raise RuntimeError(
                f"Expected ValueError for prompt: {repr(prompt)}"
            )


def main() -> None:
    """
    Run GeminiClient integration tests.
    """

    print("=" * 70)
    print(" Intelligent Hospital Information Assistant")
    print(" Gemini Client Integration Test")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1 : Initialize Client
    # ---------------------------------------------------------

    print("\n[1/3] Initializing Gemini Client...")

    client = GeminiClient()

    print("✓ Gemini Client Initialized")

    # ---------------------------------------------------------
    # Step 2 : Generate Response
    # ---------------------------------------------------------

    print("\n[2/3] Generating sample response...")

    prompt = """
You are a hospital assistant.

Question:
Where is the cardiology department?

Context:
Cardiology Department is located in Block A,
Floor 1,
Room 203.

Answer:
"""

    response = client.generate_response(prompt)

    validate_response(response)

    print("\nGenerated Response\n")
    print("=" * 70)
    print(response)
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 3 : Validation Tests
    # ---------------------------------------------------------

    print("\n[3/3] Running validation tests...")

    validate_invalid_prompt(client)

    print("\n" + "=" * 70)
    print(" GEMINI CLIENT VALIDATION SUCCESSFUL")
    print("=" * 70)

    print("Initialization : PASSED")
    print("API Connection : PASSED")
    print("Response        : PASSED")
    print("Validation      : PASSED")

    print("\nAll Gemini Client integration tests passed successfully!")

    print("=" * 70)


if __name__ == "__main__":
    main()