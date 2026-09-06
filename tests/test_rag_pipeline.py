"""
tests/test_rag_pipeline_fallback.py
=============================================================================
Focused regression tests for the deterministic retrieval fallback added to
RAGPipeline.ask():
- Gemini success is returned exactly as before;
- GeminiUnavailableError triggers the deterministic fallback (Gemini is
  not called again);
- the fallback uses only existing retrieved-document metadata
  (department_name / recommended_department / record_type == "doctor")
  and contains no hardcoded department or doctor name;
- a report with nothing usable returns the safe generic message;
- any OTHER exception from Gemini still raises RuntimeError as before
  (no change to that existing behavior);
- RAGResponse's existing fields (response_time_ms, retrieval_time_ms,
  retrieved_documents, source_documents) are still populated normally in
  the fallback path.

RAGPipeline.__init__ initializes real embedding/vector-store/retriever
components, so these tests construct an instance without calling
__init__ and inject fake _retriever / _prompt_builder / _gemini_client
collaborators, exactly like the existing architecture expects.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.rag_pipeline import RAGPipeline
from modules.gemini_client import GeminiUnavailableError


class _FakeRetriever:
    def __init__(self, documents: List[Document]) -> None:
        self._documents = documents

    def retrieve(self, question: str) -> List[Document]:
        return list(self._documents)

    def retrieve_by_metadata(self, filters: dict, limit: Optional[int] = None):
        return []


class _FakePromptBuilder:
    def build_prompt(self, question, documents, conversation_history=None) -> str:
        return f"PROMPT for: {question}"


class _FakeGeminiClient:
    def __init__(self, *, answer: Optional[str] = None, error: Optional[Exception] = None):
        self._answer = answer
        self._error = error
        self.calls = 0

    def generate_response(self, prompt: str) -> str:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._answer


def _make_pipeline(retriever, prompt_builder, gemini_client) -> RAGPipeline:
    """A RAGPipeline instance bypassing __init__'s real component setup."""
    pipeline = object.__new__(RAGPipeline)
    pipeline._embedding_model = None
    pipeline._vector_store = None
    pipeline._retriever = retriever
    pipeline._prompt_builder = prompt_builder
    pipeline._gemini_client = gemini_client
    return pipeline


# =============================================================================
# Normal Gemini success is unaffected
# =============================================================================


def test_ask_returns_gemini_answer_unchanged_on_success():
    documents = [Document(page_content="Cardiology is on floor 2.", metadata={})]
    gemini_client = _FakeGeminiClient(answer="Cardiology is on floor 2.")
    pipeline = _make_pipeline(_FakeRetriever(documents), _FakePromptBuilder(), gemini_client)

    result = pipeline.ask("Where is Cardiology?")

    assert result.answer == "Cardiology is on floor 2."
    assert gemini_client.calls == 1
    assert result.response_time_ms >= 0
    assert result.retrieval_time_ms >= 0
    assert result.retrieved_documents == documents


# =============================================================================
# GeminiUnavailableError triggers the deterministic fallback
# =============================================================================


def test_ask_falls_back_when_gemini_is_unavailable_and_does_not_call_gemini_again():
    documents = [
        Document(
            page_content="Dr. Meera Rao - Cardiologist, 10 years experience.",
            metadata={"record_type": "doctor", "department_name": "Cardiology"},
        )
    ]
    gemini_client = _FakeGeminiClient(error=GeminiUnavailableError("still unavailable"))
    pipeline = _make_pipeline(_FakeRetriever(documents), _FakePromptBuilder(), gemini_client)

    result = pipeline.ask("My father has chest pain, which department?")

    assert "Cardiology" in result.answer
    assert gemini_client.calls == 1  # Gemini is not retried again after the fallback
    assert result.retrieved_documents == documents


def test_fallback_uses_department_name_metadata_field():
    documents = [Document(page_content="", metadata={"department_name": "Neurology"})]
    gemini_client = _FakeGeminiClient(error=GeminiUnavailableError("down"))
    pipeline = _make_pipeline(_FakeRetriever(documents), _FakePromptBuilder(), gemini_client)

    result = pipeline.ask("A question.")

    assert "Neurology" in result.answer
    assert "recommended department" in result.answer


def test_fallback_uses_recommended_department_metadata_field():
    documents = [Document(page_content="", metadata={"recommended_department": "Orthopedics"})]
    gemini_client = _FakeGeminiClient(error=GeminiUnavailableError("down"))
    pipeline = _make_pipeline(_FakeRetriever(documents), _FakePromptBuilder(), gemini_client)

    result = pipeline.ask("A question.")

    assert "Orthopedics" in result.answer


def test_fallback_is_driven_by_retrieved_data_not_hardcoded():
    # Swap the department name in the fixture and confirm the output
    # changes correspondingly - proving nothing is hardcoded.
    gemini_client = _FakeGeminiClient(error=GeminiUnavailableError("down"))

    documents_a = [Document(page_content="", metadata={"department_name": "Dermatology"})]
    pipeline_a = _make_pipeline(_FakeRetriever(documents_a), _FakePromptBuilder(), gemini_client)
    result_a = pipeline_a.ask("A question.")
    assert "Dermatology" in result_a.answer
    assert "Neurology" not in result_a.answer

    documents_b = [Document(page_content="", metadata={"department_name": "Urology"})]
    pipeline_b = _make_pipeline(
        _FakeRetriever(documents_b), _FakePromptBuilder(), _FakeGeminiClient(error=GeminiUnavailableError("down"))
    )
    result_b = pipeline_b.ask("A question.")
    assert "Urology" in result_b.answer
    assert "Dermatology" not in result_b.answer


def test_fallback_includes_doctor_description_from_retrieved_document():
    documents = [
        Document(
            page_content="Dr. Arjun Nair - Neurologist, OPD Mon-Fri 10am-2pm.",
            metadata={"record_type": "doctor", "department_name": "Neurology"},
        )
    ]
    gemini_client = _FakeGeminiClient(error=GeminiUnavailableError("down"))
    pipeline = _make_pipeline(_FakeRetriever(documents), _FakePromptBuilder(), gemini_client)

    result = pipeline.ask("A question.")

    assert "Dr. Arjun Nair" in result.answer
    assert "OPD Mon-Fri 10am-2pm" in result.answer


def test_fallback_contains_no_hardcoded_doctor_or_department_name():
    # A completely made-up, fictional department/doctor name that could
    # not possibly be hardcoded anywhere in production logic.
    documents = [
        Document(
            page_content="Dr. Zzyzx Q. Fictionalperson - Xenology specialist.",
            metadata={"record_type": "doctor", "department_name": "Xenology Wing 9"},
        )
    ]
    gemini_client = _FakeGeminiClient(error=GeminiUnavailableError("down"))
    pipeline = _make_pipeline(_FakeRetriever(documents), _FakePromptBuilder(), gemini_client)

    result = pipeline.ask("A question.")

    assert "Xenology Wing 9" in result.answer
    assert "Dr. Zzyzx Q. Fictionalperson" in result.answer


def test_fallback_returns_safe_message_when_nothing_useful_is_retrieved():
    documents = [Document(page_content="Some unrelated FAQ text.", metadata={})]
    gemini_client = _FakeGeminiClient(error=GeminiUnavailableError("down"))
    pipeline = _make_pipeline(_FakeRetriever(documents), _FakePromptBuilder(), gemini_client)

    result = pipeline.ask("A question.")

    assert "limited" in result.answer.lower()
    assert "information desk" in result.answer.lower()


def test_fallback_with_no_documents_at_all_returns_safe_message():
    gemini_client = _FakeGeminiClient(error=GeminiUnavailableError("down"))
    pipeline = _make_pipeline(_FakeRetriever([]), _FakePromptBuilder(), gemini_client)

    result = pipeline.ask("A question.")

    assert "limited" in result.answer.lower()


# =============================================================================
# Any OTHER exception from Gemini still fails exactly as before
# =============================================================================


def test_other_gemini_exceptions_still_raise_runtime_error_not_fallback():
    documents = [Document(page_content="", metadata={"department_name": "Cardiology"})]
    gemini_client = _FakeGeminiClient(error=ValueError("invalid prompt"))
    pipeline = _make_pipeline(_FakeRetriever(documents), _FakePromptBuilder(), gemini_client)

    with pytest.raises(RuntimeError) as exc_info:
        pipeline.ask("A question.")

    assert not isinstance(exc_info.value, GeminiUnavailableError)


# =============================================================================
# RAGResponse structure is preserved in the fallback path
# =============================================================================


def test_response_metrics_and_source_documents_still_populated_in_fallback_path():
    documents = [
        Document(
            page_content="Dr. Priya Sharma - Cardiologist.",
            metadata={
                "record_type": "doctor",
                "department_name": "Cardiology",
                "source": "doctors.csv",
                "doctor_id": "D-101",
            },
        )
    ]
    gemini_client = _FakeGeminiClient(error=GeminiUnavailableError("down"))
    pipeline = _make_pipeline(_FakeRetriever(documents), _FakePromptBuilder(), gemini_client)

    result = pipeline.ask("A question.")

    assert result.response_time_ms >= 0
    assert result.retrieval_time_ms >= 0
    assert result.retrieved_documents == documents
    assert result.source_documents == [
        {"record_type": "doctor", "department_name": "Cardiology", "source": "doctors.csv", "doctor_id": "D-101"}
    ]
    assert result.confidence_score is None