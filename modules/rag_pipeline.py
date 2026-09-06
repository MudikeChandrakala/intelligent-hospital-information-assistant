"""
rag_pipeline.py

Orchestrates the Retrieval-Augmented Generation (RAG) backend for the
Intelligent Hospital Information Assistant.

This module has a single responsibility: coordinate the existing
embedding, retrieval, prompt-building, and generation components to
answer a user's question.

This module DOES NOT:
- Implement document retrieval logic
- Implement prompt construction logic
- Implement embedding generation logic
- Create or manage the vector store's underlying data
- Handle Streamlit or any other UI concerns
"""

from __future__ import annotations


from pathlib import Path
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core import documents

from modules import retriever
from modules.embedding_generator import EmbeddingGenerator
from modules.chroma_vector_store import ChromaVectorStore
from modules.document_loader import DocumentLoader
from modules.text_chunker import TextChunker
from modules.retriever import Retriever
from modules.prompt_builder import PromptBuilder
from modules.gemini_client import GeminiClient, GeminiUnavailableError

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Structured Result (Phase 9B)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RAGResponse:
    """
    Structured result of a single `RAGPipeline.ask()` execution.

    Introduced in Phase 9B so the backend can expose the runtime
    metrics the metrics/insights panel (`ui/metrics.py`) needs, without
    the pipeline duplicating any retrieval or generation logic — every
    field here is data already produced by the existing
    `Retriever` / `PromptBuilder` / `GeminiClient` collaborators during
    the same `ask()` call, simply retained and returned instead of
    being discarded once the method returns.

    Attributes:
        answer: The generated response text — identical to what
            `ask()` returned prior to Phase 9B. This is the only field
            most callers need.
        response_time_ms: Total wall-clock time, in milliseconds, for
            the full `ask()` execution (validation through the Gemini
            call).
        retrieval_time_ms: Wall-clock time, in milliseconds, spent
            specifically inside `self._retriever.retrieve(...)`.
        retrieved_documents: The list of `Document` objects returned by
            the retriever for this question, unmodified.
        source_documents: A list of metadata dictionaries, one per
            entry in `retrieved_documents`, derived directly from each
            document's existing `.metadata` attribute (e.g. `source`,
            `record_type`, `source_file`). No new fields are computed
            or inferred beyond what each document's metadata already
            contains.
        confidence_score: A similarity/confidence score in the 0.0-1.0
            range, or `None` if no such score is available. The
            current `Retriever` does not expose similarity scores (see
            `retriever.py`'s scoreless `as_retriever().invoke()` path),
            so this is `None` unless and until that changes — it is
            never fabricated to have a numeric value.
    """

    answer: str
    response_time_ms: float
    retrieval_time_ms: float
    retrieved_documents: List[Any] = field(default_factory=list)
    source_documents: List[dict] = field(default_factory=list)
    confidence_score: Optional[float] = None


class RAGPipeline:
    """
    Coordinates the RAG backend components to answer user questions.

    This class is responsible for initializing the embedding model,
    vector store, retriever, prompt builder, and Gemini client, and for
    orchestrating the end-to-end flow of validating a question,
    retrieving relevant documents, building a prompt, and generating a
    response. It does not implement any of the underlying embedding,
    retrieval, prompt construction, or generation logic itself.

    Attributes
    ----------
    _embedding_model : Any
        The initialized embedding model used by the vector store.

    _vector_store : Any
        The initialized vector store used for document retrieval.

    _retriever : Any
        The initialized retriever used to fetch relevant documents.

    _prompt_builder : Any
        The initialized prompt builder used to construct prompts.

    _gemini_client : Any
        The initialized Gemini client used to generate responses.
    """

    #: Metadata fields kept in each `source_documents` entry produced by
    #: `_extract_source_documents()`. These are the actual field names
    #: `DocumentLoader`'s `_parse_*_dataset` methods already write into
    #: every retrieved `Document`'s `.metadata` (doctor/department/
    #: symptom/disease/medicine/navigation/appointment/insurance/
    #: emergency records all set `source`, `source_file`, and
    #: `record_type`; doctor and several other structured records set
    #: `department_name`; doctor records set `doctor_id`; unstructured
    #: records set `title` or, for FAQ, `question`; several structured
    #: and unstructured records set `record_id`). No field is renamed,
    #: invented, or fabricated — this is only a filter over keys that
    #: already exist in the metadata.
    _SOURCE_DOCUMENT_METADATA_FIELDS: tuple[str, ...] = (
        "source",
        "source_file",
        "record_type",
        "department_name",
        "doctor_id",
        "title",
        "question",
        "record_id",
    )

    def __init__(self) -> None:
        """
        Initialize the RAG pipeline.

        Initializes the embedding model, vector store, retriever,
        prompt builder, and Gemini client required to answer questions.

        Raises:
            RuntimeError: If any component fails to initialize.
        """

        self._embedding_model: Any = self._initialize_embedding_model()
        self._vector_store: Any = self._initialize_vector_store()
        self._retriever: Any = self._initialize_retriever()
        self._prompt_builder: Any = self._initialize_prompt_builder()
        self._gemini_client: Any = self._initialize_gemini_client()

        logger.info("RAGPipeline initialized successfully.")

    # -----------------------------------------------------------------
    # Internal Helper Methods
    # -----------------------------------------------------------------

    def _initialize_embedding_model(self) -> Any:
        """
        Initialize the embedding model component.

        Returns:
            The initialized embedding model instance.

        Raises:
            RuntimeError: If the embedding model cannot be initialized.
        """

        try:
            generator = EmbeddingGenerator()
            embedding_model = generator.get_embedding_model()
        except Exception as exc:
            logger.exception("Failed to initialize the embedding model.")
            raise RuntimeError(f"Failed to initialize embedding model: {exc}") from exc

        logger.info("Embedding model initialized successfully.")
        return embedding_model

    def _initialize_vector_store(self) -> Any:
        """
        Initialize the vector store component.

        If a persisted vector store already exists, load it normally.
        If it does not exist, build it automatically from the hospital
        knowledge base.

        Returns:
            The initialized vector store instance.

        Raises:
            RuntimeError: If the vector store cannot be initialized.
        """
        try:
            vector_store = ChromaVectorStore(
                embedding_model=self._embedding_model
            )

            if vector_store.persist_directory.exists():
                logger.info(
                    "Existing vector store found. Loading persisted vector store."
                )
                vector_store.load_vector_store()
            else:
                logger.info(
                    "Vector store not found. Building it from the hospital "
                    "knowledge base."
                )

                project_root = Path(__file__).resolve().parent.parent

                document_loader = DocumentLoader(
                    project_root=project_root
                )
                documents = document_loader.load_all_documents()

                text_chunker = TextChunker()
                chunks = text_chunker.split_documents(documents)

                logger.info(
                    "Loaded %d documents and created %d chunks.",
                    len(documents),
                    len(chunks),
                )

                vector_store.build_vector_store(
                    chunks,
                    overwrite=False,
                )

            loaded_vector_store = vector_store.get_vector_store()

        except Exception as exc:
            logger.exception("Failed to initialize the vector store.")
            raise RuntimeError(
                f"Failed to initialize vector store: {exc}"
            ) from exc

        logger.info("Vector store initialized successfully.")
        return loaded_vector_store
    def _initialize_retriever(self) -> Any:
        """
        Initialize the retriever component.

        Returns:
            The initialized retriever instance.

        Raises:
            RuntimeError: If the retriever cannot be initialized.
        """

        try:
            retriever = Retriever(self._vector_store)
        except Exception as exc:
            logger.exception("Failed to initialize the retriever.")
            raise RuntimeError(
                f"Failed to initialize retriever: {exc}"
            ) from exc

        logger.info("Retriever initialized successfully.")
        return retriever

    def _initialize_prompt_builder(self) -> Any:
        """
        Initialize the prompt builder component.

        Returns:
            The initialized prompt builder instance.

        Raises:
            RuntimeError: If the prompt builder cannot be initialized.
        """

        try:
            prompt_builder = PromptBuilder()
        except Exception as exc:
            logger.exception("Failed to initialize the prompt builder.")
            raise RuntimeError(
                f"Failed to initialize prompt builder: {exc}"
            ) from exc

        logger.info("Prompt builder initialized successfully.")
        return prompt_builder


    def _enrich_with_department_doctors(
        self,
        documents: List[Any],
        question: str,
    ) -> List[Any]:
        """
        Add doctor records for departments identified by the retrieved
        documents.

        Doctor records are retrieved using existing metadata rather than
        hardcoded doctor names.
        """
        enriched_documents = list(documents)

        department_names: set[str] = set()

        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}

            department_name = metadata.get("department_name")
            if department_name:
                department_names.add(str(department_name))

            recommended_department = metadata.get("recommended_department")
            if recommended_department:
                department_names.add(str(recommended_department))

        if not department_names:
            return enriched_documents

        existing_doctor_ids = {
            str(document.metadata.get("doctor_id"))
            for document in documents
            if getattr(document, "metadata", None)
            and document.metadata.get("doctor_id")
        }

        for department_name in sorted(department_names):
            try:
                                doctor_documents = self._retriever.retrieve_by_metadata(
                    {
                        "$and": [
                            {"record_type": "doctor"},
                            {"department_name": department_name},
                        ]
                    },
                    limit=5,
                )
            except Exception:
                logger.exception(
                    "Failed to retrieve doctors for department '%s'.",
                    department_name,
                )
                continue

            for doctor_document in doctor_documents:
                doctor_id = str(
                    doctor_document.metadata.get("doctor_id", "")
                )

                if doctor_id and doctor_id in existing_doctor_ids:
                    continue

                enriched_documents.append(doctor_document)

                if doctor_id:
                    existing_doctor_ids.add(doctor_id)

        logger.info(
            "Department doctor enrichment added %d documents.",
            len(enriched_documents) - len(documents),
        )

        return enriched_documents

    def _initialize_gemini_client(self) -> Any:
        """
        Initialize the Gemini client component.

        Returns:
            The initialized Gemini client instance.

        Raises:
            RuntimeError: If the Gemini client cannot be initialized.
        """

        try:
            gemini_client = GeminiClient()
        except Exception as exc:
            logger.exception("Failed to initialize the Gemini client.")
            raise RuntimeError(f"Failed to initialize Gemini client: {exc}") from exc

        logger.info("Gemini client initialized successfully.")
        return gemini_client

    def _validate_question(self, question: str) -> None:
        """
        Validate that the supplied question is a non-empty string.

        Args:
            question: The question to validate.

        Raises:
            ValueError: If ``question`` is empty, contains only
                whitespace, or is not a string.
        """

        if not isinstance(question, str) or not question.strip():
            logger.error("Invalid question provided to RAGPipeline.")
            raise ValueError("question cannot be empty.")

        logger.debug("Question validated successfully.")

    # -----------------------------------------------------------------
    # Public APIs
    # -----------------------------------------------------------------

    def ask(self, question: str, conversation_history: Optional[List[Any]] = None) -> RAGResponse:
        """
        Answer a user's question using the RAG pipeline.

        Validates the question, retrieves relevant documents, builds a
        prompt, generates a response via Gemini, and returns a
        structured `RAGResponse` bundling the generated answer together
        with the runtime metrics (Phase 9B) the metrics/insights panel
        needs: total response time, retrieval time, the retrieved
        documents themselves, and source-document metadata derived from
        them. No retrieval, prompt-construction, or generation logic is
        duplicated to produce these metrics — every value is measured
        or read directly from the same `Retriever` / `PromptBuilder` /
        `GeminiClient` calls this method already makes.

        Args:
            question: The user's question.
            conversation_history: Optional list of previous conversation messages
                to provide context for the current question. When provided,
                helps Gemini understand follow-up questions and their resolution.

        Returns:
            A `RAGResponse` containing the generated answer text plus
            its associated runtime metrics. `confidence_score` is
            `None` unless the underlying `Retriever` exposes a genuine
            similarity/confidence score — it is never fabricated.

        Raises:
            ValueError: If ``question`` fails validation.
            RuntimeError: If any pipeline stage fails.
        """

        self._validate_question(question)
        cleaned_question = question.strip()

        logger.info("Starting RAG pipeline execution.")

        pipeline_start_time: float = time.perf_counter()

        try:
            retrieval_start_time: float = time.perf_counter()
            documents: List[Any] = self._retriever.retrieve(cleaned_question)

            documents = self._enrich_with_department_doctors(
                documents,
                cleaned_question,
            )

            retrieval_time_ms: float = (
                time.perf_counter() - retrieval_start_time
            ) * 1000
        except Exception as exc:
            logger.exception("Failed to retrieve documents.")
            raise RuntimeError(f"Failed to retrieve documents: {exc}") from exc

        logger.info("Documents retrieved successfully in %.2f ms.", retrieval_time_ms)

        try:
            prompt: str = self._prompt_builder.build_prompt(
                cleaned_question, documents, conversation_history=conversation_history
            )
            print("\n" + "=" * 100)
            print("FINAL PROMPT SENT TO GEMINI")
            print("=" * 100)
            print(prompt)
            print("=" * 100)
        except Exception as exc:
            logger.exception("Failed to build prompt.")
            raise RuntimeError(f"Failed to build prompt: {exc}") from exc

        logger.info("Prompt built successfully.")

        try:
            response: str = self._gemini_client.generate_response(prompt)
        except GeminiUnavailableError as exc:
            logger.warning(
                "Gemini generation failed because the service is "
                "temporarily unavailable (%s). Using a deterministic "
                "answer built from the already-retrieved hospital "
                "documents instead of failing the request. Gemini will "
                "not be called again for this request.",
                exc,
            )
            response = self._build_retrieval_fallback_response(
                documents, cleaned_question
            )
        except Exception as exc:
            logger.exception("Failed to generate response.")
            raise RuntimeError(f"Failed to generate response: {exc}") from exc

        response_time_ms: float = (time.perf_counter() - pipeline_start_time) * 1000

        source_documents: List[dict] = self._extract_source_documents(documents)

        logger.info(
            "RAG pipeline execution completed successfully in %.2f ms (retrieval: %.2f ms).",
            response_time_ms,
            retrieval_time_ms,
        )

        return RAGResponse(
            answer=response,
            response_time_ms=response_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            retrieved_documents=documents,
            source_documents=source_documents,
            confidence_score=None,
        )

    def _extract_source_documents(self, documents: List[Any]) -> List[dict]:
        """
        Derive lightweight, UI-friendly source-document metadata from
        retrieved documents.

        Reads only the existing `.metadata` attribute already present
        on each retrieved `Document` (populated upstream by
        `DocumentLoader` when the knowledge base was built), and keeps
        only the subset of fields listed in
        `_SOURCE_DOCUMENT_METADATA_FIELDS` — ``source``,
        ``source_file``, ``record_type``, ``department_name``,
        ``doctor_id``, ``title``, ``question``, and ``record_id`` —
        instead of copying every raw metadata field a document happens
        to carry (e.g. internal linkage IDs like ``navigation_id`` or
        operational flags like ``wheelchair_accessible``). These are
        the actual field names `DocumentLoader` already writes into
        metadata; this method renames nothing and computes nothing new,
        it only filters. A field is included only when it is present
        on the document's metadata and its value is not `None`. A
        document with no `metadata` attribute, an empty one, or none of
        the whitelisted fields present, contributes an empty dict
        rather than being skipped, so `source_documents` always aligns
        one-to-one with `retrieved_documents`.

        Args:
            documents: The `Document` objects returned by
                `self._retriever.retrieve(...)` for this question.

        Returns:
            A list of lightweight metadata dictionaries, one per
            document, in the same order as ``documents``.
        """

        source_documents: List[dict] = []
        for document in documents:
            metadata = getattr(document, "metadata", None)
            if not metadata:
                source_documents.append({})
                continue

            lightweight_metadata = {
                field_name: metadata[field_name]
                for field_name in self._SOURCE_DOCUMENT_METADATA_FIELDS
                if field_name in metadata and metadata[field_name] is not None
            }
            source_documents.append(lightweight_metadata)

        return source_documents

    def _build_retrieval_fallback_response(
        self, documents: List[Any], question: str
    ) -> str:
        """
        Build a deterministic answer directly from already-retrieved
        documents, used ONLY when Gemini itself is temporarily
        unavailable after `GeminiClient.generate_response()`'s bounded
        retries are exhausted (`GeminiUnavailableError`). Gemini is not
        called again to produce this answer.

        Reads only the existing metadata fields
        `_enrich_with_department_doctors` already relies on -
        `department_name`, `recommended_department`, `record_type`, and
        the `page_content` of any retrieved doctor record - so no
        department, doctor, or symptom is invented or hardcoded here.
        If the retrieved documents contain none of this information,
        a safe, generic message is returned instead.

        Args:
            documents: The (already retrieved and enriched) `Document`
                objects for this question.
            question: The user's cleaned question. Unused for now -
                accepted for a stable signature and to keep this
                clearly scoped to what was actually retrieved, not to
                re-interpret the question itself.

        Returns:
            A concise, deterministic answer built only from
            `documents`, or a safe "limited information" message if
            nothing usable was retrieved.
        """

        department_names: List[str] = []
        doctor_descriptions: List[str] = []

        for document in documents:
            metadata = getattr(document, "metadata", None) or {}

            for field_name in ("department_name", "recommended_department"):
                department_name = metadata.get(field_name)
                if department_name and str(department_name) not in department_names:
                    department_names.append(str(department_name))

            if metadata.get("record_type") == "doctor":
                content = (getattr(document, "page_content", "") or "").strip()
                if content and content not in doctor_descriptions:
                    doctor_descriptions.append(content)

        if not department_names and not doctor_descriptions:
            logger.info(
                "Deterministic retrieval fallback: no department or doctor "
                "information was identifiable in the retrieved documents."
            )
            return (
                "The hospital information retrieved for this question is "
                "limited. Please contact the hospital information desk "
                "for assistance."
            )

        response_sections: List[str] = []

        if department_names:
            if len(department_names) == 1:
                response_sections.append(
                    f"Based on the hospital information available, "
                    f"{department_names[0]} is the recommended department "
                    f"for this concern. Please consult a doctor in that "
                    f"department for an appropriate evaluation."
                )
            else:
                joined_departments = ", ".join(department_names)
                response_sections.append(
                    f"Based on the hospital information available, the "
                    f"following departments may be relevant to this "
                    f"concern: {joined_departments}. Please consult a "
                    f"doctor in the most appropriate department for an "
                    f"evaluation."
                )

        if doctor_descriptions:
            response_sections.append(
                "The following doctor information from the hospital "
                "knowledge base may help:\n"
                + "\n".join(f"- {description}" for description in doctor_descriptions)
            )

        response_sections.append(
            "This information is based on the hospital knowledge base "
            "only. If your concern is severe or urgent, please seek "
            "emergency care immediately."
        )

        return "\n\n".join(response_sections)