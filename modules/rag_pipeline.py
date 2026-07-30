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

import logging
from typing import Any, List

from modules.embedding_generator import EmbeddingGenerator
from modules.chroma_vector_store import ChromaVectorStore
from modules.retriever import Retriever
from modules.prompt_builder import PromptBuilder
from modules.gemini_client import GeminiClient

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)


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

        Returns:
            The loaded vector store instance.

        Raises:
            RuntimeError: If the vector store cannot be initialized.
        """

        try:
            vector_store = ChromaVectorStore(embedding_model=self._embedding_model)
            vector_store.load_vector_store()
            loaded_vector_store = vector_store.get_vector_store()
        except Exception as exc:
            logger.exception("Failed to initialize the vector store.")
            raise RuntimeError(f"Failed to initialize vector store: {exc}") from exc

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
            raise RuntimeError(f"Failed to initialize retriever: {exc}") from exc

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
            raise RuntimeError(f"Failed to initialize prompt builder: {exc}") from exc

        logger.info("Prompt builder initialized successfully.")
        return prompt_builder

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

    def ask(self, question: str) -> str:
        """
        Answer a user's question using the RAG pipeline.

        Validates the question, retrieves relevant documents, builds a
        prompt, generates a response via Gemini, and returns the
        generated response.

        Args:
            question: The user's question.

        Returns:
            The generated response text.

        Raises:
            ValueError: If ``question`` fails validation.
            RuntimeError: If any pipeline stage fails.
        """

        self._validate_question(question)
        cleaned_question = question.strip()

        logger.info("Starting RAG pipeline execution.")

        try:
            documents: List[Any] = self._retriever.retrieve(cleaned_question)
        except Exception as exc:
            logger.exception("Failed to retrieve documents.")
            raise RuntimeError(f"Failed to retrieve documents: {exc}") from exc

        logger.info("Documents retrieved successfully.")

        try:
            prompt: str = self._prompt_builder.build_prompt(cleaned_question, documents)
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
        except Exception as exc:
            logger.exception("Failed to generate response.")
            raise RuntimeError(f"Failed to generate response: {exc}") from exc

        logger.info("RAG pipeline execution completed successfully.")
        return response