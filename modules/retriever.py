"""
retriever.py

Wraps an already-built vector store's LangChain retriever interface to
fetch relevant documents for a given query.

This module has a single responsibility: validate queries and retrieve
matching chunked documents from a vector store supplied via dependency
injection.

This module DOES NOT:
- Create or load a Chroma vector store
- Call Google Gemini
- Build prompts
- Implement any other part of the RAG pipeline
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_SEARCH_TYPE = "similarity"
DEFAULT_K = 5


class Retriever:
    """
    Retrieves relevant documents from an injected vector store.

    This class is responsible for validating queries and wrapping a
    vector store's `as_retriever()` interface so callers can fetch
    relevant chunked documents without knowing the underlying vector
    store implementation. The vector store instance is supplied via
    dependency injection and is never created or loaded inside this
    class.

    Attributes
    ----------
    _vector_store : Any
        The pre-built vector store instance used to create the retriever.

    search_type : str
        The search strategy used by the retriever (e.g. ``"similarity"``).

    k : int
        The number of documents to return per query.

    _retriever : Any
        The underlying LangChain retriever instance.
    """

    def __init__(
        self,
        vector_store: Any,
        search_type: str = DEFAULT_SEARCH_TYPE,
        k: int = DEFAULT_K,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            vector_store: A pre-built vector store instance (e.g. a
                `Chroma` instance) exposing an `as_retriever()` method.
                This class does not create or own the vector store's
                lifecycle.
            search_type: The search strategy to use, passed to
                `vector_store.as_retriever()`. Defaults to
                ``"similarity"``.
            k: The number of documents to return per query. Defaults to
                ``5``.

        Raises:
            ValueError: If ``vector_store`` is ``None``, ``search_type``
                is empty, or ``k`` is not a positive integer.
            RuntimeError: If the underlying retriever cannot be created.
        """

        if vector_store is None:
            logger.error("No vector store provided to Retriever.")
            raise ValueError("vector_store cannot be None.")

        if not isinstance(search_type, str) or not search_type.strip():
            logger.error("Invalid search_type provided to Retriever: %r", search_type)
            raise ValueError("search_type cannot be empty.")

        search_type = search_type.strip()

        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            logger.error("Invalid k provided to Retriever: %r", k)
            raise ValueError("k must be a positive integer.")

        self._vector_store: Any = vector_store
        self.search_type: str = search_type
        self.k: int = k
        self._retriever: Any = self._create_retriever()

        logger.info(
            "Retriever initialized successfully with search_type='%s', k=%s.",
            self.search_type,
            self.k,
        )

    # -----------------------------------------------------------------
    # Internal Helper Methods
    # -----------------------------------------------------------------

    def _validate_query(self, query: str) -> None:
        """
        Validate that the supplied query is a non-empty string.

        Args:
            query: The user query to validate.

        Raises:
            ValueError: If ``query`` is empty, contains only whitespace,
                or is not a string.
        """

        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query provided to Retriever: %r", query)
            raise ValueError("query cannot be empty.")

        logger.debug("Query validated successfully.")

    def _create_retriever(self) -> Any:
        """
        Create the underlying LangChain retriever from the vector store.

        Returns:
            The retriever instance returned by
            `vector_store.as_retriever()`.

        Raises:
            RuntimeError: If the retriever cannot be created.
        """

        logger.info(
            "Creating retriever with search_type='%s', k=%s.",
            self.search_type,
            self.k,
        )

        try:
            retriever = self._vector_store.as_retriever(
                search_type=self.search_type,
                search_kwargs={"k": self.k},
            )
        except Exception as exc:
            logger.exception("Failed to create retriever from vector store.")
            raise RuntimeError(f"Failed to create retriever: {exc}") from exc

        logger.info("Successfully created retriever.")
        return retriever

    def _validate_results(self, results: Any) -> None:
        """
        Validate that retrieval results are a list of `Document` objects.

        Args:
            results: The raw value returned by the underlying retriever.

        Raises:
            RuntimeError: If ``results`` is not a list, or if any item
                in ``results`` is not a `Document` instance.
        """

        if not isinstance(results, list):
            logger.error(
                "Retriever returned a non-list result of type %s.",
                type(results).__name__,
            )
            raise RuntimeError("Retriever must return a list of Document objects.")

        for index, document in enumerate(results):
            if not isinstance(document, Document):
                logger.error(
                    "Retriever result at index %d is not a Document instance (got %s).",
                    index,
                    type(document).__name__,
                )
                raise RuntimeError(
                    "Retriever must return a list of Document objects."
                )

        logger.debug("Retrieval results validated successfully.")

    # -----------------------------------------------------------------
    # Public APIs
    # -----------------------------------------------------------------

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve relevant documents for the given query.

        Args:
            query: The user query to retrieve relevant documents for.

        Returns:
            A list of `Document` objects relevant to the query.

        Raises:
            ValueError: If ``query`` fails validation.
            RuntimeError: If retrieval fails or returns malformed results.
        """

        self._validate_query(query)
        cleaned_query = query.strip()

        logger.info("Retrieving documents for query.")

        try:
            results = self._retriever.invoke(cleaned_query)
        except Exception as exc:
            logger.exception("Failed to retrieve documents for query.")
            raise RuntimeError(f"Failed to retrieve documents: {exc}") from exc

        self._validate_results(results)

        logger.info(
            "Retrieved %d document(s) using '%s' search.",
            len(results),
            self.search_type,
        )
        return results

    def get_retriever(self) -> Any:
        """
        Return the underlying LangChain retriever instance.

        Returns:
            The retriever instance created from the injected vector store.
        """

        return self._retriever