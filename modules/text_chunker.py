"""
text_chunker.py

Splits LangChain Document objects into retrieval-ready text chunks while
preserving their source metadata.

This module does not load data, generate embeddings, or perform retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Split LangChain documents into overlapping, metadata-preserving chunks.

    Parameters
    ----------
    chunk_size : int, default=500
        Maximum number of characters in each text chunk.

    chunk_overlap : int, default=100
        Number of trailing characters shared by consecutive chunks.

    Raises
    ------
    TypeError
        If ``chunk_size`` or ``chunk_overlap`` is not an integer.

    ValueError
        If the chunk configuration is invalid.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        """Initialize the text splitter with the requested chunk settings."""

        self._validate_chunk_configuration(chunk_size, chunk_overlap)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        logger.info("TextChunker initialized successfully.")
        logger.debug("Chunk Size    : %s", self.chunk_size)
        logger.debug("Chunk Overlap : %s", self.chunk_overlap)

    # -----------------------------------------------------------------
    # Internal Helper Methods
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_chunk_configuration(chunk_size: int, chunk_overlap: int) -> None:
        """Validate the configured chunk size and overlap values."""

        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer greater than zero.")

        if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int):
            raise TypeError("chunk_overlap must be a non-negative integer.")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to zero.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

    @staticmethod
    def _validate_documents(documents: list[Document]) -> None:
        """Validate that the input is a list of LangChain Document objects."""

        if not isinstance(documents, list):
            raise TypeError("documents must be a list of LangChain Document objects.")

        for index, document in enumerate(documents):
            if not isinstance(document, Document):
                raise TypeError(
                    "Every item in documents must be a LangChain Document "
                    f"object. Invalid item at index {index}: "
                    f"{type(document).__name__}."
                )

            if not document.page_content.strip():
                raise ValueError(
                    f"Document at index {index} contains empty page_content."
                )

    @staticmethod
    def _create_chunk_metadata(
        original_metadata: dict[str, Any],
        chunk_index: int,
        total_chunks: int,
    ) -> dict[str, Any]:
        """Create chunk metadata without changing the original document metadata."""

        metadata = dict(original_metadata)
        metadata["chunk_index"] = chunk_index
        metadata["total_chunks"] = total_chunks
        return metadata

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into overlapping chunks and enrich their metadata.

        Each returned chunk retains all metadata from its source document and
        includes ``chunk_index`` and ``total_chunks`` fields. Chunk indices
        are zero-based and reset for each source document.

        Parameters
        ----------
        documents : list[Document]
            LangChain documents to split.

        Returns
        -------
        list[Document]
            The resulting metadata-preserving text chunks.

        Raises
        ------
        TypeError
            If ``documents`` is not a list of LangChain Document objects.
        """

        self._validate_documents(documents)
        logger.info("Starting document chunking for %s documents.", len(documents))

        chunks: list[Document] = []

        for document in documents:
            document_chunks = self._text_splitter.split_documents([document])
            total_chunks = len(document_chunks)

            for chunk_index, chunk in enumerate(document_chunks):
                chunk.metadata = self._create_chunk_metadata(
                    document.metadata,
                    chunk_index,
                    total_chunks,
                )
                chunks.append(chunk)

        logger.info("Generated %s text chunks.", len(chunks))
        logger.info("Document chunking completed successfully.")
        return chunks
