"""
chroma_vector_store.py

Improved Chroma vector store manager.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIRECTORY = Path("vector_store")
DEFAULT_COLLECTION_NAME = "hospital_information"


class ChromaVectorStore:
    """Builds and loads a persistent Chroma vector database."""

    def __init__(
        self,
        embedding_model: Any,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        if embedding_model is None:
            raise ValueError("embedding_model cannot be None.")

        self.embedding_model = embedding_model
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._vector_store: Chroma | None = None

        logger.info(
            "Initialized ChromaVectorStore(collection=%s, path=%s)",
            self.collection_name,
            self.persist_directory,
        )

    def _validate_documents(self, documents: list[Document]) -> None:
        if not isinstance(documents, list) or not documents:
            raise ValueError("documents must be a non-empty list of Document objects.")

        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise ValueError(f"documents[{i}] is not a Document.")
            if not doc.page_content or not doc.page_content.strip():
                raise ValueError(f"documents[{i}] has empty page_content.")
              

    def _create_vector_store(self, documents: list[Document]) -> Chroma:
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name,
        )

    def build_vector_store(self, documents: list[Document], overwrite: bool = False) -> Chroma:
        self._validate_documents(documents)

        if overwrite and self.persist_directory.exists():
            logger.info("Removing existing vector store: %s", self.persist_directory)
            shutil.rmtree(self.persist_directory)

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            self._vector_store = self._create_vector_store(documents)
            logger.info("Vector store created successfully with %d documents.", len(documents))
            return self._vector_store
        except Exception as exc:
            logger.exception("Failed to build vector store.")
            raise RuntimeError(f"Failed to build vector store: {exc}") from exc

    def load_vector_store(self) -> Chroma:
        if not self.persist_directory.exists():
            raise RuntimeError(
                f"Vector store directory '{self.persist_directory}' does not exist."
            )

        try:
            self._vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
            )
            logger.info("Vector store loaded successfully.")
            return self._vector_store
        except Exception as exc:
            logger.exception("Failed to load vector store.")
            raise RuntimeError(f"Failed to load vector store: {exc}") from exc

    def get_vector_store(self) -> Chroma:
        if self._vector_store is None:
            raise RuntimeError(
                "Vector store not initialized. Call build_vector_store() or load_vector_store() first."
            )
        return self._vector_store
