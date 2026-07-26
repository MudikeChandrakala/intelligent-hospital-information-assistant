"""
embedding_generator.py

Loads and exposes the sentence embedding model used by the RAG pipeline.

Responsibilities
----------------
- Validate the embedding model name.
- Load the HuggingFace embedding model.
- Expose the loaded embedding model.

This module DOES NOT:
- Build or query a vector database
- Perform document retrieval
- Call Gemini
- Implement any RAG logic
- Provide a Streamlit interface
"""

from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingGenerator:
    """
    Loads and exposes a HuggingFace sentence embedding model.

    This class validates the requested embedding model name,
    loads the model, and provides access to it for later
    components of the RAG pipeline.

    Attributes
    ----------
    model_name : str
        HuggingFace model name.

    _embedding_model : HuggingFaceEmbeddings
        Loaded embedding model instance.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """
        Initialize the embedding generator.

        Parameters
        ----------
        model_name : str, optional
            HuggingFace embedding model name.

        Raises
        ------
        ValueError
            If the model name is invalid.

        RuntimeError
            If the embedding model cannot be loaded.
        """

        self._validate_model_name(model_name)

        self.model_name: str = model_name.strip()

        self._embedding_model: HuggingFaceEmbeddings = (
            self._load_embedding_model(self.model_name)
        )

        logger.info(
            "EmbeddingGenerator initialized successfully with model: %s",
            self.model_name,
        )

    # -----------------------------------------------------------------
    # Private Methods
    # -----------------------------------------------------------------

    def _validate_model_name(self, model_name: str) -> None:
        """
        Validate the embedding model name.

        Parameters
        ----------
        model_name : str
            Model name to validate.

        Raises
        ------
        ValueError
            If the model name is empty or invalid.
        """

        if not isinstance(model_name, str):
            logger.error("Model name must be a string.")
            raise ValueError("Embedding model name must be a string.")

        if not model_name.strip():
            logger.error("Embedding model name cannot be empty.")
            raise ValueError("Embedding model name cannot be empty.")

        logger.debug("Validated embedding model: %s", model_name.strip())

    def _load_embedding_model(
        self,
        model_name: str,
    ) -> HuggingFaceEmbeddings:
        """
        Load the HuggingFace embedding model.

        Parameters
        ----------
        model_name : str
            Name of the HuggingFace model.

        Returns
        -------
        HuggingFaceEmbeddings
            Loaded embedding model.

        Raises
        ------
        RuntimeError
            If the model cannot be loaded.
        """

        logger.info("Loading embedding model: %s", model_name)

        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    "device": "cpu",
                },
                encode_kwargs={
                    "normalize_embeddings": False,
                },
            )

            logger.info(
                "Successfully loaded embedding model: %s",
                model_name,
            )

            return embedding_model

        except Exception as exc:
            logger.exception(
                "Failed to load embedding model: %s",
                model_name,
            )

            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'."
            ) from exc

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        """
        Return the loaded embedding model.

        Returns
        -------
        HuggingFaceEmbeddings
            Loaded embedding model instance.
        """

        return self._embedding_model