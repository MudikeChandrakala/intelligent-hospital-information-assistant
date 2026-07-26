"""
prompt_builder.py

Converts a user's question and retrieved LangChain Document objects into
a single structured prompt for Google Gemini.

This module has a single responsibility: build a well-formed RAG prompt
from a question and a list of already-retrieved documents.

This module DOES NOT:
- Retrieve documents
- Call Google Gemini
- Access Chroma
- Create embeddings
- Manage chat history
- Handle Streamlit
- Perform any vector search
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DOCUMENT_HEADER = "=" * 60

PROMPT_TEMPLATE = """You are an AI-powered Hospital Information Assistant.

Your job is to answer ONLY using the hospital information provided below.

Instructions:

- Use ONLY the provided hospital context.
- Combine information from multiple retrieved documents whenever appropriate.
- Do NOT use outside knowledge.
- Do NOT make assumptions.
- Do NOT hallucinate.
- If the answer is not available in the provided context, respond exactly:

"I couldn't find that information in the hospital knowledge base."

Provide clear, concise and helpful responses.

==================================================
Hospital Context
==================================================

{context}

==================================================
User Question
==================================================

{question}

==================================================
Answer
==================================================
"""


class PromptBuilder:
    """
    Builds a structured RAG prompt from a question and retrieved documents.

    This class is responsible for validating a user's question and a list
    of retrieved `Document` objects, formatting the documents into a
    readable context block, and combining the question and context into a
    single prompt string ready to be sent to Google Gemini. It does not
    retrieve documents, call Gemini, or perform any other part of the RAG
    pipeline.
    """

    # -----------------------------------------------------------------
    # Public APIs
    # -----------------------------------------------------------------

    def build_prompt(self, question: str, documents: list[Document]) -> str:
        """
        Build a structured prompt from a question and retrieved documents.

        Args:
            question: The user's question.
            documents: The list of `Document` objects retrieved for the
                question.

        Returns:
            A single structured prompt string ready to be sent to
            Google Gemini.

        Raises:
            ValueError: If ``question`` or ``documents`` fail validation.
            RuntimeError: If prompt construction fails unexpectedly.
        """

        self._validate_question(question)
        self._validate_documents(documents)

        cleaned_question = question.strip()

        logger.info(
            "Building prompt using %d retrieved document(s).",
            len(documents),
        )

        try:
            context = self._format_context(documents)
            prompt = self._create_prompt(cleaned_question, context)
        except Exception as exc:
            logger.exception("Failed to build prompt.")
            raise RuntimeError(f"Failed to build prompt: {exc}") from exc

        logger.info("Prompt built successfully.")
        return prompt

    # -----------------------------------------------------------------
    # Internal Helper Methods
    # -----------------------------------------------------------------

    def _validate_question(self, question: str) -> None:
        """
        Validate that the supplied question is a non-empty string.

        Args:
            question: The user question to validate.

        Raises:
            ValueError: If ``question`` is empty, contains only
                whitespace, or is not a string.
        """

        if not isinstance(question, str) or not question.strip():
            logger.error("Invalid question provided to PromptBuilder.")
            raise ValueError("question cannot be empty.")

        logger.debug("Question validated successfully.")

    def _validate_documents(self, documents: list[Document]) -> None:
        """
        Validate that the supplied documents are a non-empty list of
        `Document` objects.

        Args:
            documents: The retrieved documents to validate.

        Raises:
            ValueError: If ``documents`` is not a list, is empty, or
                contains an item that is not a `Document` instance.
        """

        if not isinstance(documents, list) or not documents:
            logger.error("Invalid documents provided to PromptBuilder.")
            raise ValueError("documents cannot be empty.")

        for index, document in enumerate(documents):
            if not isinstance(document, Document):
                logger.error(
                    "Document at index %d is not a Document instance (got %s).",
                    index,
                    type(document).__name__,
                )
                raise ValueError("documents must contain only Document objects.")

        logger.debug("Documents validated successfully.")

    def _format_context(self, documents: list[Document]) -> str:
        """
        Format retrieved documents into a single readable context block.

        Each document contributes only its ``page_content``. Documents
        are separated by labeled section boundaries.

        Args:
            documents: The retrieved documents to format.

        Returns:
            A single string containing the formatted context.
        """

        sections: list[str] = []

        for index, document in enumerate(documents, start=1):
            section = (
                f"{DOCUMENT_HEADER}\n"
                f"Document {index}\n"
                f"{DOCUMENT_HEADER}\n"
                f"{document.page_content}\n"
                f"{DOCUMENT_HEADER}"
            )
            sections.append(section)

        return "\n\n".join(sections)

    def _create_prompt(self, question: str, context: str) -> str:
        """
        Combine the formatted context and question into the final prompt.

        Args:
            question: The cleaned user question.
            context: The formatted hospital context block.

        Returns:
            The final structured prompt string.
        """

        return PROMPT_TEMPLATE.format(context=context, question=question)