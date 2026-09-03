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

{conversation_history}

Instructions:

- Use ONLY the provided hospital context.
- Combine information from multiple retrieved documents whenever appropriate.
- Do NOT use outside knowledge.
- Do NOT make assumptions.
- Do NOT hallucinate.
- If the hospital context contains an explicit symptom-to-department mapping or directly answers the department recommendation, use that information directly.
- For a direct department recommendation, do NOT ask for additional symptom details such as severity, duration, associated symptoms, or age unless the hospital context explicitly requires that information to determine the department.
- When the user says "I have", "I need", or otherwise describes their own symptom or condition, treat the user as the patient unless another person is explicitly identified.
- For the user's own symptom, use the hospital's explicit symptom-to-department mapping when available.
- Do not request the user's age merely to identify a department unless the hospital context explicitly requires age for that recommendation.
- When the user asks which doctor they should consult for a symptom, first identify the recommended department from the symptom-to-department mapping, then provide the available doctors from that department using the retrieved hospital context.
- If the retrieved hospital context contains doctor names, qualifications, specializations, experience, consultation timings, fees, or locations for the recommended department, include the relevant available doctor information in the answer.
- Do not stop after identifying the department when the user explicitly asks for a doctor.
- For department recommendations involving a family member, ask for the patient's age before recommending a department when the age is not provided.
- Do not assume that terms such as "daughter", "son", "mother", or "father" indicate a particular age.
- If the user provides the requested age in a follow-up message, use the conversation context to preserve the original symptom, condition, and recommendation request.
- Do not treat a short follow-up such as an age ("43", "14", etc.) as a standalone hospital question.
- After receiving the age, answer the original question using ONLY the hospital context.
- If the hospital context contains a clear department recommendation, provide that recommendation even when optional clinical details are not available.
- If emergency instructions are relevant to the user's question, provide them in addition to the department recommendation rather than replacing the department answer.
- Treat the "Current User Question" as the resolved version of the user's request when it contains information carried forward from the conversation.
- Conversation history is provided only to resolve references, follow-up answers, and missing context. It is not a source of hospital facts.
- When the current question identifies the patient's age, use that age when interpreting retrieved hospital symptom and department information.
- For a child or adolescent patient, prefer a retrieved hospital record that explicitly associates the symptom/condition with Pediatrics when such a record is available.
- If multiple retrieved records provide different department recommendations for the same symptom, select the recommendation that best matches the patient's documented age/group and the most specific matching hospital record.
- Do not reject an answer merely because another retrieved document gives a more general department recommendation.
- If the answer is not available in the provided context, respond exactly:


"I couldn't find that information in the hospital knowledge base."

Provide clear, concise and helpful responses.

==================================================
Hospital Context
==================================================

{context}

==================================================
Current User Question
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

    def build_prompt(self, question: str, documents: list[Document], conversation_history: list = None) -> str:
        """
        Build a structured prompt from a question and retrieved documents.

        Args:
            question: The user's question.
            documents: The list of `Document` objects retrieved for the
                question.
            conversation_history: Optional list of previous conversation messages
                to provide context. If provided, recent messages will be included
                in the prompt to help Gemini understand follow-up questions.

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
            prompt = self._create_prompt(cleaned_question, context, conversation_history=conversation_history)
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

    def _create_prompt(self, question: str, context: str, conversation_history: list = None) -> str:
        """
        Combine the formatted context and question into the final prompt.

        Args:
            question: The cleaned user question.
            context: The formatted hospital context block.
            conversation_history: Optional list of conversation messages to format
                into the prompt for context.

        Returns:
            The final structured prompt string.
        """

        formatted_history = self._format_conversation_history(conversation_history)

        return PROMPT_TEMPLATE.format(
            conversation_history=formatted_history,
            context=context,
            question=question
        )

    def _format_conversation_history(self, conversation_history: list = None) -> str:
        """
        Format conversation history into a readable section for the prompt.

        If conversation_history is empty or None, returns an empty string
        (which the template will handle gracefully).

        Args:
            conversation_history: List of ChatMessage objects from the conversation.

        Returns:
            Formatted conversation history string, or empty string if no history.
        """

        if not conversation_history:
            return ""

        try:
            history_lines = []

            # Keep only recent messages (e.g., last 10 turns) to avoid excessive context
            recent_messages = conversation_history[-10:]

            for message in recent_messages:
                # Handle both dict-like and object-like message formats
                role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
                content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)

                if role and content:
                    # Format as "User: ..." or "Assistant: ..."
                    formatted_role = role.capitalize()
                    history_lines.append(f"{formatted_role}: {content}")

            if history_lines:
                history_section = "CONVERSATION HISTORY\n" + "=" * 60 + "\n"
                history_section += "\n".join(history_lines)
                history_section += "\n" + "=" * 60 + "\n"
                return history_section

            return ""

        except Exception as exc:
            logger.warning("Failed to format conversation history: %s", exc)
            return ""