"""
gemini_client.py

Sends a fully constructed prompt to Google Gemini and returns the
generated response text.

This module has a single responsibility: accept a prompt, send it to
Google Gemini via the google-genai SDK, and return the generated
response.

This module DOES NOT:
- Retrieve documents
- Build prompts
- Use ChromaDB
- Access the knowledge base
- Handle Streamlit
"""

from __future__ import annotations

import logging
import os
import time
import httpx
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash"
API_KEY_ENV_VAR = "GOOGLE_API_KEY"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class GeminiUnavailableError(RuntimeError):
    """
    Raised by ``generate_response()`` only when Gemini's service itself
    was temporarily unavailable (e.g. HTTP 503 / ``google.genai.errors.
    ServerError``) and every bounded retry attempt was exhausted.

    This is a ``RuntimeError`` subclass, so any existing code that
    already catches ``RuntimeError`` continues to work unchanged.  It
    exists only so a caller such as ``RAGPipeline`` can distinguish "the
    service is temporarily down, a deterministic fallback may be
    reasonable" from every other failure (invalid prompt, invalid API
    key, or any other error), which must continue to fail immediately.
    """


class GeminiClient:
    """
    Sends prompts to Google Gemini and returns generated responses.

    This class is responsible for loading the Google API key, configuring
    the google-genai SDK, loading the generative model, and
    generating a response for a given prompt. It does not retrieve
    documents, build prompts, or perform any other part of the RAG
    pipeline.

    Attributes
    ----------
    _api_key : str
        The Google API key used to authenticate with Gemini.

    _client : Any
        The configured google-genai Client instance used to
        communicate with the Gemini API.

    _model : Any
        The generative model identifier used to generate responses.
    """

    def __init__(self) -> None:
        """
        Initialize the Gemini client.

        Loads the Google API key from the environment, configures the
        google-genai SDK, and loads the generative model.

        Raises:
            ValueError: If the API key is missing or empty.
            RuntimeError: If the SDK cannot be configured or the model
                cannot be loaded.
        """

        self._api_key: str = self._load_api_key()
        self._client: Any = self._configure_genai()
        self._model: Any = self._load_model()

        logger.info("GeminiClient initialized successfully with model '%s'.", MODEL_NAME)

    # -----------------------------------------------------------------
    # Internal Helper Methods
    # -----------------------------------------------------------------

    def _load_api_key(self) -> str:
        """
        Load the Google API key from a .env file.

        Returns:
            The loaded API key.

        Raises:
            ValueError: If the API key is missing or empty.
        """

        load_dotenv(dotenv_path=ENV_FILE_PATH)
        api_key = os.getenv(API_KEY_ENV_VAR)

        if not isinstance(api_key, str) or not api_key.strip():
            logger.error("No API key found in environment variable '%s'.", API_KEY_ENV_VAR)
            raise ValueError(f"{API_KEY_ENV_VAR} is not set or is empty.")

        logger.info("API key loaded successfully.")
        return api_key.strip()

    def _configure_genai(self) -> Any:
        """
        Configure the google-genai SDK with the loaded API key.

        Returns:
            The configured google-genai Client instance.

        Raises:
            RuntimeError: If the SDK cannot be configured.
        """

        try:
            client = genai.Client(api_key=self._api_key)
        except Exception as exc:
            logger.exception("Failed to configure the google-genai SDK.")
            raise RuntimeError(f"Failed to configure Gemini SDK: {exc}") from exc

        logger.info("Gemini SDK configured successfully.")
        return client

    def _load_model(self) -> Any:
        """
        Load the configured Gemini generative model.

        Returns:
            The generative model identifier used for content generation.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """

        try:
            if not isinstance(MODEL_NAME, str) or not MODEL_NAME.strip():
                raise ValueError("MODEL_NAME is not set or is empty.")
            model = MODEL_NAME
        except Exception as exc:
            logger.exception("Failed to load Gemini model '%s'.", MODEL_NAME)
            raise RuntimeError(f"Failed to load Gemini model: {exc}") from exc

        logger.info("Gemini model '%s' loaded successfully.", MODEL_NAME)
        return model

    def _validate_prompt(self, prompt: str) -> None:
        """
        Validate that the supplied prompt is a non-empty string.

        Args:
            prompt: The prompt to validate.

        Raises:
            ValueError: If ``prompt`` is empty, contains only whitespace,
                or is not a string.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            logger.error("Invalid prompt provided to GeminiClient.")
            raise ValueError("prompt cannot be empty.")

        logger.debug("Prompt validated successfully.")

    def _extract_response_text(self, response: Any) -> str:
        """
        Extract the generated text from a Gemini SDK response object.

        Args:
            response: The raw response object returned by the Gemini
                SDK.

        Returns:
            The extracted response text.

        Raises:
            RuntimeError: If no text can be extracted from the response.
        """

        try:
            candidates = getattr(response, "candidates", None)

            if not candidates:
                logger.error("Gemini response did not contain any candidates.")
                raise RuntimeError("Gemini response did not contain any candidates.")

            text = response.text
        except RuntimeError:
            raise
        except Exception as exc:
            logger.exception("Failed to extract text from Gemini response.")
            raise RuntimeError(f"Failed to extract response text: {exc}") from exc

        if not isinstance(text, str) or not text.strip():
            logger.error("Gemini response did not contain usable text.")
            raise RuntimeError("Gemini response did not contain usable text.")

        return text.strip()

    # -----------------------------------------------------------------
    # Public APIs
    # -----------------------------------------------------------------

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from Gemini for the given prompt.

        Automatically retries on transient failures that indicate a
        temporary problem reaching or receiving a response from Gemini,
        rather than a problem with the request itself:
          - ``httpx.RemoteProtocolError`` (connection dropped before a
            response was received), and
          - ``google.genai.errors.ServerError`` (Gemini's own 5xx
            responses, e.g. HTTP 503 "model is currently experiencing
            high demand").

        Up to ``max_attempts`` calls are made, waiting
        ``retry_delay_seconds`` between attempts. If every attempt is
        exhausted, ``GeminiUnavailableError`` is raised so a caller can
        choose to fall back instead of failing outright. Any other
        exception (e.g. an invalid API key or invalid prompt, both of
        which are permanent, not transient) is raised immediately,
        unchanged from before.

        Args:
            prompt: The fully constructed prompt to send to Gemini.

        Returns:
            The generated response text.

        Raises:
            ValueError: If ``prompt`` fails validation.
            GeminiUnavailableError: If every retry attempt for a
                transient failure is exhausted.
            RuntimeError: If response generation fails for any other
                reason.
        """

        self._validate_prompt(prompt)
        cleaned_prompt = prompt.strip()

        logger.info("Sending prompt to Gemini model '%s'.", MODEL_NAME)

        max_attempts = 3
        retry_delay_seconds = 2
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=cleaned_prompt,
                )
                break
            except (httpx.RemoteProtocolError, genai_errors.ServerError) as exc:
                last_error = exc
                if attempt < max_attempts:
                    logger.warning(
                        "Transient error on attempt %d/%d while calling "
                        "Gemini (%s). Retrying in %d second(s)...",
                        attempt,
                        max_attempts,
                        exc,
                        retry_delay_seconds,
                    )
                    time.sleep(retry_delay_seconds)
                else:
                    logger.error(
                        "All %d attempt(s) to call Gemini failed due to a "
                        "transient error: %s",
                        max_attempts,
                        exc,
                    )
            except Exception as exc:
                logger.exception("Failed to generate response from Gemini.")
                raise RuntimeError(f"Failed to generate response: {exc}") from exc
        else:
            raise GeminiUnavailableError(
                f"Gemini was still unavailable after {max_attempts} attempts "
                f"due to a transient error: {last_error}"
            ) from last_error

        response_text = self._extract_response_text(response)

        logger.info("Response generated successfully.")
        return response_text