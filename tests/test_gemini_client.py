"""
tests/test_gemini_client.py
=============================================================================
Focused regression tests for the Gemini reliability fix:
- retries on google.genai.errors.ServerError (e.g. HTTP 503), same as the
  pre-existing httpx.RemoteProtocolError retry behavior;
- raises GeminiUnavailableError (a RuntimeError subclass) only when every
  retry attempt is exhausted for a transient failure;
- permanent errors (e.g. an invalid-request ClientError) are NOT retried
  and are NOT turned into GeminiUnavailableError;
- a normal successful call is completely unaffected.

GeminiClient.__init__ talks to real infrastructure (loads a .env API key,
configures the real google-genai SDK client), so these tests construct an
instance without calling __init__ and inject a fake `_client`/`_model`,
consistent with GeminiClient's own attributes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.gemini_client import GeminiClient, GeminiUnavailableError


def _make_client_without_init() -> GeminiClient:
    """A GeminiClient instance with _client/_model set directly, bypassing
    __init__'s real API-key loading and SDK configuration."""
    client = object.__new__(GeminiClient)
    client._api_key = "test-key"
    client._client = SimpleNamespace(models=SimpleNamespace())
    client._model = "gemini-2.5-flash"
    return client


def _server_error(message: str = "high demand") -> genai_errors.ServerError:
    return genai_errors.ServerError(503, {"error": {"message": message, "status": "UNAVAILABLE"}})


def _client_error(message: str = "invalid api key") -> genai_errors.ClientError:
    return genai_errors.ClientError(401, {"error": {"message": message, "status": "UNAUTHENTICATED"}})


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.candidates = [object()]
        self.text = text


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retries call time.sleep(2) - patch it so tests run instantly."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


# =============================================================================
# Normal success is unaffected
# =============================================================================


def test_successful_response_is_returned_unchanged():
    client = _make_client_without_init()
    calls = {"count": 0}

    def _generate_content(model, contents):
        calls["count"] += 1
        return _FakeGeminiResponse("The Cardiology department is on the 2nd floor.")

    client._client.models.generate_content = _generate_content

    result = client.generate_response("Where is Cardiology?")

    assert result == "The Cardiology department is on the 2nd floor."
    assert calls["count"] == 1


# =============================================================================
# ServerError (503) retry behavior
# =============================================================================


def test_retries_on_server_error_then_succeeds():
    client = _make_client_without_init()
    calls = {"count": 0}

    def _generate_content(model, contents):
        calls["count"] += 1
        if calls["count"] < 3:
            raise _server_error()
        return _FakeGeminiResponse("Recovered answer.")

    client._client.models.generate_content = _generate_content

    result = client.generate_response("A question.")

    assert result == "Recovered answer."
    assert calls["count"] == 3


def test_raises_gemini_unavailable_error_after_exhausting_retries_on_server_error():
    client = _make_client_without_init()
    calls = {"count": 0}

    def _generate_content(model, contents):
        calls["count"] += 1
        raise _server_error()

    client._client.models.generate_content = _generate_content

    with pytest.raises(GeminiUnavailableError):
        client.generate_response("A question.")

    assert calls["count"] == 3  # bounded: exactly max_attempts, not unlimited


def test_gemini_unavailable_error_is_a_runtime_error_subclass():
    # Any existing code that already catches bare RuntimeError must
    # continue to work unchanged.
    assert issubclass(GeminiUnavailableError, RuntimeError)


# =============================================================================
# Pre-existing httpx.RemoteProtocolError retry behavior is preserved
# =============================================================================


def test_retries_on_remote_protocol_error_then_succeeds_unchanged():
    client = _make_client_without_init()
    calls = {"count": 0}

    def _generate_content(model, contents):
        calls["count"] += 1
        if calls["count"] < 2:
            raise httpx.RemoteProtocolError("connection dropped")
        return _FakeGeminiResponse("Recovered after network drop.")

    client._client.models.generate_content = _generate_content

    result = client.generate_response("A question.")

    assert result == "Recovered after network drop."
    assert calls["count"] == 2


def test_remote_protocol_error_exhaustion_now_raises_gemini_unavailable_error():
    client = _make_client_without_init()

    def _generate_content(model, contents):
        raise httpx.RemoteProtocolError("connection dropped")

    client._client.models.generate_content = _generate_content

    with pytest.raises(GeminiUnavailableError):
        client.generate_response("A question.")


# =============================================================================
# Permanent errors are NOT retried and NOT treated as "unavailable"
# =============================================================================


def test_client_error_is_raised_immediately_without_retry():
    client = _make_client_without_init()
    calls = {"count": 0}

    def _generate_content(model, contents):
        calls["count"] += 1
        raise _client_error()

    client._client.models.generate_content = _generate_content

    with pytest.raises(RuntimeError) as exc_info:
        client.generate_response("A question.")

    assert calls["count"] == 1  # no retry for a permanent/client error
    assert not isinstance(exc_info.value, GeminiUnavailableError)


def test_invalid_prompt_still_raises_value_error_immediately():
    client = _make_client_without_init()
    with pytest.raises(ValueError):
        client.generate_response("   ")


def test_unrecognized_exception_type_is_raised_immediately_without_retry():
    client = _make_client_without_init()
    calls = {"count": 0}

    def _generate_content(model, contents):
        calls["count"] += 1
        raise KeyError("unexpected SDK error")

    client._client.models.generate_content = _generate_content

    with pytest.raises(RuntimeError) as exc_info:
        client.generate_response("A question.")

    assert calls["count"] == 1
    assert not isinstance(exc_info.value, GeminiUnavailableError)