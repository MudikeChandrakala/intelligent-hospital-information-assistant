"""
modules/voice_assistant.py
=============================================================================
Speech-to-text helper for the Intelligent Hospital Information Assistant.

This module provides a single, small, reusable class — `VoiceAssistant` —
whose only job is turning a burst of microphone audio into recognized
text using Google's speech recognition API via the `SpeechRecognition`
package.

`VoiceAssistant` deliberately knows NOTHING about:
    - Gemini
    - The RAG pipeline
    - LangChain
    - ChromaDB
    - Chat history / conversation state
    - Streamlit itself (no `st.*` calls anywhere in this file)

It is a pure speech-to-text utility. The UI layer (`app.py`) is
responsible for deciding what to do with the recognized text (e.g.
feeding it into the exact same prompt-submission flow a typed message
already goes through) — this module never makes that decision itself.

-----------------------------------------------------------------------------
Responsibilities
-----------------------------------------------------------------------------
    1. Start recording  -> `record_audio()`
    2. Convert speech to text -> `transcribe_audio()`
    3. Return recognized text -> both return small, typed result objects;
       `listen_and_transcribe()` is a convenience method combining steps
       1 and 2 for callers that don't need the two-stage split.

Every failure mode (missing microphone, denied permission, timeout, no
speech detected, network failure, recognition failure, or the
`SpeechRecognition` package not being installed at all) is caught and
converted into a friendly `status_message` on the returned result object
— this module never raises out of `record_audio()`, `transcribe_audio()`,
or `listen_and_transcribe()`, so it can never crash the calling app.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from io import BytesIO
from dataclasses import dataclass
from typing import Optional

try:
    from gtts import gTTS
except ImportError:  # pragma: no cover - handled gracefully by `text_to_speech()`
    gTTS = None  # type: ignore[assignment]

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover - exercised only when the optional
    # dependency is missing; handled gracefully by `is_available()` below
    # rather than failing the whole application at import time.
    sr = None  # type: ignore[assignment]

logger = logging.getLogger("hospital_assistant.voice_assistant")

# =============================================================================
# CONSTANTS
# =============================================================================

#: How many seconds to wait for speech to *begin* before giving up.
DEFAULT_TIMEOUT_SECONDS: float = 5.0

#: Maximum length, in seconds, of a single recorded phrase once speech
#: has begun (prevents an indefinitely long recording).
DEFAULT_PHRASE_TIME_LIMIT_SECONDS: float = 15.0

#: How long to sample ambient noise before listening, so
#: `Recognizer.energy_threshold` adapts to the room instead of using a
#: fixed default that may be too sensitive or not sensitive enough.
DEFAULT_AMBIENT_NOISE_DURATION_SECONDS: float = 0.5

#: BCP-47 language tag passed to Google's speech recognition API.
DEFAULT_LANGUAGE: str = "en-US"
#: Language used for Text-to-Speech
DEFAULT_TTS_LANGUAGE = "en"

#: Speech speed
DEFAULT_TTS_SLOW = False

# --- User-friendly status messages ------------------------------------------
_MESSAGE_UNAVAILABLE_PACKAGE: str = (
    "\U0001F399\uFE0F Voice input is unavailable: the 'SpeechRecognition' "
    "package is not installed."
)
_MESSAGE_MICROPHONE_UNAVAILABLE: str = (
    "\U0001F399\uFE0F Microphone unavailable. Please check that a "
    "microphone is connected and that this app has permission to use it."
)
_MESSAGE_TIMEOUT: str = "\u23F1\uFE0F Speech timeout. No speech was detected — please try again."
_MESSAGE_NOT_UNDERSTOOD: str = "\u2753 Could not recognize speech. Please try again."
_MESSAGE_NETWORK_ERROR: str = (
    "\U0001F310 Network error while recognizing speech. Please check your connection and try again."
)
_MESSAGE_RECOGNITION_FAILED: str = "\u26A0\uFE0F Speech recognition failed. Please try again."
_MESSAGE_SUCCESS: str = "\u2705 Voice recognized successfully."
_MESSAGE_EMPTY_RECORDING: str = "\u2753 No speech detected. Please try again."


# =============================================================================
# RESULT MODELS
# =============================================================================


@dataclass(frozen=True)
class AudioCaptureResult:
    """
    Result of attempting to record audio from the microphone.

    Attributes:
        success: Whether audio was captured successfully.
        audio: The captured audio data (an `sr.AudioData` instance) if
            `success` is True, otherwise `None`. Typed as `object` here
            (rather than importing `sr.AudioData` unconditionally) so
            this module still imports cleanly when `speech_recognition`
            itself is not installed.
        status_message: A short, user-friendly message describing the
            outcome — safe to display directly in the UI.
    """

    success: bool
    audio: Optional[object]
    status_message: str


@dataclass(frozen=True)
class VoiceRecognitionResult:
    """
    Result of a full voice-input attempt (recording plus transcription).

    Attributes:
        success: Whether recognizable text was produced.
        text: The recognized, stripped text if `success` is True,
            otherwise `None`.
        status_message: A short, user-friendly message describing the
            outcome — safe to display directly in the UI, e.g. "Voice
            recognized successfully." or "Could not recognize speech."
    """

    success: bool
    text: Optional[str]
    status_message: str


# =============================================================================
# VOICE ASSISTANT
# =============================================================================


class VoiceAssistant:
    """
    Small, reusable speech-to-text helper built on the `SpeechRecognition`
    package and Google's speech recognition API.

    Responsibilities are intentionally narrow: start recording, convert
    speech to text, and return the recognized text. This class knows
    nothing about Gemini, the RAG pipeline, LangChain, ChromaDB, or chat
    history — callers decide what the recognized text means.

    Every public method catches its own failure modes (missing
    microphone, denied permission, timeout, no speech detected, network
    failure, recognition failure, missing optional dependency) and
    reports them via a result object's `status_message` rather than
    raising, so a caller never needs to wrap calls to this class in its
    own try/except to stay crash-free.
    """

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        phrase_time_limit_seconds: float = DEFAULT_PHRASE_TIME_LIMIT_SECONDS,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        """
        Construct a `VoiceAssistant`.

        Args:
            timeout_seconds: Seconds to wait for speech to begin before
                giving up (maps to `sr.WaitTimeoutError` on expiry).
            phrase_time_limit_seconds: Maximum length of a single
                recorded phrase once speech has begun.
            language: BCP-47 language tag passed to Google's speech
                recognition API (e.g. "en-US").
        """
        self._timeout_seconds = timeout_seconds
        self._phrase_time_limit_seconds = phrase_time_limit_seconds
        self._language = language
        self._recognizer = sr.Recognizer() if sr is not None else None

    def is_available(self) -> bool:
        """
        Check whether voice input can be used at all.

        Returns:
            True if the `SpeechRecognition` package is installed, False
            otherwise. This does not check for microphone hardware —
            that is only discoverable when `record_audio()` actually
            tries to open one.
        """
        return sr is not None

    def record_audio(self) -> AudioCaptureResult:
        """
        Record a single phrase from the system's default microphone.

        Adjusts briefly for ambient noise before listening so the
        recognizer's energy threshold adapts to the current room.
        Blocks the calling thread until speech is captured, the phrase
        time limit is reached, or `timeout_seconds` elapses with no
        speech detected.

        Returns:
            An `AudioCaptureResult`. On failure, `success` is False,
            `audio` is `None`, and `status_message` describes what went
            wrong (missing package, no microphone / permission denied,
            or a timeout with no speech detected).
        """
        if not self.is_available():
            logger.warning("record_audio() called but 'speech_recognition' is not installed.")
            return AudioCaptureResult(success=False, audio=None, status_message=_MESSAGE_UNAVAILABLE_PACKAGE)

        try:
            with sr.Microphone() as source:
                self._recognizer.adjust_for_ambient_noise(
                    source, duration=DEFAULT_AMBIENT_NOISE_DURATION_SECONDS
                )
                audio = self._recognizer.listen(
                    source,
                    timeout=self._timeout_seconds,
                    phrase_time_limit=self._phrase_time_limit_seconds,
                )
        except sr.WaitTimeoutError:
            logger.info("Voice input timed out waiting for speech to begin.")
            return AudioCaptureResult(success=False, audio=None, status_message=_MESSAGE_TIMEOUT)
        except (OSError, AttributeError) as exc:
            # `sr.Microphone()` raises OSError when no input device is
            # available or the OS denies access to it, and AttributeError
            # when PyAudio (its own dependency) is not installed.
            logger.error("Microphone unavailable: %s", exc)
            return AudioCaptureResult(success=False, audio=None, status_message=_MESSAGE_MICROPHONE_UNAVAILABLE)
        except Exception as exc:  # noqa: BLE001 - last-resort guard, never let this crash the app
            logger.error("Unexpected error while recording audio: %s", exc)
            return AudioCaptureResult(success=False, audio=None, status_message=_MESSAGE_RECOGNITION_FAILED)

        logger.debug("Audio captured successfully.")
        return AudioCaptureResult(success=True, audio=audio, status_message="Audio captured.")
    def transcribe_audio_bytes(
        self, audio_bytes: bytes
    ) -> VoiceRecognitionResult:
        """
        Convert browser-recorded WAV bytes into SpeechRecognition AudioData
        and reuse the existing transcription method.
        """
        if not self.is_available():
            logger.warning(
                "transcribe_audio_bytes() called but "
                "'speech_recognition' is not installed."
            )
            return VoiceRecognitionResult(
                success=False,
                text=None,
                status_message=_MESSAGE_UNAVAILABLE_PACKAGE,
            )

        if not audio_bytes:
            logger.warning("No audio bytes were received.")
            return VoiceRecognitionResult(
                success=False,
                text=None,
                status_message=_MESSAGE_EMPTY_RECORDING,
            )

        try:
            audio_buffer = BytesIO(audio_bytes)

            with sr.AudioFile(audio_buffer) as source:
                audio = self._recognizer.record(source)

            return self.transcribe_audio(audio)

        except Exception as exc:
            logger.error(
                "Failed to process browser-recorded audio: %s",
                exc,
            )
            return VoiceRecognitionResult(
                success=False,
                text=None,
                status_message=_MESSAGE_RECOGNITION_FAILED,
            )    
    def transcribe_audio(self, audio: object) -> VoiceRecognitionResult:
        """
        Convert previously captured audio into text via Google's speech
        recognition API.

        Args:
            audio: An `sr.AudioData` instance, typically the `audio`
                field of a successful `AudioCaptureResult`.

        Returns:
            A `VoiceRecognitionResult`. On failure, `success` is False,
            `text` is `None`, and `status_message` describes what went
            wrong (unrecognizable speech, a network/API failure, or an
            unexpected recognition error).
        """
        if not self.is_available():
            logger.warning("transcribe_audio() called but 'speech_recognition' is not installed.")
            return VoiceRecognitionResult(success=False, text=None, status_message=_MESSAGE_UNAVAILABLE_PACKAGE)

        try:
            recognized_text = self._recognizer.recognize_google(audio, language=self._language)
        except sr.UnknownValueError:
            logger.info("Speech was not intelligible.")
            return VoiceRecognitionResult(success=False, text=None, status_message=_MESSAGE_NOT_UNDERSTOOD)
        except sr.RequestError as exc:
            logger.error("Speech recognition request failed: %s", exc)
            return VoiceRecognitionResult(success=False, text=None, status_message=_MESSAGE_NETWORK_ERROR)
        except Exception as exc:  # noqa: BLE001 - last-resort guard, never let this crash the app
            logger.error("Unexpected error during speech recognition: %s", exc)
            return VoiceRecognitionResult(success=False, text=None, status_message=_MESSAGE_RECOGNITION_FAILED)

        cleaned_text = (recognized_text or "").strip()
        if not cleaned_text:
            return VoiceRecognitionResult(success=False, text=None, status_message=_MESSAGE_EMPTY_RECORDING)

        logger.info("Voice input recognized successfully (%d characters).", len(cleaned_text))
        return VoiceRecognitionResult(success=True, text=cleaned_text, status_message=_MESSAGE_SUCCESS)

    def listen_and_transcribe(self) -> VoiceRecognitionResult:
        """
        Convenience method: record a phrase and transcribe it in one call.

        Equivalent to calling `record_audio()` followed by
        `transcribe_audio()`, for callers that don't need to show a
        distinct "listening" vs. "processing" status between the two
        stages (the caller in `app.py` uses the two-stage methods
        directly instead, precisely so it can show that distinction).

        Returns:
            A `VoiceRecognitionResult`. If recording itself fails, its
            `status_message` is propagated unchanged and `text` is
            `None`.
        """
        capture_result = self.record_audio()
        if not capture_result.success:
            return VoiceRecognitionResult(success=False, text=None, status_message=capture_result.status_message)

        return self.transcribe_audio(capture_result.audio)

    def text_to_speech(self, text: str) -> Optional[bytes]:
        """
        Convert text into MP3 audio using gTTS.

        Returns:
            MP3 bytes if successful, otherwise None.
        """
        logger.info("Entering text_to_speech()")

        if gTTS is None:
            logger.warning("text_to_speech() called but 'gTTS' is not installed.")
            return None

        cleaned_text = (text or "").strip()
        if not cleaned_text:
            logger.debug("text_to_speech() called with empty text.")
            return None

        try:
            audio_buffer = BytesIO()
            logger.info("Initialized in-memory audio buffer.")

            tts = gTTS(
                text=cleaned_text,
                lang=DEFAULT_TTS_LANGUAGE,
                slow=DEFAULT_TTS_SLOW,
            )
            logger.info("gTTS initialized successfully.")

            logger.info("Writing MP3 data to buffer.")
            tts.write_to_fp(audio_buffer)
            buffer_size = audio_buffer.tell()
            logger.info("MP3 write complete; buffer position=%d bytes.", buffer_size)

            audio_buffer.seek(0)
            logger.info("Audio buffer rewound to start.")

            audio_bytes = audio_buffer.read()
            logger.info("Read MP3 bytes from buffer (byte_length=%d).", len(audio_bytes))

            if not audio_bytes:
                logger.warning("text_to_speech() produced an empty MP3 payload.")
                return None

            logger.info("Generated speech successfully (%d characters, %d bytes).", len(cleaned_text), len(audio_bytes))
            return audio_bytes
        except Exception as exc:  # noqa: BLE001 - never let TTS failures crash the app
            logger.error("Text-to-speech failed: %s", exc)
            return None