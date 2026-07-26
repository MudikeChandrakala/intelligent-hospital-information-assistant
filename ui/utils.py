"""
ui/utils.py
=============================================================================
Shared utility library for the Intelligent Hospital Information Assistant
frontend.

This module provides small, PURE, side-effect-free helper functions used
across `ui/chat.py`, `ui/sidebar.py`, `ui/metrics.py`, `ui/components.py`,
and `app.py`: timestamp/duration/number formatting, input validation,
status/document/confidence display mappings, and unit-conversion helpers.

This module explicitly does NOT:
    - Render any UI (no `st.markdown`, no `st.button`, no Streamlit import
      of any kind)
    - Call Gemini or any LLM
    - Access ChromaDB or any vector store
    - Call the Retriever
    - Access the RAG pipeline
    - Read or write Streamlit session state
    - Perform any business logic (routing, decision-making about what a
      button click means, etc.)

Every function here takes plain values in and returns plain values out —
strings, floats, ints, bools, tuples. Nothing in this module has ever
seen a Streamlit `DeltaGenerator`. This makes it trivially unit-testable
and safe to import from anywhere in the frontend (or a future backend
adapter layer) without pulling in any rendering or session-state
dependency.

-----------------------------------------------------------------------------
Relationship to `ui.styles` / `ui.components`
-----------------------------------------------------------------------------
Where a display-mapping helper here needs an actual color value (e.g.
`get_status_color`), it imports the relevant token constants from the
locked `ui.styles.Colors` class rather than hard-coding hex values, so a
future re-theme of `ui/styles.py` automatically propagates here too.
`ui.components` already has its own *internal*, private color-mapping
helpers for building HTML — the functions in this module are a public,
Streamlit-free equivalent for any other part of the frontend (or
`app.py`) that needs the same classification/color logic without needing
to render anything.

-----------------------------------------------------------------------------
Public API (grouped)
-----------------------------------------------------------------------------
Formatting:
    format_timestamp, format_duration, format_percentage,
    format_confidence, format_similarity_score, format_token_count,
    format_file_size, format_number

Validation:
    is_valid_message, is_valid_confidence, is_valid_percentage,
    is_non_empty, truncate_text, safe_strip

Display / mapping:
    get_status_color, get_status_icon, get_document_icon,
    get_message_icon, get_confidence_level

Conversion:
    ms_to_seconds, seconds_to_readable, confidence_to_percentage,
    score_to_string

Shared constants:
    DEFAULT_CONFIDENCE, DEFAULT_STATUS, DEFAULT_PLACEHOLDER,
    DEFAULT_EXAMPLE_QUESTIONS, SUPPORTED_DOCUMENT_TYPES,
    SUPPORTED_STATUSES, CONFIDENCE_LEVELS
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Tuple, Union

from ui.styles import Colors

# =============================================================================
# TYPE ALIASES
# =============================================================================

StatusKind = Literal["online", "offline", "warning", "processing", "error"]
ConfidenceLevel = Literal["Low", "Medium", "High", "Excellent"]
MessageRole = Literal["user", "assistant"]
DocumentType = Literal[
    "Doctor",
    "Department",
    "Disease",
    "Medicine",
    "FAQ",
    "Appointment",
    "Insurance",
    "Emergency Protocol",
    "Patient Guideline",
    "Hospital Information",
]


# =============================================================================
# SECTION 1 — SHARED CONSTANTS
# =============================================================================
# Centralized default values and lookup tables used throughout the
# frontend, so `app.py` and every `ui/*.py` module agree on the same
# defaults without redefining them locally.

#: Default confidence value used when no score is available yet.
DEFAULT_CONFIDENCE: float = 0.0

#: Default status used when a component's real status is not yet known.
DEFAULT_STATUS: StatusKind = "offline"

#: Default placeholder text for the chat input box.
DEFAULT_PLACEHOLDER: str = "Ask about hospital services..."

#: Default example questions shown on the chat welcome screen.
DEFAULT_EXAMPLE_QUESTIONS: Tuple[str, ...] = (
    "Which doctor should I consult for chest pain?",
    "What are the visiting hours?",
    "Where is Cardiology?",
    "How can I book an appointment?",
)

#: The ten knowledge-base document categories recognized by the frontend.
SUPPORTED_DOCUMENT_TYPES: Tuple[DocumentType, ...] = (
    "Doctor",
    "Department",
    "Disease",
    "Medicine",
    "FAQ",
    "Appointment",
    "Insurance",
    "Emergency Protocol",
    "Patient Guideline",
    "Hospital Information",
)

#: The five system/component statuses recognized by the frontend.
SUPPORTED_STATUSES: Tuple[StatusKind, ...] = (
    "online",
    "offline",
    "processing",
    "warning",
    "error",
)

#: The four confidence bands used to classify RAG retrieval/answer scores.
CONFIDENCE_LEVELS: Tuple[ConfidenceLevel, ...] = ("Low", "Medium", "High", "Excellent")


# =============================================================================
# SECTION 2 — MAPPING TABLES
# =============================================================================
# Private lookup dictionaries backing the display-helper functions in
# Section 4. Kept as module-level constants (rather than rebuilt inside
# each function call) so they are defined exactly once.

# Status -> a (Colors token, emoji) pair.
_STATUS_DISPLAY_MAP: dict[StatusKind, Tuple[str, str]] = {
    "online": (Colors.SUCCESS, "\u2705"),        # ✅
    "offline": (Colors.TEXT_MUTED, "\u26AB"),     # ⚫
    "processing": (Colors.INFO, "\u23F3"),        # ⏳
    "warning": (Colors.WARNING, "\u26A0\uFE0F"),  # ⚠️
    "error": (Colors.ERROR, "\u274C"),            # ❌
}

# Document type -> emoji.
_DOCUMENT_ICON_MAP: dict[str, str] = {
    "Doctor": "\U0001FA7A",               # 🩺
    "Department": "\U0001F3E2",           # 🏢
    "Disease": "\U0001FA7B",              # 🩻
    "Medicine": "\U0001F48A",             # 💊
    "FAQ": "\u2753",                      # ❓
    "Appointment": "\U0001F4C5",          # 📅
    "Insurance": "\U0001F4C4",            # 📄
    "Emergency Protocol": "\U0001F6A8",   # 🚨
    "Patient Guideline": "\U0001F4D8",    # 📘
    "Hospital Information": "\U0001F3E5",  # 🏥
}

# Message role -> emoji.
_MESSAGE_ICON_MAP: dict[MessageRole, str] = {
    "user": "\U0001F464",       # 👤
    "assistant": "\U0001F916",  # 🤖
}

# Confidence level -> (lower_bound_inclusive, Colors token).
# Bands: Low < 0.50 <= Medium < 0.70 <= High < 0.90 <= Excellent.
_CONFIDENCE_BANDS: Tuple[Tuple[float, ConfidenceLevel, str], ...] = (
    (0.90, "Excellent", Colors.SUCCESS),
    (0.70, "High", Colors.INFO),
    (0.50, "Medium", Colors.WARNING),
    (0.0, "Low", Colors.ERROR),
)

# Fallback used by every mapping helper when given an unrecognized key,
# so callers always get a sensible default instead of a KeyError.
_UNKNOWN_ICON: str = "\u2022"  # •
_UNKNOWN_COLOR: str = Colors.TEXT_MUTED


# =============================================================================
# SECTION 3 — FORMATTING HELPERS
# =============================================================================


def format_timestamp(timestamp: Union[str, datetime, None], fmt: str = "%I:%M %p") -> str:
    """
    Format a timestamp value into a short, human-readable string.

    Args:
        timestamp: A `datetime` object, an already-formatted string
            (returned as-is), or `None`.
        fmt: `strftime`-style format string used when `timestamp` is a
            `datetime`. Defaults to 12-hour clock with AM/PM
            (e.g. "10:42 AM").

    Returns:
        The formatted timestamp string, or an empty string if
        `timestamp` is `None` or empty.
    """
    if not timestamp:
        return ""
    if isinstance(timestamp, datetime):
        return timestamp.strftime(fmt).lstrip("0")
    return str(timestamp)


def format_duration(milliseconds: Optional[Union[int, float]]) -> str:
    """
    Format a millisecond duration as a compact, human-readable string.

    Args:
        milliseconds: Duration in milliseconds, or `None`.

    Returns:
        A string like "480ms" (values under 1000ms), "1.2s" (values
        under 60s), or "1m 05s" (values at or above 60s). Returns an
        em-dash placeholder ("\u2014") if `milliseconds` is `None`.
    """
    if milliseconds is None:
        return "\u2014"

    if milliseconds < 1000:
        return f"{milliseconds:.0f}ms"

    total_seconds = milliseconds / 1000.0
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"

    minutes, seconds = divmod(int(round(total_seconds)), 60)
    return f"{minutes}m {seconds:02d}s"


def format_percentage(value: Optional[float], decimals: int = 0) -> str:
    """
    Format a 0.0-1.0 fraction as a percentage string.

    Args:
        value: Fraction in the 0.0-1.0 range, or `None`.
        decimals: Number of decimal places to show (default 0).

    Returns:
        A string like "91%" (or "91.3%" with `decimals=1`), or an
        em-dash placeholder if `value` is `None`.
    """
    if value is None:
        return "\u2014"
    clamped = max(0.0, min(1.0, value))
    return f"{clamped * 100:.{decimals}f}%"


def format_confidence(score: Optional[float]) -> str:
    """
    Format a confidence score as a combined percentage + level label.

    Args:
        score: Confidence score in the 0.0-1.0 range, or `None`.

    Returns:
        A string like "91% (Excellent)", or an em-dash placeholder if
        `score` is `None`.
    """
    if score is None:
        return "\u2014"
    level, _ = get_confidence_level(score)
    return f"{format_percentage(score)} ({level})"


def format_similarity_score(score: Optional[float], decimals: int = 2) -> str:
    """
    Format a raw similarity score (e.g. cosine similarity) as a fixed-
    precision decimal string.

    Args:
        score: Similarity score, typically in the 0.0-1.0 range, or
            `None`.
        decimals: Number of decimal places to show (default 2).

    Returns:
        A string like "0.87", or an em-dash placeholder if `score` is
        `None`.
    """
    if score is None:
        return "\u2014"
    return f"{score:.{decimals}f}"


def format_token_count(count: Optional[int]) -> str:
    """
    Format a token count for compact display, abbreviating large values.

    Args:
        count: Number of tokens, or `None`.

    Returns:
        A string like "512", "1.2K" (for >= 1,000), or "3.4M"
        (for >= 1,000,000). Returns an em-dash placeholder if `count`
        is `None`.
    """
    if count is None:
        return "\u2014"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def format_file_size(size_bytes: Optional[Union[int, float]]) -> str:
    """
    Format a byte count as a human-readable file size.

    Args:
        size_bytes: Size in bytes, or `None`.

    Returns:
        A string like "482 B", "12.4 KB", "3.1 MB", or "1.2 GB".
        Returns an em-dash placeholder if `size_bytes` is `None`.
    """
    if size_bytes is None:
        return "\u2014"

    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"  # pragma: no cover — defensive fallback


def format_number(value: Optional[Union[int, float]], decimals: int = 0) -> str:
    """
    Format a number with thousands separators.

    Args:
        value: The number to format, or `None`.
        decimals: Number of decimal places to show (default 0, i.e.
            integers).

    Returns:
        A string like "1,556" or "1,556.40", or an em-dash placeholder
        if `value` is `None`.
    """
    if value is None:
        return "\u2014"
    return f"{value:,.{decimals}f}"


# =============================================================================
# SECTION 4 — VALIDATION HELPERS
# =============================================================================


def safe_strip(value: Optional[str]) -> str:
    """
    Strip surrounding whitespace from a string, tolerating `None`.

    Args:
        value: The string to strip, or `None`.

    Returns:
        The stripped string, or an empty string if `value` is `None`.
    """
    return value.strip() if isinstance(value, str) else ""


def is_non_empty(value: Optional[str]) -> bool:
    """
    Check whether a string has non-whitespace content.

    Args:
        value: The string to check, or `None`.

    Returns:
        True if `value` is a string containing at least one
        non-whitespace character, False otherwise.
    """
    return bool(safe_strip(value))


def is_valid_message(text: Optional[str], max_length: int = 2000) -> bool:
    """
    Validate that a chat message is non-empty and within a reasonable
    length limit.

    Args:
        text: The candidate message text, or `None`.
        max_length: Maximum allowed character length (default 2000).

    Returns:
        True if `text` is non-empty (after stripping) and no longer than
        `max_length` characters, False otherwise.
    """
    stripped = safe_strip(text)
    return bool(stripped) and len(stripped) <= max_length


def is_valid_confidence(score: Optional[float]) -> bool:
    """
    Validate that a confidence score is a real number in the 0.0-1.0
    range.

    Args:
        score: The candidate confidence score, or `None`.

    Returns:
        True if `score` is a number in [0.0, 1.0], False otherwise
        (including when `score` is `None`).
    """
    if score is None or isinstance(score, bool):
        return False
    if not isinstance(score, (int, float)):
        return False
    return 0.0 <= float(score) <= 1.0


def is_valid_percentage(value: Optional[float]) -> bool:
    """
    Validate that a value is a real number in the 0-100 range.

    Args:
        value: The candidate percentage value, or `None`.

    Returns:
        True if `value` is a number in [0, 100], False otherwise
        (including when `value` is `None`).
    """
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return 0.0 <= float(value) <= 100.0


def truncate_text(text: Optional[str], max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length, appending a suffix if cut.

    Args:
        text: The text to truncate, or `None`.
        max_length: Maximum length of the returned string, including the
            suffix (default 100).
        suffix: Text appended when truncation occurs (default "...").

    Returns:
        The original text if it already fits within `max_length`,
        otherwise a truncated version ending in `suffix`. Returns an
        empty string if `text` is `None`.
    """
    stripped = safe_strip(text)
    if not stripped:
        return ""
    if len(stripped) <= max_length:
        return stripped

    cutoff = max(0, max_length - len(suffix))
    return stripped[:cutoff].rstrip() + suffix


# =============================================================================
# SECTION 5 — DISPLAY / MAPPING HELPERS
# =============================================================================


def get_status_color(status: Optional[str]) -> str:
    """
    Map a status keyword to its representative design-token color.

    Args:
        status: One of "online", "offline", "warning", "processing",
            "error", or an unrecognized value.

    Returns:
        A color token from `ui.styles.Colors`. Unrecognized statuses
        return the neutral muted-text color.
    """
    color, _ = _STATUS_DISPLAY_MAP.get(status, (_UNKNOWN_COLOR, _UNKNOWN_ICON))
    return color


def get_status_icon(status: Optional[str]) -> str:
    """
    Map a status keyword to a representative emoji.

    Args:
        status: One of "online", "offline", "warning", "processing",
            "error", or an unrecognized value.

    Returns:
        A single emoji glyph. Unrecognized statuses return a generic
        bullet ("\u2022").
    """
    _, icon = _STATUS_DISPLAY_MAP.get(status, (_UNKNOWN_COLOR, _UNKNOWN_ICON))
    return icon


def get_document_icon(document_type: Optional[str]) -> str:
    """
    Map a knowledge-base document type to a representative emoji.

    Args:
        document_type: One of the values in `SUPPORTED_DOCUMENT_TYPES`
            (e.g. "Doctor", "FAQ", "Emergency Protocol"), or an
            unrecognized value.

    Returns:
        A single emoji glyph. Unrecognized document types return a
        generic document glyph ("\U0001F4C4").
    """
    return _DOCUMENT_ICON_MAP.get(document_type or "", "\U0001F4C4")


def get_message_icon(role: Optional[str]) -> str:
    """
    Map a chat message role to a representative emoji.

    Args:
        role: "user" or "assistant", or an unrecognized value.

    Returns:
        A single emoji glyph. Unrecognized roles return a generic
        bullet ("\u2022").
    """
    return _MESSAGE_ICON_MAP.get(role, _UNKNOWN_ICON)


def get_confidence_level(score: Optional[float]) -> Tuple[ConfidenceLevel, str]:
    """
    Classify a 0.0-1.0 confidence/retrieval score into a human-readable
    band and its representative color.

    Bands:
        score < 0.50            -> "Low"        (error color)
        0.50 <= score < 0.70    -> "Medium"     (warning color)
        0.70 <= score < 0.90    -> "High"        (info color)
        score >= 0.90            -> "Excellent"  (success color)

    Args:
        score: Confidence score, expected in the 0.0-1.0 range. `None`
            and out-of-range values are treated as 0.0 ("Low").

    Returns:
        A 2-tuple of (level label, color token).
    """
    clamped = max(0.0, min(1.0, score)) if score is not None else 0.0
    for lower_bound, level, color in _CONFIDENCE_BANDS:
        if clamped >= lower_bound:
            return level, color
    return "Low", Colors.ERROR  # pragma: no cover — unreachable safety net


# =============================================================================
# SECTION 6 — CONVERSION HELPERS
# =============================================================================


def ms_to_seconds(milliseconds: Optional[Union[int, float]]) -> float:
    """
    Convert a millisecond duration to seconds.

    Args:
        milliseconds: Duration in milliseconds, or `None`.

    Returns:
        The equivalent duration in seconds as a float. Returns `0.0` if
        `milliseconds` is `None`.
    """
    if milliseconds is None:
        return 0.0
    return milliseconds / 1000.0


def seconds_to_readable(seconds: Optional[Union[int, float]]) -> str:
    """
    Convert a duration in seconds to a compact, human-readable string.

    Args:
        seconds: Duration in seconds, or `None`.

    Returns:
        A string like "42s", "3m 05s", or "1h 02m", scaling the unit to
        the magnitude of the input. Returns an em-dash placeholder
        ("\u2014") if `seconds` is `None`.
    """
    if seconds is None:
        return "\u2014"

    total_seconds = int(round(seconds))
    if total_seconds < 60:
        return f"{total_seconds}s"

    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m"


def confidence_to_percentage(score: Optional[float]) -> float:
    """
    Convert a 0.0-1.0 confidence score to a 0-100 percentage value.

    Args:
        score: Confidence score in the 0.0-1.0 range, or `None`.

    Returns:
        The equivalent percentage as a float in [0, 100]. Returns `0.0`
        if `score` is `None`. Out-of-range inputs are clamped.
    """
    if score is None:
        return 0.0
    return max(0.0, min(1.0, score)) * 100.0


def score_to_string(score: Optional[float], as_percentage: bool = True, decimals: int = 0) -> str:
    """
    Convert a numeric score into a display-ready string, as either a
    percentage or a fixed-precision decimal.

    Args:
        score: The score to convert, or `None`.
        as_percentage: If True (default), formats as a percentage via
            `format_percentage`. If False, formats as a fixed-precision
            decimal via `format_similarity_score`.
        decimals: Number of decimal places to show.

    Returns:
        A formatted string, or an em-dash placeholder if `score` is
        `None`.
    """
    if as_percentage:
        return format_percentage(score, decimals=decimals)
    return format_similarity_score(score, decimals=decimals if decimals else 2)