"""
ui/metrics.py
=============================================================================
Right-column "AI Insights" panel for the Intelligent Hospital Information
Assistant frontend.

This module renders — and ONLY renders — a healthcare-focused AI insights
panel: assistant/module status, response performance, response-source
insights, and the assistant's available AI modules.

It does NOT:
    - Call Gemini or any LLM
    - Access ChromaDB or any vector store
    - Access the Retriever
    - Calculate metrics, confidence scores, or coverage percentages
    - Read or write Streamlit session state
    - Perform any other business logic

Every value shown here is supplied by the caller as a plain function
argument (typically forwarded from `app.py`, which itself gets them from
the `RAGPipeline`). This keeps the module a pure, testable, reusable
presentation layer — exactly like `ui/sidebar.py`.

-----------------------------------------------------------------------------
Design note (healthcare AI dashboard redesign)
-----------------------------------------------------------------------------
This panel intentionally no longer exposes developer-analytics content
(knowledge-base document counts, token counts, embedding/chunk
configuration, similarity scores, coverage percentages, pipeline/model
identifiers, or system uptime). Those numbers meant little to a
non-technical hospital-assistant user and have been removed entirely —
both their UI and any helper function that existed only to support them.
The panel now reads as a compact status/insights summary for an AI
healthcare product: "Assistant Status", "Performance", "Response
Insights", and "AI Modules".

-----------------------------------------------------------------------------
Backward compatibility note
-----------------------------------------------------------------------------
`app.py` (unmodified, per the redesign brief) calls this module exactly
as follows:

    render_metrics(
        response_time_ms=st.session_state.response_time,
        retrieval_time_ms=st.session_state.retrieval_time,
        confidence_score=st.session_state.confidence_score,
        pipeline_status="online" if st.session_state.backend_initialized else "offline",
    )

`render_metrics()` keeps this exact name and keeps all four of those
keyword arguments, with compatible types and defaults, so that call
continues to work unmodified. `confidence_score` is still accepted (so
the call above never raises), but — per the redesign brief's explicit
"Do NOT display Confidence" instruction — its value is not rendered
anywhere in the new layout.

Every other removed section's parameters (`total_documents`, `doctors`,
`prompt_tokens`, `coverage_*`, `chunk_size`, `system_uptime`, ...) have
been deleted along with the sections that displayed them, since nothing
in `app.py` ever passed them and keeping dozens of dead parameters would
contradict the brief's "avoid clutter" requirement. A handful of new,
fully optional keyword arguments (`gemini_status`,
`knowledge_base_status`, `retrieved_sources`, `chunks_retrieved`,
`primary_source`, `sources_used`, `document_types`) were added to
support the new sections; none of them break the existing call above,
and each renders "Not Available" (never a fabricated `0` or `—`) when
`app.py` doesn't supply it.

-----------------------------------------------------------------------------
Layout note
-----------------------------------------------------------------------------
`ui/layout.py` builds a three-column skeleton (`LayoutColumns`) whose right
column (`columns.insights`) is reserved for this module. `app.py` is
expected to call `render_metrics()` inside `with columns.insights:`, e.g.:

    from ui.layout import render_layout
    from ui.sidebar import render_sidebar
    from ui.metrics import render_metrics

    columns = render_layout(online=True)
    with columns.sidebar:
        render_sidebar(...)
    with columns.chat:
        ...  # ui/chat.py
    with columns.insights:
        render_metrics(
            response_time_ms=980,
            retrieval_time_ms=142,
            confidence_score=0.91,
            pipeline_status="online",
        )

-----------------------------------------------------------------------------
Public API
-----------------------------------------------------------------------------
    render_metrics_header()        -> None
    render_assistant_status()       -> None
    render_performance()             -> None
    render_response_insights()        -> None
    render_ai_modules()                -> None
    render_metrics()                    -> None   (orchestrator; call from app.py)
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import streamlit as st

from ui.components import render_divider, render_info_panel, render_metric_card, render_section_header

# =============================================================================
# TYPE ALIASES
# =============================================================================
# Mirrors `ui.components.StatusKind` — kept as a local alias (rather than
# an import) since it is a typing-only construct, not a reusable function
# or a design token, matching the convention already established in
# `ui/sidebar.py` and `ui/chat.py`.

StatusKind = Literal["online", "offline", "warning", "processing", "error"]
StatValue = Union[int, str]

#: Text shown in place of a genuinely missing value. Never a fabricated
#: `0` or an em-dash — an explicit, honest "we don't have this yet".
NOT_AVAILABLE: str = "Not Available"

#: Text shown for a feature that is planned but not yet implemented.
COMING_SOON: str = "\U0001F6A7 Coming Soon"

#: Text shown for a feature that is implemented and usable today.
AVAILABLE: str = "\u2705 Available"


# =============================================================================
# CONSTANTS
# =============================================================================


@dataclass(frozen=True)
class MetricCardSpec:
    """
    A single metric card's display data, used internally to render groups
    of metric cards through one shared grid helper instead of duplicating
    `render_metric_card` calls throughout this module.

    Attributes:
        title: Metric label (e.g. "Retrieval Time").
        value: Pre-formatted display value (e.g. "142 ms").
        icon: Optional single emoji/glyph shown above the label.
        status: Optional accent color — "success", "warning", or "error".
            `None` renders the card with no special accent.
    """

    title: str
    value: StatValue
    icon: Optional[str] = None
    status: Optional[Literal["success", "warning", "error"]] = None


#: The high-level AI services summarized in the dashboard.
_AI_SERVICE_ROWS: tuple[tuple[str, str], ...] = (
    ("Gemini", "\u2728"),
    ("Retriever", "\U0001F9E0"),
    ("Vector Store", "\U0001F4DA"),
    ("Voice Input", "\U0001F3A4"),
    ("Voice Output", "\U0001F50A"),
)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _render_metric_grid(cards: Sequence[MetricCardSpec], num_columns: int = 2) -> None:
    """
    Render a sequence of metric cards laid out in an evenly-spaced grid.

    Centralizing this loop avoids repeating the "create N st.columns and
    cycle through them" pattern across the "Assistant Status",
    "Performance", "Response Insights", and "AI Modules" sections, which
    all need the same grid behavior.

    Args:
        cards: Metric card specifications to render, in display order.
        num_columns: Number of columns in the grid (defaults to 2, which
            fits comfortably in the narrower right-hand insights column).

    Returns:
        None. Renders each card via `render_metric_card`.
    """
    columns = st.columns(num_columns)
    for index, card in enumerate(cards):
        with columns[index % num_columns]:
            render_metric_card(
                title=card.title,
                value=str(card.value),
                icon=card.icon,
                status=card.status,
            )


def _format_ms(value: Optional[Union[int, float]]) -> str:
    """
    Format a millisecond duration for display, tolerating `None`.

    Args:
        value: Duration in milliseconds, or `None` if not yet available.

    Returns:
        A string like "142 ms", or `NOT_AVAILABLE` when `value` is
        `None`.
    """
    if value is None:
        return NOT_AVAILABLE
    return f"{value:.0f} ms" if isinstance(value, float) else f"{value} ms"


def _format_count(value: Optional[StatValue]) -> str:
    """
    Format a plain count for display, tolerating `None` or an
    already-formatted string.

    Args:
        value: A count, a pre-formatted string, or `None`.

    Returns:
        The string representation, or `NOT_AVAILABLE` when `value` is
        `None`.
    """
    if value is None:
        return NOT_AVAILABLE
    return str(value)


def _format_text(value: Optional[str]) -> str:
    """
    Format a free-text value for display, tolerating `None` or blank text.

    Args:
        value: The text to display, or `None`/empty if unavailable.

    Returns:
        The stripped text, or `NOT_AVAILABLE` when `value` is `None` or
        blank.
    """
    if value is None or not value.strip():
        return NOT_AVAILABLE
    return value.strip()


def _format_status(value: bool) -> str:
    """Format a boolean feature state as Enabled or Disabled."""
    return "Enabled" if value else "Disabled"


def _format_confidence_score(score: Optional[float]) -> str:
    """Format a confidence score as a percentage when available."""
    if score is None:
        return NOT_AVAILABLE
    normalized = score * 100 if score <= 1 else score
    return f"{normalized:.0f}%"


def _format_response_length(value: Optional[StatValue]) -> str:
    """Format the latest assistant response length for display."""
    if value is None:
        return NOT_AVAILABLE
    return f"{value} chars"


def _format_document_types(document_types: Optional[Sequence[str]]) -> str:
    """
    Format a list of retrieved-document types for display.

    Args:
        document_types: Distinct document/source types behind the most
            recent response (e.g. ``["Doctor Profile", "FAQ"]``), or
            `None`/empty if unavailable.

    Returns:
        A comma-separated string of the distinct, non-empty types, or
        `NOT_AVAILABLE` when nothing was supplied.
    """
    if not document_types:
        return NOT_AVAILABLE

    distinct_types = [doc_type.strip() for doc_type in document_types if doc_type and doc_type.strip()]
    if not distinct_types:
        return NOT_AVAILABLE

    # Preserve order while removing duplicates.
    seen: set[str] = set()
    ordered_unique_types = []
    for doc_type in distinct_types:
        if doc_type not in seen:
            seen.add(doc_type)
            ordered_unique_types.append(doc_type)

    return ", ".join(ordered_unique_types)


def _status_emoji_label(status: StatusKind) -> tuple[str, str]:
    """
    Map a `StatusKind` to its display emoji and short label.

    Args:
        status: One of "online", "offline", "warning", "processing",
            "error".

    Returns:
        A `(emoji, label)` tuple — `("\U0001F7E2", "Online")` for
        "online"; `("\U0001F7E1", "Initializing")` for "warning" or
        "processing"; `("\U0001F534", "Offline")` for anything else
        (including "offline" and "error").
    """
    if status == "online":
        return "\U0001F7E2", "Online"
    if status in ("warning", "processing"):
        return "\U0001F7E1", "Initializing"
    return "\U0001F534", "Offline"


def _status_card_accent(status: StatusKind) -> Optional[Literal["success", "warning", "error"]]:
    """
    Map a `StatusKind` to the metric-card accent color that matches
    `_status_emoji_label`'s emoji for the same status.

    Args:
        status: One of "online", "offline", "warning", "processing",
            "error".

    Returns:
        "success" for "online", "warning" for "warning"/"processing",
        "error" for anything else.
    """
    if status == "online":
        return "success"
    if status in ("warning", "processing"):
        return "warning"
    return "error"


def _render_section_label(label: str) -> None:
    """
    Render a small bold section label above a group of cards.

    Kept as a thin wrapper (rather than repeating the same `st.markdown`
    call in every section function) so every section's label uses
    identical spacing/typography.

    Args:
        label: The section label text (e.g. "Performance").

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown(
        f'<p style="font-weight:600; margin-bottom:0.5rem;">{label}</p>',
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 1 — PANEL HEADER
# =============================================================================


def render_metrics_header() -> None:
    """
    Render the AI Insights panel header.

    Displays the panel title ("AI Insights") and a subtitle describing
    the assistant, via the shared `render_section_header` component.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    render_section_header(
        title="AI Insights",
        subtitle="Your AI healthcare assistant, at a glance",
        icon="\U0001F3E5",
    )


# =============================================================================
# SECTION 2 — ASSISTANT STATUS
# =============================================================================


def render_ai_services(
    pipeline_status: StatusKind = "offline",
    voice_input_enabled: bool = True,
    voice_output_enabled: bool = True,
    gemini_status: Optional[StatusKind] = None,
    vector_store_status: Optional[StatusKind] = None,
) -> None:
    """Render the compact AI services overview section."""
    _render_section_label("AI Services")

    resolved_gemini_status = gemini_status if gemini_status is not None else pipeline_status
    resolved_vector_store_status = vector_store_status if vector_store_status is not None else pipeline_status

    status_by_component: dict[str, str] = {
        "Gemini": _status_emoji_label(resolved_gemini_status)[1],
        "Retriever": _status_emoji_label(pipeline_status)[1],
        "Vector Store": _status_emoji_label(resolved_vector_store_status)[1],
        "Voice Input": _format_status(voice_input_enabled),
        "Voice Output": _format_status(voice_output_enabled),
    }

    cards: list[MetricCardSpec] = []
    for component_label, icon in _AI_SERVICE_ROWS:
        value = status_by_component[component_label]
        accent: Optional[Literal["success", "warning", "error"]] = None
        if value == "Online" or value == "Enabled":
            accent = "success"
        elif value == "Initializing":
            accent = "warning"
        else:
            accent = "error"
        cards.append(MetricCardSpec(title=component_label, value=value, icon=icon, status=accent))

    _render_metric_grid(cards, num_columns=2)


# =============================================================================
# SECTION 3 — PERFORMANCE
# =============================================================================


def render_performance(
    response_time_ms: Optional[Union[int, float]] = None,
    retrieval_time_ms: Optional[Union[int, float]] = None,
    retrieved_sources: Optional[StatValue] = None,
    chunks_retrieved: Optional[StatValue] = None,
    response_length: Optional[StatValue] = None,
) -> None:
    """
    Render the "Performance" section as a grid of metric cards.

    Deliberately limited to the values a user actually benefits from
    seeing: response time, retrieval time, retrieved sources, retrieved
    chunks, and the latest response length. Token counts and other
    development metrics are intentionally not shown here. This function
    only displays values already computed by the backend `RAGPipeline`
    or derived in `app.py` — it never measures timing or counts anything
    itself.

    Args:
        response_time_ms: Time spent generating the response, in
            milliseconds, or `None` if unavailable.
        retrieval_time_ms: Time spent on retrieval, in milliseconds, or
            `None` if unavailable.
        retrieved_sources: Number of distinct source documents
            retrieved, or `None` if unavailable.
        chunks_retrieved: Number of text chunks retrieved from the
            vector store, or `None` if unavailable.
        response_length: Length of the latest assistant response, in
            characters, or `None` if unavailable.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _render_section_label("Performance")
    cards = (
        MetricCardSpec(title="Response Time", value=_format_ms(response_time_ms), icon="\u26A1"),
        MetricCardSpec(title="Retrieval Time", value=_format_ms(retrieval_time_ms), icon="\u23F1\uFE0F"),
        MetricCardSpec(title="Retrieved Sources", value=_format_count(retrieved_sources), icon="\U0001F4DA"),
        MetricCardSpec(title="Retrieved Chunks", value=_format_count(chunks_retrieved), icon="\U0001F9E9"),
        MetricCardSpec(title="Response Length", value=_format_response_length(response_length), icon="\u270D\uFE0F"),
    )
    _render_metric_grid(cards, num_columns=2)


# =============================================================================
# SECTION 4 — RESPONSE INSIGHTS
# =============================================================================


def render_retrieval(
    primary_source: Optional[str] = None,
    document_types: Optional[Sequence[str]] = None,
    ranking_method: Optional[str] = None,
    confidence_score: Optional[float] = None,
) -> None:
    """
    Render the "Retrieval" section as a grid of metric cards.

    Summarizes what the most recent answer was grounded in — the top
    source, document type(s), ranking method, and confidence — without
    exposing raw retrieval internals. This function only displays
    values already computed by the backend or derived in `app.py`.

    Args:
        primary_source: The single most relevant retrieved source for
            the most recent answer (e.g. a document name or title), or
            `None` if unavailable.
        document_types: Distinct document/source types behind the most
            recent answer, or `None` if unavailable.
        ranking_method: Human-readable ranking/similarity method name,
            or `None` if unavailable.
        confidence_score: Confidence score from the latest answer, or
            `None` if unavailable.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _render_section_label("Retrieval")
    cards = (
        MetricCardSpec(title="Primary Source", value=_format_text(primary_source), icon="\U0001F4CC"),
        MetricCardSpec(
            title="Document Type",
            value=_format_document_types(document_types),
            icon="\U0001F4C4",
        ),
        MetricCardSpec(title="Ranking Method", value=_format_text(ranking_method), icon="\U0001F4CB"),
        MetricCardSpec(title="Confidence", value=_format_confidence_score(confidence_score), icon="\U0001F3AF"),
    )
    _render_metric_grid(cards, num_columns=2)


def render_technical_details(
    pipeline_status: StatusKind = "offline",
    gemini_status: Optional[StatusKind] = None,
    vector_store_status: Optional[StatusKind] = None,
    confidence_score: Optional[float] = None,
) -> None:
    """Render low-level technical metrics inside an expander."""
    resolved_gemini_status = gemini_status if gemini_status is not None else pipeline_status
    resolved_vector_store_status = vector_store_status if vector_store_status is not None else pipeline_status

    with st.expander("Technical Details", expanded=False):
        _render_section_label("Technical Details")
        cards = (
            MetricCardSpec(title="Pipeline Status", value=_status_emoji_label(pipeline_status)[1], icon="\U0001F9E0", status=_status_card_accent(pipeline_status)),
            MetricCardSpec(title="Gemini Status", value=_status_emoji_label(resolved_gemini_status)[1], icon="\u2728", status=_status_card_accent(resolved_gemini_status)),
            MetricCardSpec(title="Vector Store Status", value=_status_emoji_label(resolved_vector_store_status)[1], icon="\U0001F4DA", status=_status_card_accent(resolved_vector_store_status)),
            MetricCardSpec(title="Confidence", value=_format_confidence_score(confidence_score), icon="\U0001F3AF"),
        )
        _render_metric_grid(cards, num_columns=2)


# =============================================================================
# SECTION 5 — AI MODULES
# =============================================================================


def render_prescription_analysis() -> None:
    """Render the display-only prescription analysis section."""
    _render_section_label("Prescription Analysis")
    render_info_panel(
        title=COMING_SOON,
        message="This section will later display OCR metrics.",
        variant="warning",
    )


# =============================================================================
# ORCHESTRATOR
# =============================================================================


def render_metrics(
    response_time_ms: Optional[Union[int, float]] = None,
    retrieval_time_ms: Optional[Union[int, float]] = None,
    confidence_score: Optional[float] = None,
    pipeline_status: StatusKind = "offline",
    voice_input_enabled: bool = True,
    voice_output_enabled: bool = True,
    gemini_status: Optional[StatusKind] = None,
    vector_store_status: Optional[StatusKind] = None,
    retrieved_sources: Optional[StatValue] = None,
    chunks_retrieved: Optional[StatValue] = None,
    response_length: Optional[StatValue] = None,
    primary_source: Optional[str] = None,
    document_types: Optional[Sequence[str]] = None,
    ranking_method: Optional[str] = None,
) -> None:
    """
    Assemble the complete "AI Insights" panel in one call.

    Convenience entry point for `app.py`: renders the panel header,
    AI Services, Performance, Retrieval, and Prescription Analysis — in
    order, separated by dividers.

    Every argument is a plain value forwarded to the corresponding
    section function below; see each section function's docstring for
    details on an individual parameter. This orchestrator performs no
    computation of its own.

    `confidence_score` is accepted (and never validated/used) solely so
    `app.py`'s existing call — which passes it as a keyword argument —
    keeps working unmodified; the redesigned panel does not display a
    confidence value anywhere, per the redesign brief.

    Typical usage in `app.py` (this is `app.py`'s actual, unmodified
    call site):

        from ui.layout import render_layout
        from ui.metrics import render_metrics

        columns = render_layout(online=True)
        with columns.insights:
            render_metrics(
                response_time_ms=st.session_state.response_time,
                retrieval_time_ms=st.session_state.retrieval_time,
                confidence_score=st.session_state.confidence_score,
                pipeline_status="online" if st.session_state.backend_initialized else "offline",
                voice_input_enabled=True,
                voice_output_enabled=True,
            )

    Returns:
        None. Renders directly into the Streamlit app.
    """
    render_metrics_header()
    render_divider()

    render_ai_services(
        pipeline_status=pipeline_status,
        voice_input_enabled=voice_input_enabled,
        voice_output_enabled=voice_output_enabled,
        gemini_status=gemini_status,
        vector_store_status=vector_store_status,
    )
    render_divider()

    render_performance(
        response_time_ms=response_time_ms,
        retrieval_time_ms=retrieval_time_ms,
        retrieved_sources=retrieved_sources,
        chunks_retrieved=chunks_retrieved,
        response_length=response_length,
    )
    render_divider()

    render_retrieval(
        primary_source=primary_source,
        document_types=document_types,
        ranking_method=ranking_method,
        confidence_score=confidence_score,
    )
    render_divider()

    render_prescription_analysis()
    render_technical_details(
        pipeline_status=pipeline_status,
        gemini_status=gemini_status,
        vector_store_status=vector_store_status,
        confidence_score=confidence_score,
    )