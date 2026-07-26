"""
ui/metrics.py
=============================================================================
Right-column "AI Insights" analytics panel for the Intelligent Hospital
Information Assistant frontend.

This module renders — and ONLY renders — the AI Insights panel: knowledge
base overview, retrieval metrics, response metrics, system health, knowledge
coverage, confidence summary, retrieved-sources summary, performance
summary, pipeline overview, and last-update information.

It does NOT:
    - Call Gemini or any LLM
    - Access ChromaDB or any vector store
    - Access the Retriever
    - Calculate metrics, confidence scores, or coverage percentages
    - Read or write Streamlit session state
    - Perform any other business logic

Every number, status, and label shown here is supplied by the caller as a
plain function argument (typically forwarded from `app.py`, which itself
gets them from the `RAGPipeline`). This keeps the module a pure, testable,
reusable presentation layer — exactly like `ui/sidebar.py`.

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
            total_documents=1556,
            doctors=24, departments=8, diseases=140,
            medicines=310, appointments=57,
            retrieved_sources=4, retrieval_confidence=0.91,
            average_similarity_score=0.87, retrieval_time_ms=142,
            chunks_retrieved=6,
            response_time_ms=980, prompt_tokens=512,
            completion_tokens=180, total_tokens=692,
            response_length=640,
            gemini_status="online", retriever_status="online",
            vector_store_status="online", embedding_model_status="online",
            pipeline_status="online",
            performance_message="Average response time below target. "
                                "High retrieval confidence. Knowledge "
                                "base healthy.",
        )

-----------------------------------------------------------------------------
Public API
-----------------------------------------------------------------------------
    render_metrics_header()       -> None
    render_kb_overview()           -> None
    render_retrieval_metrics()     -> None
    render_response_metrics()      -> None
    render_system_health()          -> None
    render_knowledge_coverage()     -> None
    render_confidence_summary()     -> None
    render_sources_summary()        -> None
    render_performance_summary()    -> None
    render_pipeline_summary()       -> None
    render_last_update()             -> None
    render_metrics()                  -> None   (orchestrator; call from app.py)
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Union

import streamlit as st

from ui.components import (
    render_confidence_badge,
    render_divider,
    render_info_panel,
    render_key_value,
    render_metric_card,
    render_progress,
    render_section_header,
    render_status_badge,
    render_tag,
)

# =============================================================================
# TYPE ALIASES
# =============================================================================
# Mirrors `ui.components.StatusKind` / `TrendDirection` — kept as local
# aliases (rather than imports) since they are typing-only constructs, not
# reusable functions or design tokens. This keeps this module's imports
# limited to exactly what the brief asks for: components from
# `ui.components`.

StatusKind = Literal["online", "offline", "warning", "processing", "error"]
StatValue = Union[int, str]
TrendDirection = Literal["up", "down", "flat"]
InfoPanelVariant = Literal["info", "success", "warning", "error"]


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
        trend: Optional "up" / "down" / "flat" direction indicator.
        delta: Optional short delta string (e.g. "+12% vs last run").
        status: Optional accent color — "default", "success", "warning",
            or "error".
    """

    title: str
    value: StatValue
    icon: Optional[str] = None
    trend: Optional[TrendDirection] = None
    delta: Optional[str] = None
    status: Optional[str] = None


# The ten knowledge-base categories tracked in the "Knowledge Coverage"
# section, in display order. Kept as a tuple of (label, icon) pairs so
# `render_knowledge_coverage` can zip them against the caller-supplied
# percentages without repeating label strings elsewhere in the module.
KNOWLEDGE_COVERAGE_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("Hospital Information", "\U0001F3E5"),
    ("Doctor Directory", "\U0001FA7A"),
    ("Departments", "\U0001F3E2"),
    ("Diseases", "\U0001FA7B"),
    ("Medicines", "\U0001F48A"),
    ("Appointments", "\U0001F4C5"),
    ("Insurance", "\U0001F4C4"),
    ("Emergency Protocols", "\U0001F6A8"),
    ("Patient Guidelines", "\U0001F4D8"),
    ("FAQ", "\u2753"),
)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _render_metric_grid(cards: Sequence[MetricCardSpec], num_columns: int = 2) -> None:
    """
    Render a sequence of metric cards laid out in an evenly-spaced grid.

    Centralizing this loop avoids repeating the "create N st.columns and
    cycle through them" pattern for the Knowledge Base Overview, Retrieval
    Metrics, and Response Metrics sections, which all need the same grid
    behavior.

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
                trend=card.trend,
                delta=card.delta,
                status=card.status,
            )


def _format_ms(value: Optional[Union[int, float]]) -> str:
    """
    Format a millisecond duration for display, tolerating `None`.

    Args:
        value: Duration in milliseconds, or `None` if not yet available.

    Returns:
        A string like "142 ms", or an em-dash placeholder when `value`
        is `None`.
    """
    if value is None:
        return "\u2014"
    return f"{value:.0f} ms" if isinstance(value, float) else f"{value} ms"


def _format_percentage(value: Optional[float]) -> str:
    """
    Format a 0.0-1.0 fraction as a whole-number percentage string.

    Args:
        value: Fraction in the 0.0-1.0 range, or `None` if unavailable.

    Returns:
        A string like "91%", or an em-dash placeholder when `value` is
        `None`.
    """
    if value is None:
        return "\u2014"
    return f"{value * 100:.0f}%"


def _format_count(value: Optional[Union[int, str]]) -> str:
    """
    Format a plain integer count for display, tolerating `None` or an
    already-formatted string.

    Args:
        value: A count, a pre-formatted string, or `None`.

    Returns:
        The string representation, or an em-dash placeholder when `value`
        is `None`.
    """
    if value is None:
        return "\u2014"
    return str(value)


# =============================================================================
# SECTION 1 — PANEL HEADER
# =============================================================================


def render_metrics_header() -> None:
    """
    Render the AI Insights panel header.

    Displays the panel title ("AI Insights"), subtitle ("Real-time
    Retrieval-Augmented Generation Analytics"), and a chart icon, via the
    shared `render_section_header` component.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    render_section_header(
        title="AI Insights",
        subtitle="Real-time Retrieval-Augmented Generation Analytics",
        icon="\U0001F4CA",
    )


# =============================================================================
# SECTION 2 — KNOWLEDGE BASE OVERVIEW
# =============================================================================


def render_kb_overview(
    total_documents: StatValue = 0,
    doctors: StatValue = 0,
    departments: StatValue = 0,
    diseases: StatValue = 0,
    medicines: StatValue = 0,
    appointments: StatValue = 0,
) -> None:
    """
    Render the "Knowledge Base Overview" section as a grid of metric cards.

    Args:
        total_documents: Total number of indexed knowledge-base records.
        doctors: Number of doctor profiles on record.
        departments: Number of hospital departments on record.
        diseases: Number of disease/condition entries on record.
        medicines: Number of medicine entries on record.
        appointments: Number of appointment entries on record.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown(
        '<p style="font-weight:600; margin-bottom:0.5rem;">Knowledge Base Overview</p>',
        unsafe_allow_html=True,
    )
    cards = (
        MetricCardSpec(title="Total Documents", value=_format_count(total_documents), icon="\U0001F4C4"),
        MetricCardSpec(title="Doctors", value=_format_count(doctors), icon="\U0001FA7A"),
        MetricCardSpec(title="Departments", value=_format_count(departments), icon="\U0001F3E2"),
        MetricCardSpec(title="Diseases", value=_format_count(diseases), icon="\U0001FA7B"),
        MetricCardSpec(title="Medicines", value=_format_count(medicines), icon="\U0001F48A"),
        MetricCardSpec(title="Appointments", value=_format_count(appointments), icon="\U0001F4C5"),
    )
    _render_metric_grid(cards, num_columns=2)


# =============================================================================
# SECTION 3 — RETRIEVAL METRICS
# =============================================================================


def render_retrieval_metrics(
    retrieved_sources: StatValue = 0,
    retrieval_confidence: Optional[float] = None,
    average_similarity_score: Optional[float] = None,
    retrieval_time_ms: Optional[Union[int, float]] = None,
    chunks_retrieved: StatValue = 0,
) -> None:
    """
    Render the "Retrieval Metrics" section as a grid of metric cards.

    All values describe a single retrieval step already performed by the
    backend `Retriever` / `RAGPipeline` — this function only displays
    them, it never queries the vector store itself.

    Args:
        retrieved_sources: Number of distinct source documents retrieved.
        retrieval_confidence: Overall retrieval confidence, 0.0-1.0.
        average_similarity_score: Mean similarity score across retrieved
            chunks, 0.0-1.0.
        retrieval_time_ms: Time spent on retrieval, in milliseconds.
        chunks_retrieved: Number of text chunks retrieved from ChromaDB.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown(
        '<p style="font-weight:600; margin-bottom:0.5rem;">Retrieval Metrics</p>',
        unsafe_allow_html=True,
    )
    cards = (
        MetricCardSpec(title="Retrieved Sources", value=_format_count(retrieved_sources), icon="\U0001F4DA"),
        MetricCardSpec(
            title="Retrieval Confidence",
            value=_format_percentage(retrieval_confidence),
            icon="\U0001F3AF",
        ),
        MetricCardSpec(
            title="Avg. Similarity Score",
            value=_format_percentage(average_similarity_score),
            icon="\U0001F4CF",
        ),
        MetricCardSpec(title="Retrieval Time", value=_format_ms(retrieval_time_ms), icon="\u23F1\uFE0F"),
        MetricCardSpec(title="Chunks Retrieved", value=_format_count(chunks_retrieved), icon="\U0001F9E9"),
    )
    _render_metric_grid(cards, num_columns=2)


# =============================================================================
# SECTION 4 — RESPONSE METRICS
# =============================================================================


def render_response_metrics(
    response_time_ms: Optional[Union[int, float]] = None,
    prompt_tokens: StatValue = 0,
    completion_tokens: StatValue = 0,
    total_tokens: StatValue = 0,
    response_length: StatValue = 0,
) -> None:
    """
    Render the "Response Metrics" section as a grid of metric cards.

    All values describe the Gemini generation step already performed by
    the backend `RAGPipeline` — this function only displays them, it
    never calls Gemini or counts tokens itself.

    Args:
        response_time_ms: Time spent generating the response, in
            milliseconds.
        prompt_tokens: Number of tokens in the prompt sent to Gemini.
        completion_tokens: Number of tokens in Gemini's completion.
        total_tokens: `prompt_tokens + completion_tokens`, supplied by the
            caller (not recomputed here).
        response_length: Character (or word) length of the final response
            text, as defined by the caller.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown(
        '<p style="font-weight:600; margin-bottom:0.5rem;">Response Metrics</p>',
        unsafe_allow_html=True,
    )
    cards = (
        MetricCardSpec(title="Response Time", value=_format_ms(response_time_ms), icon="\u26A1"),
        MetricCardSpec(title="Prompt Tokens", value=_format_count(prompt_tokens), icon="\U0001F4DD"),
        MetricCardSpec(title="Completion Tokens", value=_format_count(completion_tokens), icon="\U0001F4AC"),
        MetricCardSpec(title="Total Tokens", value=_format_count(total_tokens), icon="\U0001F522"),
        MetricCardSpec(title="Response Length", value=_format_count(response_length), icon="\U0001F4CF"),
    )
    _render_metric_grid(cards, num_columns=2)


# =============================================================================
# SECTION 5 — SYSTEM HEALTH
# =============================================================================


def render_system_health(
    gemini_status: StatusKind = "offline",
    retriever_status: StatusKind = "offline",
    vector_store_status: StatusKind = "offline",
    embedding_model_status: StatusKind = "offline",
    pipeline_status: StatusKind = "offline",
) -> None:
    """
    Render the "System Health" section as a set of status badges.

    Summarizes the health of the five moving parts behind the assistant:
    Gemini, the Retriever, the vector store, the embedding model, and the
    overall RAG pipeline. This function only displays the statuses it is
    given — it performs no health checks or API calls itself.

    Args:
        gemini_status: One of "online", "offline", "warning",
            "processing", "error".
        retriever_status: Same status vocabulary as `gemini_status`.
        vector_store_status: Same status vocabulary as `gemini_status`.
        embedding_model_status: Same status vocabulary as `gemini_status`.
        pipeline_status: Same status vocabulary as `gemini_status`;
            represents the overall `RAGPipeline` health.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown(
        '<p style="font-weight:600; margin-bottom:0.5rem;">System Health</p>',
        unsafe_allow_html=True,
    )
    status_rows: tuple[tuple[str, StatusKind], ...] = (
        ("Gemini", gemini_status),
        ("Retriever", retriever_status),
        ("Vector Store", vector_store_status),
        ("Embedding Model", embedding_model_status),
        ("Pipeline", pipeline_status),
    )
    for component_label, component_status in status_rows:
        row_markup = (
            '<div style="display:flex; justify-content:space-between; '
            'align-items:center; padding:0.3rem 0;">'
            f"<span>{component_label}</span><span></span>"
            "</div>"
        )
        st.markdown(row_markup, unsafe_allow_html=True)
        render_status_badge(label=component_status.title(), status=component_status)


# =============================================================================
# SECTION 6 — KNOWLEDGE COVERAGE
# =============================================================================


def render_knowledge_coverage(
    hospital_information: float = 0.0,
    doctor_directory: float = 0.0,
    departments: float = 0.0,
    diseases: float = 0.0,
    medicines: float = 0.0,
    appointments: float = 0.0,
    insurance: float = 0.0,
    emergency_protocols: float = 0.0,
    patient_guidelines: float = 0.0,
    faq: float = 0.0,
) -> None:
    """
    Render the "Knowledge Coverage" section as animated progress bars.

    Each percentage represents how complete/populated a given knowledge-
    base category is, as determined by the caller (e.g. `app.py`,
    reasoning about the 13-file / 1,556-record knowledge base). This
    function performs no coverage calculation of its own.

    Args:
        hospital_information: Coverage percentage (0-100) for general
            hospital information.
        doctor_directory: Coverage percentage for the doctor directory.
        departments: Coverage percentage for department records.
        diseases: Coverage percentage for disease/condition records.
        medicines: Coverage percentage for medicine records.
        appointments: Coverage percentage for appointment records.
        insurance: Coverage percentage for insurance information.
        emergency_protocols: Coverage percentage for emergency protocols.
        patient_guidelines: Coverage percentage for patient guidelines.
        faq: Coverage percentage for the FAQ knowledge base.

    Returns:
        None. Renders directly into the Streamlit app via
        `render_progress` for each category.
    """
    st.markdown(
        '<p style="font-weight:600; margin-bottom:0.5rem;">Knowledge Coverage</p>',
        unsafe_allow_html=True,
    )
    percentages = (
        hospital_information,
        doctor_directory,
        departments,
        diseases,
        medicines,
        appointments,
        insurance,
        emergency_protocols,
        patient_guidelines,
        faq,
    )
    for (label, icon), percentage in zip(KNOWLEDGE_COVERAGE_CATEGORIES, percentages):
        render_progress(percentage=percentage, label=f"{icon} {label}", animated=True)


# =============================================================================
# SECTION 7 — CONFIDENCE SUMMARY
# =============================================================================


def render_confidence_summary(confidence_score: float, explanation: str = "") -> None:
    """
    Render the "Confidence Summary" section.

    Displays a confidence badge (which internally classifies the score
    into Low / Medium / High / Excellent — that classification lives in
    `ui.components`, not here) alongside a short caller-supplied
    explanation of what the score means for this particular answer.

    Args:
        confidence_score: Overall answer confidence, 0.0-1.0. Not
            recalculated here — passed straight to `render_confidence_badge`.
        explanation: A short, caller-written sentence explaining the
            score in context (e.g. "Based on 4 highly relevant sources
            from the FAQ and Patient Guidelines knowledge base.").

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown(
        '<p style="font-weight:600; margin-bottom:0.5rem;">Confidence Summary</p>',
        unsafe_allow_html=True,
    )
    render_confidence_badge(confidence_score)
    if explanation:
        st.markdown(
            f'<p style="font-size:0.85rem; color:#475569; margin-top:0.5rem;">{explanation}</p>',
            unsafe_allow_html=True,
        )


# =============================================================================
# SECTION 8 — RETRIEVED SOURCES SUMMARY
# =============================================================================


def render_sources_summary(
    source_count: StatValue = 0,
    top_source: str = "\u2014",
    document_type: str = "\u2014",
    ranking_method: str = "\u2014",
) -> None:
    """
    Render the "Retrieved Sources Summary" section.

    Args:
        source_count: Total number of sources retrieved for the current
            answer.
        top_source: Name of the highest-ranked retrieved document.
        document_type: Category of the top-ranked document (e.g. "FAQ",
            "Doctor Profile", "Patient Guidelines").
        ranking_method: Human-readable name of the ranking/similarity
            method used (e.g. "Cosine Similarity", "MMR").

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown(
        '<p style="font-weight:600; margin-bottom:0.5rem;">Retrieved Sources Summary</p>',
        unsafe_allow_html=True,
    )
    render_key_value(
        {
            "Number of Sources": _format_count(source_count),
            "Top Source": top_source,
            "Document Type": document_type,
            "Ranking Method": ranking_method,
        }
    )


# =============================================================================
# SECTION 9 — PERFORMANCE SUMMARY
# =============================================================================


def render_performance_summary(
    message: str,
    variant: InfoPanelVariant = "success",
    title: str = "Performance Summary",
) -> None:
    """
    Render the "Performance Summary" section as an information panel.

    The message is fully composed by the caller (typically `app.py`,
    after inspecting the metrics it already has) — this function does
    not evaluate thresholds or decide what "excellent performance" means.

    Args:
        message: The performance summary text, e.g. "Average response
            time below target. High retrieval confidence. Knowledge base
            healthy."
        variant: Visual tone of the panel — one of "info", "success",
            "warning", "error". Defaults to "success".
        title: Panel heading. Defaults to "Performance Summary".

    Returns:
        None. Renders directly into the Streamlit app via
        `render_info_panel`.
    """
    render_info_panel(message=message, variant=variant, title=title)


# =============================================================================
# SECTION 10 — PIPELINE OVERVIEW
# =============================================================================


def render_pipeline_summary(
    embedding_model: str = "\u2014",
    llm: str = "\u2014",
    vector_database: str = "\u2014",
    chunk_size: StatValue = "\u2014",
    chunk_overlap: StatValue = "\u2014",
    embedding_dimension: StatValue = "\u2014",
) -> None:
    """
    Render the "Pipeline Overview" section as a key-value summary.

    All configuration values are supplied by the caller (typically read
    from the backend `RAGPipeline`'s configuration) — this function does
    not introspect the pipeline itself.

    Args:
        embedding_model: Name of the configured embedding model (e.g.
            "all-MiniLM-L6-v2").
        llm: Name of the configured generation model (e.g.
            "gemini-1.5-flash").
        vector_database: Name of the vector store in use (e.g. "ChromaDB").
        chunk_size: Configured text-chunk size (e.g. 500).
        chunk_overlap: Configured chunk overlap (e.g. 50).
        embedding_dimension: Dimensionality of the embedding vectors
            (e.g. 384).

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown(
        '<p style="font-weight:600; margin-bottom:0.5rem;">Pipeline Overview</p>',
        unsafe_allow_html=True,
    )
    render_key_value(
        {
            "Embedding Model": embedding_model,
            "LLM": llm,
            "Vector Database": vector_database,
            "Chunk Size": _format_count(chunk_size),
            "Chunk Overlap": _format_count(chunk_overlap),
            "Embedding Dimension": _format_count(embedding_dimension),
        }
    )


# =============================================================================
# SECTION 11 — LAST UPDATE
# =============================================================================


def render_last_update(
    knowledge_base_updated: str = "Never",
    model_loaded: str = "Never",
    system_uptime: str = "\u2014",
) -> None:
    """
    Render the "Last Update" section as a key-value summary.

    Args:
        knowledge_base_updated: Human-readable recency string for the
            last knowledge-base refresh (e.g. "2 minutes ago").
        model_loaded: Human-readable recency string for when the
            embedding/generation models were loaded.
        system_uptime: Human-readable system uptime (e.g. "3h 42m").

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown(
        '<p style="font-weight:600; margin-bottom:0.5rem;">Last Update</p>',
        unsafe_allow_html=True,
    )
    render_key_value(
        {
            "Knowledge Base Updated": knowledge_base_updated,
            "Model Loaded": model_loaded,
            "System Uptime": system_uptime,
        }
    )


# =============================================================================
# ORCHESTRATOR
# =============================================================================


def render_metrics(
    # Knowledge Base Overview
    total_documents: StatValue = 0,
    doctors: StatValue = 0,
    departments: StatValue = 0,
    diseases: StatValue = 0,
    medicines: StatValue = 0,
    appointments: StatValue = 0,
    # Retrieval Metrics
    retrieved_sources: StatValue = 0,
    retrieval_confidence: Optional[float] = None,
    average_similarity_score: Optional[float] = None,
    retrieval_time_ms: Optional[Union[int, float]] = None,
    chunks_retrieved: StatValue = 0,
    # Response Metrics
    response_time_ms: Optional[Union[int, float]] = None,
    prompt_tokens: StatValue = 0,
    completion_tokens: StatValue = 0,
    total_tokens: StatValue = 0,
    response_length: StatValue = 0,
    # System Health
    gemini_status: StatusKind = "offline",
    retriever_status: StatusKind = "offline",
    vector_store_status: StatusKind = "offline",
    embedding_model_status: StatusKind = "offline",
    pipeline_status: StatusKind = "offline",
    # Knowledge Coverage
    coverage_hospital_information: float = 0.0,
    coverage_doctor_directory: float = 0.0,
    coverage_departments: float = 0.0,
    coverage_diseases: float = 0.0,
    coverage_medicines: float = 0.0,
    coverage_appointments: float = 0.0,
    coverage_insurance: float = 0.0,
    coverage_emergency_protocols: float = 0.0,
    coverage_patient_guidelines: float = 0.0,
    coverage_faq: float = 0.0,
    # Confidence Summary
    confidence_score: float = 0.0,
    confidence_explanation: str = "",
    # Retrieved Sources Summary
    source_count: StatValue = 0,
    top_source: str = "\u2014",
    document_type: str = "\u2014",
    ranking_method: str = "\u2014",
    # Performance Summary
    performance_message: str = "",
    performance_variant: InfoPanelVariant = "success",
    # Pipeline Overview
    embedding_model: str = "\u2014",
    llm: str = "\u2014",
    vector_database: str = "\u2014",
    chunk_size: StatValue = "\u2014",
    chunk_overlap: StatValue = "\u2014",
    embedding_dimension: StatValue = "\u2014",
    # Last Update
    knowledge_base_updated: str = "Never",
    model_loaded: str = "Never",
    system_uptime: str = "\u2014",
) -> None:
    """
    Assemble the complete "AI Insights" panel in one call.

    Convenience entry point for `app.py`: renders the panel header,
    knowledge base overview, retrieval metrics, response metrics, system
    health, knowledge coverage, confidence summary, retrieved sources
    summary, performance summary, pipeline overview, and last-update
    information — in order, separated by dividers.

    Every argument is a plain value forwarded to the corresponding
    section function below; see each section function's docstring for
    details on an individual parameter. This orchestrator performs no
    computation of its own.

    Typical usage in `app.py`:

        from ui.layout import render_layout
        from ui.metrics import render_metrics

        columns = render_layout(online=True)
        with columns.insights:
            render_metrics(
                total_documents=1556, doctors=24, departments=8,
                diseases=140, medicines=310, appointments=57,
                retrieved_sources=4, retrieval_confidence=0.91,
                average_similarity_score=0.87, retrieval_time_ms=142,
                chunks_retrieved=6, response_time_ms=980,
                prompt_tokens=512, completion_tokens=180,
                total_tokens=692, response_length=640,
                gemini_status="online", retriever_status="online",
                vector_store_status="online",
                embedding_model_status="online", pipeline_status="online",
                performance_message="Average response time below target.",
            )

    Returns:
        None. Renders directly into the Streamlit app.
    """
    render_metrics_header()
    render_divider()

    render_kb_overview(
        total_documents=total_documents,
        doctors=doctors,
        departments=departments,
        diseases=diseases,
        medicines=medicines,
        appointments=appointments,
    )
    render_divider()

    render_retrieval_metrics(
        retrieved_sources=retrieved_sources,
        retrieval_confidence=retrieval_confidence,
        average_similarity_score=average_similarity_score,
        retrieval_time_ms=retrieval_time_ms,
        chunks_retrieved=chunks_retrieved,
    )
    render_divider()

    render_response_metrics(
        response_time_ms=response_time_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        response_length=response_length,
    )
    render_divider()

    render_system_health(
        gemini_status=gemini_status,
        retriever_status=retriever_status,
        vector_store_status=vector_store_status,
        embedding_model_status=embedding_model_status,
        pipeline_status=pipeline_status,
    )
    render_divider()

    render_knowledge_coverage(
        hospital_information=coverage_hospital_information,
        doctor_directory=coverage_doctor_directory,
        departments=coverage_departments,
        diseases=coverage_diseases,
        medicines=coverage_medicines,
        appointments=coverage_appointments,
        insurance=coverage_insurance,
        emergency_protocols=coverage_emergency_protocols,
        patient_guidelines=coverage_patient_guidelines,
        faq=coverage_faq,
    )
    render_divider()

    render_confidence_summary(
        confidence_score=confidence_score,
        explanation=confidence_explanation,
    )
    render_divider()

    render_sources_summary(
        source_count=source_count,
        top_source=top_source,
        document_type=document_type,
        ranking_method=ranking_method,
    )
    render_divider()

    if performance_message:
        render_performance_summary(
            message=performance_message,
            variant=performance_variant,
        )
        render_divider()

    render_pipeline_summary(
        embedding_model=embedding_model,
        llm=llm,
        vector_database=vector_database,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_dimension=embedding_dimension,
    )
    render_divider()

    render_last_update(
        knowledge_base_updated=knowledge_base_updated,
        model_loaded=model_loaded,
        system_uptime=system_uptime,
    )