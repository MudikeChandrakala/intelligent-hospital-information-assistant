"""
app.py
=============================================================================
Intelligent Hospital Information Assistant — application entry point.

PHASE 1 — Application Initialization
-----------------------------------------------------------------------------
This phase wires up only what the app needs to boot and render its page
shell: logging, lightweight startup configuration, and the page layout
from the locked `ui.layout` module.

This phase does NOT:
    - Initialize the RAG pipeline
    - Load any embedding/LLM models
    - Create or touch `st.session_state`
    - Render the sidebar (`ui/sidebar.py`)
    - Render the chat panel (`ui/chat.py`)
    - Render metrics/insights (`ui/metrics.py`)
    - Call Gemini, ChromaDB, or the Retriever

Those all arrive in later phases, once this shell is in place. Calling
`render_layout()` here only builds the page chrome (header, three-column
skeleton, footer) defined in the locked `ui.layout` module — it does not
render any sidebar/chat/metrics content itself.

PHASE 2 — Backend Initialization
-----------------------------------------------------------------------------
This phase constructs the backend RAG services into a single
`BackendServices` bundle: the document loader and text chunker (kept
available outside the pipeline), plus the `RAGPipeline`. `RAGPipeline`
already constructs and owns its own embedding generator, vector store,
retriever, prompt builder, and Gemini client internally, so those
collaborators are intentionally not constructed a second time here.

This phase does NOT:
    - Render the sidebar, chat panel, or metrics/insights
    - Create or touch `st.session_state`
    - Process user input
    - Query the RAG pipeline or retrieve any documents
    - Compute any embeddings
    - Call Gemini or answer any question

Every `_initialize_*` helper only constructs its service object with its
configuration — it does not load documents, build an index, compute
embeddings, or make any outbound API call. Those actions belong to later
phases, once the sidebar/chat/metrics UI and session state exist to
drive them.

PHASE 3 — Session State Management
-----------------------------------------------------------------------------
This phase centralizes `st.session_state` initialization via
`initialize_session_state()`. It seeds chat state, user state, backend
state, generation state, retrieved-document state, metrics, system
status, conversation metadata, and application flags — each only if
not already present — so later phases can rely on every key existing
from the very first rerun.

This phase does NOT:
    - Render the sidebar, chat panel, or metrics/insights
    - Process user input
    - Call `RAGPipeline.ask()` or retrieve any documents
    - Call Gemini
    - Define callbacks or any conversation logic

PHASE 4 — Layout Integration
-----------------------------------------------------------------------------
This phase populates the three containers already returned by the
locked `ui.layout.render_layout()` — `columns.sidebar`, `columns.chat`,
and `columns.insights` — with minimal placeholder content only. It
establishes the structure that later phases (5, 6, and 9A) replace
with `ui.sidebar.render_sidebar()`, `ui.chat.render_chat()`, and
`ui.metrics.render_metrics()`.

This phase does NOT:
    - Render `ui.sidebar.render_sidebar()`, `ui.chat.render_chat()`,
      or `ui.metrics.render_metrics()`
    - Process user input, prompts, or forms
    - Create widgets, buttons, or callbacks
    - Display chat history
    - Retrieve documents or call Gemini
    - Call `RAGPipeline.ask()`

PHASE 5 — Sidebar Integration
-----------------------------------------------------------------------------
This phase replaces only the sidebar placeholder introduced in Phase 4.
`columns.sidebar` now renders the existing, unmodified
`ui.sidebar.render_sidebar()`. `columns.chat` and `columns.insights`
continue to show their Phase 4 placeholder content unchanged.

This phase does NOT:
    - Modify `ui/sidebar.py` or recreate any of its functionality
    - Render `ui.chat.render_chat()` or `ui.metrics.render_metrics()`
    - Render chat history or responses
    - Process user input, prompts, or forms
    - Create callbacks, or any button/widget beyond what
      `render_sidebar()` itself already provides
    - Retrieve documents or call Gemini
    - Call `RAGPipeline.ask()`
    - Modify session state

PHASE 6 — Chat Interface Integration
-----------------------------------------------------------------------------
This phase replaces only the chat placeholder introduced in Phase 4.
`columns.chat` now renders the existing, unmodified
`ui.chat.render_chat()`. `columns.sidebar` continues rendering the real
sidebar from Phase 5, and `columns.insights` continues to show its
Phase 4 placeholder content unchanged.

This phase does NOT:
    - Modify `ui/chat.py` or recreate any of its functionality
    - Render `ui.metrics.render_metrics()`
    - Process prompts or send messages
    - Call `RAGPipeline.ask()`, retrieve documents, or call Gemini
    - Generate responses or implement streaming
    - Implement conversation logic
    - Modify session state
    - Render metrics
    - Create callbacks beyond what `render_chat()` itself already
      provides

PHASE 7A — User Interaction
-----------------------------------------------------------------------------
This phase captures the submitted user prompt returned by the existing,
unmodified `ui.chat.render_chat()` and records it into
`st.session_state`. It appends the prompt to `messages` — the
application's single source of truth for the conversation — increments
`conversation_count`, updates `current_question`, resets
`current_response`, and marks `is_generating` so a later phase knows a
response is owed.

This phase does NOT:
    - Call `RAGPipeline.ask()` or retrieve any documents
    - Call Gemini
    - Compute response time, retrieval time, or confidence
    - Display assistant responses
    - Implement streaming or any conversation/generation logic
    - Modify `ui/chat.py` or `ui/sidebar.py`

PHASE 7B + PHASE 8 — Backend Communication Layer
-----------------------------------------------------------------------------
This phase connects the chat UI to the existing `BackendServices` bundle
(`st.session_state.backend_services`, constructed once in Phase 2) so a
submitted prompt is routed through the backend rather than answered with
a value fabricated inline in `ui/chat.py`. The backend now returns the
real answer produced by `backend.rag_pipeline.ask(prompt)` — the
already-initialized `RAGPipeline` instance's public API — which
internally performs retrieval, prompt assembly, and the Gemini call.
No hardcoded or placeholder assistant reply remains anywhere in this
module.

The flow is: submitting a prompt (Phase 7A) marks `is_generating=True`
and triggers a rerun so the typing indicator renders; the following
rerun then calls the backend communication layer, which calls
`backend.rag_pipeline.ask(prompt)` and appends the resulting
`ChatMessage` (assistant, real generated content) to `messages`, clears
`is_generating`, and reruns once more so the typing indicator is
replaced by the assistant bubble. If the backend is unavailable or the
call raises, the exception is logged and a friendly assistant message is
shown instead — the app never crashes.

This phase does NOT:
    - Implement retrieval, prompt assembly, or Gemini calls itself —
      those already exist inside `RAGPipeline.ask()` and are only
      invoked, not reimplemented, here
    - Compute confidence, response time, or any other metric
    - Implement streaming responses
    - Modify `ui/chat.py` or `ui/sidebar.py`
    - Construct a second `BackendServices` / `RAGPipeline` instance

PHASE 9A — Metrics/Insights Integration
-----------------------------------------------------------------------------
This phase replaces only the insights placeholder introduced in Phase 4.
`columns.insights` now renders the existing, unmodified
`ui.metrics.render_metrics()`, called with the handful of values that
already exist in `st.session_state` (`response_time`, `retrieval_time`,
`confidence_score`) plus a `pipeline_status` derived from
`backend_initialized` the same way `_render_layout()` already derives
`ai_status` for `render_chat()`. Every other `render_metrics()` argument
is left at its own default — this phase does not fabricate knowledge-base
counts, coverage percentages, or source summaries that nothing in the
app currently computes.

This phase does NOT:
    - Modify `ui/metrics.py` or recreate any of its functionality
    - Call Gemini, ChromaDB, or the Retriever directly
    - Compute new metrics, confidence scores, or coverage percentages
    - Modify session state or introduce new session-state keys
    - Process user input, prompts, or forms

-----------------------------------------------------------------------------
Public API
-----------------------------------------------------------------------------
    AppConfig        -> immutable startup configuration
    BackendServices   -> immutable bundle of initialized backend services
    initialize_backend() -> constructs and returns a `BackendServices` bundle
    initialize_session_state() -> seeds default `st.session_state` values
    main()             -> application entry point (called via `streamlit run`)
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional
import streamlit as st

from ui.layout import LayoutColumns, PAGE_TITLE, PROJECT_VERSION, render_layout
from ui.chat import (
    ChatMessage,
    ChatInputResult,
    ChatRenderResult,
    ConversationMetadata,
    render_assistant_message,
    render_chat_footer,
    render_chat_header,
    render_divider,
    render_input_box,
    render_typing_indicator,
    render_user_message,
    render_welcome,
)
from ui.components import render_info_panel, render_metric_card, render_section_header
from ui.sidebar import render_sidebar
from ui.metrics import render_metrics
from ui.prescription import render_prescription_page

# --- Phase 2: backend service classes (constructed, not invoked, here) ----
# These live in the `modules` package alongside `ui`. `RAGPipeline` already
# constructs and owns its own `EmbeddingGenerator`, `ChromaVectorStore`,
# `Retriever`, `PromptBuilder`, and `GeminiClient` internally (see
# `modules/rag_pipeline.py`), so those collaborator classes are not
# imported or constructed again here — doing so would duplicate several
# expensive initializations (loading the embedding model, opening the
# vector store, configuring the Gemini SDK) for no benefit.
from modules.document_loader import DocumentLoader
from modules.rag_pipeline import RAGPipeline, RAGResponse
from modules.text_chunker import TextChunker
from modules.voice_assistant import VoiceAssistant, VoiceRecognitionResult

# =============================================================================
# PROJECT CONSTANTS
# =============================================================================
# Application-level identity. `APP_NAME` / `APP_VERSION` are aliased from
# the locked `ui.layout` constants rather than restated, so the browser
# title, header banner, and startup log line can never drift out of sync.

APP_NAME: str = PAGE_TITLE
APP_VERSION: str = PROJECT_VERSION

#: Logger name used throughout `app.py`. `ui/*.py` modules do their own
#: thing (they are Streamlit-render-only); this is the application-level
#: logger for startup/lifecycle events.
LOGGER_NAME: str = "hospital_assistant"

#: Module-level logger. `_configure_logging()` attaches handlers to this
#: same named logger (looked up by name) when `main()` runs — defining
#: it here lets Phase 2's module-level `_initialize_*` helpers log
#: without a logger having to be threaded through every function call.
logger = logging.getLogger(LOGGER_NAME)

#: Environment variable names Phase 1 reads at startup. No other module
#: should need to read these directly.
_ENV_APP_ENVIRONMENT: str = "APP_ENV"
_ENV_LOG_LEVEL: str = "LOG_LEVEL"

# =============================================================================
# HELPER CONSTANTS
# =============================================================================

#: Fallback environment name when `APP_ENV` is not set.
DEFAULT_ENVIRONMENT: str = "development"

#: Fallback log level when `LOG_LEVEL` is not set or is invalid.
DEFAULT_LOG_LEVEL: str = "INFO"

#: Standard log line format: timestamp | level | logger name | message.
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

#: Backend/pipeline health has not been checked yet before initialization
#: runs, so the header status badge starts in a neutral "not yet
#: initialized" state rather than falsely claiming the system is online.
DEFAULT_BACKEND_ONLINE: bool = False
DEFAULT_BACKEND_STATUS_LABEL: str = "Initializing..."

#: Status badge label shown once `initialize_backend()` has completed
#: successfully (see `main()`).
BACKEND_ONLINE_STATUS_LABEL: str = "\U0001F7E2 Online"

#: `system_status` value (an existing Phase 3 session-state key) shown
#: once `initialize_backend()` has completed successfully.
BACKEND_ONLINE_SYSTEM_STATUS: str = "Online"


# =============================================================================
# BACKEND CONFIGURATION (Phase 2)
# =============================================================================
# Configuration for the RAG backend services. All values are overridable
# via environment variables so deployment settings never need a code
# change; the literals below are only development-time fallbacks.

#: Project root passed to `DocumentLoader`. `DocumentLoader` itself
#: resolves `<project_root>/knowledge_base/structured` and
#: `<project_root>/knowledge_base/unstructured` beneath this path, so
#: this should point at the project root, not the knowledge-base
#: directory itself.
PROJECT_ROOT_DIR: str = os.environ.get("PROJECT_ROOT_DIR", ".")

#: Target size (in characters, per `TextChunker`'s own convention) for
#: each chunk, and how much consecutive chunks overlap.
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "50"))

# NOTE: Embedding model, Chroma persistence, retrieval top-k, and Gemini
# API/model configuration are intentionally not duplicated here.
# `RAGPipeline` constructs and owns its own `EmbeddingGenerator`,
# `ChromaVectorStore`, `Retriever`, `PromptBuilder`, and `GeminiClient`
# internally (loading `GOOGLE_API_KEY` from the project's `.env` file
# itself), so there is nothing left for `app.py` to configure or pass
# in for those components.


# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AppConfig:
    """
    Immutable, resolved application configuration for this run.

    Holds only lightweight startup metadata — never secrets, model
    handles, or pipeline objects (those belong to backend modules in
    later phases).

    Attributes:
        app_name: Display name of the application.
        version: Application/project version string.
        environment: Deployment environment label (e.g. "development",
            "staging", "production"), read from `APP_ENV`.
        log_level: Resolved logging level name (e.g. "INFO", "DEBUG"),
            read from `LOG_LEVEL`.
    """

    app_name: str
    version: str
    environment: str
    log_level: str


def _resolve_app_config() -> AppConfig:
    """
    Build the immutable `AppConfig` for this run from constants and
    environment variables.

    Reads `APP_ENV` and `LOG_LEVEL` from the process environment,
    falling back to `DEFAULT_ENVIRONMENT` / `DEFAULT_LOG_LEVEL` when
    unset. Performs no I/O beyond `os.environ` lookups.

    Returns:
        A populated `AppConfig` instance.
    """
    environment = os.environ.get(_ENV_APP_ENVIRONMENT, DEFAULT_ENVIRONMENT).strip() or DEFAULT_ENVIRONMENT
    raw_log_level = os.environ.get(_ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL).strip().upper()
    log_level = raw_log_level if raw_log_level in logging._nameToLevel else DEFAULT_LOG_LEVEL

    return AppConfig(
        app_name=APP_NAME,
        version=APP_VERSION,
        environment=environment,
        log_level=log_level,
    )


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================


def _configure_logging(log_level: str) -> logging.Logger:
    """
    Configure application-wide logging and return the app's named logger.

    Safe to call more than once (e.g. across Streamlit reruns): reuses
    the same named logger and avoids attaching duplicate handlers.

    Args:
        log_level: Logging level name (e.g. "INFO", "DEBUG", "WARNING").

    Returns:
        The configured `logging.Logger` instance for `LOGGER_NAME`.
    """
    resolved_level = logging.getLevelName(log_level)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# =============================================================================
# BACKEND SERVICES (Phase 2)
# =============================================================================


@dataclass(frozen=True)
class BackendServices:
    """
    Immutable bundle of every initialized backend RAG service.

    Constructed once by `initialize_backend()` and handed to later
    phases (chat handling, metrics) to use — this module never calls
    into any of these services itself beyond constructing them.

    `RAGPipeline` already constructs and owns its own
    `EmbeddingGenerator`, `ChromaVectorStore`, `Retriever`,
    `PromptBuilder`, and `GeminiClient` internally, so this bundle does
    not hold separate instances of those — doing so would initialize
    each of those expensive components twice for no benefit. Only the
    components that have a genuine reason to exist outside the
    pipeline (e.g. for a future knowledge-base ingestion/admin flow)
    are kept here alongside the pipeline itself.

    Attributes:
        document_loader: Reads source knowledge-base files from disk.
        text_chunker: Splits loaded documents into retrieval-sized chunks.
        rag_pipeline: End-to-end pipeline tying retrieval and generation
            together; owns its own embedding model, vector store,
            retriever, prompt builder, and Gemini client internally.
    """

    document_loader: DocumentLoader
    text_chunker: TextChunker
    rag_pipeline: RAGPipeline


def _initialize_document_loader() -> DocumentLoader:
    """
    Construct the `DocumentLoader` for the configured project root.

    `DocumentLoader` resolves its own `knowledge_base/structured` and
    `knowledge_base/unstructured` subdirectories beneath `project_root`
    and validates they exist. It does not read or parse any dataset
    files at construction time — that only happens if/when
    `load_all_documents()` is called, which Phase 2 does not do.

    Returns:
        An initialized `DocumentLoader` instance.

    Raises:
        RuntimeError: If the loader cannot be constructed (including
            when the expected knowledge-base directories are missing).
    """
    try:
        loader = DocumentLoader(project_root=PROJECT_ROOT_DIR)
        logger.debug("DocumentLoader initialized (project_root=%s).", PROJECT_ROOT_DIR)
        return loader
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain-specific error below
        logger.error("Failed to initialize DocumentLoader: %s", exc)
        raise RuntimeError("Failed to initialize DocumentLoader.") from exc


def _initialize_text_chunker() -> TextChunker:
    """
    Construct the `TextChunker` used to split loaded documents into
    retrieval-sized chunks.

    Only constructs the object with its configured chunk size/overlap —
    it does not chunk any text yet.

    Returns:
        An initialized `TextChunker` instance.

    Raises:
        RuntimeError: If the chunker cannot be constructed.
    """
    try:
        chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        logger.debug(
            "TextChunker initialized (chunk_size=%d, chunk_overlap=%d).",
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )
        return chunker
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize TextChunker: %s", exc)
        raise RuntimeError("Failed to initialize TextChunker.") from exc


def _initialize_rag_pipeline() -> RAGPipeline:
    """
    Construct the `RAGPipeline` that ties retrieval, prompt assembly,
    and Gemini generation together.

    `RAGPipeline` takes no constructor arguments: it assembles and owns
    its own internal `EmbeddingGenerator`, `ChromaVectorStore`,
    `Retriever`, `PromptBuilder`, and `GeminiClient` instances end to
    end (loading the embedding model, connecting to the persisted
    vector store, and configuring the Gemini SDK itself). Those
    collaborators are therefore intentionally not constructed
    separately in `app.py` — doing so would initialize each of them a
    second time for no benefit. `RAGPipeline` is a self-contained,
    independently usable service rather than one wired from other
    `BackendServices` members — this helper only constructs it, it
    does not call `ask()` or answer any question.

    Returns:
        An initialized `RAGPipeline` instance.

    Raises:
        RuntimeError: If the pipeline (or any of the internal
            components it constructs) cannot be initialized.
    """
    try:
        pipeline = RAGPipeline()
        logger.debug("RAGPipeline initialized.")
        return pipeline
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize RAGPipeline: %s", exc)
        raise RuntimeError("Failed to initialize RAGPipeline.") from exc


def initialize_backend() -> BackendServices:
    """
    Construct and wire together every backend service required by the
    RAG pipeline.

    Builds the document loader and text chunker (kept available outside
    the pipeline, e.g. for a future knowledge-base ingestion/admin
    flow), then constructs `RAGPipeline`, which builds and owns its own
    embedding generator, vector store, retriever, prompt builder, and
    Gemini client internally. Each step only constructs its object: no
    documents are loaded, no embeddings are computed, no vectors are
    queried, and no call is made to Gemini here.

    Returns:
        A fully populated `BackendServices` instance.

    Raises:
        RuntimeError: If any individual service fails to initialize.
            The originating `_initialize_*` helper logs the specific
            failure before this propagates, so the caller only needs
            to handle (or let crash) a single exception type.
    """
    logger.info("Initializing backend services...")

    document_loader = _initialize_document_loader()
    text_chunker = _initialize_text_chunker()
    rag_pipeline = _initialize_rag_pipeline()

    logger.info("All backend services initialized successfully.")

    return BackendServices(
        document_loader=document_loader,
        text_chunker=text_chunker,
        rag_pipeline=rag_pipeline,
    )


# =============================================================================
# SESSION STATE (Phase 3)
# =============================================================================
# Centralized `st.session_state` initialization. This section only ever
# *declares* state — it never renders UI, reads user input, retrieves
# documents, or calls `RAGPipeline.ask()` / Gemini. Every key is seeded
# exactly once per session; reruns are safe because each assignment is
# guarded by an `if "key" not in st.session_state` check.

#: Default values for every Streamlit session-state key this app uses.
#: Centralized here (rather than inlined across many `if` statements) so
#: the full set of state the app depends on is visible in one place.
SESSION_STATE_DEFAULTS: dict[str, object] = {
    # --- Chat state ---------------------------------------------------
    # AI Assistant and Voice Assistant each keep their own, completely
    # independent conversation history. Both are lists of `ChatMessage`
    # turns; which one a given helper reads/writes is decided by its
    # `target` parameter ("ai" or "voice") rather than by a single
    # shared list.
    "ai_messages": [],
    "voice_messages": [],
    # --- User state ------------------------------------------------------
    "current_question": "",
    "current_response": "",
    # --- Backend state ---------------------------------------------------
    "backend_services": None,
    "backend_initialized": False,
    # --- Generation state --------------------------------------------------
    "is_generating": False,
    "pending_target": "ai",
    # --- Retrieved information ---------------------------------------------
    "retrieved_documents": [],
    "source_documents": [],
    # --- Metrics -------------------------------------------------------
    "response_time": 0.0,
    "retrieval_time": 0.0,
    "confidence_score": 0.0,
    # --- System status ---------------------------------------------------
    "system_status": "Initializing",
    # --- Conversation metadata ---------------------------------------------
    "conversation_count": 0,
    "sidebar_active_page": "AI Assistant",
    # --- Application flags -------------------------------------------------
    "sidebar_expanded": True,
    "metrics_expanded": True,
    "voice_input_enabled": True,
    "speak_assistant_responses": True,
    "voice_speech_status": "Idle",
    "last_voice_command": "",
    "last_spoken_response": "",
    # --- Assistant audio cache --------------------------------------------
    "assistant_response_audio_cache": {},
}


def initialize_session_state() -> None:
    """
    Safely initialize every required `st.session_state` variable.

    Each key in `SESSION_STATE_DEFAULTS` is only ever assigned if it is
    not already present in `st.session_state`, so calling this function
    on every Streamlit rerun is a no-op for state that already exists
    (e.g. `backend_services` / `backend_initialized`, which `main()` may
    have already populated with real values before this runs).

    This function only declares state — it does not render any UI,
    process user input, retrieve documents, or call the RAG pipeline or
    Gemini.

    Returns:
        None.
    """
    initialized_keys: list[str] = []

    for key, default_value in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            initialized_keys.append(key)

    if initialized_keys:
        logger.debug("Session state initialized for keys: %s", ", ".join(initialized_keys))
    else:
        logger.debug("Session state already initialized; no new keys added.")


# =============================================================================
# LAYOUT INTEGRATION (Phase 4 + Phase 5 + Phase 6 + Phase 9A)
# =============================================================================
# Populates the three containers returned by the locked `ui.layout
# .render_layout()`. `columns.sidebar` renders the real, unmodified
# `ui.sidebar.render_sidebar()`; `columns.chat` renders the real,
# unmodified `ui.chat.render_chat()`; `columns.insights` renders the
# real, unmodified `ui.metrics.render_metrics()`. This section renders
# no business logic (no prompt processing, no retrieval, no Gemini
# calls, no `RAGPipeline.ask()`) beyond what `render_sidebar()`,
# `render_chat()`, and `render_metrics()` themselves already provide.

#: Retained for reference; no longer rendered now that Phase 9A wires up
#: `ui.metrics.render_metrics()` in its place.
_INSIGHTS_PLACEHOLDER_TEXT: str = "Metrics panel ready. Waiting for Phase 9."


# =============================================================================
# USER INTERACTION (Phase 7A)
# =============================================================================
# Captures the submitted user prompt returned by the existing,
# unmodified `ui.chat.render_chat()` and records it into
# `st.session_state`. This section does not call `RAGPipeline.ask()`,
# retrieve documents, call Gemini, compute any metrics, or display an
# assistant response — it only stores the prompt and prepares session
# state for the generation phase that follows.


def _handle_user_prompt(prompt: str, target: Literal["ai", "voice"] = "ai") -> None:
    """
    Record a newly submitted user prompt into `st.session_state`.

    Uses only the existing session-state keys seeded by
    `initialize_session_state()` (Phase 3), plus the independent
    `ai_messages` / `voice_messages` conversation lists. Appends the
    prompt to whichever list `target` selects — the application's
    single source of truth for that page's conversation — updates
    `current_question`, resets `current_response` for the upcoming
    answer, increments `conversation_count`, and sets `is_generating`
    to `True` so a later phase knows a response is owed.

    `current_question`, `current_response`, `conversation_count`, and
    `is_generating` remain shared, page-agnostic bookkeeping: only one
    prompt can ever be "pending" at a time in Streamlit's synchronous
    execution model, so a single shared pending-generation flag is
    sufficient — `_process_pending_generation()` determines which
    conversation that pending prompt belongs to from
    `st.session_state.sidebar_active_page`, which cannot change between
    this call and the rerun that resolves it.

    This function does not call `RAGPipeline.ask()`, retrieve
    documents, call Gemini, compute response/retrieval time or
    confidence, or render anything itself.

    Args:
        prompt: The submitted user prompt text, already known to be
            non-empty.
        target: Which conversation this prompt belongs to — "ai" for
            `st.session_state.ai_messages` (the AI Assistant page) or
            "voice" for `st.session_state.voice_messages` (the Voice
            Assistant page). Defaults to "ai" to preserve prior
            behavior for any caller that does not pass it explicitly.

    Returns:
        None.
    """
    user_turn = ChatMessage(
        role="user",
        content=prompt,
        timestamp=datetime.now(),
    )

    target_messages = st.session_state.ai_messages if target == "ai" else st.session_state.voice_messages
    target_messages.append(user_turn)

    st.session_state.current_question = prompt
    st.session_state.current_response = ""
    st.session_state.pending_target = target
    st.session_state.conversation_count += 1
    st.session_state.is_generating = True
    logger.info(
        "User prompt captured for %s conversation (conversation_count=%d).",
        target,
        st.session_state.conversation_count,
    )
    logger.debug("Session state updated for new prompt: %r", prompt)


# =============================================================================
# BACKEND COMMUNICATION LAYER (Phase 7B + Phase 8 + Phase 9B)
# =============================================================================
# Routes a captured prompt (Phase 7A) through the existing
# `BackendServices` bundle and records the resulting assistant message.
# `_generate_assistant_response()` calls the real, already-initialized
# `RAGPipeline.ask()` — owned by `st.session_state.backend_services` and
# constructed exactly once in `initialize_backend()` — so every prompt is
# answered by the actual Retriever -> PromptBuilder -> Gemini flow. No
# second `RAGPipeline` (or any of its internal collaborators) is created
# here, and no hardcoded/placeholder reply remains anywhere in this
# module.
#
# As of Phase 9B, `RAGPipeline.ask()` returns a structured `RAGResponse`
# rather than a bare string. `_generate_assistant_response()` unpacks
# that structure into the existing Phase 3 metrics session-state keys
# (`response_time`, `retrieval_time`, `retrieved_documents`,
# `source_documents`, `confidence_score`) and returns only the answer
# text, so every other caller in this module continues to work with a
# plain string exactly as before — `_handle_assistant_response()` and
# `_process_pending_generation()` are unmodified.

#: Friendly, non-crashing message shown when the backend communication
#: layer is unavailable or raises an exception.
_BACKEND_UNAVAILABLE_MESSAGE: str = (
    "I'm sorry, I'm having trouble reaching the assistant service right "
    "now. Please try again in a moment."
)


def _record_response_metrics(response: RAGResponse) -> None:
    """
    Populate the existing Phase 3 metrics session-state keys from a
    `RAGResponse`.

    Reads only fields already present on `response` — computed inside
    `RAGPipeline.ask()` itself — and writes them into the corresponding
    `st.session_state` keys seeded by `initialize_session_state()`. No
    metric is computed, estimated, or fabricated here.

    `confidence_score` is deliberately only overwritten when
    `response.confidence_score` is not `None`: the current `Retriever`
    does not expose a genuine similarity/confidence score, so
    `RAGResponse.confidence_score` is `None` for now, and
    `st.session_state.confidence_score` is left at its existing
    (Phase 3 default `0.0`, or whatever it already held) value rather
    than being overwritten with a fabricated number.

    Args:
        response: The structured result returned by
            `RAGPipeline.ask()` for the prompt just answered.

    Returns:
        None.
    """
    st.session_state.response_time = response.response_time_ms
    st.session_state.retrieval_time = response.retrieval_time_ms
    st.session_state.retrieved_documents = response.retrieved_documents
    st.session_state.source_documents = response.source_documents

    if response.confidence_score is not None:
        st.session_state.confidence_score = response.confidence_score

    logger.debug(
        "Recorded response metrics (response_time_ms=%.2f, retrieval_time_ms=%.2f, "
        "retrieved_documents=%d, source_documents=%d, confidence_score=%r).",
        response.response_time_ms,
        response.retrieval_time_ms,
        len(response.retrieved_documents),
        len(response.source_documents),
        response.confidence_score,
    )


def _generate_assistant_response(prompt: str, backend: Optional[BackendServices]) -> str:
    """
    Route a prompt through the existing `BackendServices` communication
    layer, record the resulting runtime metrics, and return the
    assistant's reply text.

    Uses the single `BackendServices` instance already constructed by
    `initialize_backend()` and stored in
    `st.session_state.backend_services` — no second backend or
    `RAGPipeline` is created here. Delegates the actual answer
    generation entirely to the existing, already-initialized
    `backend.rag_pipeline.ask(prompt)`, which internally performs
    retrieval, prompt building, and the Gemini call; this function does
    not implement any of that logic itself, it only orchestrates the
    call, records the metrics `ask()` already computed, and validates
    that the backend is available first.

    As of Phase 9B, `ask()` returns a structured `RAGResponse` rather
    than a bare string. This function unpacks that structure via
    `_record_response_metrics()` into the existing Phase 3
    session-state keys and returns only `response.answer`, so callers
    of this function (`_process_pending_generation()`) continue to
    receive a plain string exactly as before.

    Args:
        prompt: The user's submitted question, already recorded by
            `_handle_user_prompt()`.
        backend: The existing `BackendServices` bundle from
            `st.session_state.backend_services`, or `None` if backend
            initialization has not completed successfully.

    Returns:
        The assistant's reply text, from `RAGResponse.answer`.

    Raises:
        RuntimeError: If `backend` (or its `rag_pipeline`) is not
            available.
        Exception: Any exception raised by `RAGPipeline.ask()` itself
            propagates unchanged — the caller
            (`_process_pending_generation()`) is responsible for
            catching it, logging it, and falling back to
            `_BACKEND_UNAVAILABLE_MESSAGE` so the app never crashes.
    """
    if backend is None or backend.rag_pipeline is None:
        raise RuntimeError("Backend service is not available.")

    logger.debug("Routing prompt through backend communication layer: %r", prompt)
    response: RAGResponse = backend.rag_pipeline.ask(prompt)
    _record_response_metrics(response)
    return response.answer


def _handle_assistant_response(response_text: str, target: Literal["ai", "voice"] = "ai") -> None:
    """
    Record a generated assistant response into `st.session_state`.

    Appends an assistant `ChatMessage` to whichever conversation list
    `target` selects (`ai_messages` or `voice_messages`) — the
    application's single source of truth for that page's conversation —
    updates `current_response`, and clears `is_generating` so the
    typing indicator is removed on the next render.

    When `target == "voice"` and `st.session_state
    .speak_assistant_responses` is enabled, this also automatically
    generates and caches speech audio for the new message via
    `_generate_voice_response_audio()` — reusing the existing
    `_voice_assistant.text_to_speech()` and
    `assistant_response_audio_cache`/`_assistant_audio_cache_key()`, no
    new audio cache. This never happens for `target == "ai"`: the AI
    Assistant remains text-only.

    Args:
        response_text: The assistant's reply text to record.
        target: Which conversation this response belongs to — "ai" for
            `st.session_state.ai_messages` or "voice" for
            `st.session_state.voice_messages`. Callers should pass the
            same target used for the matching `_handle_user_prompt()`
            call; `_process_pending_generation()` does this by deriving
            it from `st.session_state.sidebar_active_page`.

    Returns:
        None.
    """
    assistant_turn = ChatMessage(
        role="assistant",
        content=response_text,
        timestamp=datetime.now(),
    )

    target_messages = st.session_state.ai_messages if target == "ai" else st.session_state.voice_messages
    target_messages.append(assistant_turn)
    message_index = len(target_messages) - 1

    st.session_state.current_response = response_text
    st.session_state.is_generating = False

    logger.info(
        "Assistant response recorded for %s conversation (conversation_count=%d).",
        target,
        st.session_state.conversation_count,
    )
    logger.debug("Session state updated with assistant response: %r", response_text)

    if target == "voice" and st.session_state.speak_assistant_responses:
        _generate_voice_response_audio(assistant_turn, message_index)


def _generate_voice_response_audio(message: ChatMessage, message_index: int) -> None:
    """
    Automatically generate and cache speech audio for a Voice Assistant
    response.

    Reuses the existing `_voice_assistant.text_to_speech()` and the
    existing `st.session_state.assistant_response_audio_cache` /
    `_assistant_audio_cache_key()` — the exact same cache
    `_render_assistant_speech_control()` already reads from, so no
    second audio cache is introduced. Only ever called for the Voice
    Assistant (`target == "voice"` in `_handle_assistant_response()`);
    never called for the AI Assistant.

    Never raises: any failure (including an unexpected exception from
    the TTS call itself) is logged and reflected in
    `st.session_state.voice_speech_status`, and the voice conversation
    continues normally — the caller does not need to guard this call.

    Args:
        message: The assistant `ChatMessage` just appended to
            `st.session_state.voice_messages`.
        message_index: That message's index within `voice_messages`,
            used to build the same cache key
            `_render_assistant_speech_control()` would use for it.

    Returns:
        None.
    """
    cache_key = _assistant_audio_cache_key(message, message_index)

    try:
        audio_bytes = _voice_assistant.text_to_speech(message.content)
    except Exception as exc:  # noqa: BLE001 - never let TTS crash the app
        logger.error("Failed to generate speech for voice response %s: %s", cache_key, exc)
        st.session_state.voice_speech_status = "Speech generation failed"
        return

    if audio_bytes is None:
        logger.warning("Text-to-speech returned no audio for voice response %s.", cache_key)
        st.session_state.voice_speech_status = "Speech generation failed"
        return

    st.session_state.assistant_response_audio_cache[cache_key] = audio_bytes
    st.session_state.last_spoken_response = message.content
    st.session_state.voice_speech_status = "Speaking"
    logger.info("Cached MP3 bytes for voice response %s (byte_length=%d).", cache_key, len(audio_bytes))


def _process_pending_generation() -> None:
    """
    Generate and record the assistant reply for an already-submitted
    prompt, if one is awaiting a response.

    Only acts when `is_generating` is `True` and `current_response` is
    still empty — i.e. this is the rerun immediately following prompt
    submission, so the chat panel has already rendered the typing
    indicator for this cycle. Calls the Phase 7B communication layer via
    `_generate_assistant_response()`; if the backend is unavailable or
    the call raises, the exception is logged and a friendly assistant
    message is recorded instead so the app never crashes. Triggers a
    rerun afterward so the typing indicator is replaced by the new
    assistant message.

    The response is routed to the same conversation the prompt came
    from by reading `st.session_state.sidebar_active_page`: "Voice
    Assistant" routes to `voice_messages`, anything else (i.e. "AI
    Assistant") routes to `ai_messages`. This is safe because
    `sidebar_active_page` cannot change between the `_handle_user_prompt()`
    call that set `is_generating=True` and the rerun that resolves it —
    no other user interaction can occur in between in Streamlit's
    synchronous execution model.

    Returns:
        None.
    """
    if not st.session_state.is_generating or st.session_state.current_response:
        return

    prompt = st.session_state.current_question
    backend: Optional[BackendServices] = st.session_state.backend_services
    target: Literal["ai", "voice"] = st.session_state.pending_target

    try:
        response_text = _generate_assistant_response(prompt=prompt, backend=backend)
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message, never crashes the app
        logger.error("Failed to generate assistant response: %s", exc)
        response_text = _BACKEND_UNAVAILABLE_MESSAGE

    _handle_assistant_response(response_text, target=target)
    st.rerun()


def _truncate_preview(text: str, max_length: int = 80) -> str:
    """Return a short preview string with an ellipsis when needed."""
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return ""
    if len(cleaned_text) <= max_length:
        return cleaned_text
    return cleaned_text[:max_length].rstrip() + "..."


def _voice_connection_status() -> str:
    """Return the current microphone availability label."""
    return "Connected" if _voice_assistant.is_available() else "Unavailable"


def _render_voice_assistant_panel() -> None:
    """Render the compact voice assistant dashboard and controls."""
    render_section_header(
        title="Voice Assistant",
        subtitle="Live voice input, spoken responses, and recent activity",
        icon="🎤",
    )

    status_cards = (
    (
        "Voice Assistant",
        "Ready" if _voice_assistant.is_available() else "Unavailable",
        "🎤",
        "success" if _voice_assistant.is_available() else "error",
    ),
)

    col = st.columns(1)[0]

    for title, value, icon, status in status_cards:
        with col:
          render_metric_card(
            title=title,
            value=value,
            icon=icon,
            status=status,
        )


def _assistant_audio_cache_key(message: ChatMessage, message_index: int) -> str:
    """Build a stable cache key for one assistant message's generated audio."""
    timestamp_value = str(message.timestamp)
    fingerprint_source = f"{message_index}|{timestamp_value}|{message.content}"
    fingerprint = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()
    return f"assistant_audio_{fingerprint}"


def _render_assistant_speech_control(message: ChatMessage, message_index: int) -> None:
    """Render a play button and cached audio player for one assistant response."""
    logger.info(
        "Entering assistant playback control (message_index=%d, cached=%s, enabled=%s).",
        message_index,
        _assistant_audio_cache_key(message, message_index) in st.session_state.assistant_response_audio_cache,
        st.session_state.speak_assistant_responses,
    )

    if not st.session_state.speak_assistant_responses:
        logger.info("Assistant speech is disabled; hiding play control.")
        return

    cache_key = _assistant_audio_cache_key(message, message_index)
    audio_cache: dict[str, bytes] = st.session_state.assistant_response_audio_cache
    audio_bytes = audio_cache.get(cache_key)

    button_key = f"play_response_{cache_key}"
    clicked = st.button("🔊 Play Response", key=button_key, use_container_width=False)
    if clicked:
        logger.info("Play Response clicked for assistant message %s.", cache_key)
        if audio_bytes is None:
            logger.info("No cached MP3 found; generating speech for assistant message %s.", cache_key)
            audio_bytes = _voice_assistant.text_to_speech(message.content)
            if audio_bytes is None:
                st.warning("⚠ Unable to generate speech.")
                logger.warning("Unable to generate speech for assistant message %s.", cache_key)
                return

            audio_cache[cache_key] = audio_bytes
            logger.info("Cached MP3 bytes for assistant message %s (byte_length=%d).", cache_key, len(audio_bytes))

        st.session_state.last_spoken_response = message.content
        st.session_state.voice_speech_status = "Speaking"

    if audio_bytes is None:
        logger.info("No cached audio available yet for assistant message %s; render button only.", cache_key)
        return

    logger.info("Rendering Streamlit audio player for assistant message %s (byte_length=%d).", cache_key, len(audio_bytes))
    st.audio(audio_bytes, format="audio/mp3")


def _render_chat_panel(target: Literal["ai", "voice"] = "ai") -> ChatRenderResult:
    """
    Render a chat panel with per-assistant-response playback controls.

    Reused for both independent conversations: reads
    `st.session_state.ai_messages` when `target="ai"` (the AI Assistant
    page) or `st.session_state.voice_messages` when `target="voice"`
    (the Voice Assistant page). Both pages share the exact same
    underlying `ui.chat` components (`render_chat_header`,
    `render_welcome`, `render_user_message`, `render_assistant_message`,
    `render_typing_indicator`, `render_input_box`, `render_chat_footer`)
    — `ui/chat.py` itself is not modified.

    Args:
        target: Which conversation to render — "ai" or "voice".

    Returns:
        The `ChatRenderResult` for this render, from whichever
        conversation's input box was rendered.
    """
    resolved_messages = list(st.session_state.ai_messages if target == "ai" else st.session_state.voice_messages)

    render_chat_header(
        metadata=ConversationMetadata(
            message_count=len(resolved_messages),
            ai_status="online" if st.session_state.backend_initialized else "offline",
        )
    )
    render_divider()

    selected_example: Optional[str] = None
    if not resolved_messages:
        selected_example = render_welcome(example_questions=[])
    else:
        for message_index, message in enumerate(resolved_messages):
            if message.role == "user":
                render_user_message(message)
            else:
                render_assistant_message(message)
                if target == "voice":
                    _render_assistant_speech_control(message, message_index)

        if st.session_state.is_generating:
            render_typing_indicator()

    render_divider()
    input_result = render_input_box(disabled=st.session_state.is_generating)
    render_chat_footer()

    return ChatRenderResult(input=input_result, selected_example=selected_example)


# =============================================================================
# VOICE INPUT (Phase 10A)
# =============================================================================
# Renders a small microphone control in the chat column, directly beneath
# the existing, unmodified `ui.chat.render_chat()`. Recording and
# speech-to-text conversion are delegated entirely to
# `modules.voice_assistant.VoiceAssistant`, which knows nothing about
# Gemini, the RAG pipeline, LangChain, ChromaDB, or chat history. When
# recognition succeeds, the recognized text is routed through the exact
# same `_handle_user_prompt()` (Phase 7A) a typed submission already
# uses, followed by the exact same `st.rerun()` — voice is a second
# *source* of a prompt, not a second submission pathway. On failure, a
# friendly status message is shown inline within the same script run;
# nothing is submitted and no new `st.session_state` key is introduced.

#: Constructed once at import time. `VoiceAssistant` only holds
#: lightweight configuration and an `sr.Recognizer()` — there is no
#: model to load, so unlike `BackendServices` this does not need to be
#: cached in `st.session_state` to avoid expensive reconstruction.
_voice_assistant = VoiceAssistant()

#: Widget key for the microphone button, so it never collides with any
#: key `ui/chat.py` or `ui/sidebar.py` use for their own widgets.
_VOICE_BUTTON_KEY: str = "voice_input_microphone_button"


def _render_voice_input_control(target: Literal["ai", "voice"] = "voice") -> None:
    """
    Render the microphone control and handle a click, if one occurred.

    Shows a two-stage status ("Listening...", then "Processing
    speech...") for the duration of the blocking record/transcribe
    calls. On success, submits the recognized text through the existing
    `_handle_user_prompt()` Phase 7A flow — routed to whichever
    conversation `target` selects — and reruns, exactly as a typed
    "Ask" click already does. On failure, shows a friendly inline
    message and leaves both conversations untouched.

    This function does not call `RAGPipeline.ask()`, retrieve
    documents, or call Gemini itself — submitting the recognized text
    only marks it pending, exactly like Phase 7A; the existing
    `_process_pending_generation()` call at the end of `_render_layout()`
    is what actually answers it.

    Args:
        target: Which conversation a recognized prompt should be
            recorded into — "ai" when this control is rendered on the
            AI Assistant page, or "voice" when rendered on the Voice
            Assistant page. Defaults to "voice" to preserve the
            behavior of the original Voice Assistant page mic button.

    Returns:
        None.
    """
    if not _voice_assistant.is_available():
        st.caption("\U0001F3A4 Voice input unavailable: the 'SpeechRecognition' package is not installed.")
        return

    clicked = st.button(
        "\U0001F3A4 Ask by voice",
        key=f"{_VOICE_BUTTON_KEY}_{target}",
        disabled=st.session_state.is_generating or not st.session_state.voice_input_enabled,
        use_container_width=False,
    )
    if not clicked:
        return

    st.session_state.voice_speech_status = "Listening"
    with st.status("\U0001F3A4 Listening...", expanded=False) as status_box:
        capture_result = _voice_assistant.record_audio()

        if not capture_result.success:
            status_box.update(label=capture_result.status_message, state="error")
            logger.info("Voice capture failed: %s", capture_result.status_message)
            st.session_state.voice_speech_status = "Idle"
            return

        status_box.update(label="Processing speech...")
        recognition_result: VoiceRecognitionResult = _voice_assistant.transcribe_audio(capture_result.audio)

        if not recognition_result.success or not recognition_result.text:
            status_box.update(label=recognition_result.status_message, state="error")
            logger.info("Voice recognition failed: %s", recognition_result.status_message)
            st.session_state.voice_speech_status = "Idle"
            return

        status_box.update(label=recognition_result.status_message, state="complete")

    logger.info("Voice input captured a prompt for %s conversation (conversation_count will be incremented).", target)
    st.session_state.last_voice_command = recognition_result.text
    st.session_state.voice_speech_status = "Idle"
    _handle_user_prompt(recognition_result.text, target=target)
    st.rerun()


def _render_voice_response_audio() -> None:
    """
    Display the audio player for the most recent Voice Assistant reply.

    Reads only `st.session_state.voice_messages` (to find the latest
    assistant message) and the existing
    `st.session_state.assistant_response_audio_cache` via
    `_assistant_audio_cache_key()` — the same cache
    `_generate_voice_response_audio()` already writes into and
    `_render_assistant_speech_control()` already reads from elsewhere;
    no second audio cache is introduced here. The written response text
    itself is never rendered by this function — only its audio.

    Renders nothing at all when there is no assistant message yet in
    `voice_messages`, or when audio for the latest one hasn't been
    generated (e.g. `speak_assistant_responses` is disabled, or
    generation failed) — no fabricated or placeholder audio is ever
    played.

    Returns:
        None.
    """
    voice_messages: list = st.session_state.voice_messages

    latest_assistant_index: Optional[int] = None
    latest_assistant_message: Optional[ChatMessage] = None
    for index in range(len(voice_messages) - 1, -1, -1):
        if voice_messages[index].role == "assistant":
            latest_assistant_index = index
            latest_assistant_message = voice_messages[index]
            break

    if latest_assistant_message is None or latest_assistant_index is None:
        return

    cache_key = _assistant_audio_cache_key(latest_assistant_message, latest_assistant_index)
    audio_bytes = st.session_state.assistant_response_audio_cache.get(cache_key)
    if audio_bytes is None:
        return

    st.audio(audio_bytes, format="audio/mp3")


# =============================================================================
# METRICS DISPLAY VALUES (Phase 9C — dashboard integration)
# =============================================================================
# `ui/metrics.py` now accepts several additional, optional display values
# (`retrieved_sources`, `chunks_retrieved`, `primary_source`, `sources_used`,
# `document_types`). This section derives them from the existing
# `st.session_state.source_documents` / `st.session_state.retrieved_documents`
# — already populated by `_record_response_metrics()` from the real
# `RAGResponse` — and nothing else. It does not call the RAG pipeline,
# ChromaDB, or Gemini, does not compute similarity/confidence/tokens, and
# never fabricates a value: whatever cannot be genuinely derived is left
# as `None` so `render_metrics()` can render its own "not available" state.
#
# Each retrieved item's `source` / `record_type` is read from LangChain
# `Document.metadata` — the same metadata keys `DocumentLoader` already
# attaches to every document (e.g. `metadata["source"] == "doctor_directory"`,
# `metadata["record_type"] == "doctor"`). The extraction helpers below also
# tolerate a plain dict or string in place of a `Document`, in case
# `RAGResponse.source_documents` exposes a lighter-weight shape than the
# raw retriever output — without guessing at values neither shape provides.


def _extract_document_source_name(document: object) -> Optional[str]:
    """
    Read a single retrieved document's source identifier, if present.

    Looks for LangChain `Document`-style `metadata["source"]` first (the
    convention already used throughout `modules/document_loader.py`),
    then tolerates a plain dict shaped the same way, then a bare string
    (used as-is). Returns `None` — never a fabricated placeholder — if no
    source identifier can be found.

    Args:
        document: A single entry from `st.session_state.source_documents`.

    Returns:
        The source identifier string, or `None` if it cannot be
        determined from the document as given.
    """
    if isinstance(document, str):
        return document.strip() or None

    metadata = getattr(document, "metadata", None)
    if metadata is None and isinstance(document, dict):
        metadata = document.get("metadata", document)

    if isinstance(metadata, dict):
        source = metadata.get("source")
        if isinstance(source, str) and source.strip():
            return source.strip()

    return None


def _extract_document_record_type(document: object) -> Optional[str]:
    """
    Read a single retrieved document's record type, if present.

    Looks for LangChain `Document`-style `metadata["record_type"]` first
    (the convention already used throughout
    `modules/document_loader.py`, e.g. `"doctor"`, `"medicine"`,
    `"faq"`), then tolerates a plain dict shaped the same way. Returns
    `None` — never a fabricated placeholder — if no record type can be
    found.

    Args:
        document: A single entry from `st.session_state.source_documents`.

    Returns:
        The record-type string, or `None` if it cannot be determined
        from the document as given.
    """
    metadata = getattr(document, "metadata", None)
    if metadata is None and isinstance(document, dict):
        metadata = document.get("metadata", document)

    if isinstance(metadata, dict):
        record_type = metadata.get("record_type")
        if isinstance(record_type, str) and record_type.strip():
            return record_type.strip()

    return None


def _build_metrics_display_values() -> dict[str, object]:
    """
    Derive the optional `render_metrics()` display values from the
    existing `st.session_state.source_documents` /
    `st.session_state.retrieved_documents` — the real retrieval results
    already recorded by `_record_response_metrics()` from the most
    recent `RAGResponse`. Computes nothing beyond counting and reading
    existing metadata; never calls the pipeline, never estimates a
    similarity/confidence/token value, and never invents a document
    type that isn't present in the retrieved metadata.

    Any value whose source list is empty (nothing retrieved yet, or the
    most recent query genuinely returned nothing) is left as `None`
    rather than reported as `0`, so the dashboard can distinguish "no
    data available" from a real zero-result answer.

    Returns:
        A dict with keys `retrieved_sources`, `chunks_retrieved`,
        `primary_source`, `sources_used`, and `document_types`, each
        either a genuine derived value or `None`.
    """
    source_documents = st.session_state.source_documents
    retrieved_documents = st.session_state.retrieved_documents
    response_length: Optional[int] = None
    ranking_method: Optional[str] = None

    if st.session_state.current_response.strip():
        response_length = len(st.session_state.current_response.strip())
    else:
        # `current_response` is shared, page-agnostic bookkeeping, but the
        # conversation it belongs to is one of two independent lists now.
        # Check whichever list actually has the more recent assistant
        # turn, so this fallback keeps working for both the AI Assistant
        # and Voice Assistant conversations.
        for candidate_messages in (st.session_state.ai_messages, st.session_state.voice_messages):
            if candidate_messages and getattr(candidate_messages[-1], "role", None) == "assistant":
                last_message = candidate_messages[-1]
                response_length = len((getattr(last_message, "content", "") or "").strip())
                break

    if source_documents:
        ranking_method = "RAG Retrieval"

    retrieved_sources: Optional[int] = len(source_documents) if source_documents else None
    chunks_retrieved: Optional[int] = len(retrieved_documents) if retrieved_documents else None
    sources_used: Optional[int] = len(source_documents) if source_documents else None

    primary_source: Optional[str] = None
    if source_documents:
        primary_source = _extract_document_source_name(source_documents[0])

    document_types: Optional[list[str]] = None
    if source_documents:
        distinct_types: list[str] = []
        for document in source_documents:
            record_type = _extract_document_record_type(document)
            if record_type and record_type not in distinct_types:
                distinct_types.append(record_type)
        document_types = distinct_types or None

    return {
        "retrieved_sources": retrieved_sources,
        "chunks_retrieved": chunks_retrieved,
        "response_length": response_length,
        "ranking_method": ranking_method,
        "primary_source": primary_source,
        "sources_used": sources_used,
        "document_types": document_types,
    }


def _render_layout(columns: LayoutColumns) -> None:
    """
    Populate each locked-layout container for the current phase.

    Renders the real sidebar into `columns.sidebar` via the existing,
    unmodified `ui.sidebar.render_sidebar()`, the real chat interface
    into `columns.chat` via the existing, unmodified
    `ui.chat.render_chat()`, and the real metrics/insights panel into
    `columns.insights` via the existing, unmodified
    `ui.metrics.render_metrics()`. `render_metrics()` is called with
    the values already present in `st.session_state`
    (`response_time`, `retrieval_time`, `confidence_score`) plus a
    `pipeline_status` derived from `backend_initialized` the same way
    `ai_status` is derived below for `render_chat()`, and — as of
    Phase 9C — the optional dashboard values
    (`retrieved_sources`, `chunks_retrieved`, `primary_source`,
    `sources_used`, `document_types`) derived by
    `_build_metrics_display_values()` from the existing
    `source_documents` / `retrieved_documents` session state. Any value
    that cannot be genuinely derived is passed through as `None` rather
    than fabricated.

    As of Phase 7A, `render_chat()` is called with the current
    conversation (`st.session_state.messages`) and generation state
    (`st.session_state.is_generating`), and returns a
    `ChatRenderResult`. When the user pressed Ask
    (`chat_result.input.ask_clicked`) with non-empty input text, that
    text is handed to `_handle_user_prompt()` to be recorded into
    `st.session_state`. Merely receiving a `ChatRenderResult` back
    from `render_chat()` — e.g. on a rerun where Ask was not clicked —
    does not trigger `_handle_user_prompt()`. The Clear button
    (`chat_result.input.clear_clicked`) and welcome-screen example
    selection (`chat_result.selected_example`) are intentionally left
    unhandled for a later phase.

    As of Phase 7B, submitting a prompt triggers a rerun (so the next
    render shows the typing indicator via `is_generating`), and
    `_process_pending_generation()` — called after every render — is
    what actually invokes the Phase 7B backend communication layer and
    records the assistant's reply once that rerun happens.

    As of Phase 10A, `_render_voice_input_control()` renders a
    microphone button directly beneath the chat panel inside
    `columns.chat`. A successful voice recognition calls
    `_handle_user_prompt()` and reruns exactly like a typed "Ask" click
    does — it is a second source of a prompt, not a second submission
    pathway — so `_process_pending_generation()` answers it identically
    either way.

    As of the AI/Voice conversation-separation refactor, the AI
    Assistant and Voice Assistant pages each read, submit to, and
    display only their own independent `st.session_state.ai_messages` /
    `st.session_state.voice_messages` conversation, while both continue
    to share the exact same `BackendServices` / `RAGPipeline` instance —
    no backend, Gemini client, retriever, or vector store is duplicated.

    As of the text/voice output split, the AI Assistant page renders
    `_render_chat_panel(target="ai")`, which never shows a playback
    control for its (text-only) responses. The Voice Assistant page no
    longer renders a text chat panel at all — it renders only
    `_render_voice_input_control(target="voice")` (the microphone
    control) and `_render_voice_response_audio()` (an audio player for
    the latest `voice_messages` assistant reply, shown only once real
    audio for it exists in `assistant_response_audio_cache` — that
    audio is generated automatically by `_handle_assistant_response()`
    via `_generate_voice_response_audio()`, not by a manual "Play
    Response" click). The Voice Assistant's written response text is
    never displayed.

    This function still renders no widget, form, or callback beyond
    what `render_sidebar()`, the chat panel, `render_metrics()`, and
    `_render_voice_input_control()` themselves already provide, and it does not retrieve documents,
    call `RAGPipeline.ask()`, or call Gemini — that generation logic
    belongs to the backend communication layer above, not here.

    Args:
        columns: The `LayoutColumns` returned by `render_layout()`.

    Returns:
        None.
    """
    active_page = st.session_state.sidebar_active_page

    with columns.sidebar:
        render_sidebar(active_page=active_page)

    ai_chat_result: Optional[ChatRenderResult] = None

    with columns.chat:
        if active_page == "AI Assistant":
            ai_chat_result = _render_chat_panel(target="ai")
            

        elif active_page == "Voice Assistant":
            st.title("🎤 Voice Assistant")
            st.caption("Speak naturally to interact with the hospital assistant.")
            _render_voice_input_control(target="voice")
            _render_voice_response_audio()

        elif active_page == "Prescription Analysis":
            render_prescription_page()

    with columns.insights:
        
        render_divider()
        metrics_display_values = _build_metrics_display_values()
        render_metrics(
            response_time_ms=st.session_state.response_time,
            retrieval_time_ms=st.session_state.retrieval_time,
            confidence_score=st.session_state.confidence_score,
            pipeline_status="online" if st.session_state.backend_initialized else "offline",
            voice_input_enabled=st.session_state.voice_input_enabled,
            voice_output_enabled=st.session_state.speak_assistant_responses,
            retrieved_sources=metrics_display_values["retrieved_sources"],
            chunks_retrieved=metrics_display_values["chunks_retrieved"],
            response_length=metrics_display_values["response_length"],
            primary_source=metrics_display_values["primary_source"],
            document_types=metrics_display_values["document_types"],
            ranking_method=metrics_display_values["ranking_method"],
        )

    if ai_chat_result is not None:
        prompt_text = ai_chat_result.input.text.strip()
        if ai_chat_result.input.ask_clicked and prompt_text:
            _handle_user_prompt(prompt_text, target="ai")
            st.rerun()

    _process_pending_generation()

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Application entry point (Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5 + Phase 6 + Phase 9A).

    Resolves startup configuration, configures logging, initializes the
    backend RAG services exactly once per session — reusing the existing
    `st.session_state.backend_services` on every subsequent rerun instead
    of calling `initialize_backend()` again — seeds the remaining
    session-state defaults, renders the page shell (page config, global
    styles, header, three-column skeleton, footer) via the locked
    `ui.layout.render_layout()`, renders the real sidebar via the
    existing `ui.sidebar.render_sidebar()`, renders the real chat
    interface via the existing `ui.chat.render_chat()`, and renders the
    real metrics/insights panel via the existing
    `ui.metrics.render_metrics()`.

    Returns:
        None.
    """
    config = _resolve_app_config()
    logger = _configure_logging(config.log_level)

    logger.info(
        "Starting %s v%s [environment=%s, log_level=%s]",
        config.app_name,
        config.version,
        config.environment,
        config.log_level,
    )

    if (
        "backend_services" not in st.session_state
        or st.session_state.backend_services is None
    ):
        st.session_state.backend_services = initialize_backend()
        st.session_state.backend_initialized = True
        st.session_state.system_status = BACKEND_ONLINE_SYSTEM_STATUS
        logger.info("Backend initialization complete.")
    else:
        logger.debug("Reusing existing backend services from session state; skipping initialize_backend().")

    backend: BackendServices = st.session_state.backend_services

    initialize_session_state()
    logger.info("Session state initialization complete.")

    columns: LayoutColumns = render_layout(
        online=st.session_state.backend_initialized,
        status_label=(
            BACKEND_ONLINE_STATUS_LABEL
            if st.session_state.backend_initialized
            else DEFAULT_BACKEND_STATUS_LABEL
        ),
    )

    logger.info("Phase 1 page shell rendered successfully.")

    _render_layout(columns)
    logger.info("Chat interface successfully integrated.")


if __name__ == "__main__":
    main()