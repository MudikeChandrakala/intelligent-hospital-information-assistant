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
establishes the structure that later phases (5, 6, and 9) will replace
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

PHASE 7B — Backend Communication Layer
-----------------------------------------------------------------------------
This phase connects the chat UI to the existing `BackendServices` bundle
(`st.session_state.backend_services`, constructed once in Phase 2) so a
submitted prompt is routed through the backend rather than answered with
a value fabricated inline in `ui/chat.py`. The backend currently returns
a temporary placeholder reply — Phase 8 will replace only that reply
with a real `RAGPipeline.ask()` call; the communication pipeline built
here does not otherwise change.

The flow is: submitting a prompt (Phase 7A) marks `is_generating=True`
and triggers a rerun so the typing indicator renders; the following
rerun then calls the backend communication layer, appends the resulting
`ChatMessage` (assistant, placeholder content) to `messages`, clears
`is_generating`, and reruns once more so the typing indicator is
replaced by the assistant bubble. If the backend is unavailable or the
call raises, the exception is logged and a friendly assistant message is
shown instead — the app never crashes.

This phase does NOT:
    - Call Gemini, ChromaDB, the Retriever, or generate embeddings
    - Implement retrieval, ranking, or knowledge-base search
    - Compute confidence, response time, or any other metric
    - Implement streaming responses
    - Modify `ui/chat.py` or `ui/sidebar.py`
    - Construct a second `BackendServices` / `RAGPipeline` instance

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
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import streamlit as st

from ui.layout import LayoutColumns, PAGE_TITLE, PROJECT_VERSION, render_layout
from ui.chat import ChatMessage, ChatRenderResult, render_chat
from ui.sidebar import render_sidebar

# --- Phase 2: backend service classes (constructed, not invoked, here) ----
# These live in the `modules` package alongside `ui`. `RAGPipeline` already
# constructs and owns its own `EmbeddingGenerator`, `ChromaVectorStore`,
# `Retriever`, `PromptBuilder`, and `GeminiClient` internally (see
# `modules/rag_pipeline.py`), so those collaborator classes are not
# imported or constructed again here — doing so would duplicate several
# expensive initializations (loading the embedding model, opening the
# vector store, configuring the Gemini SDK) for no benefit.
from modules.document_loader import DocumentLoader
from modules.rag_pipeline import RAGPipeline
from modules.text_chunker import TextChunker

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
    # `messages` is the application's single source of truth for the
    # conversation (a list of `{"role": ..., "content": ...}` turns).
    "messages": [],
    # --- User state ------------------------------------------------------
    "current_question": "",
    "current_response": "",
    # --- Backend state ---------------------------------------------------
    "backend_services": None,
    "backend_initialized": False,
    # --- Generation state --------------------------------------------------
    "is_generating": False,
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
    # --- Application flags -------------------------------------------------
    "sidebar_expanded": True,
    "metrics_expanded": True,
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
# LAYOUT INTEGRATION (Phase 4 + Phase 5 + Phase 6)
# =============================================================================
# Populates the three containers returned by the locked `ui.layout
# .render_layout()`. As of Phase 6, `columns.sidebar` renders the real,
# unmodified `ui.sidebar.render_sidebar()`, and `columns.chat` renders
# the real, unmodified `ui.chat.render_chat()`; `columns.insights`
# still shows its Phase 4 placeholder content. This section renders no
# business logic (no prompt processing, no retrieval, no Gemini calls,
# no `RAGPipeline.ask()`) beyond what `render_sidebar()` and
# `render_chat()` themselves already provide.

#: Placeholder copy shown in the insights container until Phase 9 wires
#: up `ui.metrics.render_metrics()`.
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


def _handle_user_prompt(prompt: str) -> None:
    """
    Record a newly submitted user prompt into `st.session_state`.

    Uses only the existing session-state keys seeded by
    `initialize_session_state()` (Phase 3). Appends the prompt to
    `messages` — the application's single source of truth for the
    conversation — updates `current_question`, resets
    `current_response` for the upcoming answer, increments
    `conversation_count`, and sets `is_generating` to `True` so a
    later phase knows a response is owed.

    This function does not call `RAGPipeline.ask()`, retrieve
    documents, call Gemini, compute response/retrieval time or
    confidence, or render anything itself.

    Args:
        prompt: The submitted user prompt text, already known to be
            non-empty.

    Returns:
        None.
    """
    user_turn = ChatMessage(
        role="user",
        content=prompt,
        timestamp=datetime.now(),
    )

    st.session_state.messages.append(user_turn)
    st.session_state.current_question = prompt
    st.session_state.current_response = ""
    st.session_state.conversation_count += 1
    st.session_state.is_generating = True

    logger.info("User prompt captured (conversation_count=%d).", st.session_state.conversation_count)
    logger.debug("Session state updated for new prompt: %r", prompt)


# =============================================================================
# BACKEND COMMUNICATION LAYER (Phase 7B)
# =============================================================================
# Routes a captured prompt (Phase 7A) through the existing
# `BackendServices` bundle and records the resulting assistant message.
# This section does not implement RAG, retrieval, embeddings, or call
# Gemini — `_generate_assistant_response()` only validates that the
# backend is reachable and returns a temporary placeholder reply. Phase 8
# will replace that one return value with a real `RAGPipeline.ask()`
# call; nothing else in this pipeline is expected to change.

#: Temporary assistant reply returned by the Phase 7B communication
#: layer. Phase 8 replaces this constant's use with a real
#: `RAGPipeline.ask()` response.
_PLACEHOLDER_ASSISTANT_REPLY: str = (
    "I received your question successfully. "
    "The RAG pipeline will be connected in Phase 8."
)

#: Friendly, non-crashing message shown when the backend communication
#: layer is unavailable or raises an exception.
_BACKEND_UNAVAILABLE_MESSAGE: str = (
    "I'm sorry, I'm having trouble reaching the assistant service right "
    "now. Please try again in a moment."
)


def _generate_assistant_response(prompt: str, backend: Optional[BackendServices]) -> str:
    """
    Route a prompt through the existing `BackendServices` communication
    layer and return the assistant's reply text.

    Uses the single `BackendServices` instance already constructed by
    `initialize_backend()` and stored in
    `st.session_state.backend_services` — no second backend or
    `RAGPipeline` is created here. This function does not call
    `RAGPipeline.ask()`, retrieve documents, compute embeddings, or call
    Gemini; it only confirms the backend is available and returns a
    temporary placeholder reply. Phase 8 will replace only that return
    value with a real `RAGPipeline.ask()` call.

    Args:
        prompt: The user's submitted question, already recorded by
            `_handle_user_prompt()`.
        backend: The existing `BackendServices` bundle from
            `st.session_state.backend_services`, or `None` if backend
            initialization has not completed successfully.

    Returns:
        The assistant's reply text.

    Raises:
        RuntimeError: If `backend` (or its `rag_pipeline`) is not
            available.
    """
    if backend is None or backend.rag_pipeline is None:
        raise RuntimeError("Backend service is not available.")

    logger.debug("Routing prompt through backend communication layer: %r", prompt)
    return _PLACEHOLDER_ASSISTANT_REPLY


def _handle_assistant_response(response_text: str) -> None:
    """
    Record a generated assistant response into `st.session_state`.

    Appends an assistant `ChatMessage` to `messages` — the application's
    single source of truth for the conversation — updates
    `current_response`, and clears `is_generating` so the typing
    indicator is removed on the next render.

    Args:
        response_text: The assistant's reply text to record.

    Returns:
        None.
    """
    assistant_turn = ChatMessage(
        role="assistant",
        content=response_text,
        timestamp=datetime.now(),
    )

    st.session_state.messages.append(assistant_turn)
    st.session_state.current_response = response_text
    st.session_state.is_generating = False

    logger.info("Assistant response recorded (conversation_count=%d).", st.session_state.conversation_count)
    logger.debug("Session state updated with assistant response: %r", response_text)


def _process_pending_generation() -> None:
    """
    Generate and record the assistant reply for an already-submitted
    prompt, if one is awaiting a response.

    Only acts when `is_generating` is `True` and `current_response` is
    still empty — i.e. this is the rerun immediately following prompt
    submission, so `render_chat()` has already rendered the typing
    indicator for this cycle. Calls the Phase 7B communication layer via
    `_generate_assistant_response()`; if the backend is unavailable or
    the call raises, the exception is logged and a friendly assistant
    message is recorded instead so the app never crashes. Triggers a
    rerun afterward so the typing indicator is replaced by the new
    assistant message.

    Returns:
        None.
    """
    if not st.session_state.is_generating or st.session_state.current_response:
        return

    prompt = st.session_state.current_question
    backend: Optional[BackendServices] = st.session_state.backend_services

    try:
        response_text = _generate_assistant_response(prompt=prompt, backend=backend)
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message, never crashes the app
        logger.error("Failed to generate assistant response: %s", exc)
        response_text = _BACKEND_UNAVAILABLE_MESSAGE

    _handle_assistant_response(response_text)
    st.rerun()


def _render_layout(columns: LayoutColumns) -> None:
    """
    Populate each locked-layout container for the current phase.

    Renders the real sidebar into `columns.sidebar` via the existing,
    unmodified `ui.sidebar.render_sidebar()`, and the real chat
    interface into `columns.chat` via the existing, unmodified
    `ui.chat.render_chat()`. `columns.insights` continues to show its
    Phase 4 placeholder content until Phase 9 wires up
    `ui.metrics.render_metrics()`.

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

    This function still renders no widget, form, or callback beyond
    what `render_sidebar()` and `render_chat()` themselves already
    provide, and it does not retrieve documents, call
    `RAGPipeline.ask()`, or call Gemini — that generation logic
    belongs to a later phase.

    Args:
        columns: The `LayoutColumns` returned by `render_layout()`.

    Returns:
        None.
    """
    with columns.sidebar:
        render_sidebar()

    with columns.chat:
        chat_result: ChatRenderResult = render_chat(
        messages=st.session_state.messages,
        ai_status="online" if st.session_state.backend_initialized else "offline",
        is_generating=st.session_state.is_generating,
    )

    with columns.insights:
        st.info(_INSIGHTS_PLACEHOLDER_TEXT)

    prompt_text: str = chat_result.input.text.strip()

    if chat_result.input.ask_clicked and prompt_text:
        _handle_user_prompt(prompt_text)
        st.rerun()

    _process_pending_generation()


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Application entry point (Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5 + Phase 6).

    Resolves startup configuration, configures logging, initializes the
    backend RAG services, stores them in `st.session_state`, seeds the
    remaining session-state defaults, renders the page shell (page
    config, global styles, header, three-column skeleton, footer) via
    the locked `ui.layout.render_layout()`, renders the real sidebar
    via the existing `ui.sidebar.render_sidebar()`, renders the real
    chat interface via the existing `ui.chat.render_chat()`, and
    populates the insights container with its Phase 4 placeholder
    content.

    Later phases will replace the remaining placeholder by rendering
    `ui.metrics.render_metrics()` into `columns.insights`, and will add
    the conversation logic (prompt processing, retrieval, and Gemini
    calls) behind the chat interface rendered here.

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

    backend: BackendServices = initialize_backend()
    logger.info("Backend initialization complete.")

    st.session_state.backend_services = backend
    st.session_state.backend_initialized = True
    st.session_state.system_status = BACKEND_ONLINE_SYSTEM_STATUS

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