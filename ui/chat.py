"""
ui/chat.py
=============================================================================
Center-column conversational interface for the Intelligent Hospital
Information Assistant frontend.

This module renders — and ONLY renders — the main chat panel: the chat
header, the welcome/empty-state screen, the scrolling message list (user
and assistant messages, with optional confidence/source/timing metadata),
retrieved-source chips, a typing indicator, the text input + Ask/Clear
controls, and the conversation footer/disclaimer.

It does NOT:
    - Call Gemini or any LLM
    - Access ChromaDB or any vector store
    - Call the Retriever
    - Generate embeddings
    - Invoke the RAG pipeline
    - Load documents
    - Calculate metrics (confidence scores, response times, token counts)
    - Read or write Streamlit session state
    - Decide what happens when "Ask" or "Clear" is pressed

Every message, status, and metric shown here is supplied by the caller
(typically `app.py`, forwarding data it obtained from the `RAGPipeline`
and its own session-state-backed conversation history) as plain function
arguments / dataclasses. `render_input_box` and `render_chat` only report
*what the user did this run* (typed text, pressed Ask, pressed Clear) —
`app.py` decides what that means. This keeps the module a pure, testable,
reusable presentation layer, exactly like `ui/sidebar.py` and
`ui/metrics.py`.

-----------------------------------------------------------------------------
Layout note
-----------------------------------------------------------------------------
`ui/layout.py` builds a three-column skeleton (`LayoutColumns`) whose
center column (`columns.chat`) is reserved for this module. `app.py` is
expected to call `render_chat()` inside `with columns.chat:`, e.g.:

    from ui.layout import render_layout
    from ui.chat import render_chat, ChatMessage, SourceDocument

    columns = render_layout(online=True)
    with columns.chat:
        result = render_chat(
            messages=st.session_state.get("messages", []),
            ai_status="online",
            is_generating=False,
        )
    if result.input.ask_clicked and result.input.text.strip():
        ...  # app.py decides what asking actually does
    if result.input.clear_clicked:
        ...  # app.py decides what clearing actually does
    if result.selected_example:
        ...  # app.py can pre-fill the input with the clicked example

-----------------------------------------------------------------------------
Public API
-----------------------------------------------------------------------------
    render_chat_header()          -> None
    render_welcome()               -> Optional[str]
    render_user_message()          -> None
    render_assistant_message()     -> None
    render_sources()                -> None
    render_typing_indicator()       -> None
    render_input_box()              -> ChatInputResult
    render_chat_footer()             -> None
    render_chat()                    -> ChatRenderResult  (orchestrator)
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional, Sequence, Union

import streamlit as st

from ui.components import (
    render_avatar,
    render_card,
    render_chat_timestamp,
    render_divider,
    render_empty_state,
    render_footer_note,
    render_info_panel,
    render_key_value,
    render_loading,
    render_section_header,
    render_source_chip,
    render_status_badge,
    render_tag,
)

# =============================================================================
# TYPE ALIASES
# =============================================================================
# Mirrors `ui.components.StatusKind` — kept as a local alias (rather than
# an import) since it is a typing-only construct, not a reusable function
# or a design token, matching the convention already established in
# `ui/sidebar.py` and `ui/metrics.py`.

StatusKind = Literal["online", "offline", "warning", "processing", "error"]
MessageRole = Literal["user", "assistant"]


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass(frozen=True)
class SourceDocument:
    """
    A single retrieved knowledge-base document, attached to an assistant
    message when the RAG pipeline cited sources for its answer.

    Attributes:
        name: Display name of the retrieved source
            (e.g. "patient_guidelines.txt").
        document_type: Short category label (e.g. "FAQ", "Doctor Profile",
            "Emergency Protocol"). Defaults to "Document".
        score: Optional retrieval confidence score for this specific
            source, in the 0.0-1.0 range.
    """

    name: str
    document_type: str = "Document"
    score: Optional[float] = None


@dataclass(frozen=True)
class ChatMessage:
    """
    A single message in the conversation, either from the user or the
    assistant.

    Attributes:
        role: Either "user" or "assistant".
        content: The message text. May contain simple Markdown (bold,
            lists, links) — rendering is the caller's responsibility to
            have already sanitized/prepared.
        timestamp: A pre-formatted timestamp string (e.g. "10:42 AM") or
            a `datetime` object (formatted automatically via
            `_format_timestamp`).
        confidence: Optional overall answer confidence, 0.0-1.0.
            Assistant messages only.
        source_count: Optional number of retrieved sources backing this
            answer. Assistant messages only.
        response_time_ms: Optional generation time, in milliseconds.
            Assistant messages only.
        sources: Optional list of `SourceDocument` citations. Assistant
            messages only.
        ranking_method: Optional human-readable name of the
            retrieval/ranking method used (e.g. "Cosine Similarity").
            Assistant messages only.
    """

    role: MessageRole
    content: str
    timestamp: Union[str, datetime] = ""
    confidence: Optional[float] = None
    source_count: Optional[int] = None
    response_time_ms: Optional[Union[int, float]] = None
    sources: Optional[Sequence[SourceDocument]] = None
    ranking_method: Optional[str] = None


@dataclass(frozen=True)
class ConversationMetadata:
    """
    Summary information about the current conversation, used to populate
    the chat header.

    Attributes:
        message_count: Total number of messages in the conversation so
            far (both user and assistant).
        ai_status: Overall AI/backend status, forwarded to
            `render_status_badge` — one of "online", "offline",
            "warning", "processing", "error".
    """

    message_count: int = 0
    ai_status: StatusKind = "offline"


@dataclass(frozen=True)
class ChatInputResult:
    """
    What the user did in the input area on this Streamlit run.

    Attributes:
        text: The current contents of the text input box.
        ask_clicked: True if the "Ask" button was clicked this run.
        clear_clicked: True if the "Clear" button was clicked this run.
    """

    text: str
    ask_clicked: bool
    clear_clicked: bool


@dataclass(frozen=True)
class ChatRenderResult:
    """
    The combined result of rendering the full chat panel via
    `render_chat()`.

    Attributes:
        input: The `ChatInputResult` describing this run's input-area
            interaction.
        selected_example: The text of a welcome-screen example question
            the user clicked, if any (only possible when the conversation
            is empty and the welcome screen was shown). `None` otherwise.
    """

    input: ChatInputResult
    selected_example: Optional[str] = None


# =============================================================================
# CONSTANTS
# =============================================================================

# Default example questions shown on the welcome screen. Callers may
# override these via `render_welcome(example_questions=...)`.
DEFAULT_EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "Which doctor should I consult for chest pain?",
    "What are the visiting hours?",
    "Where is Cardiology?",
    "How can I book an appointment?",
)

# Widget keys used for the input-area controls, centralized so they are
# never duplicated or typo'd across the module.
_INPUT_TEXT_KEY = "chat_input_text_area"
_ASK_BUTTON_KEY = "chat_input_ask_button"
_CLEAR_BUTTON_KEY = "chat_input_clear_button"
_EXAMPLE_BUTTON_KEY_PREFIX = "chat_welcome_example_"


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _format_timestamp(timestamp: Union[str, datetime]) -> str:
    """
    Normalize a timestamp value into a short display string.

    Args:
        timestamp: Either an already-formatted string (returned as-is,
            e.g. "10:42 AM") or a `datetime` object (formatted as
            "%I:%M %p", e.g. "10:42 AM").

    Returns:
        A short, human-readable timestamp string. Returns an empty string
        if `timestamp` is falsy.
    """
    if not timestamp:
        return ""
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%I:%M %p").lstrip("0")
    return str(timestamp)


def _format_response_time(response_time_ms: Optional[Union[int, float]]) -> Optional[str]:
    """
    Format a response-time value for display in assistant-message metadata.

    Args:
        response_time_ms: Duration in milliseconds, or `None`.

    Returns:
        A string like "1.2s" (for values >= 1000 ms) or "480ms", or
        `None` if `response_time_ms` is `None`.
    """
    if response_time_ms is None:
        return None
    if response_time_ms >= 1000:
        return f"{response_time_ms / 1000:.1f}s"
    return f"{response_time_ms:.0f}ms"


def _build_message_row(avatar_kind: str, bubble_html: str, align_right: bool) -> None:
    """
    Render one chat row (avatar + bubble) using a two-column layout.

    Streamlit wraps every `st.markdown` call in its own block-level
    container, so placing an avatar and a bubble truly "inline" inside a
    single flex row requires either raw nested HTML (risking double
    rendering if `render_avatar` already renders as a side effect) or
    Streamlit's own column primitives. This helper uses `st.columns`,
    which is the safer, composition-friendly choice and keeps this module
    from having to assume implementation details of `ui.components`
    functions beyond their public parameters.

    Args:
        avatar_kind: Forwarded to `render_avatar` (e.g. "user",
            "assistant").
        bubble_html: Pre-built HTML for the message bubble (produced by
            `_build_bubble_html`), rendered via a single `st.markdown`
            call inside the wider column.
        align_right: Reserved for future right-aligned layouts. Currently
            both roles use the same avatar-then-bubble column order,
            distinguished visually by bubble color and avatar glyph
            instead of side placement, which keeps the layout simple and
            robust at narrow (mobile) widths.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    avatar_col, bubble_col = st.columns([1, 11], gap="small")
    with avatar_col:
        render_avatar(kind=avatar_kind, size="sm")
    with bubble_col:
        st.markdown(bubble_html, unsafe_allow_html=True)


def _build_bubble_html(role: MessageRole, content: str) -> str:
    """
    Build the HTML for a single message bubble.

    Reuses the `.chat-bubble` / `.chat-bubble--user` / `.chat-bubble--assistant`
    classes already defined in the locked `ui/styles.py` — this function
    does not define any new CSS, it only references existing class names,
    exactly as `ui/layout.py` and `ui/sidebar.py` already do for their own
    structural containers (e.g. `.app-card`, `.sidebar-item`).

    Args:
        role: "user" or "assistant" — selects the bubble's color variant.
        content: The message text to display inside the bubble.

    Returns:
        An HTML string for the bubble `<div>`.
    """
    variant_class = "chat-bubble--user" if role == "user" else "chat-bubble--assistant"
    return f'<div class="chat-bubble {variant_class}">{content}</div>'


def _build_metadata_tags(
    confidence: Optional[float],
    source_count: Optional[int],
    response_time_ms: Optional[Union[int, float]],
) -> None:
    """
    Render an assistant message's optional metadata row (confidence,
    source count, response time) as a set of small tags.

    Args:
        confidence: Optional overall answer confidence, 0.0-1.0.
        source_count: Optional number of retrieved sources.
        response_time_ms: Optional generation time, in milliseconds.

    Returns:
        None. Renders directly into the Streamlit app via `render_tag`.
        Does nothing if all three values are `None`.
    """
    formatted_time = _format_response_time(response_time_ms)
    if confidence is None and source_count is None and formatted_time is None:
        return

    if confidence is not None:
        render_tag(f"Confidence: {confidence * 100:.0f}%", variant="info")
    if source_count is not None:
        render_tag(f"{source_count} source{'s' if source_count != 1 else ''}", variant="teal")
    if formatted_time is not None:
        render_tag(f"\u26A1 {formatted_time}", variant="neutral")


# =============================================================================
# SECTION 1 — CHAT HEADER
# =============================================================================


def render_chat_header(metadata: Optional[ConversationMetadata] = None) -> None:
    """
    Render the chat panel header.

    Displays the "AI Assistant" title, the "Ask hospital-related
    questions" subtitle, a live AI status badge, and the current
    conversation's message count.

    Args:
        metadata: Conversation summary info (message count, AI status).
            Defaults to an empty/offline `ConversationMetadata` if not
            supplied.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    resolved_metadata = metadata if metadata is not None else ConversationMetadata()

    render_section_header(
        title="AI Assistant",
        subtitle="Ask hospital-related questions",
        icon="\U0001F916",
    )

    status_col, count_col = st.columns([1, 1])
    with status_col:
        render_status_badge(
            label=resolved_metadata.ai_status.title(),
            status=resolved_metadata.ai_status,
        )
    with count_col:
        render_tag(
            f"{resolved_metadata.message_count} message"
            f"{'s' if resolved_metadata.message_count != 1 else ''}",
            variant="neutral",
        )


# =============================================================================
# SECTION 2 — WELCOME SCREEN
# =============================================================================


def render_welcome(
    example_questions: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    Render the welcome/empty-state screen, shown when no conversation
    exists yet.

    Displays a hospital-assistant icon, a welcome message, and a set of
    clickable example questions the user can use to start the
    conversation.

    Args:
        example_questions: Optional custom list of example questions.
            Defaults to `DEFAULT_EXAMPLE_QUESTIONS`.

    Returns:
        The text of the example question the user clicked this run, or
        `None` if none was clicked. This module does not act on the
        selection — the caller decides whether to pre-fill the input box
        or submit it directly.
    """
    questions = example_questions if example_questions is not None else DEFAULT_EXAMPLE_QUESTIONS

    render_empty_state(
        title="Welcome to the Hospital Information Assistant",
        subtitle=(
            "Ask me anything about doctors, departments, appointments, "
            "or hospital services — I'll find the answer for you."
        ),
        icon="\U0001F3E5",
    )

    st.markdown(
        '<p style="text-align:center; font-weight:600; margin:0.5rem 0;">Try asking:</p>',
        unsafe_allow_html=True,
    )

    selected: Optional[str] = None
    for index, question in enumerate(questions):
        clicked = st.button(
            f"\U0001F4AC {question}",
            key=f"{_EXAMPLE_BUTTON_KEY_PREFIX}{index}",
            use_container_width=True,
        )
        if clicked:
            selected = question

    return selected


# =============================================================================
# SECTION 3-5 — CONVERSATION MESSAGES
# =============================================================================


def render_user_message(message: ChatMessage) -> None:
    """
    Render a single user message: avatar, message content, and timestamp.

    Args:
        message: The `ChatMessage` to render. Its `role` field is not
            checked here — callers are expected to route messages to
            `render_user_message` / `render_assistant_message` themselves
            (see `render_chat`'s dispatch loop) — but the content is
            always rendered as a user-styled bubble regardless.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    bubble_html = _build_bubble_html("user", message.content)
    _build_message_row(avatar_kind="user", bubble_html=bubble_html, align_right=True)

    timestamp_text = _format_timestamp(message.timestamp)
    if timestamp_text:
        render_chat_timestamp(timestamp_text)


def render_assistant_message(message: ChatMessage) -> None:
    """
    Render a single assistant message: avatar, message content,
    timestamp, and optional confidence / source-count / response-time
    metadata and retrieved-source chips.

    Args:
        message: The `ChatMessage` to render.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    bubble_html = _build_bubble_html("assistant", message.content)
    _build_message_row(avatar_kind="assistant", bubble_html=bubble_html, align_right=False)

    timestamp_text = _format_timestamp(message.timestamp)
    if timestamp_text:
        render_chat_timestamp(timestamp_text)

    _build_metadata_tags(
        confidence=message.confidence,
        source_count=message.source_count,
        response_time_ms=message.response_time_ms,
    )

    if message.sources:
        render_sources(message.sources, ranking_method=message.ranking_method)


# =============================================================================
# SECTION 6 — RETRIEVED SOURCES
# =============================================================================


def render_sources(
    sources: Sequence[SourceDocument],
    ranking_method: Optional[str] = None,
) -> None:
    """
    Render the retrieved-source citations attached to an assistant
    message.

    Displays one source chip per document (name, document type, and
    confidence score), plus the ranking method used, if supplied.

    Args:
        sources: The `SourceDocument` citations to display. Does nothing
            if empty.
        ranking_method: Optional human-readable ranking/similarity method
            name (e.g. "Cosine Similarity").

    Returns:
        None. Renders directly into the Streamlit app via
        `render_source_chip` for each document.
    """
    if not sources:
        return

    st.markdown(
        '<p style="font-size:0.8rem; color:#94A3B8; margin:0.4rem 0 0.2rem 0;">'
        "\U0001F4DA Sources</p>",
        unsafe_allow_html=True,
    )
    for source in sources:
        render_source_chip(
            document_name=source.name,
            document_type=source.document_type,
            score=source.score,
        )

    if ranking_method:
        render_key_value({"Ranking Method": ranking_method})


# =============================================================================
# SECTION 7 — TYPING INDICATOR
# =============================================================================


def render_typing_indicator() -> None:
    """
    Render the "assistant is generating a response" typing indicator.

    Displays an assistant avatar next to the existing three-dot typing
    animation from `ui.components` / `ui.styles`. Callers should show
    this while waiting for the backend `RAGPipeline` to return a result,
    and remove it once the real assistant message is available.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    avatar_col, indicator_col = st.columns([1, 11], gap="small")
    with avatar_col:
        render_avatar(kind="assistant", size="sm")
    with indicator_col:
        render_loading(mode="typing")


# =============================================================================
# SECTION 8 — INPUT AREA
# =============================================================================


def render_input_box(
    placeholder: str = "Ask about hospital services...",
    disabled: bool = False,
) -> ChatInputResult:
    """
    Render the message input area: a text input plus "Ask" and "Clear"
    buttons.

    This function only renders UI and reports what happened — it never
    processes the entered text, calls the RAG pipeline, or mutates any
    conversation state. The caller (`app.py`) is responsible for reading
    `ChatInputResult` and deciding what to do.

    Args:
        placeholder: Placeholder text shown in the empty input box.
        disabled: If True, disables the text input and both buttons
            (e.g. while a response is already being generated).

    Returns:
        A `ChatInputResult` describing the current input text and
        whether "Ask" or "Clear" was clicked on this run.
    """
    text_value = st.text_input(
        label="Message",
        placeholder=placeholder,
        key=_INPUT_TEXT_KEY,
        disabled=disabled,
        label_visibility="collapsed",
    )

    ask_col, clear_col = st.columns([3, 1])
    with ask_col:
        ask_clicked = st.button(
            "\U0001F4E4 Ask",
            key=_ASK_BUTTON_KEY,
            use_container_width=True,
            disabled=disabled,
            type="primary",
        )
    with clear_col:
        clear_clicked = st.button(
            "\U0001F5D1\uFE0F Clear",
            key=_CLEAR_BUTTON_KEY,
            use_container_width=True,
            disabled=disabled,
        )

    return ChatInputResult(
        text=text_value or "",
        ask_clicked=ask_clicked,
        clear_clicked=clear_clicked,
    )


# =============================================================================
# SECTION 9 — CONVERSATION FOOTER
# =============================================================================


def render_chat_footer() -> None:
    """
    Render the conversation footer: project attribution and a medical
    disclaimer.

    Returns:
        None. Renders directly into the Streamlit app via
        `render_footer_note` and `render_info_panel`.
    """
    render_footer_note("Hospital Information Assistant \u2022 Responses generated using RAG")
    render_info_panel(
        message=(
            "This assistant provides general hospital information and does not "
            "replace professional medical advice. In an emergency, please "
            "contact hospital emergency services immediately."
        ),
        variant="warning",
        title="Medical Disclaimer",
    )


# =============================================================================
# ORCHESTRATOR
# =============================================================================


def render_chat(
    messages: Optional[Sequence[ChatMessage]] = None,
    ai_status: StatusKind = "offline",
    is_generating: bool = False,
    example_questions: Optional[Sequence[str]] = None,
    input_placeholder: str = "Ask about hospital services...",
) -> ChatRenderResult:
    """
    Assemble the complete chat panel in one call.

    Convenience entry point for `app.py`: renders the header, then either
    the welcome screen (if `messages` is empty) or the full conversation
    (dispatching each message to `render_user_message` /
    `render_assistant_message`, plus a typing indicator if
    `is_generating` is True), followed by the input area and footer.

    Typical usage in `app.py`:

        from ui.layout import render_layout
        from ui.chat import render_chat

        columns = render_layout(online=True)
        with columns.chat:
            result = render_chat(
                messages=st.session_state.get("messages", []),
                ai_status="online",
                is_generating=False,
            )
        if result.input.ask_clicked and result.input.text.strip():
            ...  # app.py owns what "ask" does
        if result.input.clear_clicked:
            ...  # app.py owns what "clear" does

    Args:
        messages: The full conversation history so far, in chronological
            order. An empty/`None` sequence shows the welcome screen
            instead.
        ai_status: Forwarded to `render_chat_header` via
            `ConversationMetadata`.
        is_generating: If True, renders a typing indicator after the
            last message and disables the input area.
        example_questions: Forwarded to `render_welcome`.
        input_placeholder: Forwarded to `render_input_box`.

    Returns:
        A `ChatRenderResult` combining this run's input-area interaction
        and (if the welcome screen was shown) any example question the
        user clicked.
    """
    resolved_messages = list(messages) if messages else []

    render_chat_header(
        metadata=ConversationMetadata(
            message_count=len(resolved_messages),
            ai_status=ai_status,
        )
    )
    render_divider()

    selected_example: Optional[str] = None
    if not resolved_messages:
        selected_example = render_welcome(example_questions=example_questions)
    else:
        for message in resolved_messages:
            if message.role == "user":
                render_user_message(message)
            else:
                render_assistant_message(message)

        if is_generating:
            render_typing_indicator()

    render_divider()
    input_result = render_input_box(
        placeholder=input_placeholder,
        disabled=is_generating,
    )
    render_chat_footer()

    return ChatRenderResult(input=input_result, selected_example=selected_example)