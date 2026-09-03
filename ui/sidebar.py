"""
ui/sidebar.py
=============================================================================
Left navigation panel for the Intelligent Hospital Information Assistant
frontend.

This module renders — and ONLY renders — the application's left sidebar:
branding, navigation, knowledge-base/AI status, quick statistics, quick
action buttons, project information, and a footer.

It does NOT own:
    - Chat rendering or conversation state       -> ui/chat.py
    - RAG pipeline invocation / Gemini calls      -> backend modules
    - Vector store / embedding logic              -> backend modules
    - Metrics calculation                          -> ui/metrics.py
    - Routing / page-switch persistence            -> app.py

Every value shown here (document counts, connection status, statistics)
is supplied by the caller as a plain function argument. This module never
reaches into a database, a session object, or a pipeline to fetch it —
that keeps it a pure, testable, reusable presentation layer.

-----------------------------------------------------------------------------
Layout note
-----------------------------------------------------------------------------
`ui/layout.py` builds a three-column skeleton (`LayoutColumns`) rather than
using Streamlit's native `st.sidebar` (which `configure_page()` starts
collapsed, since navigation lives in this custom left column instead).
`app.py` is expected to call this module's `render_sidebar()` inside
`with columns.sidebar:`, e.g.:

    from ui.layout import render_layout
    from ui.sidebar import render_sidebar

    columns = render_layout(online=True)
    with columns.sidebar:
        render_sidebar(active_page="AI Assistant")

The dark, "console-like" `.sidebar-*` CSS classes used throughout this
module are defined as standalone (non-scoped) rules in `ui/styles.py`
specifically so they render correctly inside this custom column, not only
inside Streamlit's native sidebar element.

-----------------------------------------------------------------------------
Public API
-----------------------------------------------------------------------------
    render_sidebar()        -> None   (orchestrator; call this from app.py)
    render_navigation()     -> None
    render_kb_status()      -> None
    render_ai_status()      -> None
    render_quick_stats()    -> None
    render_quick_actions()  -> Dict[str, bool]
    render_project_info()   -> None
    render_sidebar_footer() -> None
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Union

import streamlit as st

from ui.components import (
    render_divider,
    render_key_value,
    render_status_badge,
    render_tag,
)
from ui.layout import PAGE_ICON, PAGE_TITLE, PROJECT_VERSION

# =============================================================================
# TYPE ALIASES
# =============================================================================

# Mirrors `ui.components.StatusKind` — kept as a local alias (rather than an
# import) since it is a typing-only construct, not a reusable function or a
# design token, and this keeps sidebar.py's imports limited to exactly what
# the brief asks for: functions from ui.components, tokens from ui.styles.
StatusKind = Literal["online", "offline", "warning", "processing", "error"]
StatValue = Union[int, str]


# =============================================================================
# CONSTANTS
# =============================================================================
# Static sidebar copy and configuration. Centralized here so navigation
# labels/icons or default statistics don't get scattered across render
# functions.


@dataclass(frozen=True)
class NavItem:
    """
    A single sidebar navigation entry.

    Attributes:
        label: Display label shown to the user (also used to match
            against `active_page` for highlighting).
        icon: A single emoji/character shown to the left of the label.
    """

    label: str
    icon: str


# The six top-level pages this application exposes. Order here is the
# order they render in the sidebar.
NAV_ITEMS: tuple[NavItem, ...] = (
    NavItem(label="AI Assistant", icon="💬"),
    NavItem(label="Voice Assistant", icon="🎤"),
    NavItem(label="Medical Report Analysis", icon="📋"),
)

# Default technology-stack tags shown in the "Project Information" section.
DEFAULT_TECH_STACK: tuple[str, ...] = (
    "Python",
    "Streamlit",
    "Google Gemini",
    "RAG",
    "Vector Database",
)


# =============================================================================
# SECTION 1 — LOGO / BRAND
# =============================================================================


def _render_logo(project_name: str, version: str, icon: str) -> None:
    """
    Render the small application logo block at the top of the sidebar.

    Displays the hospital icon, the project name, and the current
    version, reusing `PAGE_ICON` / `PAGE_TITLE` / `PROJECT_VERSION` from
    the locked `ui/layout.py` so the sidebar never drifts out of sync
    with the main page header's branding.

    Args:
        project_name: Project/application name to display.
        version: Version string to display (e.g. "1.0").
        icon: Emoji/character shown inside the logo mark.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    markup = f"""
    <div class="u-flex u-items-center u-gap-md" style="margin-bottom: 0.25rem;">
        <div class="app-header__logo" style="width:36px;height:36px;font-size:1.1rem;">
            {icon}
        </div>
        <div style="min-width:0;">
            <p class="sidebar-card__title" style="margin:0;font-size:0.95rem;">
                {project_name}
            </p>
            <p class="sidebar-project-info" style="margin:0;">
                Version {version}
            </p>
        </div>
    </div>
    """
    st.markdown(markup, unsafe_allow_html=True)


# =============================================================================
# SECTION 2 — NAVIGATION
# =============================================================================


def render_navigation(
    active_page: str = "AI Assistant",
    items: Optional[Sequence[NavItem]] = None,
) -> None:
    """
    Render the sidebar's main navigation list.

    Purely presentational: highlights whichever entry matches
    `active_page` using the `.sidebar-item--active` modifier class
    already defined in `ui/styles.py`. This function does not read or
    write any session state and does not perform routing — the caller
    (`app.py`) owns deciding what "active" means and what happens on
    navigation.

    Args:
        active_page: Label of the currently active page. Compared
            case-insensitively against each item's label.
        items: Optional custom list of `NavItem`s. Defaults to the
            application's standard six-page `NAV_ITEMS`.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    nav_items = items if items is not None else NAV_ITEMS
    normalized_active = active_page.strip().lower()

    st.markdown('<p class="sidebar-heading">Navigation</p>', unsafe_allow_html=True)

    option_labels = [item.label for item in nav_items]
    icon_map = {item.label: item.icon for item in nav_items}
    selected_index = next((index for index, item in enumerate(nav_items) if item.label.strip().lower() == normalized_active), 0)

    selected_page = st.radio(
        label="Navigation",
        options=option_labels,
        key="sidebar_active_page",
        label_visibility="collapsed",
        format_func=lambda label: f"{icon_map.get(label, '')} {label}".strip(),
    )


# =============================================================================
# SECTION 3 — KNOWLEDGE BASE STATUS
# =============================================================================


def render_kb_status(
    documents_loaded: StatValue = "\u2014",
    vector_store_status: StatusKind = "offline",
    embedding_model: str = "Not configured",
    last_updated: str = "Never",
) -> None:
    """
    Render the "Knowledge Base Status" sidebar panel.

    Shows how many documents are indexed, whether the vector store is
    reachable, which embedding model is configured, and when the
    knowledge base was last refreshed. All values are supplied by the
    caller — this function performs no vector-store or file-system
    access of its own.

    Args:
        documents_loaded: Number of documents currently indexed (int or
            a pre-formatted string, e.g. "128" or "\u2014" when unknown).
        vector_store_status: One of "online", "offline", "warning",
            "processing", "error" — passed straight through to
            `render_status_badge`.
        embedding_model: Name of the configured embedding model
            (e.g. "text-embedding-004").
        last_updated: Pre-formatted recency string (e.g. "2 minutes ago").

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown('<p class="sidebar-heading">Knowledge Base Status</p>', unsafe_allow_html=True)

    render_key_value(
        {
            "Documents Loaded": str(documents_loaded),
            "Embedding Model": embedding_model,
            "Last Updated": last_updated,
        }
    )
    render_status_badge(label=f"Vector Store: {vector_store_status.title()}", status=vector_store_status)


# =============================================================================
# SECTION 4 — AI STATUS
# =============================================================================


def render_ai_status(
    gemini_status: StatusKind = "offline",
    retriever_status: StatusKind = "offline",
    rag_pipeline_status: StatusKind = "offline",
    connection_status: StatusKind = "offline",
) -> None:
    """
    Render the "AI Status" sidebar panel.

    Summarizes the health of the four moving parts behind the assistant:
    the Gemini API connection, the document retriever, the overall RAG
    pipeline, and the general backend connection. This function only
    displays the statuses it is given — it performs no health checks or
    API calls itself.

    Args:
        gemini_status: One of "online", "offline", "warning",
            "processing", "error".
        retriever_status: Same status vocabulary as `gemini_status`.
        rag_pipeline_status: Same status vocabulary as `gemini_status`.
        connection_status: Same status vocabulary as `gemini_status`;
            represents the overall backend connection health.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown('<p class="sidebar-heading">AI Status</p>', unsafe_allow_html=True)

    status_rows = (
        ("Gemini", gemini_status),
        ("Retriever", retriever_status),
        ("RAG Pipeline", rag_pipeline_status),
        ("Connection", connection_status),
    )
    for row_label, row_status in status_rows:
        row_markup = (
            '<div class="sidebar-card__row" style="align-items:center;">'
            f"<span>{row_label}</span><span></span>"
            "</div>"
        )
        st.markdown(row_markup, unsafe_allow_html=True)
        render_status_badge(label=row_status.title(), status=row_status)


# =============================================================================
# SECTION 5 — QUICK STATISTICS
# =============================================================================


def render_quick_stats(
    doctors: StatValue = 0,
    departments: StatValue = 0,
    diseases: StatValue = 0,
    medicines: StatValue = 0,
    appointments: StatValue = 0,
) -> None:
    """
    Render the "Quick Statistics" sidebar panel.

    A compact key-value summary of the hospital knowledge base's scale
    (how many doctors, departments, diseases, medicines, and
    appointments are known to the system). All counts are supplied by
    the caller.

    Args:
        doctors: Number of doctors on record.
        departments: Number of hospital departments on record.
        diseases: Number of disease/condition entries on record.
        medicines: Number of medicine entries on record.
        appointments: Number of appointments on record.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    st.markdown('<p class="sidebar-heading">Quick Statistics</p>', unsafe_allow_html=True)

    render_key_value(
        {
            "Doctors": str(doctors),
            "Departments": str(departments),
            "Diseases": str(diseases),
            "Medicines": str(medicines),
            "Appointments": str(appointments),
        }
    )


# =============================================================================
# SECTION 6 — QUICK ACTIONS
# =============================================================================


def render_quick_actions() -> Dict[str, bool]:
    """
    Render the "Quick Actions" sidebar buttons.

    These are UI controls ONLY — pressing a button here does not refresh
    a knowledge base, reload a vector store, clear a chat, or export
    anything. This function simply renders four `st.button`s and returns
    which one (if any) was pressed on this run, leaving the caller
    (`app.py` / future backend wiring) fully responsible for acting on
    that signal.

    Returns:
        A dict with keys "refresh_knowledge_base", "reload_vector_store",
        "clear_chat", "export_chat", each mapped to `True` if that
        button was clicked on this run, `False` otherwise.
    """
    st.markdown('<p class="sidebar-heading">Quick Actions</p>', unsafe_allow_html=True)

    refresh_clicked = st.button(
        "\U0001F504 Refresh Knowledge Base",
        key="sidebar_action_refresh_kb",
        use_container_width=True,
    )
    
    
    clear_clicked = st.button(
        "\U0001F5D1\uFE0F Clear Chat",
        key="sidebar_action_clear_chat",
        use_container_width=True,
    )
   

    return {
      "refresh_knowledge_base": refresh_clicked,
      "clear_chat": clear_clicked,
}


# =============================================================================
# SECTION 7 — PROJECT INFORMATION
# =============================================================================


def render_project_info(
    project_name: str = "Intelligent Hospital Information Assistant",
    version: Optional[str] = None,
    college: str = "Add College / Institution Name",
    tech_stack: Optional[Sequence[str]] = None,
) -> None:
    """
    Render the "Project Information" sidebar panel.

    Displays static research-project metadata: project name, version,
    institution, and the technology stack (as tags).

    Args:
        project_name: Full project/research title.
        version: Version string. Defaults to `PROJECT_VERSION` imported
            from the locked `ui/layout.py` so both modules always agree.
        college: Institution/college name to credit.
        tech_stack: Optional sequence of technology names rendered as
            tags. Defaults to `DEFAULT_TECH_STACK`.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    resolved_version = version if version is not None else PROJECT_VERSION
    stack = tech_stack if tech_stack is not None else DEFAULT_TECH_STACK

    st.markdown('<p class="sidebar-heading">Project Information</p>', unsafe_allow_html=True)

    render_key_value(
        {
            "Research Project": project_name,
            "Version": resolved_version,
            "College": college,
        }
    )
    for technology in stack:
        render_tag(technology, variant="neutral")


# =============================================================================
# SECTION 8 — SIDEBAR FOOTER
# =============================================================================


def render_sidebar_footer(version: Optional[str] = None) -> None:
    """
    Render the small, muted footer pinned to the bottom of the sidebar.

    Displays attribution ("Made with \u2764\uFE0F", Google Gemini,
    Streamlit) and the current version.

    Args:
        version: Version string to display. Defaults to
            `PROJECT_VERSION` imported from the locked `ui/layout.py`.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    resolved_version = version if version is not None else PROJECT_VERSION

    markup = f"""
    <div class="sidebar-footer">
        <p style="margin:0;">Made with \u2764\uFE0F using Google Gemini &amp; Streamlit</p>
        <p style="margin:0.15rem 0 0 0;">Version {resolved_version}</p>
    </div>
    """
    st.markdown(markup, unsafe_allow_html=True)


# =============================================================================
# ORCHESTRATOR
# =============================================================================


def render_sidebar(
    active_page: str = "AI Assistant",
    documents_loaded: StatValue = "\u2014",
    vector_store_status: StatusKind = "offline",
    embedding_model: str = "Not configured",
    last_updated: str = "Never",
    gemini_status: StatusKind = "offline",
    retriever_status: StatusKind = "offline",
    rag_pipeline_status: StatusKind = "offline",
    connection_status: StatusKind = "offline",
    doctors: StatValue = 0,
    departments: StatValue = 0,
    diseases: StatValue = 0,
    medicines: StatValue = 0,
    appointments: StatValue = 0,
    college: str = "Add College / Institution Name",
    tech_stack: Optional[Sequence[str]] = None,
) -> Dict[str, bool]:
    """
    Assemble the complete sidebar in one call.

    Convenience entry point for `app.py`: renders the logo, navigation,
    knowledge-base status, AI status, quick statistics, quick actions,
    project information, and footer — in the correct order, separated
    by dividers.

    Typical usage in `app.py`:

        from ui.layout import render_layout
        from ui.sidebar import render_sidebar

        columns = render_layout(online=pipeline_is_healthy)
        with columns.sidebar:
            actions = render_sidebar(
                active_page="AI Assistant",
                documents_loaded=128,
                vector_store_status="online",
                embedding_model="text-embedding-004",
                last_updated="2 minutes ago",
                gemini_status="online",
                retriever_status="online",
                rag_pipeline_status="online",
                connection_status="online",
                doctors=24, departments=8, diseases=140,
                medicines=310, appointments=57,
            )
        if actions["clear_chat"]:
            ...  # app.py decides what "clear chat" actually does

    Args:
        active_page: Label of the currently active page, forwarded to
            `render_navigation`.
        documents_loaded: Forwarded to `render_kb_status`.
        vector_store_status: Forwarded to `render_kb_status`.
        embedding_model: Forwarded to `render_kb_status`.
        last_updated: Forwarded to `render_kb_status`.
        gemini_status: Forwarded to `render_ai_status`.
        retriever_status: Forwarded to `render_ai_status`.
        rag_pipeline_status: Forwarded to `render_ai_status`.
        connection_status: Forwarded to `render_ai_status`.
        doctors: Forwarded to `render_quick_stats`.
        departments: Forwarded to `render_quick_stats`.
        diseases: Forwarded to `render_quick_stats`.
        medicines: Forwarded to `render_quick_stats`.
        appointments: Forwarded to `render_quick_stats`.
        college: Forwarded to `render_project_info`.
        tech_stack: Forwarded to `render_project_info`.

    Returns:
        The dict returned by `render_quick_actions()` — which quick
        action button (if any) was clicked on this run.
    """
    _render_logo(project_name=PAGE_TITLE, version=PROJECT_VERSION, icon=PAGE_ICON)
    render_divider()

    render_navigation(active_page=active_page)
    render_divider()

    actions = render_quick_actions()
   
    return actions