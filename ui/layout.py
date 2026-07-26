"""
ui/layout.py
=============================================================================
Page shell and layout scaffolding for the Intelligent Hospital Information
Assistant frontend.

This module owns ONLY structural page concerns:
    - Streamlit page configuration (title, icon, layout mode)
    - The top application header (branding, title, subtitle, status badge)
    - The three-column main layout skeleton (sidebar / chat / insights)
    - The application footer

It does NOT own:
    - Sidebar content or navigation logic       -> ui/sidebar.py
    - Chat rendering or conversation state      -> ui/chat.py
    - Metrics / analytics rendering             -> ui/metrics.py
    - RAG pipeline invocation                   -> backend modules

All visual styling (colors, typography, spacing, shadows, gradients,
animation, responsive breakpoints) comes exclusively from the locked
design system in `ui.styles`. This module never hard-codes a CSS value —
it only injects the pre-built stylesheet via `apply_global_styles()` and
composes small HTML snippets using tokens/helpers exported from
`ui.styles`.

-----------------------------------------------------------------------------
Public API
-----------------------------------------------------------------------------
    configure_page()      -> None
    render_header()        -> None
    render_main_layout()   -> LayoutColumns
    render_footer()         -> None
    render_layout()          -> LayoutColumns   (orchestrator / convenience)

`render_layout()` is the single entry point `app.py` needs: it configures
the page, injects global styles, renders the header, builds the three-
column skeleton, and renders the footer — returning the three column
containers so `sidebar.py`, `chat.py`, and `metrics.py` can render their
own content into the correct region.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from ui.styles import (
    Colors,
    Gradients,
    Spacing,
    apply_global_styles,
    get_status_badge_html,
)

# =============================================================================
# CONSTANTS
# =============================================================================
# Static project metadata shown in the header and footer. Centralized here
# so copy changes don't require hunting through render functions.

PAGE_TITLE: str = "Intelligent Hospital Information Assistant"
PAGE_ICON: str = "🏥"
PROJECT_SUBTITLE: str = "AI-Powered Retrieval-Augmented Generation Assistant"
PROJECT_VERSION: str = "1.0"

# Relative width ratios for the three-column main layout:
# (sidebar-reserved, chat-reserved, insights-reserved).
COLUMN_RATIOS: tuple[float, float, float] = (1.0, 2.4, 1.2)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class LayoutColumns:
    """
    Container holding the three Streamlit column handles produced by
    `render_main_layout()`.

    Downstream modules render into these handles using a `with` block, e.g.:

        columns = render_layout()
        with columns.sidebar:
            render_sidebar(...)   # from ui/sidebar.py
        with columns.chat:
            render_chat(...)      # from ui/chat.py
        with columns.insights:
            render_metrics(...)   # from ui/metrics.py

    Attributes:
        sidebar: Left column, reserved for `ui/sidebar.py`.
        chat: Center column, reserved for `ui/chat.py`.
        insights: Right column, reserved for `ui/metrics.py`.
    """

    sidebar: DeltaGenerator
    chat: DeltaGenerator
    insights: DeltaGenerator


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================


def configure_page() -> None:
    """
    Configure Streamlit's page-level settings.

    Must be called exactly once, before any other Streamlit command, per
    Streamlit's `set_page_config` requirement. `render_layout()` calls this
    first automatically; call it directly only if you are assembling the
    page manually instead of using `render_layout()`.

    Sets the browser tab title/icon, requests the wide layout (needed for
    the three-column sidebar/chat/insights skeleton), and starts with the
    native Streamlit sidebar collapsed since navigation lives in the
    custom left column instead.

    Returns:
        None.
    """
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )


# =============================================================================
# HEADER
# =============================================================================


def render_header(
    online: bool = True,
    status_label: str = "System Online",
) -> None:
    """
    Render the top application header.

    Displays, in a single bordered banner (styled via `ui.styles`):
        - A logo/icon mark for the assistant
        - The project title: "Intelligent Hospital Information Assistant"
        - The subtitle: "AI-Powered Retrieval-Augmented Generation Assistant"
        - A live system-status badge (e.g. "System Online" / "System Offline")

    Args:
        online: Whether the backend/RAG pipeline is currently reachable.
            Controls whether the status badge renders as success (green)
            or error (red) styling. Callers (e.g. app.py, after a health
            check against the RAGPipeline) should pass the real status.
        status_label: Text shown inside the status badge. Defaults to
            "System Online"; pass e.g. "System Offline" or
            "Reconnecting..." alongside `online=False` for other states.

    Returns:
        None. Renders directly into the Streamlit app via `st.markdown`.
    """
    status_badge_html = get_status_badge_html(label=status_label, online=online)

    header_html = f"""
    <div class="app-header anim-fade-in">
        <div class="app-header__brand">
            <div class="app-header__logo">{PAGE_ICON}</div>
            <div>
                <p class="app-header__title">{PAGE_TITLE}</p>
                <p class="app-header__subtitle">{PROJECT_SUBTITLE}</p>
            </div>
        </div>
        {status_badge_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


# =============================================================================
# MAIN LAYOUT
# =============================================================================


def render_main_layout() -> LayoutColumns:
    """
    Build the three-column main layout skeleton.

    Creates a wide row split into three regions using Streamlit's native
    `st.columns`, sized by `COLUMN_RATIOS`:

        Left    -> reserved for `ui/sidebar.py` (navigation, KB status, settings)
        Center  -> reserved for `ui/chat.py` (conversation with the RAGPipeline)
        Right   -> reserved for `ui/metrics.py` (retrieval/AI insight panels)

    This function does not render any sidebar, chat, or metrics content
    itself — it only creates the containers and returns them for
    `ui/sidebar.py`, `ui/chat.py`, and `ui/metrics.py` to render into.

    Returns:
        A `LayoutColumns` instance exposing `.sidebar`, `.chat`, and
        `.insights` Streamlit column handles for other modules to render
        into via `with columns.<region>:`.
    """
    left_col, center_col, right_col = st.columns(
        COLUMN_RATIOS,
        gap="medium",
    )

    return LayoutColumns(sidebar=left_col, chat=center_col, insights=right_col)


def _region_placeholder_html(title: str, subtitle: str) -> str:
    """
    Build a small, dismissible-looking placeholder card for an empty
    layout region.

    This is only a structural stand-in: once `sidebar.py`, `chat.py`, or
    `metrics.py` render real content into the returned column, Streamlit
    will simply stack their content below (or, in practice, those modules
    should be the only thing writing into the column — see module
    docstring). Kept intentionally minimal and on-brand via the shared
    `.app-card` styling from `ui.styles`.

    Args:
        title: Short label naming the region (e.g. "Navigation").
        subtitle: One line describing which module owns the region.

    Returns:
        An HTML string using the `.app-card--flat` design-system class.
    """
    return f"""
    <div class="app-card app-card--flat" style="text-align:center; color:{Colors.TEXT_MUTED};">
        <p class="app-card__title" style="font-size:1rem;">{title}</p>
        <p class="app-card__body">{subtitle}</p>
    </div>
    """


# =============================================================================
# FOOTER
# =============================================================================


def render_footer() -> None:
    """
    Render the application footer.

    Displays project/version metadata and technology attribution:
        - Project Version 1.0
        - Powered by Google Gemini
        - Retrieval-Augmented Generation (RAG)
        - Built using Streamlit

    Uses the shared `.app-card` / utility classes from `ui.styles` for a
    horizontally-centered row of small, muted labels separated by
    vertical dividers, consistent with the rest of the design system.

    Returns:
        None. Renders directly into the Streamlit app via `st.markdown`.
    """
    footer_items = (
        f"Version {PROJECT_VERSION}",
        "Powered by Google Gemini",
        "Retrieval-Augmented Generation (RAG)",
        "Built using Streamlit",
    )
    separator = '<span class="u-divider-vertical"></span>'
    items_html = separator.join(f"<span>{item}</span>" for item in footer_items)

    footer_html = f"""
    <div class="u-flex u-items-center u-justify-center u-gap-md"
         style="margin-top:{Spacing.XL}; padding-top:{Spacing.MD};
                border-top:1px solid {Colors.BORDER};
                color:{Colors.TEXT_MUTED}; font-size:0.8rem; text-align:center;">
        {items_html}
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


# =============================================================================
# ORCHESTRATOR
# =============================================================================


def render_layout(
    online: bool = True,
    status_label: str = "System Online",
) -> LayoutColumns:
    """
    Assemble the full page shell in one call.

    Convenience entry point for `app.py`: configures the page, injects the
    global design-system stylesheet, renders the header, builds the
    three-column skeleton, and renders the footer — in the correct order.

    Typical usage in `app.py`:

        from ui.layout import render_layout

        columns = render_layout(online=pipeline_is_healthy)
        with columns.sidebar:
            render_sidebar(...)
        with columns.chat:
            render_chat(...)
        with columns.insights:
            render_metrics(...)

    Args:
        online: Forwarded to `render_header` — whether the backend/RAG
            pipeline is currently reachable.
        status_label: Forwarded to `render_header` — text shown in the
            status badge.

    Returns:
        A `LayoutColumns` instance for `sidebar.py`, `chat.py`, and
        `metrics.py` to render their content into.
    """
    configure_page()
    apply_global_styles()
    render_header(online=online, status_label=status_label)
    columns = render_main_layout()
    render_footer()
    return columns