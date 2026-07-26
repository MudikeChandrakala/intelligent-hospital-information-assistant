"""
ui/components.py
=============================================================================
Reusable UI component library for the Intelligent Hospital Information
Assistant frontend.

This module is the single place where small, composable, presentation-only
building blocks live: section headers, cards, badges, metric tiles, source
chips, doctor/department cards, empty states, loading indicators, avatars,
info panels, dividers, progress bars, confidence badges, tags, key-value
rows, chat timestamps, and footer notes.

Think of this as a small component library in the spirit of Material UI /
Ant Design / Fluent UI, purpose-built for a healthcare AI SaaS product and
rendered through Streamlit's `st.markdown(..., unsafe_allow_html=True)`.

-----------------------------------------------------------------------------
What this module does NOT do
-----------------------------------------------------------------------------
    - No sidebar navigation logic                -> ui/sidebar.py
    - No chat rendering / conversation state      -> ui/chat.py
    - No RAG pipeline invocation                  -> backend modules
    - No metrics calculation                      -> ui/metrics.py
    - No `st.session_state` business logic (other than the one internal
      flag used to inject component CSS exactly once per session)

Every function here takes plain data in and renders (or returns) HTML out.
Callers own the data; this module only owns presentation.

-----------------------------------------------------------------------------
Design tokens
-----------------------------------------------------------------------------
All colors, spacing, radii, shadows, typography, gradients, and animation
timings are imported from the locked `ui.styles` module. Nothing in this
file hard-codes a hex value, pixel size, or timing curve — every visual
value traces back to a token defined in `ui/styles.py`.

Some components (status badge, doctor/department cards, avatars, tags,
key-value rows, dividers-with-titles, progress bars, info panels, section
headers) need CSS classes that don't already exist in the locked
stylesheet. Those extra rules are defined in this module (still built
entirely from `ui.styles` tokens) and injected once per session via
`_inject_component_styles()`.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import html as _html_lib
import textwrap
from datetime import datetime
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import streamlit as st

from ui.styles import (
    Animation,
    Colors,
    FontSize,
    Gradients,
    Radius,
    Shadow,
    Spacing,
    Typography,
    get_badge_html,
    get_empty_state_html,
    get_skeleton_card_html,
    get_spinner_html,
)

# =============================================================================
# TYPE ALIASES
# =============================================================================

CardVariant = Literal["default", "flat", "accent", "gradient", "borderless"]
StatusKind = Literal["online", "offline", "warning", "processing", "error"]
TrendDirection = Literal["up", "down", "flat"]
LoadingMode = Literal["spinner", "skeleton", "typing"]
AvatarKind = Literal["ai", "user", "doctor", "hospital", "custom"]
InfoVariant = Literal["info", "success", "warning", "error"]
TagVariant = Literal["default", "primary", "teal", "neutral", "success", "warning", "error"]
KeyValuePairs = Union[Dict[str, str], Sequence[Tuple[str, str]]]

# Session-state key used to ensure the extra component stylesheet is
# injected only once per browser session (mirrors the pattern used by
# `apply_global_styles()` in ui/styles.py, but scoped to this module so it
# never collides with sidebar/chat/metrics session state).
_COMPONENT_CSS_FLAG: str = "_ui_components_css_injected"


# =============================================================================
# SECTION 1 — EXTRA COMPONENT CSS
# =============================================================================
# The locked ui/styles.py already ships CSS for cards, badges, metrics,
# chat bubbles, loading, and buttons. The helpers below add the handful of
# classes this module needs that styles.py does not already define
# (status badge, section header, doctor/department cards, avatars, info
# panels, dividers-with-titles, progress bars, tags, key-value rows, and
# footer notes) — built exclusively from styles.py tokens.


def _section_header_css() -> str:
    """Return CSS for `render_section_header()`."""
    return f"""
    .comp-section-header {{
        display: flex;
        align-items: flex-start;
        gap: {Spacing.MD};
        margin-bottom: {Spacing.LG};
    }}
    .comp-section-header__icon {{
        flex-shrink: 0;
        width: 40px;
        height: 40px;
        border-radius: {Radius.MD};
        background: {Gradients.HEADER};
        color: {Colors.TEXT_ON_PRIMARY};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: {FontSize.LG};
    }}
    .comp-section-header__text {{
        min-width: 0;
    }}
    .comp-section-header__title {{
        margin: 0;
        font-size: {FontSize.XL};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_PRIMARY};
        line-height: {Typography.LINE_HEIGHT_TIGHT};
    }}
    .comp-section-header__subtitle {{
        margin: 0.2rem 0 0 0;
        font-size: {FontSize.SM};
        color: {Colors.TEXT_SECONDARY};
    }}
    .comp-section-header__description {{
        margin: 0.5rem 0 0 0;
        font-size: {FontSize.SM};
        color: {Colors.TEXT_MUTED};
        line-height: {Typography.LINE_HEIGHT_RELAXED};
    }}
    """


def _card_extra_css() -> str:
    """Return CSS for the `render_card()` variants not already in styles.py."""
    return f"""
    .app-card--borderless {{
        border: none;
        box-shadow: none;
        padding-left: 0;
        padding-right: 0;
    }}
    .app-card--borderless:hover {{
        transform: none;
        box-shadow: none;
    }}
    .comp-card__header {{
        display: flex;
        align-items: center;
        gap: {Spacing.SM};
        margin-bottom: {Spacing.SM};
    }}
    .comp-card__icon {{
        font-size: {FontSize.LG};
        flex-shrink: 0;
    }}
    .comp-card__footer {{
        margin-top: {Spacing.MD};
        padding-top: {Spacing.SM};
        border-top: 1px solid {Colors.BORDER};
        font-size: {FontSize.XS};
        color: {Colors.TEXT_MUTED};
    }}
    """


def _status_badge_css() -> str:
    """Return CSS for `render_status_badge()` (online/offline/warning/processing/error)."""
    return f"""
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        border-radius: {Radius.PILL};
        font-size: {FontSize.XS};
        font-weight: {Typography.WEIGHT_MEDIUM};
        border: 1px solid transparent;
        white-space: nowrap;
    }}
    .status-badge__dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        display: inline-block;
        flex-shrink: 0;
    }}
    .status-badge--processing .status-badge__dot {{
        animation: status-pulse 1.2s {Animation.EASE_IN_OUT} infinite;
    }}
    @keyframes status-pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.4; transform: scale(0.75); }}
    }}
    """


def _doctor_department_card_css() -> str:
    """Return CSS for `render_doctor_card()` and `render_department_card()`."""
    return f"""
    .comp-doctor-card {{
        background: {Colors.SURFACE};
        border: 1px solid {Colors.BORDER};
        border-radius: {Radius.LG};
        padding: {Spacing.LG};
        box-shadow: {Shadow.SM};
        transition: {Animation.CARD_LIFT};
    }}
    .comp-doctor-card:hover {{
        box-shadow: {Shadow.MD};
        transform: translateY(-2px);
    }}
    .comp-doctor-card__header {{
        display: flex;
        align-items: center;
        gap: {Spacing.MD};
        margin-bottom: {Spacing.MD};
    }}
    .comp-doctor-card__name {{
        margin: 0;
        font-size: {FontSize.MD};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_PRIMARY};
    }}
    .comp-doctor-card__department {{
        margin: 0.15rem 0 0 0;
        font-size: {FontSize.SM};
        color: {Colors.TEAL_DARK};
        font-weight: {Typography.WEIGHT_MEDIUM};
    }}
    .comp-doctor-card__rating {{
        margin-left: auto;
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-size: {FontSize.SM};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.WARNING_DARK};
        flex-shrink: 0;
    }}
    .comp-doctor-card__stats {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: {Spacing.SM} {Spacing.MD};
        margin-bottom: {Spacing.MD};
    }}
    .comp-doctor-card__stat-label {{
        font-size: {FontSize.XS};
        color: {Colors.TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin: 0 0 0.15rem 0;
    }}
    .comp-doctor-card__stat-value {{
        font-size: {FontSize.SM};
        color: {Colors.TEXT_PRIMARY};
        font-weight: {Typography.WEIGHT_MEDIUM};
        margin: 0;
    }}
    .comp-doctor-card__footer {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: {Spacing.SM};
        padding-top: {Spacing.SM};
        border-top: 1px solid {Colors.BORDER};
    }}

    .comp-dept-card {{
        background: {Colors.SURFACE};
        border: 1px solid {Colors.BORDER};
        border-left: 4px solid {Colors.PRIMARY};
        border-radius: {Radius.LG};
        padding: {Spacing.LG};
        box-shadow: {Shadow.SM};
        transition: {Animation.CARD_LIFT};
    }}
    .comp-dept-card:hover {{
        box-shadow: {Shadow.MD};
        transform: translateY(-2px);
    }}
    .comp-dept-card__title {{
        margin: 0 0 0.35rem 0;
        font-size: {FontSize.MD};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_PRIMARY};
    }}
    .comp-dept-card__description {{
        margin: 0 0 {Spacing.MD} 0;
        font-size: {FontSize.SM};
        color: {Colors.TEXT_SECONDARY};
        line-height: {Typography.LINE_HEIGHT_RELAXED};
    }}
    .comp-dept-card__footer {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: {Spacing.SM};
        padding-top: {Spacing.SM};
        border-top: 1px solid {Colors.BORDER};
        font-size: {FontSize.SM};
        color: {Colors.TEXT_SECONDARY};
    }}
    """


def _avatar_css() -> str:
    """Return CSS for `render_avatar()`."""
    return f"""
    .comp-avatar {{
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: {Colors.TEXT_ON_PRIMARY};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        flex-shrink: 0;
        line-height: 1;
    }}
    .comp-avatar--sm {{ width: 28px; height: 28px; font-size: {FontSize.XS}; }}
    .comp-avatar--md {{ width: 40px; height: 40px; font-size: {FontSize.SM}; }}
    .comp-avatar--lg {{ width: 56px; height: 56px; font-size: {FontSize.LG}; }}
    .comp-avatar--ai {{ background: {Gradients.TEAL}; }}
    .comp-avatar--user {{ background: {Gradients.PRIMARY}; }}
    .comp-avatar--doctor {{ background: {Gradients.HEADER}; }}
    .comp-avatar--hospital {{ background: {Colors.SIDEBAR_BG}; }}
    .comp-avatar--custom {{ background: {Gradients.PRIMARY}; }}
    """


def _info_panel_css() -> str:
    """Return CSS for `render_info_panel()`."""
    return f"""
    .comp-info-panel {{
        display: flex;
        align-items: flex-start;
        gap: {Spacing.SM};
        padding: {Spacing.MD};
        border-radius: {Radius.MD};
        border: 1px solid transparent;
        font-size: {FontSize.SM};
        line-height: {Typography.LINE_HEIGHT_RELAXED};
        margin-bottom: {Spacing.MD};
    }}
    .comp-info-panel__icon {{
        flex-shrink: 0;
        font-size: {FontSize.MD};
        line-height: 1;
    }}
    .comp-info-panel__title {{
        margin: 0 0 0.2rem 0;
        font-weight: {Typography.WEIGHT_SEMIBOLD};
    }}
    .comp-info-panel__message {{
        margin: 0;
    }}
    .comp-info-panel--info {{
        background: {Colors.INFO_BG}; border-color: {Colors.INFO_BORDER}; color: {Colors.INFO_DARK};
    }}
    .comp-info-panel--success {{
        background: {Colors.SUCCESS_BG}; border-color: {Colors.SUCCESS_BORDER}; color: {Colors.SUCCESS_DARK};
    }}
    .comp-info-panel--warning {{
        background: {Colors.WARNING_BG}; border-color: {Colors.WARNING_BORDER}; color: {Colors.WARNING_DARK};
    }}
    .comp-info-panel--error {{
        background: {Colors.ERROR_BG}; border-color: {Colors.ERROR_BORDER}; color: {Colors.ERROR_DARK};
    }}
    """


def _divider_css() -> str:
    """Return CSS for `render_divider()` (the optional-title variant)."""
    return f"""
    .comp-divider {{
        display: flex;
        align-items: center;
        gap: {Spacing.SM};
        margin: {Spacing.MD} 0;
    }}
    .comp-divider__line {{
        flex: 1;
        height: 1px;
        background: {Colors.BORDER};
    }}
    .comp-divider__title {{
        font-size: {FontSize.XS};
        font-weight: {Typography.WEIGHT_MEDIUM};
        color: {Colors.TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        white-space: nowrap;
    }}
    """


def _progress_css() -> str:
    """Return CSS for `render_progress()`."""
    return f"""
    .comp-progress {{
        margin-bottom: {Spacing.MD};
    }}
    .comp-progress__label-row {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
        font-size: {FontSize.XS};
        color: {Colors.TEXT_SECONDARY};
    }}
    .comp-progress__track {{
        width: 100%;
        height: 8px;
        border-radius: {Radius.PILL};
        background: {Colors.BORDER};
        overflow: hidden;
    }}
    .comp-progress__fill {{
        height: 100%;
        border-radius: {Radius.PILL};
        background: {Gradients.METRIC};
    }}
    .comp-progress__fill--animated {{
        transition: {Animation.transition("width", duration=Animation.SLOW)};
    }}
    """


def _tag_css() -> str:
    """Return CSS for `render_tag()`."""
    return f"""
    .comp-tag {{
        display: inline-flex;
        align-items: center;
        padding: 0.15rem 0.6rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        border-radius: {Radius.SM};
        font-size: {FontSize.XS};
        font-weight: {Typography.WEIGHT_MEDIUM};
        border: 1px solid transparent;
    }}
    .comp-tag--default {{
        background: {Colors.SURFACE_ALT}; color: {Colors.TEXT_SECONDARY}; border-color: {Colors.BORDER};
    }}
    .comp-tag--primary {{
        background: {Colors.PRIMARY_SOFT}; color: {Colors.PRIMARY_DARK}; border-color: {Colors.PRIMARY_LIGHT};
    }}
    .comp-tag--teal {{
        background: {Colors.TEAL_SOFT}; color: {Colors.TEAL_DARK}; border-color: {Colors.TEAL_LIGHT};
    }}
    .comp-tag--neutral {{
        background: {Colors.SURFACE_HOVER}; color: {Colors.TEXT_MUTED}; border-color: {Colors.BORDER};
    }}
    .comp-tag--success {{
        background: {Colors.SUCCESS_BG}; color: {Colors.SUCCESS_DARK}; border-color: {Colors.SUCCESS_BORDER};
    }}
    .comp-tag--warning {{
        background: {Colors.WARNING_BG}; color: {Colors.WARNING_DARK}; border-color: {Colors.WARNING_BORDER};
    }}
    .comp-tag--error {{
        background: {Colors.ERROR_BG}; color: {Colors.ERROR_DARK}; border-color: {Colors.ERROR_BORDER};
    }}
    """


def _key_value_css() -> str:
    """Return CSS for `render_key_value()`."""
    return f"""
    .comp-kv-list {{
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
        margin-bottom: {Spacing.MD};
    }}
    .comp-kv-row {{
        display: flex;
        justify-content: space-between;
        gap: {Spacing.MD};
        font-size: {FontSize.SM};
        padding-bottom: 0.4rem;
        border-bottom: 1px dashed {Colors.BORDER};
    }}
    .comp-kv-row:last-child {{
        border-bottom: none;
        padding-bottom: 0;
    }}
    .comp-kv-key {{
        color: {Colors.TEXT_MUTED};
        flex-shrink: 0;
    }}
    .comp-kv-value {{
        color: {Colors.TEXT_PRIMARY};
        font-weight: {Typography.WEIGHT_MEDIUM};
        text-align: right;
    }}
    """


def _footer_note_css() -> str:
    """Return CSS for `render_footer_note()`."""
    return f"""
    .comp-footer-note {{
        font-size: {FontSize.XS};
        color: {Colors.TEXT_MUTED};
        text-align: center;
        margin-top: {Spacing.MD};
    }}
    """


def _inject_component_styles() -> None:
    """
    Inject this module's extra CSS classes into the page, exactly once.

    `ui/styles.py` already injects the shared design system via
    `apply_global_styles()` (called once from `ui/layout.py`). This
    function layers on the additional classes this module needs (status
    badges, section headers, doctor/department cards, avatars, info
    panels, titled dividers, progress bars, tags, key-value rows, and
    footer notes) — all built from the same tokens.

    Guarded by a session-state flag so the extra `<style>` block is only
    written to the page once, no matter how many components are rendered.

    Returns:
        None.
    """
    if st.session_state.get(_COMPONENT_CSS_FLAG):
        return

    css_sections = (
        _section_header_css(),
        _card_extra_css(),
        _status_badge_css(),
        _doctor_department_card_css(),
        _avatar_css(),
        _info_panel_css(),
        _divider_css(),
        _progress_css(),
        _tag_css(),
        _key_value_css(),
        _footer_note_css(),
    )
    st.markdown(f"<style>{''.join(css_sections)}</style>", unsafe_allow_html=True)
    st.session_state[_COMPONENT_CSS_FLAG] = True


# =============================================================================
# SECTION 2 — PRIVATE HELPERS
# =============================================================================


def _escape(value: object) -> str:
    """
    HTML-escape a value before interpolating it into a markup string.

    Every piece of caller-supplied text (doctor names, chat content,
    document names, etc.) is routed through this helper before it is
    placed inside an f-string of HTML, so the component library never
    accidentally renders unescaped user/LLM content as markup.

    Args:
        value: Any value; it is coerced to `str` before escaping.

    Returns:
        The HTML-escaped string representation of `value`.
    """
    return _html_lib.escape(str(value))


def _get_status_color(status: StatusKind) -> Tuple[str, str, str]:
    """
    Resolve a status keyword to its (text color, background, border) tokens.

    Args:
        status: One of "online", "offline", "warning", "processing", "error".

    Returns:
        A 3-tuple of (color, background, border) hex/rgba tokens from
        `ui.styles.Colors`. Unknown statuses fall back to the "offline"
        (neutral) palette rather than raising, so a typo never crashes
        a page render.
    """
    mapping: Dict[str, Tuple[str, str, str]] = {
        "online": (Colors.SUCCESS, Colors.SUCCESS_BG, Colors.SUCCESS_BORDER),
        "offline": (Colors.TEXT_MUTED, Colors.SURFACE_ALT, Colors.BORDER),
        "warning": (Colors.WARNING, Colors.WARNING_BG, Colors.WARNING_BORDER),
        "processing": (Colors.INFO, Colors.INFO_BG, Colors.INFO_BORDER),
        "error": (Colors.ERROR, Colors.ERROR_BG, Colors.ERROR_BORDER),
    }
    return mapping.get(status, mapping["offline"])


def _format_trend(trend: Optional[TrendDirection]) -> Tuple[str, str]:
    """
    Resolve a trend direction to its delta-color and arrow-glyph CSS classes.

    Args:
        trend: "up", "down", "flat", or None.

    Returns:
        A tuple of (delta_class, arrow_class) matching the
        `.metric-card__delta--*` / `.metric-card__trend-arrow--*` classes
        already defined in `ui/styles.py`. Defaults to "flat" for None or
        an unrecognized value.
    """
    safe_trend = trend if trend in ("up", "down", "flat") else "flat"
    return f"metric-card__delta--{safe_trend}", f"metric-card__trend-arrow--{safe_trend}"


def _get_confidence_level(score: float) -> Tuple[str, InfoVariant]:
    """
    Map a confidence score to a human-readable level and a badge variant.

    Args:
        score: Confidence score. Accepts either a 0-1 fraction (e.g. 0.87)
            or a 0-100 percentage (e.g. 87); values `<= 1` are treated as
            fractions and scaled up automatically.

    Returns:
        A tuple of (level_label, badge_variant) where level_label is one
        of "Low", "Medium", "High", "Excellent" and badge_variant is one
        of "error", "warning", "info", "success" (matching the `.badge`
        classes from `ui/styles.py`).
    """
    normalized = score * 100 if score <= 1 else score
    if normalized >= 85:
        return "Excellent", "success"
    if normalized >= 65:
        return "High", "info"
    if normalized >= 40:
        return "Medium", "warning"
    return "Low", "error"


def _document_type_icon(document_type: str) -> str:
    """
    Resolve a document type string to a representative emoji icon.

    Args:
        document_type: A free-form document type label (e.g. "pdf",
            "docx", "webpage", "faq").

    Returns:
        A single emoji character. Falls back to a generic document icon
        for unrecognized types so the chip never renders blank.
    """
    icons: Dict[str, str] = {
        "pdf": "\U0001F4C4",
        "doc": "\U0001F4DD",
        "docx": "\U0001F4DD",
        "sheet": "\U0001F4CA",
        "xlsx": "\U0001F4CA",
        "csv": "\U0001F4CA",
        "url": "\U0001F310",
        "web": "\U0001F310",
        "webpage": "\U0001F310",
        "faq": "\u2753",
        "policy": "\U0001F4D1",
        "image": "\U0001F5BC\uFE0F",
    }
    return icons.get(document_type.strip().lower(), "\U0001F4C4")


def _normalize_key_value_pairs(pairs: KeyValuePairs) -> List[Tuple[str, str]]:
    """
    Normalize a `dict` or a sequence of `(key, value)` tuples into a list
    of tuples, preserving order.

    Args:
        pairs: Either a `dict` (Python 3.7+ preserves insertion order) or
            a sequence of `(key, value)` tuples.

    Returns:
        A list of `(key, value)` tuples in their original order.
    """
    if isinstance(pairs, dict):
        return list(pairs.items())
    return list(pairs)


def _render_html(markup: str) -> None:
    """
    Thin wrapper around `st.markdown(..., unsafe_allow_html=True)`.

    Centralizing this one-liner keeps every render function's intent
    obvious at the call site and gives future refactors (e.g. adding a
    sanitizer, or swapping renderers) exactly one place to change.

    Args:
        markup: A complete HTML fragment to render.

    Returns:
        None.
    """
    st.markdown(
        textwrap.dedent(markup).strip(),
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 3 — SECTION HEADER
# =============================================================================


def render_section_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    """
    Render a professional section heading.

    Used to introduce a distinct region of the page (e.g. "Retrieved
    Sources", "Available Doctors", "Pipeline Metrics") with an optional
    icon, subtitle, and longer descriptive copy.

    Args:
        title: The main heading text.
        subtitle: Optional one-line supporting text shown under the title.
        icon: Optional emoji/character shown in a rounded icon badge to
            the left of the title.
        description: Optional longer explanatory paragraph shown below
            the subtitle.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    icon_html = (
        f'<div class="comp-section-header__icon">{_escape(icon)}</div>' if icon else ""
    )
    subtitle_html = (
        f'<p class="comp-section-header__subtitle">{_escape(subtitle)}</p>' if subtitle else ""
    )
    description_html = (
        f'<p class="comp-section-header__description">{_escape(description)}</p>'
        if description
        else ""
    )

    markup = f"""
    <div class="comp-section-header anim-fade-in">
        {icon_html}
        <div class="comp-section-header__text">
            <h2 class="comp-section-header__title">{_escape(title)}</h2>
            {subtitle_html}
            {description_html}
        </div>
    </div>
    """
    # description (and icon/subtitle) are optional, so their placeholder
    # lines above collapse to whitespace-only when omitted. Left in place,
    # such a line breaks Markdown's raw-HTML-block recognition and causes
    # the remaining closing tags to be mis-parsed as an indented code
    # block. Drop those blank lines here rather than reshaping the
    # template.
    markup = "\n".join(line for line in markup.splitlines() if line.strip())
    _render_html(markup)


# =============================================================================
# SECTION 4 — CARD
# =============================================================================


def _build_card_html(
    variant: CardVariant,
    title: Optional[str],
    body: Optional[str],
    footer: Optional[str],
    icon: Optional[str],
) -> str:
    """
    Compose the HTML for `render_card()`.

    Args:
        variant: One of "default", "flat", "accent", "gradient", "borderless".
        title: Optional card title.
        body: Optional card body text (may contain simple inline HTML the
            caller already trusts, e.g. from `render_key_value` output).
        footer: Optional small footer text.
        icon: Optional emoji/character shown beside the title.

    Returns:
        A complete `<div class="app-card ...">...</div>` HTML string.
    """
    variant_class = "" if variant == "default" else f" app-card--{variant}"

    header_html = ""
    if title or icon:
        icon_html = f'<span class="comp-card__icon">{_escape(icon)}</span>' if icon else ""
        header_html = (
            f'<div class="comp-card__header">{icon_html}'
            f'<p class="app-card__title" style="margin:0;">{_escape(title) if title else ""}</p>'
            f"</div>"
        )

    body_html = f'<div class="app-card__body">{body}</div>' if body else ""
    footer_html = f'<div class="comp-card__footer">{footer}</div>' if footer else ""

    return (
        f'<div class="app-card{variant_class} hover-lift">'
        f"{header_html}{body_html}{footer_html}"
        f"</div>"
    )


def render_card(
    title: Optional[str] = None,
    body: Optional[str] = None,
    footer: Optional[str] = None,
    icon: Optional[str] = None,
    variant: CardVariant = "default",
) -> None:
    """
    Render a reusable card container.

    Args:
        title: Optional card title.
        body: Optional body content. Plain text is rendered as-is inside
            the card body region; callers that need rich content should
            build it with this module's other `render_*` helpers or pass
            pre-built, trusted HTML.
        footer: Optional small muted footer line (e.g. a timestamp or
            attribution).
        icon: Optional emoji/character shown next to the title.
        variant: One of "default", "flat", "accent", "gradient",
            "borderless". Defaults to "default" (standard elevated card).

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()
    _render_html(_build_card_html(variant, title, body, footer, icon))


# =============================================================================
# SECTION 5 — STATUS BADGE
# =============================================================================


def _build_badge_html(label: str, status: StatusKind) -> str:
    """
    Compose the HTML for `render_status_badge()`.

    Args:
        label: Text shown next to the status dot.
        status: One of "online", "offline", "warning", "processing", "error".

    Returns:
        A `<span class="status-badge status-badge--{status}">...</span>` HTML string.
    """
    color, bg, border = _get_status_color(status)
    return (
        f'<span class="status-badge status-badge--{status}" '
        f'style="color:{color};background:{bg};border-color:{border};">'
        f'<span class="status-badge__dot"></span>{_escape(label)}</span>'
    )


def render_status_badge(label: str, status: StatusKind) -> None:
    """
    Render a reusable status badge/pill.

    Args:
        label: Text shown inside the badge (e.g. "System Online",
            "Reconnecting", "3 Errors").
        status: One of "online", "offline", "warning", "processing",
            "error". Controls the badge's color and (for "processing")
            a subtle pulsing dot animation.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()
    _render_html(_build_badge_html(label, status))


# =============================================================================
# SECTION 6 — METRIC CARD
# =============================================================================


def render_metric_card(
    title: str,
    value: Union[str, int, float],
    icon: Optional[str] = None,
    trend: Optional[TrendDirection] = None,
    delta: Optional[str] = None,
    status: Optional[Literal["success", "warning", "error"]] = None,
    loading: bool = False,
) -> None:
    """
    Render a compact KPI/metric summary card.

    Args:
        title: Metric label (e.g. "Avg. Retrieval Latency").
        value: Metric value (e.g. "240ms", 12, 98.4). Ignored visually
            (but still reserves layout space) when `loading=True`.
        icon: Optional emoji/character shown above the label.
        trend: Optional "up", "down", or "flat" — renders a small arrow
            and colors the delta text accordingly.
        delta: Optional short delta string shown next to the trend arrow
            (e.g. "+4.2%", "-12ms").
        status: Optional "success", "warning", or "error" — tints the
            card's top accent bar to flag an out-of-range metric.
        loading: If True, renders a shimmering skeleton in place of the
            label/value (used while a metric is still being computed).

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    status_class = f" metric-card--{status}" if status else ""
    loading_class = " metric-card--loading" if loading else ""
    icon_html = f'<div class="metric-card__icon">{_escape(icon)}</div>' if icon else ""

    delta_html = ""
    if delta and not loading:
        delta_class, arrow_class = _format_trend(trend)
        delta_html = (
            f'<div class="metric-card__delta {delta_class}">'
            f'<span class="{arrow_class}"></span>{_escape(delta)}</div>'
        )

    markup = f"""
    <div class="metric-card{status_class}{loading_class} anim-fade-in">
        {icon_html}
        <p class="metric-card__label">{_escape(title)}</p>
        <p class="metric-card__value">{_escape(value) if not loading else ""}</p>
        {delta_html}
    </div>
    """
    _render_html(markup)


# =============================================================================
# SECTION 7 — SOURCE CHIP (RAG retrieved documents)
# =============================================================================


def render_source_chip(
    document_name: str,
    document_type: str,
    score: Optional[float] = None,
) -> None:
    """
    Render a chip representing a single retrieved knowledge-base document.

    Args:
        document_name: Display name of the retrieved document
            (e.g. "cardiology_department.pdf").
        document_type: Document type/category, used to pick an icon
            (e.g. "pdf", "docx", "url", "faq").
        score: Optional retrieval confidence/relevance score (0-1 or
            0-100). When provided, a small confidence label is appended.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    icon = _document_type_icon(document_type)
    score_html = ""
    if score is not None:
        level, _variant = _get_confidence_level(score)
        normalized = score * 100 if score <= 1 else score
        score_html = f' <span style="opacity:0.75;">&middot; {level} ({normalized:.0f}%)</span>'

    markup = (
        f'<span class="source-chip">{icon} {_escape(document_name)}{score_html}</span>'
    )
    _render_html(markup)


# =============================================================================
# SECTION 8 — DOCTOR CARD
# =============================================================================


def render_doctor_card(
    name: str,
    department: str,
    experience: str,
    availability: str,
    consultation_fee: str,
    rating: Union[str, float],
    appointment_required: bool = True,
) -> None:
    """
    Render a professional hospital doctor profile card.

    Args:
        name: Doctor's full name (e.g. "Dr. Anjali Sharma").
        department: Department/specialty (e.g. "Cardiology").
        experience: Years of experience, pre-formatted (e.g. "12 years").
        availability: Availability summary (e.g. "Mon-Fri, 10 AM - 4 PM").
        consultation_fee: Consultation fee, pre-formatted (e.g. "₹800").
        rating: Numeric or pre-formatted rating (e.g. 4.7 or "4.7/5").
        appointment_required: Whether an appointment is required before
            visiting. Renders a small tag indicating "Appointment
            Required" or "Walk-in Available".

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    avatar_html = _avatar_html(kind="doctor", size="md")
    appointment_tag = (
        _tag_html("Appointment Required", "warning")
        if appointment_required
        else _tag_html("Walk-in Available", "success")
    )

    markup = f"""
    <div class="comp-doctor-card hover-lift">
        <div class="comp-doctor-card__header">
            {avatar_html}
            <div>
                <p class="comp-doctor-card__name">{_escape(name)}</p>
                <p class="comp-doctor-card__department">{_escape(department)}</p>
            </div>
            <div class="comp-doctor-card__rating">&#11088; {_escape(rating)}</div>
        </div>
        <div class="comp-doctor-card__stats">
            <div>
                <p class="comp-doctor-card__stat-label">Experience</p>
                <p class="comp-doctor-card__stat-value">{_escape(experience)}</p>
            </div>
            <div>
                <p class="comp-doctor-card__stat-label">Consultation Fee</p>
                <p class="comp-doctor-card__stat-value">{_escape(consultation_fee)}</p>
            </div>
            <div>
                <p class="comp-doctor-card__stat-label">Availability</p>
                <p class="comp-doctor-card__stat-value">{_escape(availability)}</p>
            </div>
        </div>
        <div class="comp-doctor-card__footer">
            {appointment_tag}
        </div>
    </div>
    """
    _render_html(markup)


# =============================================================================
# SECTION 9 — DEPARTMENT CARD
# =============================================================================


def render_department_card(
    department_name: str,
    description: str,
    doctor_count: int,
    availability: Union[str, bool],
) -> None:
    """
    Render a hospital department summary card.

    Args:
        department_name: Department name (e.g. "Cardiology").
        description: Short description of the department's services.
        doctor_count: Number of doctors currently listed in the department.
        availability: Either a free-form availability string (e.g. "Open
            Today") or a bool (True -> "Available Today", False ->
            "Currently Unavailable").

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    if isinstance(availability, bool):
        availability_label = "Available Today" if availability else "Currently Unavailable"
        availability_variant: InfoVariant = "success" if availability else "error"
    else:
        availability_label = availability
        availability_variant = "info"

    doctor_label = "Doctor" if doctor_count == 1 else "Doctors"

    markup = f"""
    <div class="comp-dept-card hover-lift">
        <p class="comp-dept-card__title">{_escape(department_name)}</p>
        <p class="comp-dept-card__description">{_escape(description)}</p>
        <div class="comp-dept-card__footer">
            <span>{doctor_count} {doctor_label}</span>
            {_build_badge_html(availability_label, availability_variant) if isinstance(availability, bool) else _tag_html(availability_label, "default")}
        </div>
    </div>
    """
    _render_html(markup)


# =============================================================================
# SECTION 10 — EMPTY STATE
# =============================================================================


def render_empty_state(
    title: str,
    subtitle: Optional[str] = None,
    icon: str = "\U0001F4C4",
    button_text: Optional[str] = None,
    key: Optional[str] = None,
) -> bool:
    """
    Render a reusable empty-state placeholder.

    Used for "no chat history yet", "no documents indexed", "no search
    results", "no doctors found", etc.

    Args:
        title: Short, bold headline (e.g. "No conversations yet").
        subtitle: Optional supporting sentence.
        icon: A single emoji/character shown above the title. Defaults to
            a generic document icon.
        button_text: Optional call-to-action button label (e.g. "Start a
            Conversation"). When provided, a Streamlit button is rendered
            beneath the empty state and its clicked state is returned.
        key: Optional explicit Streamlit widget key for the button, useful
            when multiple empty states are rendered on the same page.

    Returns:
        `True` if `button_text` was provided and the button was clicked
        on this run, `False` otherwise (including when no button was
        rendered at all).
    """
    _inject_component_styles()
    _render_html(get_empty_state_html(title=title, subtitle=subtitle or "", icon=icon))

    if button_text:
        return st.button(button_text, key=key, use_container_width=False)
    return False


# =============================================================================
# SECTION 11 — LOADING
# =============================================================================


def render_loading(
    mode: LoadingMode = "spinner",
    lines: int = 3,
    text: Optional[str] = None,
) -> None:
    """
    Render a loading indicator.

    Args:
        mode: One of "spinner" (small inline spinner, optionally with
            text), "skeleton" (a shimmering placeholder card with `lines`
            text rows), or "typing" (three bouncing dots, used for an
            assistant "thinking" indicator in chat).
        lines: Number of skeleton lines to render when `mode="skeleton"`.
            Ignored for other modes.
        text: Optional label shown next to the spinner when
            `mode="spinner"`. Ignored for other modes.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    if mode == "skeleton":
        _render_html(get_skeleton_card_html(lines=lines))
        return

    if mode == "typing":
        markup = (
            '<div class="typing-indicator">'
            "<span></span><span></span><span></span>"
            "</div>"
        )
        _render_html(markup)
        return

    # Default: spinner, optionally with a text label.
    label_html = f'<span style="color:{Colors.TEXT_SECONDARY};font-size:{FontSize.SM};">{_escape(text)}</span>' if text else ""
    markup = (
        f'<div class="u-flex u-items-center u-gap-sm">{get_spinner_html()}{label_html}</div>'
    )
    _render_html(markup)


# =============================================================================
# SECTION 12 — AVATAR
# =============================================================================


def _avatar_html(kind: AvatarKind, initials: Optional[str] = None, size: str = "md") -> str:
    """
    Compose the HTML for `render_avatar()`.

    Args:
        kind: One of "ai", "user", "doctor", "hospital", "custom".
        initials: Required when `kind="custom"`; up to 2 characters shown
            inside the avatar circle.
        size: One of "sm", "md", "lg".

    Returns:
        A `<div class="comp-avatar ...">...</div>` HTML string.
    """
    glyphs: Dict[str, str] = {
        "ai": "\U0001F916",
        "user": "\U0001F9D1",
        "doctor": "\U0001FA7A",
        "hospital": "\U0001F3E5",
    }
    if kind == "custom":
        content = _escape((initials or "?")[:2].upper())
    else:
        content = glyphs.get(kind, "\U0001F464")

    safe_size = size if size in ("sm", "md", "lg") else "md"
    return f'<div class="comp-avatar comp-avatar--{kind} comp-avatar--{safe_size}">{content}</div>'


def render_avatar(kind: AvatarKind, initials: Optional[str] = None, size: str = "md") -> None:
    """
    Render a circular avatar for chat participants or entities.

    Args:
        kind: One of "ai" (assistant), "user", "doctor", "hospital", or
            "custom" (renders `initials` instead of an icon glyph).
        initials: Up to 2 characters to display when `kind="custom"`.
            Ignored for other kinds.
        size: One of "sm" (28px), "md" (40px, default), "lg" (56px).

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()
    _render_html(_avatar_html(kind=kind, initials=initials, size=size))


# =============================================================================
# SECTION 13 — INFO PANEL
# =============================================================================


def render_info_panel(
    message: str,
    variant: InfoVariant = "info",
    title: Optional[str] = None,
) -> None:
    """
    Render a reusable information/alert box.

    Args:
        message: The main message body.
        variant: One of "info", "success", "warning", "error". Controls
            the panel's color scheme and default icon.
        title: Optional bold lead-in line shown above the message.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    icons: Dict[InfoVariant, str] = {
        "info": "\u2139\uFE0F",
        "success": "\u2705",
        "warning": "\u26A0\uFE0F",
        "error": "\u274C",
    }
    title_html = f'<p class="comp-info-panel__title">{_escape(title)}</p>' if title else ""

    markup = f"""
    <div class="comp-info-panel comp-info-panel--{variant}">
        <span class="comp-info-panel__icon">{icons.get(variant, icons["info"])}</span>
        <div>
            {title_html}
            <p class="comp-info-panel__message">{_escape(message)}</p>
        </div>
    </div>
    """
    _render_html(markup)


# =============================================================================
# SECTION 14 — DIVIDER
# =============================================================================


def render_divider(title: Optional[str] = None) -> None:
    """
    Render a horizontal divider, optionally with a centered title.

    Args:
        title: Optional short label centered within the divider line
            (e.g. "OR", "Earlier Today"). When omitted, a plain
            full-width divider is rendered.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    if not title:
        _render_html('<hr class="u-divider" />')
        return

    markup = f"""
    <div class="comp-divider">
        <div class="comp-divider__line"></div>
        <span class="comp-divider__title">{_escape(title)}</span>
        <div class="comp-divider__line"></div>
    </div>
    """
    _render_html(markup)


# =============================================================================
# SECTION 15 — PROGRESS BAR
# =============================================================================


def render_progress(
    percentage: float,
    label: Optional[str] = None,
    animated: bool = True,
) -> None:
    """
    Render a horizontal progress bar.

    Args:
        percentage: Progress value from 0 to 100. Values outside that
            range are clamped.
        label: Optional label shown above the bar, with the percentage
            printed at the right-hand side of the same row.
        animated: Whether the fill's width transitions smoothly (True) or
            snaps immediately to its value (False).

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    clamped = max(0.0, min(100.0, float(percentage)))
    fill_class = "comp-progress__fill comp-progress__fill--animated" if animated else "comp-progress__fill"

    label_row_html = ""
    if label:
        label_row_html = (
            f'<div class="comp-progress__label-row">'
            f"<span>{_escape(label)}</span><span>{clamped:.0f}%</span>"
            f"</div>"
        )

    markup = f"""
    <div class="comp-progress">
        {label_row_html}
        <div class="comp-progress__track">
            <div class="{fill_class}" style="width:{clamped:.0f}%;"></div>
        </div>
    </div>
    """
    _render_html(markup)


# =============================================================================
# SECTION 16 — CONFIDENCE BADGE
# =============================================================================


def render_confidence_badge(score: float) -> None:
    """
    Render a badge summarizing a RAG confidence/relevance score.

    Args:
        score: Confidence score. Accepts either a 0-1 fraction or a 0-100
            percentage; maps to "Low" (<40%), "Medium" (40-64%), "High"
            (65-84%), or "Excellent" (>=85%), each with a distinct color.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()
    level, variant = _get_confidence_level(score)
    _render_html(get_badge_html(level, variant))


# =============================================================================
# SECTION 17 — TAG
# =============================================================================


def _tag_html(label: str, variant: TagVariant = "default") -> str:
    """
    Compose the HTML for `render_tag()`.

    Args:
        label: Tag text.
        variant: One of "default", "primary", "teal", "neutral",
            "success", "warning", "error".

    Returns:
        A `<span class="comp-tag comp-tag--{variant}">...</span>` HTML string.
    """
    valid_variants = {"default", "primary", "teal", "neutral", "success", "warning", "error"}
    safe_variant = variant if variant in valid_variants else "default"
    return f'<span class="comp-tag comp-tag--{safe_variant}">{_escape(label)}</span>'


def render_tag(label: str, variant: TagVariant = "default") -> None:
    """
    Render a small reusable tag/chip.

    Args:
        label: Tag text (e.g. "Cardiology", "New", "Follow-up").
        variant: One of "default", "primary", "teal", "neutral",
            "success", "warning", "error". Controls color only.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()
    _render_html(_tag_html(label, variant))


# =============================================================================
# SECTION 18 — KEY-VALUE DISPLAY
# =============================================================================


def render_key_value(pairs: KeyValuePairs) -> None:
    """
    Render a professional key-value display (e.g. an appointment summary).

    Args:
        pairs: Either a `dict` of `{key: value}` or a sequence of
            `(key, value)` tuples, rendered in order, one per row
            (e.g. {"Department": "Cardiology", "Doctor": "Dr. Sharma"}).

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    rows_html = "".join(
        f'<div class="comp-kv-row">'
        f'<span class="comp-kv-key">{_escape(key)}</span>'
        f'<span class="comp-kv-value">{_escape(value)}</span>'
        f"</div>"
        for key, value in _normalize_key_value_pairs(pairs)
    )
    _render_html(f'<div class="comp-kv-list">{rows_html}</div>')


# =============================================================================
# SECTION 19 — CHAT TIMESTAMP
# =============================================================================


def render_chat_timestamp(timestamp: Union[str, datetime]) -> None:
    """
    Render a small formatted timestamp badge for a chat message.

    Args:
        timestamp: Either a pre-formatted string (rendered as-is) or a
            `datetime` object (formatted as `HH:MM AM/PM`).

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()

    display_value = timestamp.strftime("%I:%M %p").lstrip("0") if isinstance(timestamp, datetime) else timestamp
    markup = f'<span class="chat-bubble__meta">\U0001F551 {_escape(display_value)}</span>'
    _render_html(markup)


# =============================================================================
# SECTION 20 — FOOTER NOTE
# =============================================================================


def render_footer_note(text: str) -> None:
    """
    Render a small, muted footer note.

    Used for disclaimers or attribution beneath a chat panel, doctor
    listing, or metrics section (e.g. "AI-generated responses may be
    inaccurate. Always consult a medical professional.").

    Args:
        text: The note text.

    Returns:
        None. Renders directly into the Streamlit app.
    """
    _inject_component_styles()
    _render_html(f'<p class="comp-footer-note">{_escape(text)}</p>')