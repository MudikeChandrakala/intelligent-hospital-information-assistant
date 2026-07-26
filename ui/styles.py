"""
ui/styles.py
=============================================================================
Production design system for the Intelligent Hospital Information Assistant
frontend.

This module is the SINGLE SOURCE OF TRUTH for every visual design token
(colors, typography, spacing, shadows, radii, animation timing, z-index
layers, breakpoints, gradients) and exposes helper functions that compile
and inject CSS into a Streamlit app.

No other UI module (layout.py, sidebar.py, chat.py, metrics.py,
components.py, utils.py) should ever hard-code a hex value, font size,
shadow, timing curve, or breakpoint. They import tokens / helpers from
here so the whole application stays visually consistent, accessible, and
easy to re-theme.

Design direction: modern SaaS / healthcare-tech product
(Microsoft Fluent + Google Material + Notion + Linear + ChatGPT),
explicitly NOT default Streamlit grey-and-red styling.

-----------------------------------------------------------------------------
Backward compatibility contract
-----------------------------------------------------------------------------
The following public symbols existed in the previous version of this file
and MUST continue to work exactly as before for downstream modules:

    Colors, Typography, FontSize, Radius, Shadow, Spacing
    apply_global_styles()
    inject_custom_css(css: str)
    load_external_stylesheet(path: str)
    get_status_badge_html(label: str = ..., online: bool = ...)

Everything below is purely additive: new token classes, new CSS sections,
and new (optional) helper functions. Nothing above has been renamed,
removed, or had its signature changed.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import streamlit as st

# =============================================================================
# SECTION 1 — COLOR PALETTE
# =============================================================================
# A healthcare-appropriate palette: a trustworthy primary blue, a calming
# healthcare teal as the accent/secondary color, clean whites for surfaces,
# and soft neutral grays for backgrounds and borders. Semantic colors cover
# status states (success / warning / error / info) so badges, alerts, and
# metric cards stay visually consistent everywhere they appear.


class Colors:
    """
    Central color palette for the entire application.

    Import this instead of hard-coding hex values anywhere in ui/*.py.
    Grouped by role (brand, neutral, semantic, chat, sidebar, buttons) so
    it's obvious which token to reach for in a given context.
    """

    # --- Brand / Primary -----------------------------------------------
    PRIMARY = "#2563EB"          # Primary Blue — main brand color, buttons, links
    PRIMARY_DARK = "#1D4ED8"     # Hover / active state for primary elements
    PRIMARY_DARKER = "#1E40AF"   # Pressed / active-active state
    PRIMARY_LIGHT = "#DBEAFE"    # Light blue tint — backgrounds, badges
    PRIMARY_SOFT = "#EFF6FF"     # Very soft blue wash — selected rows, highlights

    # --- Secondary / Healthcare Teal -------------------------------------
    TEAL = "#0D9488"             # Healthcare Teal — secondary brand accent
    TEAL_DARK = "#0F766E"        # Hover state for teal elements
    TEAL_DARKER = "#115E59"      # Pressed state for teal elements
    TEAL_LIGHT = "#CCFBF1"       # Light teal tint — success chips, icons
    TEAL_SOFT = "#F0FDFA"        # Very soft teal wash — subtle section backgrounds

    # --- Neutrals / Surfaces ---------------------------------------------
    BACKGROUND = "#F5F7FA"       # Soft gray app background (not stark white)
    SURFACE = "#FFFFFF"          # White card / panel surface
    SURFACE_ALT = "#FAFBFC"      # Slightly off-white surface for nested panels
    SURFACE_HOVER = "#F8FAFC"    # Hover background for list rows / nav items
    BORDER = "#E5E9F0"           # Soft, low-contrast border color
    BORDER_STRONG = "#D1D9E6"    # Slightly stronger border for emphasis

    # --- Text -------------------------------------------------------------
    TEXT_PRIMARY = "#0F172A"     # Near-black slate — headings, primary text
    TEXT_SECONDARY = "#475569"   # Muted slate — body copy, descriptions
    TEXT_MUTED = "#94A3B8"       # Light slate — placeholders, timestamps
    TEXT_DISABLED = "#CBD5E1"    # Disabled text / icon color
    TEXT_ON_PRIMARY = "#FFFFFF"  # Text placed on top of primary-colored surfaces

    # --- Semantic / Status --------------------------------------------------
    SUCCESS = "#059669"
    SUCCESS_DARK = "#047857"
    SUCCESS_BG = "#ECFDF5"
    SUCCESS_BORDER = "#A7F3D0"

    WARNING = "#D97706"
    WARNING_DARK = "#B45309"
    WARNING_BG = "#FFFBEB"
    WARNING_BORDER = "#FDE68A"

    ERROR = "#DC2626"
    ERROR_DARK = "#B91C1C"
    ERROR_BG = "#FEF2F2"
    ERROR_BORDER = "#FECACA"

    INFO = "#0284C7"
    INFO_DARK = "#0369A1"
    INFO_BG = "#F0F9FF"
    INFO_BORDER = "#BAE6FD"

    # --- Chat bubble specific ----------------------------------------------
    USER_BUBBLE_BG = PRIMARY
    USER_BUBBLE_TEXT = "#FFFFFF"
    ASSISTANT_BUBBLE_BG = "#FFFFFF"
    ASSISTANT_BUBBLE_TEXT = TEXT_PRIMARY
    ASSISTANT_BUBBLE_BORDER = BORDER
    CODE_BLOCK_BG = "#0F172A"
    CODE_BLOCK_TEXT = "#E2E8F0"
    CODE_INLINE_BG = "#F1F5F9"
    CODE_INLINE_TEXT = "#BE185D"

    # --- Sidebar -------------------------------------------------------------
    SIDEBAR_BG = "#0F172A"        # Deep slate navy for sidebar (contrast anchor)
    SIDEBAR_TEXT = "#E2E8F0"
    SIDEBAR_TEXT_MUTED = "#94A3B8"
    SIDEBAR_ACTIVE_BG = "rgba(37, 99, 235, 0.18)"
    SIDEBAR_HOVER_BG = "rgba(255, 255, 255, 0.06)"
    SIDEBAR_BORDER = "rgba(255, 255, 255, 0.08)"

    # --- Buttons (danger / success variants) --------------------------------
    DANGER = ERROR
    DANGER_DARK = ERROR_DARK
    SUCCESS_BTN = SUCCESS
    SUCCESS_BTN_DARK = SUCCESS_DARK

    # --- Misc -----------------------------------------------------------------
    SELECTION_BG = PRIMARY_LIGHT
    SELECTION_TEXT = "#0F172A"
    SKELETON_BASE = "#E5E9F0"
    SKELETON_HIGHLIGHT = "#F3F5F8"
    OVERLAY_SCRIM = "rgba(15, 23, 42, 0.45)"


# =============================================================================
# SECTION 2 — TYPOGRAPHY
# =============================================================================
# Uses a modern system-font stack (same family GitHub/Notion/Linear use) so
# the app renders crisply on every OS without loading external fonts, with
# an optional Google Font import for a slightly more "designed" feel.


class Typography:
    """Font families and weights used across the app."""

    FONT_FAMILY = (
        "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Roboto, Helvetica, Arial, sans-serif"
    )
    FONT_FAMILY_MONO = (
        "'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', "
        "Menlo, monospace"
    )

    WEIGHT_REGULAR = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700

    LINE_HEIGHT_TIGHT = 1.2
    LINE_HEIGHT_NORMAL = 1.5
    LINE_HEIGHT_RELAXED = 1.65


# =============================================================================
# SECTION 3 — FONT SIZES
# =============================================================================
# A restrained modular type scale using `clamp()` where fluid sizing helps
# (page title), and fixed rem tokens elsewhere for predictability.


class FontSize:
    """Type scale, expressed in rem for accessibility (respects browser zoom)."""

    XS = "0.75rem"     # 12px — captions, timestamps, badges
    SM = "0.875rem"    # 14px — secondary text, helper text
    BASE = "1rem"      # 16px — body text
    MD = "1.125rem"    # 18px — chat message text
    LG = "1.25rem"     # 20px — card titles, section subheadings
    XL = "1.5rem"      # 24px — panel headings
    XXL = "1.875rem"   # 30px — page title (fixed)
    XXXL = "2.25rem"   # 36px — hero / app title (fixed)

    # Fluid variant of the page title: scales smoothly between mobile and
    # desktop instead of jumping at a single breakpoint.
    FLUID_TITLE = "clamp(1.5rem, 1.1rem + 1.5vw, 2.25rem)"


# =============================================================================
# SECTION 4 — BORDER RADIUS
# =============================================================================
# Consistent rounding tokens for a soft, modern, "Notion-like" feel.


class Radius:
    """Border-radius tokens applied to cards, buttons, inputs, bubbles."""

    SM = "8px"      # Inputs, small chips
    MD = "12px"     # Buttons, badges
    LG = "16px"     # Cards, panels
    XL = "20px"     # Chat bubbles, hero panels
    PILL = "999px"  # Fully rounded — pills, avatars, toggle switches


# =============================================================================
# SECTION 5 — SHADOWS
# =============================================================================
# Soft, multi-layered shadows (like Material/Fluent elevation levels) rather
# than a single harsh drop-shadow. Gives cards a subtle "floating" depth.


class Shadow:
    """Elevation tokens. Higher tier = more elevated / prominent."""

    XS = "0 1px 2px rgba(15, 23, 42, 0.04)"
    SM = "0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04)"
    MD = (
        "0 4px 6px -1px rgba(15, 23, 42, 0.07), "
        "0 2px 4px -2px rgba(15, 23, 42, 0.05)"
    )
    LG = (
        "0 10px 15px -3px rgba(15, 23, 42, 0.08), "
        "0 4px 6px -4px rgba(15, 23, 42, 0.05)"
    )
    XL = (
        "0 20px 25px -5px rgba(15, 23, 42, 0.10), "
        "0 8px 10px -6px rgba(15, 23, 42, 0.05)"
    )
    FOCUS_RING = f"0 0 0 3px {Colors.PRIMARY_LIGHT}"
    FOCUS_RING_ERROR = f"0 0 0 3px {Colors.ERROR_BORDER}"


# =============================================================================
# SECTION 6 — SPACING
# =============================================================================
# A small spacing scale so paddings/margins across components stay on-grid.


class Spacing:
    """Spacing tokens (rem-based) for padding/margins/gaps."""

    XS = "0.25rem"   # 4px
    SM = "0.5rem"    # 8px
    MD = "1rem"      # 16px
    LG = "1.5rem"    # 24px
    XL = "2rem"       # 32px
    XXL = "3rem"      # 48px


# =============================================================================
# SECTION 7 — ANIMATION TOKENS  (new)
# =============================================================================
# Centralizes every timing value and easing curve used in transitions and
# keyframe animations. Nothing in the CSS below should hard-code a duration
# or cubic-bezier curve outside of this class.


class Animation:
    """
    Motion tokens: durations, easing curves, and reusable transition
    shorthands. Import these instead of hard-coding timing anywhere.
    """

    # --- Durations -----------------------------------------------------
    FAST = "120ms"      # Micro-interactions: button press, checkbox toggle
    NORMAL = "200ms"    # Standard UI transitions: hover, focus, color change
    SLOW = "350ms"      # Larger movements: panel slide-in, modal open

    # --- Easing curves ---------------------------------------------------
    EASE = "ease"
    EASE_IN = "cubic-bezier(0.4, 0, 1, 1)"
    EASE_OUT = "cubic-bezier(0, 0, 0.2, 1)"
    EASE_IN_OUT = "cubic-bezier(0.4, 0, 0.2, 1)"

    # --- Reusable transition shorthands -----------------------------------
    HOVER_SCALE = "transform 200ms cubic-bezier(0, 0, 0.2, 1)"
    CARD_LIFT = (
        "box-shadow 200ms cubic-bezier(0, 0, 0.2, 1), "
        "transform 200ms cubic-bezier(0, 0, 0.2, 1)"
    )
    BUTTON_PRESS = "transform 120ms cubic-bezier(0.4, 0, 1, 1)"

    @staticmethod
    def transition(*properties: str, duration: str = NORMAL, easing: str = EASE_IN_OUT) -> str:
        """
        Build a multi-property CSS `transition` value from tokens.

        Args:
            *properties: CSS property names to transition (e.g. "background", "color").
            duration: One of Animation.FAST / NORMAL / SLOW (defaults to NORMAL).
            easing: One of the Animation easing curves (defaults to EASE_IN_OUT).

        Returns:
            A comma-separated `transition` declaration value, e.g.
            "background 200ms cubic-bezier(0.4, 0, 0.2, 1), color 200ms cubic-bezier(0.4, 0, 0.2, 1)".
        """
        return ", ".join(f"{prop} {duration} {easing}" for prop in properties)


# =============================================================================
# SECTION 8 — Z-INDEX TOKENS  (new)
# =============================================================================
# A single stacking-context scale so overlapping elements (sticky header,
# sidebar, floating chat launcher, modals, tooltips, overlays, toasts)
# never fight each other over z-index values scattered across files.


class ZIndex:
    """Stacking-order tokens. Higher value renders on top of lower values."""

    HEADER = 100
    SIDEBAR = 200
    FLOATING_CHAT = 300
    OVERLAY = 400
    MODAL = 500
    NOTIFICATION = 600
    TOOLTIP = 700


# =============================================================================
# SECTION 9 — RESPONSIVE BREAKPOINTS  (new)
# =============================================================================
# Breakpoint tokens consumed by the media queries in the "Responsive" CSS
# section further down, and available for any ui/*.py module that needs to
# branch layout logic in Python (e.g. via `st.session_state` viewport hints).


class Breakpoints:
    """
    Responsive breakpoint tokens (min-width values used in media queries).

    MOBILE is the base/default (mobile-first); the others describe the
    minimum viewport width at which the corresponding layout kicks in.
    """

    MOBILE = "480px"    # Base styles apply below this; phones
    TABLET = "768px"    # Small tablets / large phones landscape
    LAPTOP = "1024px"   # Small laptops / large tablets
    DESKTOP = "1280px"  # Standard desktop monitors

    @classmethod
    def up(cls, breakpoint_px: str) -> str:
        """Return a `min-width` media query prefix for the given breakpoint."""
        return f"@media (min-width: {breakpoint_px})"

    @classmethod
    def down(cls, breakpoint_px: str) -> str:
        """Return a `max-width` media query prefix for the given breakpoint."""
        return f"@media (max-width: {breakpoint_px})"


# =============================================================================
# SECTION 10 — GRADIENT TOKENS  (new)
# =============================================================================
# Every `linear-gradient(...)` used anywhere in the CSS is defined exactly
# once here and referenced by name, so brand gradients stay consistent and
# are trivial to re-theme.


class Gradients:
    """Reusable gradient tokens. Never repeat `linear-gradient()` inline."""

    PRIMARY = f"linear-gradient(135deg, {Colors.PRIMARY}, {Colors.PRIMARY_DARK})"
    TEAL = f"linear-gradient(135deg, {Colors.TEAL}, {Colors.TEAL_DARK})"
    HEADER = f"linear-gradient(135deg, {Colors.PRIMARY}, {Colors.TEAL})"
    METRIC = f"linear-gradient(90deg, {Colors.PRIMARY}, {Colors.TEAL})"
    CARD = f"linear-gradient(180deg, {Colors.SURFACE} 0%, {Colors.SURFACE_ALT} 100%)"
    BACKGROUND = (
        f"linear-gradient(160deg, {Colors.BACKGROUND} 0%, {Colors.PRIMARY_SOFT} 100%)"
    )
    SKELETON_SHIMMER = (
        f"linear-gradient(90deg, {Colors.SKELETON_BASE} 25%, "
        f"{Colors.SKELETON_HIGHLIGHT} 37%, {Colors.SKELETON_BASE} 63%)"
    )


# =============================================================================
# SECTION 11 — BASE / RESET CSS
# =============================================================================
# Resets Streamlit's default chrome (hides hamburger-menu clutter where
# sensible, restyles the main container, buttons, inputs, scrollbars) so the
# whole app reads as a bespoke product rather than an out-of-the-box
# Streamlit demo. Also carries global accessibility rules: focus rings,
# selection color, placeholder color, and `prefers-reduced-motion` support.


def _base_css() -> str:
    """Return global resets, base element styling, and accessibility rules."""
    return f"""
    /* =========================================================
       1. BASE
       ========================================================= */

    /* Optional modern font import (Inter). Safe no-op if offline/blocked. */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ---- Global resets ------------------------------------------------ */
    html, body, [class*="css"] {{
        font-family: {Typography.FONT_FAMILY};
        color: {Colors.TEXT_PRIMARY};
    }}

    .stApp {{
        background: {Colors.BACKGROUND};
    }}

    /* Hide default Streamlit chrome for a cleaner, branded shell */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        background: transparent;
        z-index: {ZIndex.HEADER};
    }}

    /* Main content container: comfortable max width + padding */
    .block-container {{
        padding-top: {Spacing.LG};
        padding-bottom: {Spacing.XL};
        max-width: 1180px;
    }}

    /* ---- Accessibility: focus visibility --------------------------------- */
    /* Every interactive element gets a visible, high-contrast focus ring
       when navigated to via keyboard (not just mouse hover). */
    a:focus-visible,
    button:focus-visible,
    input:focus-visible,
    textarea:focus-visible,
    [tabindex]:focus-visible {{
        outline: none;
        box-shadow: {Shadow.FOCUS_RING};
        border-radius: {Radius.SM};
    }}

    /* ---- Accessibility: text selection color ------------------------------ */
    ::selection {{
        background: {Colors.SELECTION_BG};
        color: {Colors.SELECTION_TEXT};
    }}

    /* ---- Accessibility: placeholder color ---------------------------------- */
    ::placeholder {{
        color: {Colors.TEXT_MUTED};
        opacity: 1;
    }}

    /* ---- Accessibility: reduced motion ------------------------------------- */
    /* Respect the OS-level "reduce motion" preference by disabling all
       transitions/animations for users who have opted out of motion. */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
            scroll-behavior: auto !important;
        }}
    }}

    /* ---- Scrollbar styling --------------------------------------------- */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {Colors.BORDER_STRONG};
        border-radius: {Radius.PILL};
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {Colors.TEXT_MUTED};
    }}

    /* ---- Native Streamlit buttons (base) ---------------------------------- */
    /* Component-specific button variants live in `_button_css()`; these
       rules style the raw `st.button` output before variant classes apply. */
    .stButton > button {{
        background: {Colors.PRIMARY};
        color: {Colors.TEXT_ON_PRIMARY};
        border: none;
        border-radius: {Radius.MD};
        padding: 0.5rem 1.25rem;
        font-weight: {Typography.WEIGHT_MEDIUM};
        font-size: {FontSize.SM};
        box-shadow: {Shadow.SM};
        transition: {Animation.transition("background", "box-shadow")}, {Animation.BUTTON_PRESS};
    }}
    .stButton > button:hover {{
        background: {Colors.PRIMARY_DARK};
        box-shadow: {Shadow.MD};
    }}
    .stButton > button:active {{
        transform: translateY(1px) scale(0.98);
    }}
    .stButton > button:disabled {{
        background: {Colors.TEXT_DISABLED};
        color: {Colors.SURFACE};
        box-shadow: none;
        cursor: not-allowed;
    }}

    /* ---- Text inputs / chat input ---------------------------------------- */
    .stTextInput input,
    .stTextArea textarea,
    .stChatInput textarea {{
        border-radius: {Radius.MD} !important;
        border: 1px solid {Colors.BORDER} !important;
        background: {Colors.SURFACE} !important;
        font-size: {FontSize.BASE} !important;
        transition: {Animation.transition("border-color", "box-shadow")} !important;
    }}
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stChatInput textarea:focus {{
        border-color: {Colors.PRIMARY} !important;
        box-shadow: {Shadow.FOCUS_RING} !important;
    }}
    .stTextInput input:disabled,
    .stTextArea textarea:disabled {{
        background: {Colors.SURFACE_ALT} !important;
        color: {Colors.TEXT_DISABLED} !important;
        cursor: not-allowed !important;
    }}

    /* ---- Divider ---------------------------------------------------------- */
    hr {{
        border: none;
        border-top: 1px solid {Colors.BORDER};
        margin: {Spacing.MD} 0;
    }}
    """


# =============================================================================
# SECTION 12 — LAYOUT & UTILITY CSS  (new)
# =============================================================================
# A small, Tailwind-inspired utility layer (flex/grid/gap/center/hidden/
# scrollable/divider/rounded/shadow helpers) so simple one-off layout needs
# in components.py don't require bespoke CSS classes for every tweak.


def _layout_css() -> str:
    """Return general page-layout containers and atomic utility classes."""
    return f"""
    /* =========================================================
       2. LAYOUT & UTILITIES
       ========================================================= */

    .app-shell {{
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        gap: {Spacing.LG};
    }}

    /* ---- Flex utilities ----------------------------------------------- */
    .u-flex {{ display: flex; }}
    .u-flex-col {{ display: flex; flex-direction: column; }}
    .u-flex-wrap {{ flex-wrap: wrap; }}
    .u-items-center {{ align-items: center; }}
    .u-items-start {{ align-items: flex-start; }}
    .u-items-end {{ align-items: flex-end; }}
    .u-justify-between {{ justify-content: space-between; }}
    .u-justify-center {{ justify-content: center; }}
    .u-justify-end {{ justify-content: flex-end; }}

    /* ---- Grid utilities ------------------------------------------------- */
    .u-grid {{ display: grid; }}
    .u-grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); }}
    .u-grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); }}
    .u-grid-auto {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }}

    /* ---- Gap scale (mirrors Spacing tokens) ------------------------------- */
    .u-gap-xs {{ gap: {Spacing.XS}; }}
    .u-gap-sm {{ gap: {Spacing.SM}; }}
    .u-gap-md {{ gap: {Spacing.MD}; }}
    .u-gap-lg {{ gap: {Spacing.LG}; }}
    .u-gap-xl {{ gap: {Spacing.XL}; }}

    /* ---- Positioning / visibility ----------------------------------------- */
    .u-center {{
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .u-hidden {{ display: none !important; }}
    .u-scrollable {{
        overflow-y: auto;
        max-height: 100%;
    }}
    .u-scrollable-x {{
        overflow-x: auto;
        white-space: nowrap;
    }}

    /* ---- Divider utility (horizontal / vertical) --------------------------- */
    .u-divider {{
        border: none;
        border-top: 1px solid {Colors.BORDER};
        margin: {Spacing.SM} 0;
    }}
    .u-divider-vertical {{
        border-left: 1px solid {Colors.BORDER};
        align-self: stretch;
        margin: 0 {Spacing.SM};
    }}

    /* ---- Rounded corner utilities ------------------------------------------- */
    .u-rounded-sm {{ border-radius: {Radius.SM}; }}
    .u-rounded-md {{ border-radius: {Radius.MD}; }}
    .u-rounded-lg {{ border-radius: {Radius.LG}; }}
    .u-rounded-xl {{ border-radius: {Radius.XL}; }}
    .u-rounded-pill {{ border-radius: {Radius.PILL}; }}

    /* ---- Shadow utilities ------------------------------------------------- */
    .u-shadow-xs {{ box-shadow: {Shadow.XS}; }}
    .u-shadow-sm {{ box-shadow: {Shadow.SM}; }}
    .u-shadow-md {{ box-shadow: {Shadow.MD}; }}
    .u-shadow-lg {{ box-shadow: {Shadow.LG}; }}
    .u-shadow-none {{ box-shadow: none; }}

    /* ---- Text truncation ----------------------------------------------------- */
    .u-truncate {{
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    """


# =============================================================================
# SECTION 13 — HEADER CSS
# =============================================================================
# App-level top banner: brand mark, title, subtitle, live-status badge, and
# a slim border so it visually separates from the content below. Uses the
# fluid title size and the shared HEADER gradient token.


def _header_css() -> str:
    """Return CSS for the top-of-app header/banner component."""
    return f"""
    /* =========================================================
       3. HEADER
       ========================================================= */

    .app-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: {Spacing.MD};
        padding: {Spacing.LG} {Spacing.XL};
        background: {Colors.SURFACE};
        border: 1px solid {Colors.BORDER};
        border-radius: {Radius.LG};
        box-shadow: {Shadow.SM};
        margin-bottom: {Spacing.LG};
        position: sticky;
        top: 0;
        z-index: {ZIndex.HEADER};
    }}

    .app-header__brand {{
        display: flex;
        align-items: center;
        gap: {Spacing.MD};
        min-width: 0; /* allow child truncation inside flex */
    }}

    .app-header__logo {{
        width: 44px;
        height: 44px;
        border-radius: {Radius.MD};
        background: {Gradients.HEADER};
        display: flex;
        align-items: center;
        justify-content: center;
        color: {Colors.TEXT_ON_PRIMARY};
        font-weight: {Typography.WEIGHT_BOLD};
        font-size: {FontSize.LG};
        flex-shrink: 0;
    }}

    .app-header__title {{
        font-size: {FontSize.FLUID_TITLE};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_PRIMARY};
        line-height: {Typography.LINE_HEIGHT_TIGHT};
        margin: 0;
    }}

    .app-header__subtitle {{
        font-size: {FontSize.SM};
        color: {Colors.TEXT_SECONDARY};
        margin: 0;
    }}

    .app-header__status {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: {Radius.PILL};
        background: {Colors.SUCCESS_BG};
        color: {Colors.SUCCESS};
        font-size: {FontSize.XS};
        font-weight: {Typography.WEIGHT_MEDIUM};
        white-space: nowrap;
    }}

    .app-header__status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {Colors.SUCCESS};
        display: inline-block;
    }}
    """


# =============================================================================
# SECTION 14 — SIDEBAR CSS
# =============================================================================
# Dark, "console-like" sidebar (deep slate navy) that reads as a control
# panel, distinct from the light main content area. Extended with nav
# groups, dividers, a project-info block, and a footer/status panel.


def _sidebar_css() -> str:
    """Return CSS for the Streamlit sidebar container and its contents."""
    return f"""
    /* =========================================================
       4. SIDEBAR
       ========================================================= */

    section[data-testid="stSidebar"] {{
        background: {Colors.SIDEBAR_BG};
        border-right: 1px solid {Colors.SIDEBAR_BORDER};
        z-index: {ZIndex.SIDEBAR};
    }}

    section[data-testid="stSidebar"] * {{
        color: {Colors.SIDEBAR_TEXT};
    }}

    section[data-testid="stSidebar"] .stMarkdown p {{
        color: {Colors.SIDEBAR_TEXT_MUTED};
    }}

    /* ---- Section heading ---------------------------------------------- */
    .sidebar-heading {{
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: {FontSize.XS};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.SIDEBAR_TEXT_MUTED};
        margin: {Spacing.MD} 0 {Spacing.SM} 0;
    }}

    /* ---- Navigation group wrapper --------------------------------------- */
    .sidebar-nav-group {{
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        margin-bottom: {Spacing.MD};
    }}

    /* ---- Nav item -------------------------------------------------------- */
    .sidebar-item {{
        display: flex;
        align-items: center;
        gap: {Spacing.SM};
        padding: 0.6rem 0.8rem;
        border-radius: {Radius.MD};
        font-size: {FontSize.SM};
        color: {Colors.SIDEBAR_TEXT};
        margin-bottom: 0.15rem;
        cursor: pointer;
        transition: {Animation.transition("background", "color", duration=Animation.FAST)};
    }}
    .sidebar-item:hover {{
        background: {Colors.SIDEBAR_HOVER_BG};
    }}
    .sidebar-item--active {{
        background: {Colors.SIDEBAR_ACTIVE_BG};
        color: {Colors.TEXT_ON_PRIMARY};
        font-weight: {Typography.WEIGHT_MEDIUM};
    }}
    .sidebar-item__icon {{
        font-size: {FontSize.BASE};
        width: 20px;
        text-align: center;
        flex-shrink: 0;
    }}

    /* ---- Section divider --------------------------------------------------- */
    .sidebar-divider {{
        border: none;
        border-top: 1px solid {Colors.SIDEBAR_BORDER};
        margin: {Spacing.MD} 0;
    }}

    /* ---- Generic sidebar card (status panel, project info) ----------------- */
    .sidebar-card {{
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid {Colors.SIDEBAR_BORDER};
        border-radius: {Radius.MD};
        padding: {Spacing.MD};
        margin-bottom: {Spacing.MD};
    }}

    .sidebar-card__title {{
        font-size: {FontSize.SM};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.SIDEBAR_TEXT};
        margin: 0 0 0.35rem 0;
    }}

    .sidebar-card__row {{
        display: flex;
        justify-content: space-between;
        font-size: {FontSize.XS};
        color: {Colors.SIDEBAR_TEXT_MUTED};
        padding: 0.2rem 0;
    }}

    /* ---- Project info block (name, version, authors) ------------------------ */
    .sidebar-project-info {{
        font-size: {FontSize.XS};
        color: {Colors.SIDEBAR_TEXT_MUTED};
        line-height: {Typography.LINE_HEIGHT_RELAXED};
    }}

    /* ---- Footer (pinned bottom-of-sidebar content) --------------------------- */
    .sidebar-footer {{
        margin-top: {Spacing.LG};
        padding-top: {Spacing.MD};
        border-top: 1px solid {Colors.SIDEBAR_BORDER};
        font-size: {FontSize.XS};
        color: {Colors.SIDEBAR_TEXT_MUTED};
        text-align: center;
    }}

    /* ---- Sidebar buttons: ghost-style to fit the dark background ------------- */
    section[data-testid="stSidebar"] .stButton > button {{
        background: {Colors.SIDEBAR_HOVER_BG};
        color: {Colors.SIDEBAR_TEXT};
        border: 1px solid {Colors.SIDEBAR_BORDER};
        box-shadow: none;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: {Colors.PRIMARY};
        border-color: {Colors.PRIMARY};
        color: {Colors.TEXT_ON_PRIMARY};
    }}
    """


# =============================================================================
# SECTION 15 — CARD CSS
# =============================================================================
# Generic white "surface" cards used throughout the app (panels, info boxes,
# source-citation cards, settings sections), plus hover-lift interaction.


def _card_css() -> str:
    """Return CSS for generic reusable card/panel surfaces."""
    return f"""
    /* =========================================================
       5. CARDS
       ========================================================= */

    .app-card {{
        background: {Colors.SURFACE};
        border: 1px solid {Colors.BORDER};
        border-radius: {Radius.LG};
        padding: {Spacing.LG};
        box-shadow: {Shadow.SM};
        margin-bottom: {Spacing.MD};
        transition: {Animation.CARD_LIFT};
    }}
    .app-card:hover {{
        box-shadow: {Shadow.MD};
        transform: translateY(-2px);
    }}

    .app-card--flat {{
        box-shadow: none;
        border: 1px solid {Colors.BORDER};
    }}
    .app-card--flat:hover {{
        transform: none;
        box-shadow: none;
    }}

    .app-card--accent {{
        border-left: 4px solid {Colors.TEAL};
    }}

    .app-card--gradient {{
        background: {Gradients.CARD};
    }}

    .app-card__title {{
        font-size: {FontSize.LG};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_PRIMARY};
        margin: 0 0 {Spacing.SM} 0;
    }}

    .app-card__body {{
        font-size: {FontSize.SM};
        color: {Colors.TEXT_SECONDARY};
        line-height: {Typography.LINE_HEIGHT_RELAXED};
    }}

    /* Source-citation chip, used to show retrieved KB documents */
    .source-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.7rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        background: {Colors.TEAL_SOFT};
        border: 1px solid {Colors.TEAL_LIGHT};
        color: {Colors.TEAL_DARK};
        border-radius: {Radius.PILL};
        font-size: {FontSize.XS};
        font-weight: {Typography.WEIGHT_MEDIUM};
        transition: {Animation.transition("background", duration=Animation.FAST)};
    }}
    .source-chip:hover {{
        background: {Colors.TEAL_LIGHT};
    }}

    /* ---- Empty-state components ------------------------------------------- */
    /* Reused for "no chat history", "no documents indexed", "no search
       results", and "no metrics yet" placeholders. */
    .empty-state {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: {Spacing.XXL} {Spacing.LG};
        color: {Colors.TEXT_MUTED};
    }}
    .empty-state__icon {{
        font-size: 2.5rem;
        margin-bottom: {Spacing.MD};
        opacity: 0.6;
    }}
    .empty-state__title {{
        font-size: {FontSize.MD};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_SECONDARY};
        margin: 0 0 0.25rem 0;
    }}
    .empty-state__subtitle {{
        font-size: {FontSize.SM};
        color: {Colors.TEXT_MUTED};
        max-width: 320px;
    }}
    """


# =============================================================================
# SECTION 16 — CHAT CSS
# =============================================================================
# Distinct visual treatment for user vs. assistant messages, styled after
# modern chat UIs (ChatGPT / Claude / Intercom). Extended with timestamps,
# avatars, a typing indicator, markdown formatting, code blocks, tables,
# and lists rendered *inside* chat bubbles.


def _chat_css() -> str:
    """Return CSS for chat message bubbles, chat container, and rich content."""
    return f"""
    /* =========================================================
       6. CHAT
       ========================================================= */

    .chat-container {{
        display: flex;
        flex-direction: column;
        gap: {Spacing.MD};
        padding: {Spacing.SM} 0;
    }}

    .chat-row {{
        display: flex;
        align-items: flex-start;
        gap: {Spacing.SM};
    }}

    .chat-row--user {{
        flex-direction: row-reverse;
    }}

    .chat-avatar {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: {FontSize.SM};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_ON_PRIMARY};
    }}

    .chat-avatar--user {{
        background: {Gradients.PRIMARY};
    }}

    .chat-avatar--assistant {{
        background: {Gradients.TEAL};
    }}

    /* ---- Bubble wrapper (holds bubble + meta so max-width applies once) ---- */
    .chat-bubble-wrapper {{
        display: flex;
        flex-direction: column;
        max-width: 72%;
    }}
    .chat-row--user .chat-bubble-wrapper {{
        align-items: flex-end;
    }}

    /* ---- Base bubble ------------------------------------------------------ */
    .chat-bubble {{
        padding: 0.85rem 1.1rem;
        font-size: {FontSize.MD};
        line-height: {Typography.LINE_HEIGHT_RELAXED};
        box-shadow: {Shadow.XS};
        word-wrap: break-word;
        overflow-wrap: break-word;
    }}

    .chat-bubble--user {{
        background: {Colors.USER_BUBBLE_BG};
        color: {Colors.USER_BUBBLE_TEXT};
        border-radius: {Radius.XL} {Radius.XL} 4px {Radius.XL};
    }}

    .chat-bubble--assistant {{
        background: {Colors.ASSISTANT_BUBBLE_BG};
        color: {Colors.ASSISTANT_BUBBLE_TEXT};
        border: 1px solid {Colors.ASSISTANT_BUBBLE_BORDER};
        border-radius: {Radius.XL} {Radius.XL} {Radius.XL} 4px;
    }}

    /* ---- Timestamp / meta row ------------------------------------------------ */
    .chat-bubble__meta {{
        font-size: {FontSize.XS};
        color: {Colors.TEXT_MUTED};
        margin-top: 0.3rem;
        display: flex;
        gap: 0.4rem;
        align-items: center;
    }}

    /* ---- Rich markdown content rendered inside a bubble ----------------------- */
    .chat-bubble p {{
        margin: 0 0 0.5rem 0;
    }}
    .chat-bubble p:last-child {{
        margin-bottom: 0;
    }}

    .chat-bubble ul,
    .chat-bubble ol {{
        margin: 0.4rem 0 0.6rem 1.1rem;
        padding: 0;
    }}
    .chat-bubble li {{
        margin-bottom: 0.25rem;
    }}

    .chat-bubble a {{
        color: inherit;
        text-decoration: underline;
        text-decoration-color: currentColor;
        text-underline-offset: 2px;
    }}

    /* ---- Inline code -------------------------------------------------------- */
    .chat-bubble code {{
        font-family: {Typography.FONT_FAMILY_MONO};
        background: {Colors.CODE_INLINE_BG};
        color: {Colors.CODE_INLINE_TEXT};
        padding: 0.1rem 0.4rem;
        border-radius: 6px;
        font-size: 0.9em;
    }}

    /* ---- Code block (fenced ``` blocks) ---------------------------------------- */
    .chat-bubble pre {{
        background: {Colors.CODE_BLOCK_BG};
        color: {Colors.CODE_BLOCK_TEXT};
        border-radius: {Radius.MD};
        padding: {Spacing.MD};
        overflow-x: auto;
        margin: 0.5rem 0;
        font-size: {FontSize.SM};
    }}
    .chat-bubble pre code {{
        background: transparent;
        color: inherit;
        padding: 0;
        font-size: inherit;
    }}

    /* ---- Tables inside chat (e.g. retrieved KB comparisons) -------------------- */
    .chat-bubble table {{
        border-collapse: collapse;
        width: 100%;
        margin: 0.5rem 0;
        font-size: {FontSize.SM};
    }}
    .chat-bubble th,
    .chat-bubble td {{
        border: 1px solid {Colors.BORDER};
        padding: 0.4rem 0.6rem;
        text-align: left;
    }}
    .chat-bubble th {{
        background: {Colors.SURFACE_ALT};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
    }}

    /* ---- Typing / thinking indicator -------------------------------------------- */
    .typing-indicator {{
        display: inline-flex;
        gap: 4px;
        padding: 0.6rem 0.9rem;
    }}
    .typing-indicator span {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: {Colors.TEXT_MUTED};
        animation: typing-bounce 1.2s infinite {Animation.EASE_IN_OUT};
    }}
    .typing-indicator span:nth-child(2) {{ animation-delay: 0.15s; }}
    .typing-indicator span:nth-child(3) {{ animation-delay: 0.3s; }}
    """


# =============================================================================
# SECTION 17 — BUTTON VARIANT CSS  (new)
# =============================================================================
# Semantic button variants layered on top of Streamlit's native button via
# a wrapper class + `:has()`-free approach (Streamlit doesn't expose a
# clean per-button class hook, so these are intended to be applied via a
# surrounding `<div class="btn-variant--X">` from components.py, or reused
# directly on custom `<button>` markup inside HTML components).


def _button_css() -> str:
    """Return CSS for semantic button variants: primary/secondary/ghost/outline/danger/success."""
    return f"""
    /* =========================================================
       7. BUTTONS
       ========================================================= */

    .btn {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        padding: 0.55rem 1.25rem;
        border-radius: {Radius.MD};
        font-size: {FontSize.SM};
        font-weight: {Typography.WEIGHT_MEDIUM};
        font-family: {Typography.FONT_FAMILY};
        border: 1px solid transparent;
        cursor: pointer;
        transition: {Animation.transition("background", "border-color", "color", "box-shadow")},
                    {Animation.BUTTON_PRESS};
    }}
    .btn:active {{ transform: translateY(1px) scale(0.98); }}
    .btn:disabled {{
        opacity: 0.55;
        cursor: not-allowed;
        transform: none !important;
    }}

    /* Primary: solid brand blue */
    .btn--primary {{
        background: {Colors.PRIMARY};
        color: {Colors.TEXT_ON_PRIMARY};
        box-shadow: {Shadow.SM};
    }}
    .btn--primary:hover {{ background: {Colors.PRIMARY_DARK}; box-shadow: {Shadow.MD}; }}

    /* Secondary: solid teal, for supportive/alternate actions */
    .btn--secondary {{
        background: {Colors.TEAL};
        color: {Colors.TEXT_ON_PRIMARY};
        box-shadow: {Shadow.SM};
    }}
    .btn--secondary:hover {{ background: {Colors.TEAL_DARK}; box-shadow: {Shadow.MD}; }}

    /* Ghost: transparent, text-only, for low-emphasis actions */
    .btn--ghost {{
        background: transparent;
        color: {Colors.TEXT_SECONDARY};
    }}
    .btn--ghost:hover {{
        background: {Colors.SURFACE_HOVER};
        color: {Colors.TEXT_PRIMARY};
    }}

    /* Outline: bordered, transparent fill */
    .btn--outline {{
        background: transparent;
        color: {Colors.PRIMARY};
        border-color: {Colors.PRIMARY};
    }}
    .btn--outline:hover {{
        background: {Colors.PRIMARY_SOFT};
    }}

    /* Danger: destructive actions (delete conversation, clear KB) */
    .btn--danger {{
        background: {Colors.DANGER};
        color: {Colors.TEXT_ON_PRIMARY};
        box-shadow: {Shadow.SM};
    }}
    .btn--danger:hover {{ background: {Colors.DANGER_DARK}; box-shadow: {Shadow.MD}; }}

    /* Success: confirmations (save, export) */
    .btn--success {{
        background: {Colors.SUCCESS_BTN};
        color: {Colors.TEXT_ON_PRIMARY};
        box-shadow: {Shadow.SM};
    }}
    .btn--success:hover {{ background: {Colors.SUCCESS_BTN_DARK}; box-shadow: {Shadow.MD}; }}
    """


# =============================================================================
# SECTION 18 — INPUT CSS
# =============================================================================
# Additional (non-native-Streamlit) input styling used by custom HTML
# components — search boxes, filter chips, toggle switches — kept distinct
# from the native `.stTextInput` rules in `_base_css()`.


def _input_css() -> str:
    """Return CSS for custom (non-native) input-like components."""
    return f"""
    /* =========================================================
       8. INPUTS
       ========================================================= */

    .input-group {{
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        margin-bottom: {Spacing.MD};
    }}

    .input-label {{
        font-size: {FontSize.SM};
        font-weight: {Typography.WEIGHT_MEDIUM};
        color: {Colors.TEXT_SECONDARY};
    }}

    .input-help-text {{
        font-size: {FontSize.XS};
        color: {Colors.TEXT_MUTED};
    }}

    .input-error-text {{
        font-size: {FontSize.XS};
        color: {Colors.ERROR};
    }}

    /* Toggle switch (used for e.g. "show sources" / "verbose mode") */
    .toggle-switch {{
        position: relative;
        display: inline-block;
        width: 40px;
        height: 22px;
    }}
    .toggle-switch input {{
        opacity: 0;
        width: 0;
        height: 0;
    }}
    .toggle-slider {{
        position: absolute;
        cursor: pointer;
        inset: 0;
        background: {Colors.BORDER_STRONG};
        border-radius: {Radius.PILL};
        transition: {Animation.transition("background", duration=Animation.FAST)};
    }}
    .toggle-slider::before {{
        content: "";
        position: absolute;
        width: 16px;
        height: 16px;
        left: 3px;
        bottom: 3px;
        background: {Colors.SURFACE};
        border-radius: 50%;
        transition: {Animation.transition("transform", duration=Animation.FAST)};
    }}
    .toggle-switch input:checked + .toggle-slider {{
        background: {Colors.PRIMARY};
    }}
    .toggle-switch input:checked + .toggle-slider::before {{
        transform: translateX(18px);
    }}
    """


# =============================================================================
# SECTION 19 — METRIC CARD CSS
# =============================================================================
# Compact analytics cards for pipeline metrics (retrieval latency, tokens
# used, documents retrieved, confidence score). Extended with trend arrows,
# mini status badges, a loading skeleton state, and hover animation.


def _metric_css() -> str:
    """Return CSS for KPI / metric summary cards, including loading skeletons."""
    return f"""
    /* =========================================================
       9. METRICS
       ========================================================= */

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: {Spacing.MD};
        margin-bottom: {Spacing.LG};
    }}

    .metric-card {{
        background: {Colors.SURFACE};
        border: 1px solid {Colors.BORDER};
        border-radius: {Radius.LG};
        padding: {Spacing.MD} {Spacing.LG};
        box-shadow: {Shadow.SM};
        position: relative;
        overflow: hidden;
        transition: {Animation.CARD_LIFT};
    }}
    .metric-card:hover {{
        box-shadow: {Shadow.MD};
        transform: translateY(-2px);
    }}

    /* Colored top accent bar, gives each metric card visual identity */
    .metric-card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: {Gradients.METRIC};
    }}

    .metric-card--success::before {{ background: {Colors.SUCCESS}; }}
    .metric-card--warning::before {{ background: {Colors.WARNING}; }}
    .metric-card--error::before {{ background: {Colors.ERROR}; }}

    .metric-card__icon {{
        font-size: {FontSize.LG};
        margin-bottom: {Spacing.SM};
    }}

    .metric-card__label {{
        font-size: {FontSize.XS};
        color: {Colors.TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: {Typography.WEIGHT_MEDIUM};
        margin: 0 0 0.25rem 0;
    }}

    .metric-card__value {{
        font-size: {FontSize.XXL};
        font-weight: {Typography.WEIGHT_BOLD};
        color: {Colors.TEXT_PRIMARY};
        margin: 0;
        line-height: {Typography.LINE_HEIGHT_TIGHT};
    }}

    .metric-card__delta {{
        display: inline-flex;
        align-items: center;
        gap: 0.2rem;
        font-size: {FontSize.XS};
        font-weight: {Typography.WEIGHT_MEDIUM};
        margin-top: 0.25rem;
    }}

    .metric-card__delta--up {{ color: {Colors.SUCCESS}; }}
    .metric-card__delta--down {{ color: {Colors.ERROR}; }}
    .metric-card__delta--flat {{ color: {Colors.TEXT_MUTED}; }}

    /* Small trend-arrow glyphs; combine with the delta classes above */
    .metric-card__trend-arrow--up::before {{ content: "\\2191"; }}
    .metric-card__trend-arrow--down::before {{ content: "\\2193"; }}
    .metric-card__trend-arrow--flat::before {{ content: "\\2192"; }}

    /* Mini status badge pinned to the top-right corner of a metric card */
    .metric-card__badge {{
        position: absolute;
        top: {Spacing.SM};
        right: {Spacing.SM};
        font-size: {FontSize.XS};
        font-weight: {Typography.WEIGHT_MEDIUM};
        padding: 0.15rem 0.5rem;
        border-radius: {Radius.PILL};
    }}

    /* ---- Metric card loading skeleton -------------------------------------- */
    .metric-card--loading .metric-card__label,
    .metric-card--loading .metric-card__value {{
        color: transparent;
        border-radius: {Radius.SM};
        background: {Gradients.SKELETON_SHIMMER};
        background-size: 400% 100%;
        animation: shimmer 1.4s ease-in-out infinite;
    }}
    .metric-card--loading .metric-card__label {{
        display: inline-block;
        width: 60%;
        height: 0.8rem;
    }}
    .metric-card--loading .metric-card__value {{
        display: inline-block;
        width: 40%;
        height: 1.6rem;
        margin-top: 0.3rem;
    }}
    """


# =============================================================================
# SECTION 20 — BADGE CSS  (new)
# =============================================================================
# Small pill-shaped status labels (info / success / warning / error) reused
# across chat metadata, metric cards, and the sidebar status panel.


def _badge_css() -> str:
    """Return CSS for semantic badge/pill variants."""
    return f"""
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.65rem;
        border-radius: {Radius.PILL};
        font-size: {FontSize.XS};
        font-weight: {Typography.WEIGHT_MEDIUM};
        border: 1px solid transparent;
        white-space: nowrap;
    }}
    .badge--info {{
        background: {Colors.INFO_BG};
        color: {Colors.INFO_DARK};
        border-color: {Colors.INFO_BORDER};
    }}
    .badge--success {{
        background: {Colors.SUCCESS_BG};
        color: {Colors.SUCCESS_DARK};
        border-color: {Colors.SUCCESS_BORDER};
    }}
    .badge--warning {{
        background: {Colors.WARNING_BG};
        color: {Colors.WARNING_DARK};
        border-color: {Colors.WARNING_BORDER};
    }}
    .badge--error {{
        background: {Colors.ERROR_BG};
        color: {Colors.ERROR_DARK};
        border-color: {Colors.ERROR_BORDER};
    }}
    .badge__dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: currentColor;
        display: inline-block;
    }}
    """


# =============================================================================
# SECTION 21 — LOADING COMPONENT CSS  (new)
# =============================================================================
# Spinner, skeleton card, and shimmer keyframes shared by chat "thinking",
# metric card loading state, and any future async panel.


def _loading_css() -> str:
    """Return CSS for spinners, skeleton placeholders, and shimmer animation."""
    return f"""
    /* ---- Spinner ------------------------------------------------------- */
    .spinner {{
        width: 20px;
        height: 20px;
        border: 2.5px solid {Colors.BORDER};
        border-top-color: {Colors.PRIMARY};
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        display: inline-block;
    }}
    .spinner--lg {{
        width: 36px;
        height: 36px;
        border-width: 3.5px;
    }}

    /* ---- Skeleton card --------------------------------------------------- */
    .skeleton-card {{
        background: {Colors.SURFACE};
        border: 1px solid {Colors.BORDER};
        border-radius: {Radius.LG};
        padding: {Spacing.LG};
        box-shadow: {Shadow.SM};
    }}
    .skeleton-line {{
        height: 0.85rem;
        border-radius: {Radius.SM};
        background: {Gradients.SKELETON_SHIMMER};
        background-size: 400% 100%;
        animation: shimmer 1.4s ease-in-out infinite;
        margin-bottom: 0.6rem;
    }}
    .skeleton-line--60 {{ width: 60%; }}
    .skeleton-line--80 {{ width: 80%; }}
    .skeleton-line--100 {{ width: 100%; }}
    """


# =============================================================================
# SECTION 22 — ANIMATION KEYFRAMES
# =============================================================================
# All @keyframes definitions live in one place. Durations/easings inside
# each keyframe still reference the Animation token class.


def _animation_css() -> str:
    """Return @keyframes definitions used by chat, metrics, and loading states."""
    return f"""
    /* =========================================================
       10. ANIMATIONS
       ========================================================= */

    @keyframes typing-bounce {{
        0%, 60%, 100% {{ transform: translateY(0); opacity: 0.5; }}
        30% {{ transform: translateY(-4px); opacity: 1; }}
    }}

    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}

    @keyframes shimmer {{
        0% {{ background-position: 100% 0; }}
        100% {{ background-position: 0 0; }}
    }}

    @keyframes fade-in {{
        from {{ opacity: 0; transform: translateY(4px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .anim-fade-in {{
        animation: fade-in {Animation.NORMAL} {Animation.EASE_OUT};
    }}

    /* Reusable hover-lift / hover-scale utility classes built from tokens */
    .hover-lift {{
        transition: {Animation.CARD_LIFT};
    }}
    .hover-lift:hover {{
        box-shadow: {Shadow.MD};
        transform: translateY(-2px);
    }}
    .hover-scale {{
        transition: {Animation.HOVER_SCALE};
    }}
    .hover-scale:hover {{
        transform: scale(1.03);
    }}
    """


# =============================================================================
# SECTION 23 — RESPONSIVE / MEDIA QUERY CSS
# =============================================================================
# Mobile-first adjustments layered on top of the desktop-default styles
# defined above. Ensures nothing overflows on tablets/phones: chat bubbles
# widen, the metric grid collapses, and the header stacks vertically.


def _responsive_css() -> str:
    """Return media-query overrides for tablet and mobile viewports."""
    return f"""
    /* =========================================================
       11. RESPONSIVE
       ========================================================= */

    {Breakpoints.down(Breakpoints.LAPTOP)} {{
        .block-container {{
            padding-left: {Spacing.MD};
            padding-right: {Spacing.MD};
        }}
        .metric-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}

    {Breakpoints.down(Breakpoints.TABLET)} {{
        .app-header {{
            flex-direction: column;
            align-items: flex-start;
            gap: {Spacing.SM};
        }}
        .chat-bubble-wrapper {{
            max-width: 90%;
        }}
        .metric-grid {{
            grid-template-columns: 1fr 1fr;
            gap: {Spacing.SM};
        }}
        .u-grid-2, .u-grid-3 {{
            grid-template-columns: 1fr;
        }}
    }}

    {Breakpoints.down(Breakpoints.MOBILE)} {{
        .app-header__logo {{
            width: 36px;
            height: 36px;
        }}
        .chat-bubble-wrapper {{
            max-width: 100%;
        }}
        .metric-grid {{
            grid-template-columns: 1fr;
        }}
        .chat-bubble {{
            font-size: {FontSize.BASE};
            padding: 0.7rem 0.9rem;
        }}
    }}
    """


# =============================================================================
# SECTION 24 — PRINT CSS
# =============================================================================
# Hides chrome that makes no sense on paper (sidebar, header status badge,
# buttons, spinners) when the user prints a conversation transcript or a
# metrics report.


def _print_css() -> str:
    """Return print-media rules that hide interactive/decorative UI chrome."""
    return """
    /* =========================================================
       12. PRINT SAFETY
       ========================================================= */
    @media print {
        section[data-testid="stSidebar"],
        header[data-testid="stHeader"],
        .app-header__status,
        .stButton,
        .typing-indicator,
        .spinner {
            display: none !important;
        }
        .app-card, .chat-bubble, .metric-card {
            box-shadow: none !important;
            border: 1px solid #ccc !important;
        }
        body {
            background: #ffffff !important;
        }
    }
    """


# =============================================================================
# SECTION 25 — CSS INJECTION HELPERS (public API)
# =============================================================================
# Public functions that other ui/ modules call to apply the design system.
# Keeping injection logic here means app.py only needs one call
# (`apply_global_styles()`) to theme the entire application.
#
# NOTE ON BACKWARD COMPATIBILITY:
#   apply_global_styles(), inject_custom_css(), load_external_stylesheet(),
#   and get_status_badge_html() keep their original names and signatures.
#   New helpers (get_badge_html, get_spinner_html, get_skeleton_card_html,
#   get_empty_state_html) are additive only.


def _wrap_style_tag(css: str) -> str:
    """
    Wrap a raw CSS string in a <style> tag for st.markdown injection.

    Args:
        css: Raw CSS text (no surrounding tag).

    Returns:
        The CSS wrapped as `<style>...</style>`, ready for
        `st.markdown(..., unsafe_allow_html=True)`.
    """
    return f"<style>{css}</style>"


def apply_global_styles() -> None:
    """
    Inject the full design system into the current Streamlit page.

    This compiles every CSS section — base/reset, layout & utilities,
    header, sidebar, cards, chat, buttons, inputs, metrics, badges,
    loading components, animation keyframes, responsive overrides, and
    print safety — into a single stylesheet and injects it once.

    Call this ONCE, near the top of app.py, before rendering any other UI:

        from ui.styles import apply_global_styles
        apply_global_styles()

    Returns:
        None. The stylesheet is injected as a side effect via `st.markdown`.
    """
    css_sections = (
        _base_css(),
        _layout_css(),
        _header_css(),
        _sidebar_css(),
        _card_css(),
        _chat_css(),
        _button_css(),
        _input_css(),
        _metric_css(),
        _badge_css(),
        _loading_css(),
        _animation_css(),
        _responsive_css(),
        _print_css(),
    )
    full_css = "".join(css_sections)
    st.markdown(_wrap_style_tag(full_css), unsafe_allow_html=True)


def inject_custom_css(css: str) -> None:
    """
    Inject an arbitrary, ad-hoc CSS string.

    Useful when a specific component (e.g. components.py) needs a one-off
    style tweak that doesn't belong in the shared design system above.

    Args:
        css: Raw CSS rules (without the surrounding <style> tag).

    Returns:
        None. The CSS is injected as a side effect via `st.markdown`.
    """
    st.markdown(_wrap_style_tag(css), unsafe_allow_html=True)


def load_external_stylesheet(path: str) -> None:
    """
    Load and inject an external .css file (e.g. static/style.css).

    This lets designers/devs tweak plain CSS without touching Python, while
    still funneling through this module's single injection point.

    Args:
        path: Filesystem path to a .css file, e.g. "static/style.css".

    Returns:
        None. On success the file's contents are injected via `st.markdown`.
        If the file does not exist, a non-fatal `st.warning` is shown instead
        of raising, so the app keeps running without the optional stylesheet.
    """
    try:
        with open(path, "r", encoding="utf-8") as css_file:
            css_content = css_file.read()
        st.markdown(_wrap_style_tag(css_content), unsafe_allow_html=True)
    except FileNotFoundError:
        # Fail gracefully: the app should still run even if the optional
        # external stylesheet hasn't been created yet.
        st.warning(f"Stylesheet not found at '{path}'. Skipping.")


def get_status_badge_html(label: str = "System Online", online: bool = True) -> str:
    """
    Build the small HTML snippet for the header's live status badge.

    Args:
        label: Text shown next to the status dot.
        online: Whether to render success (green) or error (red) styling.

    Returns:
        An HTML string ready to be passed to
        `st.markdown(..., unsafe_allow_html=True)`.
    """
    color = Colors.SUCCESS if online else Colors.ERROR
    bg = Colors.SUCCESS_BG if online else Colors.ERROR_BG
    return (
        f'<span class="app-header__status" '
        f'style="background:{bg};color:{color};">'
        f'<span class="app-header__status-dot" style="background:{color};"></span>'
        f'{label}</span>'
    )


def get_badge_html(label: str, variant: str = "info") -> str:
    """
    Build an HTML snippet for a semantic status badge/pill.

    Args:
        label: Text shown inside the badge (e.g. "3 sources", "Degraded").
        variant: One of "info", "success", "warning", "error".

    Returns:
        An HTML string using the `.badge` / `.badge--{variant}` classes,
        ready for `st.markdown(..., unsafe_allow_html=True)`.
    """
    valid_variants = {"info", "success", "warning", "error"}
    safe_variant = variant if variant in valid_variants else "info"
    return (
        f'<span class="badge badge--{safe_variant}">'
        f'<span class="badge__dot"></span>{label}</span>'
    )


def get_spinner_html(size: str = "md") -> str:
    """
    Build an HTML snippet for an inline loading spinner.

    Args:
        size: "md" (default, 20px) or "lg" (36px, uses `.spinner--lg`).

    Returns:
        An HTML string using the `.spinner` class, ready for
        `st.markdown(..., unsafe_allow_html=True)`.
    """
    size_class = " spinner--lg" if size == "lg" else ""
    return f'<span class="spinner{size_class}"></span>'


def get_skeleton_card_html(lines: int = 3) -> str:
    """
    Build an HTML snippet for a shimmering skeleton placeholder card.

    Useful while the RAG pipeline is retrieving/generating and there is no
    real content to render yet (e.g. a placeholder chat bubble or metric
    panel).

    Args:
        lines: Number of skeleton text lines to render (default 3).

    Returns:
        An HTML string using `.skeleton-card` / `.skeleton-line`, ready
        for `st.markdown(..., unsafe_allow_html=True)`.
    """
    widths = ["skeleton-line--100", "skeleton-line--80", "skeleton-line--60"]
    rendered_lines = "".join(
        f'<div class="skeleton-line {widths[i % len(widths)]}"></div>'
        for i in range(max(lines, 1))
    )
    return f'<div class="skeleton-card">{rendered_lines}</div>'


def get_empty_state_html(title: str, subtitle: str = "", icon: str = "\U0001F4C4") -> str:
    """
    Build an HTML snippet for an empty-state placeholder.

    Reusable for "no chat yet", "no documents indexed", "no search
    results", and "no metrics yet" panels.

    Args:
        title: Short, bold headline (e.g. "No conversations yet").
        subtitle: Optional supporting sentence.
        icon: A single emoji/character shown above the title.

    Returns:
        An HTML string using the `.empty-state` classes, ready for
        `st.markdown(..., unsafe_allow_html=True)`.
    """
    subtitle_html = (
        f'<p class="empty-state__subtitle">{subtitle}</p>' if subtitle else ""
    )
    return (
        '<div class="empty-state">'
        f'<div class="empty-state__icon">{icon}</div>'
        f'<p class="empty-state__title">{title}</p>'
        f"{subtitle_html}"
        "</div>"
    )