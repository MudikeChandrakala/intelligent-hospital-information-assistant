"""
ui/report_analysis.py
=============================================================================
Medical report upload and OCR preview page for the Intelligent Hospital
Information Assistant frontend.

This module renders only the medical report upload UI and display logic.
OCR and structured lab-test-result extraction live in
`modules/report_analyzer.py`.

This page presents COMPUTER-PRINTED laboratory report results (CBC, lipid
profile, glucose, liver/kidney function, etc.) — not handwritten
prescriptions. Deterministic extraction remains owned by
`report_analyzer.py`; this page does not diagnose, recommend treatment, or
fabricate any value, unit, reference range, or status. An optional AI
explanation is generated only from the structured `tests` returned by the
analyzer and can never override those deterministic results.

The "Download Report (PDF)" and "Copy Report" controls, and the plain-text
report they both share, are generated only from the same structured
`tests` list already used to render the Report Summary / Important
Findings / Laboratory Results sections on this page — never recalculated
separately.

-----------------------------------------------------------------------------
Public API
-----------------------------------------------------------------------------
    render_report_analysis_page() -> None
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import textwrap
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError
import streamlit as st

from ui.components import (
    render_info_panel,
    render_section_header,
)
from modules.report_analyzer import get_report_analyzer
from modules.report_ai_service import analyze_report_findings

logger = logging.getLogger("hospital_assistant.report_ui")

# =============================================================================
# PRIVATE HELPERS
# =============================================================================


_ANALYSIS_RESULT_KEY = "report_analysis_result"
_ANALYSIS_FILE_HASH_KEY = "report_analysis_file_hash"
_ANALYSIS_DURATION_MS_KEY = "report_analysis_duration_ms"
_AI_REPORT_EXPLANATION_KEY = "report_ai_explanation"
_AI_REPORT_FILE_HASH_KEY = "report_ai_file_hash"

# report_analyzer.py returns confidence on a 0.0-1.0 scale (the mean of
# EasyOCR's raw per-block scores), not a percentage.
_LOW_CONFIDENCE_THRESHOLD = 0.70

_SAFETY_NOTICE = (
    "Values outside the reference range may require clinical attention. "
    "This analysis is informational and does not provide a diagnosis."
)

_ABNORMAL_STATUSES = ("High", "Low")

# The section headings that make up the complete report, in order. Both the
# PDF export and the "Copy Report" control are built from exactly these
# sections and nothing else, so a downstream test can assert the full report
# is always present.
_VISIBLE_REPORT_SECTIONS = {
    "Report Summary",
    "Important Findings",
    "Laboratory Results",
    "Informational Only",
}

_INFORMATIONAL_ONLY_LINES = (
    "Values outside reference ranges may require clinical attention.",
    "This analysis is informational and does not provide a diagnosis.",
)


def _format_file_size(size_bytes: int) -> str:
    """Format a file size in bytes as a readable KB/MB string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / 1024:.1f} KB"


def _get_image_resolution(uploaded_bytes: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Safely read the uploaded image resolution."""
    try:
        with Image.open(BytesIO(uploaded_bytes)) as image:
            return image.size
    except Exception:
        return None, None


def _load_uploaded_image(image_bytes: bytes) -> Optional[Image.Image]:
    """Load and validate the uploaded image exactly once."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            return image.copy()
    except UnidentifiedImageError:
        return None
    except Exception:
        return None


def _hash_bytes(data: bytes) -> str:
    """Return a stable hash for the uploaded medical report image."""
    return hashlib.sha256(data).hexdigest()


@st.cache_data(show_spinner=False)
def _analyze_report_cached(image_bytes: bytes) -> Dict[str, Any]:
    """Cache OCR and structured lab-result extraction for an unchanged upload."""
    return get_report_analyzer().analyze(image_bytes)


def _clear_cached_analysis_if_needed(file_hash: str) -> None:
    """Reset cached analysis when the uploaded file changes."""
    cached_hash = st.session_state.get(_ANALYSIS_FILE_HASH_KEY)
    if cached_hash and cached_hash != file_hash:
        st.session_state[_ANALYSIS_RESULT_KEY] = None
        st.session_state[_ANALYSIS_DURATION_MS_KEY] = None
        st.session_state[_AI_REPORT_EXPLANATION_KEY] = None
        st.session_state[_AI_REPORT_FILE_HASH_KEY] = file_hash
    st.session_state[_ANALYSIS_FILE_HASH_KEY] = file_hash


def _summarize_tests(tests: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count parameters detected and their status breakdown."""
    summary = {"total": len(tests), "normal": 0, "high": 0, "low": 0, "unknown": 0}
    for test in tests:
        status = test.get("status", "Unknown")
        if status == "Normal":
            summary["normal"] += 1
        elif status == "High":
            summary["high"] += 1
        elif status == "Low":
            summary["low"] += 1
        else:
            summary["unknown"] += 1
    return summary


def _render_report_summary(tests: List[Dict[str, Any]]) -> None:
    """Render the report-summary heading without duplicating right-side metrics."""
    render_section_header(
        title="Report Summary",
        subtitle="Overview of the laboratory parameters detected in this report.",
        icon="📊",
    )


def _render_important_findings(tests: List[Dict[str, Any]]) -> None:
    """Render a focused list of the High/Low results that stand out."""
    render_section_header(
        title="Important Findings",
        subtitle="Parameters that fall outside their reference range.",
        icon="🔎",
    )

    abnormal_tests = [test for test in tests if test.get("status") in _ABNORMAL_STATUSES]


    if not tests:
        render_info_panel(
            title="Abnormality Status Unavailable",
            message=(
                "No reliable laboratory parameters were extracted, so abnormality "
                "status cannot be determined. Manual verification is required."
            ),
            variant="warning",
        )
        return

    if not abnormal_tests:
        render_info_panel(
            title="No Abnormal Results",
            message="All detected parameters with a known reference range are within range.",
            variant="info",
        )
        return

    for test in abnormal_tests:
        status = test.get("status", "Unknown")
        icon = "🔺" if status == "High" else "🔻"
        test_name = test.get("test_name") or "Unnamed Parameter"
        value = test.get("value", "")
        unit = test.get("unit") or ""
        reference_range = test.get("reference_range") or "Not Available"

        with st.container(border=True):
            st.markdown(
                f"{icon} **{test_name}** — {value} {unit}  \n"
                f"Reference Range: {reference_range}  •  Status: **{status}**"
            )


def _format_result_value(value: Any) -> str:
    """Format a numeric test value for display without a stray '.0'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _render_laboratory_results_table(tests: List[Dict[str, Any]]) -> None:
    """Render the full structured Test | Result | Unit | Reference Range | Status table."""
    render_section_header(
        title="Laboratory Results",
        subtitle="All structured parameters extracted from the uploaded report.",
        icon="🧪",
    )

    if not tests:
        render_info_panel(
            title="No Structured Results",
            message="OCR completed, but no structured laboratory parameters could be extracted.",
            variant="warning",
        )
        return

    table_rows = [
        {
            "Test": test.get("test_name", ""),
            "Result": _format_result_value(test.get("value")),
            "Unit": test.get("unit") or "—",
            "Reference Range": test.get("reference_range") or "Not Available",
            "Status": test.get("status", "Unknown"),
        }
        for test in tests
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def _render_ai_report_analysis(tests: List[Dict[str, Any]], file_hash: str) -> None:
    """
    Render the optional AI explanation for the already-structured lab results.

    The AI service receives only `tests` produced by report_analyzer.py.
    It never replaces or recalculates deterministic values/statuses. The
    generated explanation is kept in session state for the current upload.
    """
    render_section_header(
        title="AI Report Analysis",
        subtitle="Plain-language explanation of the structured laboratory findings.",
        icon="🤖",
    )

    cached_hash = st.session_state.get(_AI_REPORT_FILE_HASH_KEY)
    cached_result = st.session_state.get(_AI_REPORT_EXPLANATION_KEY)

    if cached_hash and cached_hash != file_hash:
        cached_result = None
        st.session_state[_AI_REPORT_EXPLANATION_KEY] = None

    if cached_result and cached_result.get("success"):
        st.markdown(cached_result.get("explanation", ""))
        return

    if cached_result and not cached_result.get("success"):
        warning = cached_result.get("warning")
        if warning:
            st.caption(f"⚠️ {warning}")

    generate_clicked = st.button(
        "Generate AI Report Explanation",
        use_container_width=True,
        key=f"generate-ai-report-{file_hash[:12]}",
    )

    if not generate_clicked:
        st.caption(
            "The explanation uses only the structured laboratory results "
            "already extracted above. It does not change their values or statuses."
        )
        return

    with st.spinner("Generating AI Report Explanation..."):
        ai_result = analyze_report_findings(tests)

    st.session_state[_AI_REPORT_EXPLANATION_KEY] = ai_result
    st.session_state[_AI_REPORT_FILE_HASH_KEY] = file_hash

    if ai_result.get("success"):
        st.markdown(ai_result.get("explanation", ""))
    else:
        warning = ai_result.get("warning") or "AI explanation is unavailable."
        st.warning(warning)


def _render_safety_notice() -> None:
    """Render the required informational-only safety message."""
    render_info_panel(
        title="Informational Only",
        message=_SAFETY_NOTICE,
        variant="warning",
    )


def _render_analysis_warnings(warnings: List[str]) -> None:
    """Surface any warnings report_analyzer.py itself returned (never invented here)."""
    for warning in warnings:
        st.caption(f"⚠️ {warning}")


def _render_report_debug_trace(analysis_result: Dict[str, Any]) -> None:
    """
    Lightweight diagnostic view of the raw OCR/extraction output for this
    upload. Reads only values already present on `analysis_result` — does
    not call OCR or extraction again and does not alter the result.
    """
    raw_text = str(analysis_result.get("raw_text", ""))
    confidence = analysis_result.get("confidence")
    warnings = list(analysis_result.get("warnings", []))
    tests = list(analysis_result.get("tests", []))

    with st.expander("🔍 Report Analysis Debug Trace", expanded=False):
        st.caption(
            "Diagnostic view of the exact values produced by report_analyzer.py "
            "for this upload."
        )

        if confidence is not None:
            st.caption(f"Reported OCR confidence: {confidence}")

        st.markdown("**1. RAW OCR TEXT**")
        st.text_area(
            "Raw OCR Text",
            value=raw_text if raw_text else "(empty)",
            height=180,
            key="debug_report_raw_ocr_text",
            label_visibility="collapsed",
        )

        st.markdown("**2. WARNINGS**")
        if not warnings:
            st.caption("No warnings returned.")
        else:
            for warning in warnings:
                st.write(f"- {warning}")

        st.markdown("**3. STRUCTURED TESTS**")
        st.json(tests, expanded=False)


# =============================================================================
# STRUCTURED REPORT TEXT (shared by "Copy Report" and the PDF export)
# =============================================================================


def _build_structured_report_text(tests: List[Dict[str, Any]]) -> str:
    """
    Build the complete plain-text medical report for Copy Report and PDF.

    The deterministic sections come from the structured laboratory results.
    If an AI explanation has already been generated for the current upload,
    the exact cached explanation is appended without making another Gemini call.
    """
    summary = _summarize_tests(tests)
    abnormal_tests = [test for test in tests if test.get("status") in _ABNORMAL_STATUSES]

    lines: List[str] = []

    lines.append("# Report Summary")
    lines.append(f"- Parameters Detected: {summary['total']}")
    lines.append(f"- Normal: {summary['normal']}")
    lines.append(f"- High: {summary['high']}")
    lines.append(f"- Low: {summary['low']}")
    lines.append(f"- Unknown: {summary['unknown']}")
    lines.append("")

    lines.append("# Important Findings")
    if abnormal_tests:
        for test in abnormal_tests:
            test_name = test.get("test_name") or "Unnamed Parameter"
            value = _format_result_value(test.get("value"))
            unit = test.get("unit") or ""
            reference_range = test.get("reference_range") or "Not Available"
            status = test.get("status", "Unknown")
            value_display = f"{value} {unit}".strip()
            lines.append(
                f"- {test_name}: {value_display} "
                f"(Reference Range: {reference_range}, Status: {status})"
            )
    else:
        lines.append("- No Abnormal Results")
    lines.append("")

    lines.append("# Laboratory Results")
    if tests:
        for test in tests:
            test_name = test.get("test_name") or "Unnamed Parameter"
            value = _format_result_value(test.get("value"))
            unit = test.get("unit") or "Not Available"
            reference_range = test.get("reference_range") or "Not Available"
            status = test.get("status", "Unknown")
            lines.append(
                f"- Test: {test_name} | Result: {value} | Unit: {unit} | "
                f"Reference Range: {reference_range} | Status: {status}"
            )
    else:
        lines.append("- No Structured Results")
    lines.append("")

    lines.append("# Informational Only")
    for notice_line in _INFORMATIONAL_ONLY_LINES:
        lines.append(f"- {notice_line}")

    # Include the already-generated AI explanation in both exports. This reads
    # only from session state; it never calls Gemini again during export/copy.
    ai_result = st.session_state.get(_AI_REPORT_EXPLANATION_KEY)
    if isinstance(ai_result, dict) and ai_result.get("success"):
        ai_explanation = str(ai_result.get("explanation") or "").strip()
        if ai_explanation:
            lines.append("")
            lines.append("# AI Report Analysis")
            lines.extend(ai_explanation.splitlines())

    return "\n".join(lines)


def _split_report_sections(report_text: str) -> List[Tuple[str, str]]:
    """Split the Markdown report into displayable heading/body sections."""
    sections: List[Tuple[str, List[str]]] = []
    heading = "Medical Report Analysis"
    body: List[str] = []

    for line in report_text.splitlines():
        match = re.match(r"^#{1,3}\s+(.+?)\s*$", line)
        if match:
            if body or heading != "Medical Report Analysis":
                sections.append((heading, body))
            heading = match.group(1).strip()
            body = []
        else:
            body.append(line)

    if body or not sections:
        sections.append((heading, body))

    return [(section_heading, "\n".join(lines).strip()) for section_heading, lines in sections]


def _pdf_safe_text(value: Any) -> str:
    """Convert report content to the built-in PDF font's safe character set."""
    return str(value).encode("latin-1", "replace").decode("latin-1")


def _build_report_pdf(report_text: str, tests: Optional[List[Dict[str, Any]]] = None) -> bytes:
    """
    Render the complete medical report, including the cached AI explanation
    when available, as a text-only PDF.

    PDF finalization (`pdf.save()`) happens exactly once, after every
    section and every row has been written to the canvas — never inside
    the section loop — so a report with more than one section is never
    truncated to just the first one.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - dependency guard for deployments
        raise RuntimeError("PDF generation is unavailable because ReportLab is not installed.") from exc

    buffer = BytesIO()
    # Uncompressed: this is a short text-only report, and keeping the
    # content stream uncompressed lets tests (and basic text search in a
    # PDF viewer) find report text directly in the file bytes.
    pdf = canvas.Canvas(buffer, pagesize=A4, pageCompression=0)
    page_width, page_height = A4
    margin = 0.65 * inch
    y_position = page_height - margin

    def write_line(text: str, *, bold: bool = False, size: int = 10) -> None:
        nonlocal y_position
        font = "Helvetica-Bold" if bold else "Helvetica"
        for wrapped_line in textwrap.wrap(_pdf_safe_text(text), width=100) or [""]:
            if y_position < margin:
                pdf.showPage()
                y_position = page_height - margin
            pdf.setFont(font, size)
            pdf.drawString(margin, y_position, wrapped_line)
            y_position -= size + 4

    write_line("Medical Report Analysis", bold=True, size=16)
    y_position -= 8

    for heading, body in _split_report_sections(report_text):
        write_line(heading, bold=True, size=12)
        for line in body.splitlines():
            write_line(line or " ")
        y_position -= 6

    # Finalize only after every section above has been written.
    pdf.save()
    return buffer.getvalue()


def _render_copy_report_control(report_text: str) -> None:
    """Render a browser-side copy-to-clipboard control for the report text."""
    button_id = f"copy-report-{hashlib.sha256(report_text.encode('utf-8')).hexdigest()[:12]}"
    payload = json.dumps(report_text).replace("</", "<\\/")
    st.components.v1.html(
        f"""
        <button id="{button_id}" style="border: 1px solid #d8dee9; border-radius: 8px; background: #ffffff; color: #1f2937; cursor: pointer; font: 600 14px sans-serif; padding: 8px 14px; width: 100%;">
            📋 Copy Report
        </button>
        <script>
        const button = document.getElementById({json.dumps(button_id)});
        button.addEventListener("click", async () => {{
            await navigator.clipboard.writeText({payload});
            button.textContent = "✓ Report Copied";
            setTimeout(() => {{ button.textContent = "📋 Copy Report"; }}, 1800);
        }});
        </script>
        """,
        height=48,
    )


def _render_report_actions(tests: List[Dict[str, Any]]) -> None:
    """Render the [Download Report (PDF)] [Copy Report] controls."""
    report_text = _build_structured_report_text(tests)

    download_col, copy_col = st.columns([1, 1], gap="small")
    with download_col:
        try:
            st.download_button(
                "Download Report (PDF)",
                data=_build_report_pdf(report_text),
                file_name="medical_report_analysis.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except RuntimeError as exc:
            st.caption(str(exc))
    with copy_col:
        _render_copy_report_control(report_text)


# =============================================================================
# PAGE RENDERER
# =============================================================================


def render_report_analysis_page() -> None:
    """Render the medical report upload, preview, and laboratory-results interface."""
    render_section_header(
        title="Medical Report Analysis",
        subtitle="Upload a computer-printed medical report to extract laboratory test results and identify values that may require attention.",
        icon="📋",
    )

    with st.container(border=True):
        upload_file = st.file_uploader(
            "Upload Medical Report",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Supported formats: JPG, JPEG, PNG",
        )
        st.caption("Supported formats: JPG, JPEG, PNG")

    if upload_file is None:
        render_info_panel(
            title="Upload Medical Report",
            message="Select a JPG, JPEG, or PNG computer-printed medical report to preview and analyze it.",
            variant="info",
        )
        return

    image_bytes = upload_file.getvalue()
    if not image_bytes:
        st.error("Unreadable image. Upload a clear JPG, JPEG, or PNG medical report.")
        return

    image = _load_uploaded_image(image_bytes)
    if image is None:
        st.error("Unsupported, invalid, or corrupted image. Upload a clear JPG, JPEG, or PNG medical report.")
        return

    file_hash = _hash_bytes(image_bytes)
    _clear_cached_analysis_if_needed(file_hash)

    image_width, image_height = _get_image_resolution(image_bytes)
    image_size_label = _format_file_size(len(image_bytes))
    resolution_label = (
        f"{image_width} × {image_height}" if image_width is not None and image_height is not None else "Not Available"
    )

    st.image(image, caption=upload_file.name, use_container_width=True)
    st.caption(f"{image_size_label} • {resolution_label}")

    analyze_clicked = st.button("Analyze Medical Report", use_container_width=True)
    if analyze_clicked:
        analysis_start_time = time.perf_counter()
        with st.spinner("Analyzing Medical Report..."):
            analysis_result = _analyze_report_cached(image_bytes)
        st.session_state[_ANALYSIS_DURATION_MS_KEY] = (time.perf_counter() - analysis_start_time) * 1000
        st.session_state[_ANALYSIS_RESULT_KEY] = analysis_result
    else:
        analysis_result = st.session_state.get(_ANALYSIS_RESULT_KEY)

    if not analysis_result:
        render_info_panel(
            title="Ready to Analyze",
            message="Click Analyze Medical Report to run OCR and extract structured laboratory results.",
            variant="info",
        )
        return

    if not analysis_result.get("success", False):
        warnings = list(analysis_result.get("warnings", []))
        error_message = warnings[0] if warnings else "Unable to analyze the uploaded medical report."
        if "no text" in error_message.lower() or "unable to extract" in error_message.lower():
            st.error("No readable text detected. The medical report may be blurry, dark, or too small to read.")
        elif "unable to decode" in error_message.lower() or "unreadable" in error_message.lower():
            st.error("Unreadable image. Upload a clearer, well-lit medical report and try again.")
        else:
            st.error(error_message)
        for extra_warning in warnings[1:]:
            st.caption(f"⚠️ {extra_warning}")
        return

    tests = list(analysis_result.get("tests", []))
    warnings = list(analysis_result.get("warnings", []))
    confidence = float(analysis_result.get("confidence", 0.0) or 0.0)

    logger.info(
        "Report analysis UI: tests=%d, confidence=%.4f, warnings=%d",
        len(tests),
        confidence,
        len(warnings),
    )

    _render_report_debug_trace(analysis_result)

    if confidence < _LOW_CONFIDENCE_THRESHOLD:
        render_info_panel(
            title="Manual Verification Recommended",
            message=(
                "OCR confidence is low. Please verify the medical report image manually before acting on the extracted details."
            ),
            variant="warning",
        )

    _render_analysis_warnings(warnings)

    _render_report_summary(tests)
    _render_important_findings(tests)
    _render_laboratory_results_table(tests)
    _render_ai_report_analysis(tests, file_hash)

    # Export/copy controls come after AI generation so the exported report
    # includes the cached AI explanation when the user has generated it.
    _render_report_actions(tests)
    _render_safety_notice()
