"""
ui/prescription.py
=============================================================================
Prescription upload and OCR preview page for the Intelligent Hospital
Information Assistant frontend.

This module renders only the prescription upload UI and display logic.
OCR and medicine extraction live in `modules/prescription_analyzer.py`.

-----------------------------------------------------------------------------
Public API
-----------------------------------------------------------------------------
    render_prescription_page() -> None
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
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError
import streamlit as st

from ui.components import (
    render_divider,
    render_info_panel,
    render_key_value,
    render_section_header,
    render_status_badge,
    render_tag,
)
from modules.prescription_ai_service import get_prescription_ai_service
from modules.prescription_analyzer import get_prescription_analyzer

logger = logging.getLogger("hospital_assistant.prescription_ui")

# =============================================================================
# PRIVATE HELPERS
# =============================================================================


_ANALYSIS_RESULT_KEY = "prescription_analysis_result"
_ANALYSIS_FILE_HASH_KEY = "prescription_analysis_file_hash"
_ANALYSIS_DURATION_MS_KEY = "prescription_analysis_duration_ms"
_REPORT_RESULT_KEY = "prescription_report_result"
_REPORT_FILE_HASH_KEY = "prescription_report_file_hash"
_REPORT_DURATION_MS_KEY = "prescription_report_duration_ms"
_LOW_CONFIDENCE_THRESHOLD = 70.0
_DISPLAY_OCR_CORRECTIONS: tuple[tuple[str, str], ...] = (
    (r"\bAften\b", "After"),
    (r"\bBefone\b", "Before"),
    (r"\bAgten\b", "After"),
    (r"\bDnink\b", "Drink"),
    (r"\bsbuess\b", "Stress"),
)
_REPORT_SECTION_ICONS = {
    "Prescription Summary": "📋",
    "Detected Medicines": "💊",
    "Medicine Details": "💊",
    "Dosage Schedule": "🕒",
    "Timing (Morning / Afternoon / Night)": "🕒",
    "Before/After Food": "🍽️",
    "Duration": "📅",
    "Precautions": "⚠️",
    "Possible Side Effects": "💥",
    "Drug Interactions": "⚠️",
    "Emergency Warnings": "🚨",
    "Overall Recommendations": "📌",
}
_MEDICINE_GUIDANCE_ICONS = {
    "Uses": "🎯",
    "Side Effects": "⚠️",
    "Contraindications": "🚫",
    "Precautions": "🛡️",
}


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
    """Return a stable hash for the uploaded prescription image."""
    return hashlib.sha256(data).hexdigest()


@st.cache_data(show_spinner=False)
def _analyze_prescription_cached(image_bytes: bytes) -> dict[str, object]:
    """Cache OCR and structured extraction for an unchanged upload."""
    return get_prescription_analyzer().analyze(image_bytes)


@st.cache_data(show_spinner=False)
def _generate_report_cached(
    ocr_text: str,
    medicines_json: str,
    confidence: float,
) -> dict[str, object]:
    """Cache a report for unchanged extracted prescription data."""
    return get_prescription_ai_service().generate_report(
        ocr_text=ocr_text,
        detected_medicines=json.loads(medicines_json),
        confidence=confidence,
    )


def _clean_ocr_display_text(text: str) -> str:
    """Improve obvious OCR wording and spacing for display only, without altering analysis data."""
    cleaned = text
    for pattern, replacement in _DISPLAY_OCR_CORRECTIONS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Display-only whitespace normalization: trim trailing spaces per line
    # and collapse runs of 3+ blank lines to a single blank line. This
    # never touches `analysis_result["ocr_text"]` itself — only the
    # string handed to the read-only preview text area below.
    lines = [line.rstrip() for line in cleaned.splitlines()]
    normalized_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


def _confidence_details(confidence: float) -> tuple[str, str, str]:
    """Return the presentation label, component status, and icon for OCR confidence."""
    if confidence >= 95:
        return "Excellent", "online", "🟢"
    if confidence >= 80:
        return "Good", "processing", "🟡"
    if confidence >= 60:
        return "Moderate", "warning", "🟠"
    return "Poor", "error", "🔴"


def _render_confidence_badge(confidence: float) -> None:
    """Render a human-readable, color-coded OCR confidence badge."""
    label, status, icon = _confidence_details(confidence)
    render_status_badge(f"{icon} {label} · {confidence:.1f}%", status)


def _split_report_sections(report_text: str) -> list[tuple[str, str]]:
    """Split a Markdown report into displayable heading/body sections."""
    sections: list[tuple[str, list[str]]] = []
    heading = "Prescription Report"
    body: list[str] = []

    for line in report_text.splitlines():
        match = re.match(r"^#{1,3}\s+(.+?)\s*$", line)
        if match:
            if body or heading != "Prescription Report":
                sections.append((heading, body))
            heading = match.group(1).strip()
            body = []
        else:
            body.append(line)

    if body or not sections:
        sections.append((heading, body))

    return [(section_heading, "\n".join(lines).strip()) for section_heading, lines in sections]


def _render_copy_report_control(report_text: str) -> None:
    """Render a browser-side copy control for the generated report text."""
    button_id = f"copy-report-{hashlib.sha256(report_text.encode('utf-8')).hexdigest()[:12]}"
    payload = json.dumps(report_text).replace("</", "<\\/")
    st.components.v1.html(
        f"""
        <button id="{button_id}" style="border: 1px solid #d8dee9; border-radius: 8px; background: #ffffff; color: #1f2937; cursor: pointer; font: 600 14px sans-serif; padding: 8px 14px;">
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


def _pdf_safe_text(value: object) -> str:
    """Convert report content to the built-in PDF font's safe character set."""
    return str(value).encode("latin-1", "replace").decode("latin-1")


def _build_report_pdf(report_text: str, matched_medicines: list[dict[str, object]]) -> bytes:
    """Create a concise text-only PDF report from existing report data."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - dependency guard for deployments
        raise RuntimeError("PDF generation is unavailable because ReportLab is not installed.") from exc

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4
    margin = 0.65 * inch
    y_position = page_height - margin

    def write_line(text: str, *, bold: bool = False, size: int = 10) -> None:
        nonlocal y_position
        font = "Helvetica-Bold" if bold else "Helvetica"
        for wrapped_line in textwrap.wrap(_pdf_safe_text(text), width=92) or [""]:
            if y_position < margin:
                pdf.showPage()
                y_position = page_height - margin
            pdf.setFont(font, size)
            pdf.drawString(margin, y_position, wrapped_line)
            y_position -= size + 4

    write_line("Hospital Prescription Report", bold=True, size=16)
    y_position -= 8
    for heading, body in _split_report_sections(report_text):
        write_line(heading, bold=True, size=12)
        for line in body.splitlines():
            write_line(line or " ")
        y_position -= 6
    

        pdf.save()
        return buffer.getvalue()


def _clear_cached_analysis_if_needed(file_hash: str) -> None:
    """Reset cached analysis when the uploaded file changes."""
    cached_hash = st.session_state.get(_ANALYSIS_FILE_HASH_KEY)
    if cached_hash and cached_hash != file_hash:
        st.session_state[_ANALYSIS_RESULT_KEY] = None
        st.session_state[_REPORT_RESULT_KEY] = None
        st.session_state[_ANALYSIS_DURATION_MS_KEY] = None
        st.session_state[_REPORT_DURATION_MS_KEY] = None
    st.session_state[_ANALYSIS_FILE_HASH_KEY] = file_hash
    st.session_state[_REPORT_FILE_HASH_KEY] = file_hash


def _render_analysis_metrics(result: dict[str, object]) -> None:
    """Render OCR summary metrics in a compact, bordered row."""
    with st.container(border=True):
        metric_col_1, metric_col_2, metric_col_3 = st.columns(3, gap="medium")
        with metric_col_1:
            st.metric("🔍 OCR Confidence", f"{float(result.get('confidence', 0.0)):.1f}%")
        with metric_col_2:
            st.metric("📄 Detected Lines", str(result.get("lines", 0)))
        with metric_col_3:
            st.metric("🔤 Characters", str(result.get("characters", 0)))

        st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
        _render_confidence_badge(float(result.get("confidence", 0.0)))


def _render_performance_section(
    analysis_result: dict[str, object],
    report_result: Optional[dict[str, object]],
    analysis_duration_ms: Optional[float],
    report_duration_ms: Optional[float],
) -> None:
    """
    Render real processing-time and coverage metrics for this analysis run.

    Every value here is either measured directly around the existing
    `_analyze_prescription_cached()` / `_generate_report_cached()` calls
    in `render_prescription_page()` (never inside the cached backend
    functions themselves) or read from `analysis_result`/`report_result`
    fields the backend already returns. A value is shown as
    "Not Available" — never a fabricated `0` — when it hasn't been
    measured yet for this session (e.g. before "Analyze Prescription"
    has been clicked).
    """
    render_section_header(
        title="Performance",
        subtitle="Processing time and coverage for this analysis run.",
        icon="⚡",
    )

    medicines = list(analysis_result.get("medicines", []))
    matched_medicines = list(report_result.get("matched_medicines", [])) if report_result else []
    matched_count = sum(1 for item in matched_medicines if item.get("matched"))

    # A 2x2 grid (rather than 4 equal columns) gives each card roughly
    # double the width, so longer values ("Not Available", "3 / 5") are
    # never forced to wrap onto a second line at typical desktop widths.
    with st.container(border=True):
        row_1_col_1, row_1_col_2 = st.columns(2, gap="large")
        with row_1_col_1:
            st.metric(
                "⏱️ OCR Time",
                f"{analysis_duration_ms:.0f} ms" if analysis_duration_ms is not None else "Not Available",
            )
        with row_1_col_2:
            st.metric(
                "⏱️ Report Time",
                f"{report_duration_ms:.0f} ms" if report_duration_ms is not None else "Not Available",
            )

        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

        row_2_col_1, row_2_col_2 = st.columns(2, gap="large")
        with row_2_col_1:
            st.metric("💊 Medicines Detected", str(len(medicines)) if medicines else "Not Available")
        with row_2_col_2:
            st.metric(
                "✅ Medicines Matched",
                f"{matched_count} / {len(matched_medicines)}" if matched_medicines else "Not Available",
            )


def _render_technical_details_section(
    analysis_result: dict[str, object],
    report_result: Optional[dict[str, object]],
    resolution_label: str,
    image_size_label: str,
) -> None:
    """
    Render runtime information already available from the analysis and
    report objects.

    No new backend field is introduced: every value here already exists
    on `analysis_result` (`lines`, `characters`), `report_result`
    (`used_fallback`, `low_confidence`, `matched_medicines`,
    `report_text`), or is derived directly from the uploaded image
    (`resolution_label`, `image_size_label`, computed in
    `render_prescription_page()`). Anything unavailable — e.g. report
    fields before a report has been generated — shows "Not Available"
    rather than a fabricated value.
    """
    render_section_header(
        title="Technical Details",
        subtitle="Runtime information from this analysis and report.",
        icon="🛠️",
    )

    report_text = str(report_result.get("report_text", "")) if report_result else ""
    matched_medicines = list(report_result.get("matched_medicines", [])) if report_result else []
    matched_count = sum(1 for item in matched_medicines if item.get("matched"))

    with st.container(border=True):
        detail_col, report_col = st.columns(2, gap="large")
        with detail_col:
            st.markdown(
                '<p style="font-weight:600; font-size:0.85rem; color:#64748b; '
                'text-transform:uppercase; letter-spacing:0.03em; margin-bottom:0.5rem;">'
                "Image &amp; OCR</p>",
                unsafe_allow_html=True,
            )
            render_key_value(
                {
                    "Image Resolution": resolution_label,
                    "Image Size": image_size_label,
                    "OCR Lines": str(analysis_result.get("lines")) if analysis_result.get("lines") is not None else "Not Available",
                    "OCR Characters": (
                        str(analysis_result.get("characters"))
                        if analysis_result.get("characters") is not None
                        else "Not Available"
                    ),
                }
            )
        with report_col:
            st.markdown(
                '<p style="font-weight:600; font-size:0.85rem; color:#64748b; '
                'text-transform:uppercase; letter-spacing:0.03em; margin-bottom:0.5rem;">'
                "Report</p>",
                unsafe_allow_html=True,
            )
            if report_result is not None:
                report_source = "Fallback Template" if report_result.get("used_fallback") else "Gemini AI"
                low_confidence_flag = "Yes" if report_result.get("low_confidence") else "No"
                medicine_matches = f"{matched_count} / {len(matched_medicines)}" if matched_medicines else "Not Available"
                report_length = f"{len(report_text)} characters" if report_text else "Not Available"
            else:
                report_source = "Not Available"
                low_confidence_flag = "Not Available"
                medicine_matches = "Not Available"
                report_length = "Not Available"

            render_key_value(
                {
                    "Report Source": report_source,
                    "Low Confidence Flag": low_confidence_flag,
                    "Medicine Matches": medicine_matches,
                    "Report Length": report_length,
                }
            )


def _render_detected_medicine(medicine: dict[str, str], index: int) -> None:
    """Render one structured medicine entry as a clean, badge-annotated card."""
    medicine_name = medicine.get("name") or f"Medicine {index + 1}"
    badge_values = [value for value in (medicine.get("strength"), medicine.get("frequency"), medicine.get("timing")) if value]

    with st.expander(f"💊 {index + 1}. {medicine_name}", expanded=index == 0):
        with st.container(border=True):
            if badge_values:
                badge_columns = st.columns(len(badge_values))
                for badge_column, value in zip(badge_columns, badge_values):
                    with badge_column:
                        render_tag(value, variant="teal")
            else:
                st.caption("Structured extraction")

            st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

            detail_col, guidance_col = st.columns(2, gap="large")
            with detail_col:
                st.markdown(
                    '<p style="font-weight:600; font-size:0.85rem; color:#64748b; '
                    'text-transform:uppercase; letter-spacing:0.03em; margin-bottom:0.5rem;">'
                    "Details</p>",
                    unsafe_allow_html=True,
                )
                render_key_value(
                    {
                        
                        "Strength": medicine.get("strength") or "Not Available",
                        "Frequency": medicine.get("frequency") or "Not Available",
                    }
                )
            with guidance_col:
                st.markdown(
                    '<p style="font-weight:600; font-size:0.85rem; color:#64748b; '
                    'text-transform:uppercase; letter-spacing:0.03em; margin-bottom:0.5rem;">'
                    "Directions</p>",
                    unsafe_allow_html=True,
                )
                render_key_value(
                    {
                        "Timing": medicine.get("timing") or "Not Available",
                        "Duration": medicine.get("duration") or "Not Available",
                        "Instructions": medicine.get("instructions") or "Not Available",
                    }
                )


def _render_report_warnings(result: dict[str, object]) -> None:
    """Render report-generation warnings and validation notes."""
    if result.get("low_confidence"):
        render_info_panel(
            title="Manual Verification Recommended",
            message=(
                "OCR confidence is low, so the prescription report may miss or misread details. "
                "Please verify the medicine names, strengths, and duration manually with the original prescription."
            ),
            variant="warning",
        )

    if result.get("used_fallback"):
        render_info_panel(
            title="Fallback Report Used",
            message=(
                "Gemini report generation was unavailable, so the app generated a structured hospital-style report "
                "from the OCR text and medicine database lookup."
            ),
            variant="info",
        )


def _render_prescription_report(report_result: dict[str, object]) -> None:
    """Render the Gemini-generated or fallback prescription report."""
    render_section_header(
        title="Professional Prescription Report",
        subtitle="Explanation generated from OCR text and retrieved medicine metadata.",
        icon="📋",
    )

    _render_report_warnings(report_result)

    report_text = str(report_result.get("report_text", "")).strip()
    if not report_text:
        render_info_panel(
            title="Report Unavailable",
            message="The prescription report could not be generated.",
            variant="warning",
        )
        return

    matched_medicines = list(report_result.get("matched_medicines", []))

    action_col, copy_col = st.columns([1, 1], gap="small")
    with action_col:
        try:
            st.download_button(
                "Download Report (PDF)",
                data=_build_report_pdf(report_text, matched_medicines),
                file_name="prescription_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except RuntimeError as exc:
            st.caption(str(exc))
    with copy_col:
        _render_copy_report_control(report_text)

    for heading, body in _split_report_sections(report_text):
        icon = _REPORT_SECTION_ICONS.get(heading, "📌")
        with st.expander(f"{icon} {heading}", expanded=heading == "Prescription Summary"):
            st.markdown(body or "Not available.")

    if matched_medicines:
        render_divider()
        

        for index, item in enumerate(matched_medicines, start=1):
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            title = metadata.get("generic_name") or item.get("detected_name") or f"Medicine {index}"
            match_score = float(item.get("match_score", 0.0))

            with st.container(border=True):
                heading_col, score_col = st.columns([3, 1], gap="medium")
                with heading_col:
                    st.subheader(f"💊 {title}")
                    st.caption(f"Detected as: {item.get('detected_name') or 'Not Available'}")
                with score_col:
                    status = "online" if item.get("matched") else "warning"
                    render_status_badge(
                        f"{'Matched' if item.get('matched') else 'Review'} · {match_score:.1f}%",
                        status,
                    )

                render_divider()

                dosage = metadata.get("dosage_information") or {}
                if not isinstance(dosage, dict):
                    dosage = {}
                detail_col, guidance_col = st.columns(2, gap="large")
                with detail_col:
                    render_key_value(
                        {
                            "Class": metadata.get("medicine_class") or "Not Available",
                            "Adult Dosage": dosage.get("adult_dose") or "Not Available",
                            "Food Guidance": metadata.get("food_interactions") or "Not Available",
                            "Storage": metadata.get("storage") or "Not Available",
                        }
                    )
                with guidance_col:
                    guidance_fields = (
                        ("Uses", metadata.get("uses")),
                        ("Side Effects", metadata.get("side_effects")),
                        ("Contraindications", metadata.get("contraindications")),
                        ("Precautions", metadata.get("precautions")),
                    )
                    for field_index, (label, values) in enumerate(guidance_fields):
                        st.markdown(
                            f'<p style="font-weight:600; font-size:0.85rem; margin-bottom:0.25rem;">'
                            f"{_MEDICINE_GUIDANCE_ICONS.get(label, '•')} {label}</p>",
                            unsafe_allow_html=True,
                        )
                        if isinstance(values, list) and values:
                            st.markdown("\n".join(f"- {value}" for value in values))
                        elif isinstance(values, str) and values.strip():
                            st.markdown(values)
                        else:
                            st.caption("Not Available")
                        if field_index < len(guidance_fields) - 1:
                            st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

            st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)


# =============================================================================
# PAGE RENDERER
# =============================================================================


def render_prescription_page() -> None:
    """Render the prescription upload and preview interface."""
    render_section_header(
        title="Prescription Analysis",
        subtitle="Upload a prescription image to extract medicines, dosage, and directions.",
        icon="🩺",
    )

    with st.container(border=True):
        upload_file = st.file_uploader(
            "Upload Prescription Image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Supported formats: JPG, JPEG, PNG",
        )
        st.caption("Supported formats: JPG, JPEG, PNG")

    if upload_file is None:
        render_info_panel(
            title="Upload Prescription Image",
            message="Select a JPG, JPEG, or PNG prescription image to preview and analyze it.",
            variant="info",
        )
        return

    image_bytes = upload_file.getvalue()
    if not image_bytes:
        st.error("Unreadable image. Upload a clear JPG, JPEG, or PNG prescription.")
        return

    image = _load_uploaded_image(image_bytes)
    if image is None:
        st.error("Unsupported, invalid, or corrupted image. Upload a clear JPG, JPEG, or PNG prescription.")
        return

    file_hash = _hash_bytes(image_bytes)
    _clear_cached_analysis_if_needed(file_hash)

    image_width, image_height = _get_image_resolution(image_bytes)
    image_size_label = _format_file_size(len(image_bytes))
    resolution_label = (
        f"{image_width} × {image_height}" if image_width is not None and image_height is not None else "Not Available"
    )

    st.image(image, caption=upload_file.name, use_container_width=True)

    
   

    analyze_clicked = st.button("Analyze Prescription", use_container_width=True)
    if analyze_clicked:
        analysis_start_time = time.perf_counter()
        with st.spinner("Analyzing Prescription..."):
            analysis_result = _analyze_prescription_cached(image_bytes)
        st.session_state[_ANALYSIS_DURATION_MS_KEY] = (time.perf_counter() - analysis_start_time) * 1000
        st.session_state[_ANALYSIS_RESULT_KEY] = analysis_result
        report_result = None
        if analysis_result and analysis_result.get("success", False):
            traced_medicines = list(analysis_result.get("medicines", []))
            logger.info(
                "Prescription trace [4/6] analysis_result medicines: count=%d, names=%s, entries=%s",
                len(traced_medicines),
                [medicine.get("name") for medicine in traced_medicines],
                traced_medicines[:3],
            )
            medicine_payload = json.dumps(traced_medicines, sort_keys=True)
            report_start_time = time.perf_counter()
            with st.spinner("Generating Prescription Report..."):
                report_result = _generate_report_cached(
                    ocr_text=str(analysis_result.get("ocr_text", "")),
                    medicines_json=medicine_payload,
                    confidence=float(analysis_result.get("confidence", 0.0)),
                )
            st.session_state[_REPORT_DURATION_MS_KEY] = (time.perf_counter() - report_start_time) * 1000
            st.session_state[_REPORT_RESULT_KEY] = report_result
        else:
            st.session_state[_REPORT_DURATION_MS_KEY] = None
    else:
        analysis_result = st.session_state.get(_ANALYSIS_RESULT_KEY)
        report_result = st.session_state.get(_REPORT_RESULT_KEY)

    if analyze_clicked:
        report_result = st.session_state.get(_REPORT_RESULT_KEY)

    analysis_duration_ms = st.session_state.get(_ANALYSIS_DURATION_MS_KEY)
    report_duration_ms = st.session_state.get(_REPORT_DURATION_MS_KEY)

    if not analysis_result:
        render_info_panel(
            title="Ready to Analyze",
            message="Click Analyze Prescription to run OCR and extract structured medicine details.",
            variant="info",
        )
        return

    if not analysis_result.get("success", False):
        error_message = str(analysis_result.get("error", "Unable to analyze the uploaded prescription image."))
        if "no text" in error_message.lower():
            st.error("No readable text detected. The prescription may be blurry, dark, or too small to read.")
        elif "unreadable" in error_message.lower():
            st.error("Unreadable image. Upload a clearer, well-lit prescription image and try again.")
        else:
            st.error(error_message)
        return

    ocr_text = str(analysis_result.get("ocr_text", ""))
    medicines = list(analysis_result.get("medicines", []))
    logger.info(
        "Prescription trace [4/6] UI medicines value: count=%d, names=%s, entries=%s",
        len(medicines),
        [medicine.get("name") for medicine in medicines],
        medicines[:3],
    )


    render_section_header(
        title="Detected Medicines",
        subtitle="Structured extraction from the cleaned OCR text.",
        icon="💊",
    )

    if not medicines:
        render_info_panel(
            title="No Medicines Detected",
            message=(
                "OCR completed successfully, but no structured medicines could be extracted. "
                "Check that the prescription is clear and that medicine names are legible."
            ),
            variant="warning",
        )
        if report_result and report_result.get("success", False):
            _render_prescription_report(report_result)
        render_divider()
        _render_technical_details_section(analysis_result, report_result, resolution_label, image_size_label)
        return

    for index, medicine in enumerate(medicines):
        _render_detected_medicine(medicine, index)

    if analysis_result.get("confidence", 0.0) < _LOW_CONFIDENCE_THRESHOLD:
        render_info_panel(
            title="Manual Verification Recommended",
            message=(
                "OCR confidence is low. Please verify the prescription image manually before acting on the extracted details."
            ),
            variant="warning",
        )

    if report_result and report_result.get("success", False):
        _render_prescription_report(report_result)
    else:
        render_info_panel(
            title="Ready for Prescription Report",
            message=(
                "Click Analyze Prescription to run OCR, retrieve medicine metadata, and generate the structured report."
            ),
            variant="info",
        )
    with st.expander("🛠️ Technical Details", expanded=False):
        _render_technical_details_section(
            analysis_result,
            report_result,
            resolution_label,
            image_size_label,
    )