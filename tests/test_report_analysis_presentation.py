"""
tests/test_report_analysis_presentation.py
=============================================================================
Presentation-layer regressions for the Medical Report Analysis feature:
Report Summary counts, Important Findings, "Copy Report" text, and the PDF
export - all derived from the same structured `tests` list, never
recalculated separately, and never overridden by anything else.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.report_analysis import (
    _VISIBLE_REPORT_SECTIONS,
    _build_report_pdf,
    _build_structured_report_text,
    _summarize_tests,
)


def _test_record(**overrides: object) -> dict[str, object]:
    record = {
        "test_name": "Haemoglobin (Hb)",
        "value": 11.2,
        "unit": "g/dL",
        "reference_range": "13.0-17.0",
        "status": "Low",
    }
    record.update(overrides)
    return record


_MIXED_TESTS = [
    _test_record(test_name="Haemoglobin (Hb)", value=11.2, status="Low", reference_range="13.0-17.0"),
    _test_record(test_name="WBC", value=15000, unit="/uL", status="High", reference_range="4000-11000"),
    _test_record(test_name="Platelet Count", value=250000, unit="/uL", status="Normal", reference_range="150000-450000"),
    _test_record(test_name="Custom Marker", value=99, unit="units", status="Unknown", reference_range=""),
]

_ALL_NORMAL_TESTS = [
    _test_record(test_name="RBC", value=5.0, unit="million/uL", status="Normal", reference_range="4.5-5.9"),
]


# =============================================================================
# Summary counts (Task 2)
# =============================================================================


def test_summary_counts_are_calculated_from_structured_results():
    summary = _summarize_tests(_MIXED_TESTS)

    assert summary["total"] == 4
    assert summary["normal"] == 1
    assert summary["high"] == 1
    assert summary["low"] == 1
    assert summary["unknown"] == 1


# =============================================================================
# Important Findings (Task 3)
# =============================================================================


def test_report_text_important_findings_lists_every_abnormal_parameter():
    report_text = _build_structured_report_text(_MIXED_TESTS)
    findings_section = report_text.split("# Important Findings", 1)[1].split("# Laboratory Results", 1)[0]

    assert "Haemoglobin (Hb)" in findings_section
    assert "WBC" in findings_section
    # Normal and Unknown parameters are not "important findings".
    assert "Platelet Count" not in findings_section
    assert "Custom Marker" not in findings_section


def test_report_text_shows_no_abnormal_results_when_nothing_is_out_of_range():
    report_text = _build_structured_report_text(_ALL_NORMAL_TESTS)
    findings_section = report_text.split("# Important Findings", 1)[1].split("# Laboratory Results", 1)[0]

    assert "No Abnormal Results" in findings_section


# =============================================================================
# Copy Report text (Task 5)
# =============================================================================


def test_copy_report_text_contains_every_required_section():
    report_text = _build_structured_report_text(_MIXED_TESTS)

    for section in _VISIBLE_REPORT_SECTIONS:
        assert f"# {section}" in report_text


def test_copy_report_text_contains_all_laboratory_rows():
    report_text = _build_structured_report_text(_MIXED_TESTS)
    lab_section = report_text.split("# Laboratory Results", 1)[1].split("# Informational Only", 1)[0]

    for record in _MIXED_TESTS:
        assert str(record["test_name"]) in lab_section
        assert str(record["status"]) in lab_section


def test_copy_report_text_includes_informational_only_disclaimer():
    report_text = _build_structured_report_text(_MIXED_TESTS)

    assert "does not provide a diagnosis" in report_text


# =============================================================================
# PDF export (Task 4)
# =============================================================================


def test_pdf_contains_all_required_sections():
    report_text = _build_structured_report_text(_MIXED_TESTS)
    pdf = _build_report_pdf(report_text)

    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    for section in _VISIBLE_REPORT_SECTIONS:
        assert section.encode("latin-1") in pdf


def test_pdf_contains_all_laboratory_rows_not_just_the_first_section():
    report_text = _build_structured_report_text(_MIXED_TESTS)
    pdf = _build_report_pdf(report_text)

    # Every test name must appear in the PDF bytes - this fails if PDF
    # finalization happens inside the section loop and truncates the
    # document after only the first section is written.
    for record in _MIXED_TESTS:
        # Parentheses are PDF string-literal delimiters, so ReportLab
        # escapes them in the content stream; escape the same way here.
        expected = str(record["test_name"]).replace("(", r"\(").replace(")", r"\)")
        assert expected.encode("latin-1") in pdf


def test_pdf_generation_survives_many_laboratory_rows_across_multiple_pages():
    many_tests = [
        _test_record(test_name=f"Marker {index}", value=index, status="Normal", reference_range="0-100")
        for index in range(60)
    ]
    report_text = _build_structured_report_text(many_tests)
    pdf = _build_report_pdf(report_text)

    assert pdf.startswith(b"%PDF")
    assert b"Marker 0" in pdf
    assert b"Marker 59" in pdf