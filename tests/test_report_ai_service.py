"""
tests/test_report_ai_service.py
=============================================================================
Focused regression tests for modules/report_ai_service.py.

NOTE ON SCOPE: only modules/report_ai_service.py was provided for this
task (no existing test_report_ai_service.py), so this is a new,
self-contained file rather than an edit to an existing one. It exercises
the real functions directly, with a fake Gemini client (no network calls),
so it is a genuine regression check - but it cannot confirm the project's
claimed "139 passed" baseline; only the project's own full suite run can.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.report_ai_service import (
    _clean_test_record,
    _clean_unit_for_patient_display,
    _validate_tests,
    analyze_report_findings,
    build_report_explanation_prompt,
)


# =============================================================================
# Fixtures
# =============================================================================


def _sample_tests():
    return [
        {
            "test_name": "Hemoglobin (Hb)",
            "value": 13.2,
            "unit": "gldL Photometric",
            "reference_range": "13-18",
            "status": "Normal",
        },
        {
            "test_name": "RBC Count",
            "value": 4.56,
            "unit": "million/uL Electrical Impedance",
            "reference_range": "4.6-6.2",
            "status": "Low",
        },
        {
            "test_name": "Platelet Count",
            "value": 2.35,
            "unit": "lakhluL Electrical Impedance",
            "reference_range": "150000-400000",
            "status": "Normal",
        },
        {
            "test_name": "Total Leucocyte Count (TLC)",
            "value": 7,
            "unit": "IpL",
            "reference_range": "",
            "status": "Unknown",
        },
    ]


class _FakeGeminiClient:
    def __init__(self, response: str = "ok", raise_error: bool = False):
        self._response = response
        self._raise_error = raise_error

    def generate_response(self, prompt: str) -> str:
        if self._raise_error:
            raise RuntimeError("simulated Gemini failure")
        return self._response


# =============================================================================
# 1. Normal/High/Low/Unknown statuses are preserved in the prompt
# =============================================================================


def test_all_four_statuses_preserved_verbatim_in_prompt():
    prompt = build_report_explanation_prompt(_sample_tests())
    assert '"status": "Normal"' in prompt
    assert '"status": "Low"' in prompt
    assert '"status": "Unknown"' in prompt


def test_validate_tests_never_changes_a_valid_status():
    validated = _validate_tests(_sample_tests())
    statuses = [t["status"] for t in validated]
    assert statuses == ["Normal", "Low", "Normal", "Unknown"]


# =============================================================================
# 2. Prompt explicitly prevents diagnosis
# =============================================================================


def test_prompt_prohibits_diagnosis():
    prompt = build_report_explanation_prompt(_sample_tests())
    assert "diagnos" in prompt.lower()
    assert "Do not diagnose" in prompt or "does not provide\na diagnosis" in prompt or "diagnose diseases" in prompt


# =============================================================================
# 3. Prompt explicitly prevents inventing values/ranges/units
# =============================================================================


def test_prompt_prohibits_inventing_values_ranges_units():
    prompt = build_report_explanation_prompt(_sample_tests())
    assert "Never invent, correct, infer, or replace" in prompt
    assert "unit" in prompt.lower()
    assert "reference range" in prompt.lower()


# =============================================================================
# 4. Patient-facing instructions prohibit methodology text
# =============================================================================


@pytest.mark.parametrize(
    "word", ["Photometric", "Electrical Impedance", "Calculated", "Microscopic"]
)
def test_prompt_instructs_against_methodology_words(word):
    prompt = build_report_explanation_prompt(_sample_tests())
    assert word in prompt  # named explicitly in the rule telling Gemini to avoid it


def test_display_unit_is_clean_for_normal_result_with_methodology_suffix():
    prompt = build_report_explanation_prompt(_sample_tests())
    # The raw OCR "unit" field is legitimately still sent as input context
    # (the AI service always receives the original displayed unit) - what
    # matters is that the CLEAN display_unit Gemini is told to use for
    # output is correct, and rule 13 instructs it to never use raw "unit".
    assert '"display_unit": "million/uL"' in prompt


def test_corrupted_unit_display_unit_is_blank_not_fixed():
    # "lakhluL" (OCR corruption of "lakh/uL") must never be guessed into
    # a corrected spelling - its display_unit must be blank, so Gemini
    # (instructed by rule 13 to use only display_unit) has nothing to
    # show for it rather than a guessed correction.
    prompt = build_report_explanation_prompt(_sample_tests())
    assert '"display_unit": "lakh/uL"' not in prompt
    payload_start = prompt.index("STRUCTURED LABORATORY RESULTS:")
    platelet_section = prompt[
        prompt.index('"test_name": "Platelet Count"', payload_start) :
    ]
    assert '"display_unit": ""' in platelet_section[:200]


# =============================================================================
# 5. Unknown results remain manual verification / no unit exposed
# =============================================================================


def test_unknown_result_display_unit_is_always_blank():
    prompt = build_report_explanation_prompt(_sample_tests())
    assert '"test_name": "Total Leucocyte Count (TLC)"' in prompt
    # The raw ambiguous unit "IpL" must never be surfaced for display.
    assert '"display_unit": "IpL"' not in prompt


def test_manual_verification_heading_present():
    prompt = build_report_explanation_prompt(_sample_tests())
    assert "## Results Requiring Manual Verification" in prompt


# =============================================================================
# 6. High/Low are the only abnormal findings
# =============================================================================


def test_important_findings_restricted_to_high_low():
    prompt = build_report_explanation_prompt(_sample_tests())
    idx = prompt.index("## Important Findings")
    next_heading = prompt.index("## Results Within Range")
    section = prompt[idx:next_heading]
    assert "High" in section and "Low" in section
    assert "Unknown" not in section or "Do not" in section  # instructional only


# =============================================================================
# 7. Required headings are present, in order
# =============================================================================


def test_all_required_headings_present_in_order():
    prompt = build_report_explanation_prompt(_sample_tests())
    headings = [
        "## Overall Summary",
        "## Important Findings",
        "## Results Within Range",
        "## Results Requiring Manual Verification",
        "## General Guidance",
        "## Informational Disclaimer",
    ]
    positions = [prompt.index(h) for h in headings]
    assert positions == sorted(positions)


# =============================================================================
# 8. Existing validation behavior remains unchanged
# =============================================================================


def test_clean_test_record_defaults_are_unchanged():
    record = _clean_test_record({})
    assert record == {
        "test_name": "Unnamed Parameter",
        "value": None,
        "unit": "Not Available",
        "reference_range": "Not Available",
        "status": "Unknown",
    }


def test_validate_tests_rejects_non_list_input():
    with pytest.raises(ValueError):
        _validate_tests("not a list")


def test_validate_tests_skips_non_dict_entries():
    validated = _validate_tests([{"test_name": "A", "status": "Normal"}, "garbage", 5])
    assert len(validated) == 1
    assert validated[0]["test_name"] == "A"


def test_validate_tests_coerces_invalid_status_to_unknown():
    validated = _validate_tests([{"test_name": "A", "status": "Extremely High"}])
    assert validated[0]["status"] == "Unknown"


def test_validate_tests_caps_test_count():
    many = [{"test_name": f"T{i}", "status": "Normal"} for i in range(150)]
    validated = _validate_tests(many)
    assert len(validated) == 100


# =============================================================================
# 9. Gemini failure behavior remains unchanged
# =============================================================================


def test_analyze_report_findings_success_path():
    result = analyze_report_findings(
        _sample_tests(), client=_FakeGeminiClient(response="Explanation text")
    )
    assert result["success"] is True
    assert result["explanation"] == "Explanation text"
    assert result["warning"] == ""


def test_analyze_report_findings_gemini_failure_path():
    result = analyze_report_findings(
        _sample_tests(), client=_FakeGeminiClient(raise_error=True)
    )
    assert result["success"] is False
    assert result["explanation"] == ""
    assert "unavailable" in result["warning"].lower()
    assert "error" in result


def test_analyze_report_findings_empty_tests():
    result = analyze_report_findings([], client=_FakeGeminiClient())
    assert result["success"] is False
    assert result["explanation"] == ""
    assert "No structured laboratory results" in result["warning"]


# =============================================================================
# Direct unit-cleaning helper tests
# =============================================================================


@pytest.mark.parametrize(
    "raw_unit,expected",
    [
        ("gldL Photometric", ""),  # corrupted g/dL spelling - never fixed
        ("gmIdL", ""),  # corrupted g/dL spelling - never fixed
        ("lakhluL Electrical Impedance", ""),  # corrupted lakh/uL - never fixed
        ("IpL Electrical Impedance", ""),  # ambiguous - never guessed
        ("million/uL Electrical Impedance", "million/uL"),
        ("lakh/uL Electrical Impedance", "lakh/uL"),
        ("% Microscopic", "%"),
        ("% Calculated", "%"),
        ("fL Calculated", "fL"),
        ("pg Calculated", "pg"),
        ("g/dL", "g/dL"),
        ("Photometric", ""),  # methodology word alone -> nothing left
        ("", ""),
        (None, ""),
    ],
)
def test_clean_unit_for_patient_display(raw_unit, expected):
    assert _clean_unit_for_patient_display(raw_unit) == expected