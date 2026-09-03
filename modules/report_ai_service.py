"""
modules/report_ai_service.py
=============================================================================
AI explanation layer for structured medical laboratory report results.

This module receives ONLY the deterministic structured results produced by
modules.report_analyzer.py. It does not perform OCR, retrieve documents,
change laboratory values/statuses, or diagnose conditions.

Gemini is used only to explain the supplied findings in plain language.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from modules.gemini_client import GeminiClient

logger = logging.getLogger("hospital_assistant.report_ai_service")

# Keep the AI output bounded and predictable.
_MAX_TESTS = 100
_MAX_TEST_NAME_LENGTH = 150
_MAX_UNIT_LENGTH = 80
_MAX_REFERENCE_LENGTH = 80

# -----------------------------------------------------------------------
# Patient-facing unit cleaning (PRESENTATION ONLY).
#
# This is deliberately separate from, and does not import, the unit
# normalization in modules/report_analyzer.py - that normalization
# feeds the deterministic reference-range lookup and is out of scope
# here. This helper never affects any value/unit/range/status; it only
# decides what unit text, if any, is safe to show a patient in the AI
# explanation for an already-Normal/High/Low result.
#
# Two conservative, deterministic steps only:
#   1. Remove known lab-methodology descriptor words (never measurement
#      units) that OCR commonly merges onto the unit column, e.g.
#      "Photometric", "Electrical Impedance". A short, explicit list -
#      not a general phrase matcher.
#   2. If what remains does not exactly match one of a small set of
#      pre-identified, unambiguous unit spellings, "" is returned. The
#      original OCR text is NEVER guessed, corrected, or displayed when
#      it doesn't match - so corrupted spellings such as "gldL",
#      "lakhluL", or "IpL" are always omitted, never "fixed" or shown
#      as-is.
# -----------------------------------------------------------------------

_UNIT_METHODOLOGY_WORDS_PATTERN = re.compile(
    r"\b(?:photometric|calculated|microscopic|electrical\s+impedance)\b",
    re.IGNORECASE,
)

# Compared case-insensitively with internal whitespace removed; the
# ORIGINAL spelling/casing (after methodology-word removal) is what
# gets displayed when matched - nothing here rewrites a unit's spelling.
_SAFE_DISPLAY_UNITS = {
    "%",
    "g/dl",
    "gm/dl",
    "fl",
    "pg",
    "mg/dl",
    "meq/l",
    "mmol/l",
    "miu/ml",
    "iu/ml",
    "ng/ml",
    "pg/ml",
    "u/l",
    "million/ul",
    "lakh/ul",
    "cells/ul",
    "/ul",
    "cumm",
    "/cumm",
    "mm/hr",
    "sec",
}


def _clean_unit_for_patient_display(raw_unit: str) -> str:
    """
    Return a patient-facing unit string, or "" if the supplied unit is
    not safely presentable as-is. See the module-level note above for
    the two-step rule this follows; it never guesses an ambiguous or
    corrupted unit.
    """
    if not raw_unit or not isinstance(raw_unit, str):
        return ""

    stripped = _UNIT_METHODOLOGY_WORDS_PATTERN.sub("", raw_unit).strip()
    if not stripped:
        return ""

    compact = re.sub(r"\s+", "", stripped).lower()
    if compact in _SAFE_DISPLAY_UNITS:
        return stripped

    return ""


_SYSTEM_RULES = """
You are an assistant that explains computer-printed laboratory report results.

STRICT RULES:
1. Use ONLY the structured laboratory results supplied in the prompt.
2. Never invent, correct, infer, or replace a test value, unit, reference range,
   or status.
3. Treat the supplied status (Normal, High, Low, Unknown) as authoritative.
4. Explain only High and Low results as abnormal findings.
5. Do not turn an Unknown result into High, Low, or Normal.
6. If a reference range is "Not Available", explicitly say that the report did
   not provide a reliably extracted reference range.
7. Do not diagnose diseases or claim that a result proves a disease.
8. Do not recommend medicines, doses, treatment plans, or stopping medicines.
9. Use cautious wording such as "may", "can be associated with", or
   "should be interpreted with clinical context".
10. Do not use patient-specific information beyond the supplied laboratory
    results.
11. Return concise, patient-friendly explanations.
12. End with an informational disclaimer that the explanation does not provide
    a diagnosis and that a qualified healthcare professional should interpret
    the results in clinical context.
13. Each result includes a "display_unit" field. Use ONLY "display_unit" when
    stating a unit to the patient - never the raw "unit" field. If
    "display_unit" is empty, state the value and reference range without any
    unit rather than guessing or repeating raw OCR/laboratory-methodology
    text (e.g. never say "Photometric", "Electrical Impedance", "Calculated",
    "Microscopic", or an OCR-corrupted unit spelling).
14. For an Unknown result, do not describe or speculate about its unit or
    reference range at all - only say that the required information could
    not be reliably extracted and that manual verification is needed.
15. Prefer simple, plain language a non-medical reader can follow (e.g.
    "within the expected range", "slightly below the supplied range") over
    technical phrasing (e.g. "reference interval", "demonstrates a decreased
    value").
16. For a High/Low result that is only marginally outside its supplied
    range, prefer cautious wording such as "slightly above/below the
    supplied range" over stronger language such as "abnormally high/low".
    Never assign a severity level that was not supplied.
""".strip()


def _clean_test_record(test: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the laboratory fields that the AI is allowed to see."""
    return {
        "test_name": str(test.get("test_name") or "Unnamed Parameter")[
            :_MAX_TEST_NAME_LENGTH
        ],
        "value": test.get("value"),
        "unit": str(test.get("unit") or "Not Available")[:_MAX_UNIT_LENGTH],
        "reference_range": str(
            test.get("reference_range") or "Not Available"
        )[:_MAX_REFERENCE_LENGTH],
        "status": str(test.get("status") or "Unknown"),
    }


def _validate_tests(tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalize analyzer output before constructing the prompt."""
    if not isinstance(tests, list):
        raise ValueError("tests must be a list of structured laboratory results.")

    cleaned: List[Dict[str, Any]] = []

    for test in tests[:_MAX_TESTS]:
        if not isinstance(test, dict):
            continue

        status = str(test.get("status") or "Unknown")
        if status not in {"Normal", "High", "Low", "Unknown"}:
            status = "Unknown"

        record = _clean_test_record(test)
        record["status"] = status
        cleaned.append(record)

    return cleaned


def _build_presentation_tests(
    validated_tests: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Attach a patient-facing "display_unit" to each already-validated
    record, WITHOUT altering test_name/value/unit/reference_range/status.

    For an Unknown result, display_unit is always "" - the supplied unit
    is frequently the very reason the result could not be interpreted
    (e.g. an ambiguous OCR unit), so it is never surfaced to the patient
    for that status regardless of what _clean_unit_for_patient_display
    would otherwise return.
    """
    presentation_tests: List[Dict[str, Any]] = []
    for record in validated_tests:
        display_record = dict(record)
        if record["status"] == "Unknown":
            display_record["display_unit"] = ""
        else:
            display_record["display_unit"] = _clean_unit_for_patient_display(
                record["unit"]
            )
        presentation_tests.append(display_record)
    return presentation_tests


def build_report_explanation_prompt(tests: List[Dict[str, Any]]) -> str:
    """
    Build the complete Gemini prompt from deterministic structured results.

    The structured data is serialized as JSON so Gemini receives exactly the
    values/statuses returned by report_analyzer.py. Each entry also carries
    a "display_unit" (see _build_presentation_tests) so Gemini has an
    explicit, pre-cleaned unit to use in patient-facing text instead of
    parsing OCR methodology/corruption out of "unit" itself.
    """
    validated_tests = _validate_tests(tests)
    presentation_tests = _build_presentation_tests(validated_tests)

    payload = json.dumps(
        presentation_tests,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    return f"""
{_SYSTEM_RULES}

STRUCTURED LABORATORY RESULTS:
{payload}

TASK:
Explain these already-determined laboratory results to an ordinary patient,
in simple, everyday language. The status of each result (Normal/High/Low/
Unknown) is already decided - you are only explaining it, never deciding it.

Return the response using exactly these headings, in this order:

## Overall Summary
A short, simple summary: how many results were detected, how many are
within their supplied range, how many are above/below it, and how many
could not be reliably interpreted. Do not say the person is healthy and do
not draw a clinical conclusion.

## Important Findings
Include ONLY High and Low results. Omit this section's body (a brief note
that none were found is fine) if there are none. For each one, in simple
terms: the test name, the patient's value, the supplied reference range,
what the parameter generally measures, and what being above/below the
range can generally indicate. Use cautious, non-alarming wording (see rule
16). Do not diagnose a condition from the result.

## Results Within Range
Briefly mention the Normal results using the supplied reference range,
without over-explaining each one and without claiming the person is
healthy overall.

## Results Requiring Manual Verification
Include ONLY Unknown results. For each, state that the value was detected
but the unit or reference range could not be reliably determined, so it
should be verified manually rather than interpreted automatically. Do not
describe or guess at its unit (see rule 14).

## General Guidance
Brief, general advice to discuss any above/below-range results with a
qualified healthcare professional who can interpret them alongside the
person's medical history. Do not prescribe treatment or medication.

## Informational Disclaimer
State clearly that this explanation is for informational purposes only,
does not provide a diagnosis, and should not replace interpretation by a
qualified healthcare professional.
""".strip()


def analyze_report_findings(
    tests: List[Dict[str, Any]],
    *,
    client: GeminiClient | None = None,
) -> Dict[str, Any]:
    """
    Generate a safe natural-language explanation from structured test results.

    Returns a dictionary so the UI can distinguish successful AI output from
    an unavailable/failed Gemini call without changing deterministic results.
    """
    validated_tests = _validate_tests(tests)

    if not validated_tests:
        return {
            "success": False,
            "explanation": "",
            "warning": "No structured laboratory results are available for AI explanation.",
        }

    try:
        gemini = client or GeminiClient()
        prompt = build_report_explanation_prompt(validated_tests)
        explanation = gemini.generate_response(prompt)

        return {
            "success": True,
            "explanation": explanation,
            "warning": "",
        }

    except Exception as exc:
        logger.exception("Medical report AI explanation failed.")
        return {
            "success": False,
            "explanation": "",
            "warning": (
                "AI explanation is currently unavailable. "
                "The deterministic laboratory results above remain unchanged."
            ),
            "error": str(exc),
        }