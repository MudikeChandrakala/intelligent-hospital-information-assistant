"""
tests/test_reference_range_integration.py
=============================================================================
Tests for the reference_ranges.py integration added to
modules/report_analyzer.py: deterministic OCR unit normalization (used
only for general-range lookup, never for the displayed unit) and
labeled-context patient-sex extraction (used only as optional resolver
context, never guessed).

This file is intentionally separate from the project's existing
tests/test_report_analyzer.py, which was not available when this file was
written - keeping the new tests isolated avoids any risk of colliding
with existing fixtures/helpers there.
"""

from __future__ import annotations

import pytest

from modules.report_analyzer import ReportAnalyzer


# =============================================================================
# Unit normalization (lookup-only; must never change the displayed unit)
# =============================================================================


@pytest.mark.parametrize(
    "raw_unit,expected",
    [
        ("gldL Photometric", "g/dL"),
        ("gldL Calculated", "g/dL"),
        ("gmIdL", "g/dL"),
        ("fL Calculated", "fL"),
        ("pg Calculated", "pg"),
        ("% Microscopic", "%"),
        ("% Calculated", "%"),
        ("million/uL Electrical Impedance", "million/uL"),
        ("lakh/uL Electrical Impedance", "lakh/uL"),
        ("cells/uL Electrical Impedance", "cells/uL"),
    ],
)
def test_unit_normalization_known_ocr_corruptions(raw_unit, expected):
    assert ReportAnalyzer._normalize_unit_for_general_range_lookup(raw_unit) == expected


@pytest.mark.parametrize("raw_unit", ["Microscopic", "Calculated"])
def test_unit_normalization_bare_methodology_word_yields_no_unit(raw_unit):
    # Do NOT invent a unit when only a methodology descriptor was present.
    assert ReportAnalyzer._normalize_unit_for_general_range_lookup(raw_unit) == ""


def test_unit_normalization_never_guesses_ambiguous_unit():
    # "IpL" is not one of the pre-identified g/dL corruptions and must be
    # left unchanged (and therefore fail to resolve downstream) rather
    # than guessed.
    result = ReportAnalyzer._normalize_unit_for_general_range_lookup(
        "IpL Electrical Impedance"
    )
    assert result == "IpL"


def test_unit_normalization_does_not_affect_displayed_unit_field():
    row = {
        "test": "MCV",
        "result": "88",
        "flag": "",
        "unit": "fL Calculated",
        "reference": "",
    }
    record = ReportAnalyzer._parse_table_row(row)
    # The displayed unit keeps the original OCR text unchanged...
    assert record["unit"] == "fL Calculated"
    # ...even though the normalized "fL" was what actually resolved a range.
    assert record["reference_range"] == "80-100"
    assert record["status"] == "Normal"


# =============================================================================
# Sex extraction (labeled-context only; must never guess)
# =============================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Age / Gender: 36 Y / Male", "male"),
        ("Age / Gender: 36 Yrs / Female", "female"),
        ("Age/Gender: 42 Years / Male", "male"),
        ("Gender: Male", "male"),
        ("Gender: Female", "female"),
        ("Sex: Male", "male"),
        ("Sex: Female", "female"),
        ("Sex: M", "male"),
        ("Sex: F", "female"),
        ("Gender M", "male"),
    ],
)
def test_sex_extraction_labeled_variants(text, expected):
    assert ReportAnalyzer._extract_patient_sex(text) == expected


def test_sex_extraction_missing_gender_returns_none():
    text = "Patient Name: John Doe\nHospital: ABC Diagnostics\nAge: 34 Y\n"
    assert ReportAnalyzer._extract_patient_sex(text) is None


def test_sex_extraction_never_guesses_unlabeled_token():
    # "Female" appears in the text but is not attached to a Sex/Gender
    # label - must not be treated as the patient's sex.
    text = "Notes: family history of a sex-linked disorder. Female relative affected."
    assert ReportAnalyzer._extract_patient_sex(text) is None


def test_sex_extraction_empty_text_returns_none():
    assert ReportAnalyzer._extract_patient_sex("") is None


# =============================================================================
# End-to-end: report's own range vs. general fallback, with/without sex
# =============================================================================


def test_a_reports_own_range_remains_authoritative():
    row = {
        "test": "Hemoglobin",
        "result": "13.2",
        "flag": "",
        "unit": "g/dL",
        "reference": "12-16",
    }
    record = ReportAnalyzer._parse_table_row(row, sex="male")
    assert record["reference_range"] == "12-16"
    assert record["status"] == "Normal"
    # The general-range resolver must never have been consulted.
    assert "reference_source" not in record


def test_b_general_range_resolved_when_report_has_none():
    row = {
        "test": "Neutrophils",
        "result": "56",
        "flag": "",
        "unit": "%",
        "reference": "",
    }
    record = ReportAnalyzer._parse_table_row(row)
    assert record["reference_range"] == "40-60"
    assert record["status"] == "Normal"
    assert record["reference_source"] == "ucsf_uf_differential"


def test_sex_dependent_range_requires_sex_and_never_guesses():
    row = {
        "test": "Hemoglobin",
        "result": "13.2",
        "flag": "",
        "unit": "gldL Photometric",
        "reference": "",
    }
    without_sex = ReportAnalyzer._parse_table_row(row, sex=None)
    assert without_sex["reference_range"] == ""
    assert without_sex["status"] == "Unknown"

    with_sex = ReportAnalyzer._parse_table_row(row, sex="male")
    assert with_sex["reference_range"] == "13-18"
    assert with_sex["status"] == "Normal"


def test_j_ambiguous_ocr_unit_remains_unknown():
    row = {
        "test": "Total Leucocyte Count",
        "result": "7",
        "flag": "",
        "unit": "IpL Electrical Impedance",
        "reference": "",
    }
    record = ReportAnalyzer._parse_table_row(row)
    assert record["reference_range"] == ""
    assert record["status"] == "Unknown"


def test_k_existing_behavior_unaffected_when_test_name_unrecognized():
    row = {
        "test": "Custom Marker XYZ",
        "result": "999",
        "flag": "",
        "unit": "units",
        "reference": "",
    }
    record = ReportAnalyzer._parse_table_row(row, sex="male")
    assert record["reference_range"] == ""
    assert record["status"] == "Unknown"


def test_end_to_end_extract_tests_threads_sex_from_raw_text():
    raw_text = (
        "Age / Gender: 36 Y / Male\n"
        "Hemoglobin 13.2 gldL Photometric\n"
        "RBC Count 4.56 million/uL Electrical Impedance\n"
    )
    records, warnings = ReportAnalyzer._extract_tests(raw_text, blocks=None)
    by_name = {r["test_name"]: r for r in records}

    assert by_name["Hemoglobin"]["reference_range"] == "13-18"
    assert by_name["Hemoglobin"]["status"] == "Normal"
    # The displayed unit is untouched OCR text, not the normalized form.
    assert by_name["Hemoglobin"]["unit"] == "gldL Photometric"

    assert by_name["RBC Count"]["reference_range"] == "4.6-6.2"
    assert by_name["RBC Count"]["status"] == "Low"