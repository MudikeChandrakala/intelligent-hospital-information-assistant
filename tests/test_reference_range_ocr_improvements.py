"""Regression tests for narrow OCR reference-range lookup improvements."""
from modules.report_analyzer import ReportAnalyzer


def test_platelet_lakhlu_ocr_normalizes_for_lookup_only():
    row = {"test":"Platelet Count","result":"2.35","flag":"","unit":"lakhluL Electrical Impedance","reference":""}
    record = ReportAnalyzer._parse_table_row(row)
    assert record["unit"] == "lakhluL Electrical Impedance"
    assert record["reference_range"] == "150000-400000"
    assert record["status"] == "Normal"


def test_neutrophils_microscopic_is_safe_percentage_lookup_only():
    row = {"test":"Neutrophils","result":"56","flag":"","unit":"Microscopic","reference":""}
    record = ReportAnalyzer._parse_table_row(row)
    assert record["unit"] == "Microscopic"
    assert record["reference_range"] == "40-60"
    assert record["status"] == "Normal"


def test_rdw_calculated_is_safe_percentage_lookup_only():
    row = {"test":"RDW-CV","result":"13.6","flag":"","unit":"Calculated","reference":""}
    record = ReportAnalyzer._parse_table_row(row)
    assert record["unit"] == "Calculated"
    assert record["reference_range"] == "11-16"
    assert record["status"] == "Normal"


def test_ambiguous_tlc_unit_is_still_not_guessed():
    row = {"test":"Total Leucocyte Count","result":"7","flag":"","unit":"IpL Electrical Impedance","reference":""}
    record = ReportAnalyzer._parse_table_row(row)
    assert record["reference_range"] == ""
    assert record["status"] == "Unknown"


def test_existing_gdl_and_parenthetical_fixes_still_work():
    row = {"test":"Hemoglobin (Hb)","result":"13.2","flag":"","unit":"gldL Photometric","reference":""}
    record = ReportAnalyzer._parse_table_row(row, sex="male")
    assert record["unit"] == "gldL Photometric"
    assert record["reference_range"] == "13-18"
    assert record["status"] == "Normal"
