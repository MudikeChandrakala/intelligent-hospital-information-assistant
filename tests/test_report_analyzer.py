"""
tests/test_report_analyzer.py
=============================================================================
Deterministic unit tests for modules/report_analyzer.py.

An in-memory FakeOCRReader stands in for EasyOCR so these tests never load
a real OCR model. Only ReportAnalyzer's own extraction/classification
logic is under test here.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, List, Sequence, Tuple

import pytest
from PIL import Image

from modules.report_analyzer import ReportAnalyzer, get_report_analyzer


# =============================================================================
# Test helpers
# =============================================================================


def _tiny_image_bytes() -> bytes:
    """A minimal valid PNG so real image decoding always succeeds."""
    buffer = BytesIO()
    Image.new("RGB", (10, 10), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def _block(
    text: str,
    top: float,
    left: float = 0.0,
    width: float = 100.0,
    height: float = 20.0,
    confidence: float = 0.9,
) -> Tuple[List[List[float]], str, float]:
    """Build one EasyOCR-style detection: (bbox, text, confidence)."""
    bbox = [
        [left, top],
        [left + width, top],
        [left + width, top + height],
        [left, top + height],
    ]
    return (bbox, text, confidence)


def _line_as_blocks(
    words: Sequence[str], top: float, confidence: float = 0.9
) -> List[Tuple[List[List[float]], str, float]]:
    """Split one logical line across several OCR blocks on the same row,
    the way a real table-formatted report is often detected."""
    blocks = []
    left = 0.0
    for word in words:
        width = max(20.0, len(word) * 12.0)
        blocks.append(_block(word, top, left=left, width=width, confidence=confidence))
        left += width + 10.0
    return blocks


class FakeOCRReader:
    """Deterministic stand-in for easyocr.Reader."""

    def __init__(self, results: List[Tuple[Any, str, float]]) -> None:
        self._results = results
        self.calls = 0

    def readtext(self, image_array, detail=1, paragraph=False):  # noqa: D401
        self.calls += 1
        return self._results


class RaisingOCRReader:
    """Simulates an EasyOCR failure."""

    def readtext(self, image_array, detail=1, paragraph=False):
        raise RuntimeError("simulated OCR engine crash")


# =============================================================================
# Representative report fixture
# =============================================================================

_SAMPLE_REPORT_LINES = [
    "Hemoglobin 13.5 g/dL 12-16",
    "WBC 12500 /µL 4000-11000",
    "RBC 4.6 million/µL 4.0-5.5",
    "Platelet Count 250000 /µL 150000-450000",
]


def _sample_results() -> List[Tuple[Any, str, float]]:
    results: List[Tuple[Any, str, float]] = []
    for index, line in enumerate(_SAMPLE_REPORT_LINES):
        results.append(_block(line, top=float(index * 30), confidence=0.9))
    return results


def _analyzer_with_lines(lines: Sequence[str], confidence: float = 0.9) -> dict:
    results = [_block(line, top=float(i * 30), confidence=confidence) for i, line in enumerate(lines)]
    analyzer = ReportAnalyzer(reader=FakeOCRReader(results))
    return analyzer.analyze(_tiny_image_bytes())


# =============================================================================
# Successful extraction
# =============================================================================


def test_successful_extraction_from_representative_printed_report():
    analyzer = ReportAnalyzer(reader=FakeOCRReader(_sample_results()))
    result = analyzer.analyze(_tiny_image_bytes())

    assert result["success"] is True
    assert len(result["tests"]) == 4
    assert result["raw_text"]
    assert result["warnings"] == []


def test_extracts_name_value_unit_reference_range_fields():
    result = _analyzer_with_lines(["Hemoglobin 13.5 g/dL 12-16"])

    assert result["success"] is True
    assert len(result["tests"]) == 1

    record = result["tests"][0]
    assert record["test_name"] == "Hemoglobin"
    assert record["value"] == 13.5
    assert isinstance(record["value"], float)
    assert record["unit"] == "g/dL"
    assert record["reference_range"] == "12-16"


def test_integer_value_is_returned_as_int_not_float():
    result = _analyzer_with_lines(["WBC 12500 /µL 4000-11000"])

    record = result["tests"][0]
    assert record["value"] == 12500
    assert isinstance(record["value"], int)


# =============================================================================
# Status classification
# =============================================================================


def test_normal_classification_within_range():
    result = _analyzer_with_lines(["Hemoglobin 13.5 g/dL 12-16"])
    assert result["tests"][0]["status"] == "Normal"


def test_high_classification_above_range():
    result = _analyzer_with_lines(["WBC 12500 /µL 4000-11000"])
    assert result["tests"][0]["status"] == "High"


def test_low_classification_below_range():
    result = _analyzer_with_lines(["Hemoglobin 9.5 g/dL 12-16"])
    assert result["tests"][0]["status"] == "Low"


def test_unknown_when_reference_range_missing():
    result = _analyzer_with_lines(["Glucose Fasting 95 mg/dL"])

    record = result["tests"][0]
    assert record["status"] == "Unknown"
    assert record["reference_range"] == ""


# =============================================================================
# Safety: no fabrication
# =============================================================================


def test_no_fabricated_reference_range_for_unknown_test():
    # A made-up test name that cannot exist in any medical knowledge base.
    # The analyzer must not invent a plausible-looking range for it.
    result = _analyzer_with_lines(["Custom Marker XYZ 999 units"])

    record = result["tests"][0]
    assert record["reference_range"] == ""
    assert record["status"] == "Unknown"


def test_no_diagnosis_or_recommendation_fields_are_generated():
    result = _analyzer_with_lines(["WBC 12500 /µL 4000-11000"])

    forbidden_keys = {"diagnosis", "recommendation", "treatment", "medicine", "dosage"}
    assert forbidden_keys.isdisjoint(result.keys())

    for record in result["tests"]:
        assert forbidden_keys.isdisjoint(record.keys())


# =============================================================================
# Duplicate handling
# =============================================================================


def test_duplicate_test_lines_are_consolidated():
    results = [
        _block("Hemoglobin 13.5 g/dL 12-16", top=0.0),
        _block("Hemoglobin 13.5 g/dL 12-16", top=30.0),
    ]
    analyzer = ReportAnalyzer(reader=FakeOCRReader(results))
    result = analyzer.analyze(_tiny_image_bytes())

    assert len(result["tests"]) == 1
    assert any("duplicate" in warning.lower() for warning in result["warnings"])


def test_multi_block_line_is_merged_and_parsed_once():
    # Simulates a tabular report row where EasyOCR returns each column as
    # a separate detection on the same row instead of one merged string.
    blocks = _line_as_blocks(["Hemoglobin", "13.5", "g/dL", "12-16"], top=0.0)
    analyzer = ReportAnalyzer(reader=FakeOCRReader(blocks))
    result = analyzer.analyze(_tiny_image_bytes())

    assert result["success"] is True
    assert len(result["tests"]) == 1
    assert result["tests"][0]["test_name"] == "Hemoglobin"
    assert result["tests"][0]["value"] == 13.5


# =============================================================================
# Error handling
# =============================================================================


def test_malformed_image_bytes_return_controlled_result():
    analyzer = ReportAnalyzer(reader=FakeOCRReader([]))
    result = analyzer.analyze(b"this-is-not-a-real-image")

    assert result["success"] is False
    assert result["tests"] == []
    assert result["confidence"] == 0.0
    assert result["warnings"]


def test_ocr_returning_no_detections_returns_controlled_result():
    analyzer = ReportAnalyzer(reader=FakeOCRReader([]))
    result = analyzer.analyze(_tiny_image_bytes())

    assert result["success"] is False
    assert result["tests"] == []
    assert result["warnings"] == ["Unable to extract readable report text."]


def test_ocr_engine_exception_returns_controlled_result_not_a_crash():
    analyzer = ReportAnalyzer(reader=RaisingOCRReader())
    result = analyzer.analyze(_tiny_image_bytes())

    assert result["success"] is False
    assert result["tests"] == []
    assert result["warnings"]
    assert "simulated OCR engine crash" in result["warnings"][0]


def test_no_readable_lab_lines_still_succeeds_with_warning():
    # Header/footer metadata with no numeric result on any line.
    result = _analyzer_with_lines(["Apollo Diagnostics", "Patient Report"])

    assert result["success"] is True
    assert result["tests"] == []
    assert any("no structured" in warning.lower() for warning in result["warnings"])


# =============================================================================
# Confidence is derived, not fabricated
# =============================================================================


def test_confidence_reflects_actual_ocr_scores():
    results = [
        _block("Hemoglobin 13.5 g/dL 12-16", top=0.0, confidence=0.8),
        _block("WBC 12500 /µL 4000-11000", top=30.0, confidence=0.6),
    ]
    analyzer = ReportAnalyzer(reader=FakeOCRReader(results))
    result = analyzer.analyze(_tiny_image_bytes())

    assert result["confidence"] == 0.7


# =============================================================================
# Singleton accessor
# =============================================================================


def test_get_report_analyzer_returns_singleton():
    first = get_report_analyzer()
    second = get_report_analyzer()
    assert first is second


# =============================================================================
# Reference-range formats (Requirements 1, 2, 6)
# =============================================================================


def test_dash_separated_range_hyphen():
    result = _analyzer_with_lines(["Hemoglobin 13.0 g/dL 13.0-17.0"])
    record = result["tests"][0]
    assert record["reference_range"] == "13.0-17.0"
    assert record["status"] == "Normal"


def test_dash_separated_range_en_dash():
    result = _analyzer_with_lines(["Hemoglobin 13.0 g/dL 13.0\u201317.0"])
    record = result["tests"][0]
    assert record["reference_range"] == "13.0-17.0"


def test_dash_separated_range_em_dash():
    result = _analyzer_with_lines(["Hemoglobin 13.0 g/dL 13.0\u201417.0"])
    record = result["tests"][0]
    assert record["reference_range"] == "13.0-17.0"


def test_dash_separated_range_spaced_minus_sign():
    result = _analyzer_with_lines(["Hemoglobin 13.0 g/dL 13.0 \u2212 17.0"])
    record = result["tests"][0]
    assert record["reference_range"] == "13.0-17.0"


def test_whitespace_separated_range_is_recovered():
    # OCR dropped the dash entirely, e.g. "13.0-17.0" -> "13.0 17.0".
    result = _analyzer_with_lines(["Haemoglobin (Hb) 11.2 g/dL 13.0 17.0 LoW"])
    record = result["tests"][0]
    assert record["test_name"] == "Haemoglobin (Hb)"
    assert record["value"] == 11.2
    assert record["unit"] == "g/dL"
    assert record["reference_range"] == "13.0-17.0"
    assert record["status"] == "Low"


def test_whitespace_separated_range_integer_bounds():
    result = _analyzer_with_lines(["RBC Count 3.8 million/uL 4.50 5.90 LoW"])
    record = result["tests"][0]
    assert record["test_name"] == "RBC Count"
    assert record["value"] == 3.8
    assert record["unit"] == "million/uL"
    assert record["reference_range"] == "4.50-5.90"
    assert record["status"] == "Low"


# =============================================================================
# Explicit status token extraction (Requirement 3)
# =============================================================================


def test_explicit_low_status_extracted_case_insensitively():
    result = _analyzer_with_lines(["Haemoglobin (Hb) 11.2 g/dL 13.0 17.0 LoW"])
    assert result["tests"][0]["status"] == "Low"


def test_explicit_normal_status_extracted():
    result = _analyzer_with_lines(["MCV 90 fL 80.0 100.0 NORMAL"])
    record = result["tests"][0]
    assert record["reference_range"] == "80.0-100.0"
    assert record["status"] == "Normal"


def test_explicit_high_status_extracted():
    result = _analyzer_with_lines(["Something 999 units 100-500 HIGH"])
    assert result["tests"][0]["status"] == "High"


# =============================================================================
# Unit is isolated from range and status (Requirement 4)
# =============================================================================


def test_unit_excludes_range_and_status():
    result = _analyzer_with_lines(["MCH 29.5 pg 27.0 33.0 NORMAL"])
    record = result["tests"][0]
    assert record["unit"] == "pg"
    assert "27.0" not in record["unit"]
    assert "33.0" not in record["unit"]
    assert "normal" not in record["unit"].lower()


def test_unit_is_empty_when_ocr_drops_unit_token():
    # Known limitation: when OCR genuinely produces no unit text at all,
    # the analyzer honestly reports an empty unit rather than guessing
    # one (e.g. "%") from the test name.
    result = _analyzer_with_lines(["Hematocrit (HCT) 34.2 40.0 50.0 Low"])
    record = result["tests"][0]
    assert record["unit"] == ""
    assert record["reference_range"] == "40.0-50.0"
    assert record["status"] == "Low"


# =============================================================================
# Metadata rejection (Requirement 7)
# =============================================================================


@pytest.mark.parametrize(
    "line",
    [
        "Patient Name John Doe",
        "Patient ID 445566",
        "MRN 8823140",
        "UHID 8823140",
        "Age 34 Y",
        "Gender Male",
        "Sex M",
        "Date 15/08/2026",
        "Doctor Dr. Rao",
        "Ref Dr. Rao",
        "Sample Type Whole Blood EDTA",
        "Specimen Type Serum",
        "Barcode 90210192837",
        "Lab No 20260815001",
        "Report No RPT-9081",
        "Registration No 5591",
        "Accession No 5591",
        "Email lab@example.com",
        "Phone 9876543210",
        "Mobile 9876543210",
        "Address 12 MG Road Hyderabad",
        "Hospital Apollo Diagnostics",
    ],
)
def test_metadata_lines_are_rejected(line):
    assert ReportAnalyzer._parse_lab_line(line) is None


def test_kmc_reg_no_rejected_even_with_two_numbers():
    line = "KMC Reg No 112233 KMC Reg No:: 445566"
    assert ReportAnalyzer._parse_lab_line(line) is None


def test_metadata_rejection_end_to_end_alongside_real_results():
    result = _analyzer_with_lines(
        [
            "Patient ID 445566",
            "Age 34 Y",
            "Haemoglobin (Hb) 11.2 g/dL 13.0 17.0 LoW",
        ]
    )
    assert len(result["tests"]) == 1
    assert result["tests"][0]["test_name"] == "Haemoglobin (Hb)"


# =============================================================================
# Generic, non-hardcoded parsing across report types (Requirement 8)
# =============================================================================


def test_cbc_row_hematocrit():
    result = _analyzer_with_lines(["Hematocrit (HCT) 34.2 % 40.0 50.0 Low"])
    record = result["tests"][0]
    assert record["test_name"] == "Hematocrit (HCT)"
    assert record["value"] == 34.2
    assert record["unit"] == "%"
    assert record["reference_range"] == "40.0-50.0"
    assert record["status"] == "Low"


def test_cbc_row_mchc():
    result = _analyzer_with_lines(["MCHC 32.7 g/dL 32.0 36.0 NORMAL"])
    record = result["tests"][0]
    assert record["test_name"] == "MCHC"
    assert record["reference_range"] == "32.0-36.0"
    assert record["status"] == "Normal"


def test_cbc_row_rdw_cv_hyphenated_name_not_mistaken_for_range():
    result = _analyzer_with_lines(["RDW-CV 13.6 % 11.5 14.5 NORMAL"])
    record = result["tests"][0]
    assert record["test_name"] == "RDW-CV"
    assert record["value"] == 13.6
    assert record["reference_range"] == "11.5-14.5"


def test_non_cbc_lipid_profile_row():
    result = _analyzer_with_lines(["Total Cholesterol 180 mg/dL 125-200"])
    record = result["tests"][0]
    assert record["test_name"] == "Total Cholesterol"
    assert record["status"] == "Normal"


def test_non_cbc_thyroid_row():
    result = _analyzer_with_lines(["TSH 2.5 mIU/L 0.4-4.0"])
    record = result["tests"][0]
    assert record["test_name"] == "TSH"
    assert record["unit"] == "mIU/L"
    assert record["status"] == "Normal"


def test_non_cbc_electrolytes_row():
    result = _analyzer_with_lines(["Sodium 138 mmol/L 135-145"])
    record = result["tests"][0]
    assert record["test_name"] == "Sodium"
    assert record["status"] == "Normal"


def test_non_cbc_urine_analysis_row():
    result = _analyzer_with_lines(["Urine pH 6.0 4.5-8.0"])
    record = result["tests"][0]
    assert record["test_name"] == "Urine pH"
    assert record["reference_range"] == "4.5-8.0"


# =============================================================================
# Malformed OCR safety: never invent structure (Requirements 9, 10)
# =============================================================================


def test_badly_corrupted_range_does_not_invent_values():
    # Two unrelated number groups plus a repeated status token - the
    # structure cannot be reliably determined, so no range or status
    # should be fabricated from it.
    line = "Neutrophils Total WBC Count 12 500 72.0 Ix 4,000 11,000 HIGH HIGH"
    result = _analyzer_with_lines([line])

    assert result["success"] is True
    record = result["tests"][0]
    assert record["test_name"] == "Neutrophils Total WBC Count"
    assert record["value"] == 12
    assert record["reference_range"] == ""
    assert record["status"] == "Unknown"


def test_three_leftover_numbers_are_not_treated_as_a_range():
    result = _analyzer_with_lines(["Something 5 units 10 20 30"])
    record = result["tests"][0]
    assert record["reference_range"] == ""
    assert record["status"] == "Unknown"


def test_conflicting_status_tokens_resolve_to_unknown():
    result = _analyzer_with_lines(["Something 5 units 1-10 LOW HIGH"])
    assert result["tests"][0]["status"] == "Unknown"

# =============================================================================
# Adaptive-threshold OCR candidate (third preprocessing pass)
# =============================================================================
#
# These tests cover: the new preprocessing variant being generated, the
# OCR pipeline considering it as a third candidate, existing pass-selection
# behavior remaining intact when no fallback is needed, and graceful
# degradation when OpenCV is unavailable. They never assert anything about
# a specific laboratory test name, keeping the change generic.


def _photo_like_array(width: int = 200, height: int = 120) -> "np.ndarray":
    """A synthetic non-blank image so real preprocessing has something to
    operate on (uniform blank images produce degenerate thresholds)."""
    import numpy as np

    rng = np.random.default_rng(0)
    array = rng.integers(180, 230, size=(height, width, 3), dtype=np.uint8)
    array[40:60, 20:150] = 20  # a dark "text-like" band
    return array


def test_adaptive_threshold_variant_is_generated_when_cv2_available():
    pytest.importorskip("cv2")
    array = _photo_like_array()
    result = ReportAnalyzer._adaptive_threshold_for_ocr(array)
    assert result is not None
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_adaptive_threshold_returns_none_gracefully_without_cv2(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("simulated missing OpenCV")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    array = _photo_like_array()
    result = ReportAnalyzer._adaptive_threshold_for_ocr(array)
    assert result is None


def test_adaptive_threshold_never_raises_on_a_tiny_blank_image():
    pytest.importorskip("cv2")
    import numpy as np

    blank = np.full((10, 10, 3), 255, dtype=np.uint8)
    result = ReportAnalyzer._adaptive_threshold_for_ocr(blank)
    # Either a valid array or None - it must never raise.
    assert result is None or result.ndim == 3


class _CallCountingOCRReader:
    """Returns different canned results depending on which OCR call this
    is, so the test can tell which preprocessing variant "won"."""

    def __init__(self, results_by_call):
        self._results_by_call = results_by_call
        self.calls = 0

    def readtext(self, image_array, detail=1, paragraph=False):
        self.calls += 1
        index = min(self.calls - 1, len(self._results_by_call) - 1)
        return self._results_by_call[index]


def test_ocr_pipeline_considers_adaptive_threshold_as_a_third_pass():
    pytest.importorskip("cv2")

    # Primary pass: very low confidence, triggers both existing fallback
    # AND the new third candidate.
    low_confidence_primary = [_block("Ix", top=0.0, confidence=0.05)]
    # Enhanced pass: still weak.
    enhanced_pass = [_block("Ix weak", top=0.0, confidence=0.2)]
    # Adaptive-threshold pass: the strongest, most structured read.
    adaptive_pass = [_block("Hemoglobin 13.5 g/dL 12-16", top=0.0, confidence=0.9)]

    reader = _CallCountingOCRReader([low_confidence_primary, enhanced_pass, adaptive_pass])
    analyzer = ReportAnalyzer(reader=reader)
    result = analyzer.analyze_debug(_tiny_image_bytes())

    assert reader.calls == 3
    assert "adaptive_threshold" in result["debug"]["ocr_passes_considered"]
    assert result["debug"]["selected_ocr_pass"] == "adaptive_threshold"
    assert result["tests"][0]["test_name"] == "Hemoglobin"


def test_existing_pass_selection_is_unaffected_when_no_fallback_is_needed():
    # A clean, high-confidence primary pass must not trigger the enhanced
    # OR the adaptive-threshold candidate at all.
    reader = FakeOCRReader(_sample_results())
    analyzer = ReportAnalyzer(reader=reader)
    result = analyzer.analyze_debug(_tiny_image_bytes())

    assert result["debug"]["ocr_passes_considered"] == ["original"]
    assert result["debug"]["selected_ocr_pass"] == "original"
    assert reader.calls == 1


def test_adaptive_threshold_pass_failure_does_not_break_analysis():
    """If the adaptive-threshold OCR call itself raises, the pipeline must
    still return the best of the remaining candidates rather than fail."""

    class _AdaptivePassFailsReader:
        def __init__(self):
            self.calls = 0

        def readtext(self, image_array, detail=1, paragraph=False):
            self.calls += 1
            if self.calls == 1:
                return [_block("Ix", top=0.0, confidence=0.05)]
            if self.calls == 2:
                return [_block("Hemoglobin 13.5 g/dL 12-16", top=0.0, confidence=0.9)]
            raise RuntimeError("simulated adaptive-threshold OCR failure")

    reader = _AdaptivePassFailsReader()
    analyzer = ReportAnalyzer(reader=reader)
    result = analyzer.analyze(_tiny_image_bytes())

    assert result["success"] is True
    assert result["tests"][0]["test_name"] == "Hemoglobin"
# Perspective-correction OCR candidate (optional fourth preprocessing pass)
# =============================================================================

def _perspective_like_image_bytes() -> bytes:
    import numpy as np
    import cv2
    from PIL import Image
    from io import BytesIO

    canvas = np.zeros((500, 700, 3), dtype=np.uint8)
    canvas[:] = 35
    source = np.full((420, 620, 3), 245, dtype=np.uint8)
    cv2.rectangle(source, (8, 8), (611, 411), (0, 0, 0), 5)
    cv2.putText(source, "REPORT", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
    src = np.float32([[0, 0], [619, 0], [619, 419], [0, 419]])
    dst = np.float32([[70, 35], [650, 75], [610, 465], [45, 430]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(source, matrix, (700, 500), borderValue=(35, 35, 35))
    buffer = BytesIO()
    Image.fromarray(warped).save(buffer, format="PNG")
    return buffer.getvalue()


def test_perspective_correction_variant_is_generated_when_cv2_available():
    pytest.importorskip("cv2")
    result = ReportAnalyzer._perspective_correct_for_ocr(
        __import__("numpy").asarray(Image.open(BytesIO(_perspective_like_image_bytes())).convert("RGB"))
    )
    assert result is not None
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_perspective_correction_uses_luminance_fallback_when_edges_fail(monkeypatch):
    pytest.importorskip("cv2")
    import cv2
    import numpy as np

    # Force the primary edge-based detector to fail. The generic Otsu
    # luminance fallback should still find this large skewed page.
    monkeypatch.setattr(cv2, "Canny", lambda *args, **kwargs: np.zeros_like(args[0]))

    image = np.zeros((500, 700, 3), dtype=np.uint8)
    source = np.full((420, 620, 3), 245, dtype=np.uint8)
    src = np.float32([[0, 0], [619, 0], [619, 419], [0, 419]])
    dst = np.float32([[70, 35], [650, 75], [610, 465], [45, 430]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(
        source, matrix, (700, 500), borderValue=(25, 25, 25)
    )

    result = ReportAnalyzer._perspective_correct_for_ocr(image)

    assert result is not None
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert result.shape[1] > 500


def test_perspective_correction_returns_none_without_cv2(monkeypatch):
    import builtins
    import numpy as np

    real_import = builtins.__import__
    def _blocked_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("simulated missing OpenCV")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    result = ReportAnalyzer._perspective_correct_for_ocr(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result is None


def test_ocr_pipeline_can_consider_perspective_corrected_as_fourth_pass():
    pytest.importorskip("cv2")
    low = [_block("Ix", top=0.0, confidence=0.05)]
    enhanced = [_block("Ix weak", top=0.0, confidence=0.2)]
    adaptive = [_block("Ix still weak", top=0.0, confidence=0.25)]
    perspective = [_block("Glucose 95 mg/dL 70-99", top=0.0, confidence=0.9)]

    reader = _CallCountingOCRReader([low, enhanced, adaptive, perspective])
    analyzer = ReportAnalyzer(reader=reader)
    result = analyzer.analyze_debug(_perspective_like_image_bytes())

    assert reader.calls == 4
    assert "perspective_corrected" in result["debug"]["ocr_passes_considered"]
    assert result["debug"]["selected_ocr_pass"] == "perspective_corrected"
    assert result["tests"][0]["test_name"] == "Glucose"

# =============================================================================
# Perspective-correction area-threshold recalibration
# =============================================================================
#
# The minimum accepted page-contour area was lowered from 35% to 15% of the
# detection frame, and a generic whole-frame-boundary rejection was added,
# so a smaller in-frame page can be detected without a binarization fallback
# silently matching the photo's own outer border as if it were the page.


def _synthetic_page_photo(
    frame_width: int,
    frame_height: int,
    page_width_fraction: float,
    skew_px: int = 40,
    background_color: int = 90,
):
    """A synthetic photographed page: a skewed white rectangle with a few
    dark text-like marks, against a flat background, at a controlled
    fraction of the frame. Returns (image_array, actual_area_ratio)."""
    import numpy as np
    import cv2

    image = np.full((frame_height, frame_width, 3), background_color, dtype=np.uint8)
    page_width = int(frame_width * page_width_fraction)
    page_height = int(page_width * 1.3)
    center_x, center_y = frame_width // 2, frame_height // 2

    points = np.array(
        [
            [center_x - page_width // 2 + skew_px, center_y - page_height // 2],
            [center_x + page_width // 2, center_y - page_height // 2 - skew_px // 2],
            [center_x + page_width // 2 - skew_px // 2, center_y + page_height // 2],
            [center_x - page_width // 2, center_y + page_height // 2 + skew_px // 2],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, points, (255, 255, 255))

    for i in range(6):
        y = center_y - page_height // 3 + i * page_height // 8
        cv2.rectangle(
            image,
            (center_x - page_width // 3, y),
            (center_x + page_width // 3, y + 6),
            (30, 30, 30),
            -1,
        )

    area_ratio = (page_width * page_height) / (frame_width * frame_height)
    return image, area_ratio


def test_perspective_correction_accepts_a_small_in_frame_page():
    """A page occupying ~21% of the frame - below the OLD 35% threshold,
    above the NEW 15% threshold - must now be detected and rectified to a
    genuinely small crop, not left undetected."""
    pytest.importorskip("cv2")
    image, area_ratio = _synthetic_page_photo(1280, 720, page_width_fraction=0.30)
    assert 0.15 < area_ratio < 0.35  # confirms the fixture targets the gap

    result = ReportAnalyzer._perspective_correct_for_ocr(image)

    assert result is not None
    assert result.ndim == 3
    assert result.shape[2] == 3
    # Must be a genuinely small crop, not the whole 1280x720 frame.
    assert result.shape[0] < 700
    assert result.shape[1] < 1200


def test_perspective_correction_rejects_whole_frame_false_positive():
    """A binarization fallback matching the photo's own outer boundary
    (not a smaller page inside it) must be rejected, even though it would
    trivially satisfy the area/shape/angle gates on its own."""
    pytest.importorskip("cv2")
    import numpy as np

    # A flat, edge-free background with no page-like foreground at all -
    # the only "rectangle" available to any fallback is the frame itself.
    blank_frame = np.full((720, 1280, 3), 90, dtype=np.uint8)

    result = ReportAnalyzer._perspective_correct_for_ocr(blank_frame)

    # Either correctly rejected, or (if something is returned) it must not
    # be the degenerate whole-frame crop.
    if result is not None:
        assert not (result.shape[0] >= 720 * 0.97 and result.shape[1] >= 1280 * 0.97)


def test_perspective_correction_still_accepts_existing_large_page_case():
    """The pre-existing large-page-fill scenario (~74% of frame, matching
    the original passing perspective test's scale) must remain accepted
    after recalibrating the area threshold."""
    pytest.importorskip("cv2")
    image, area_ratio = _synthetic_page_photo(1280, 720, page_width_fraction=0.75)
    assert area_ratio > 0.35  # confirms this still exceeds the OLD threshold too

    result = ReportAnalyzer._perspective_correct_for_ocr(image)

    assert result is not None
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_perspective_correction_rejects_page_below_new_minimum_threshold():
    """A page occupying well under the new 15% minimum must still be
    correctly rejected - the threshold was recalibrated, not removed."""
    pytest.importorskip("cv2")
    image, area_ratio = _synthetic_page_photo(1280, 720, page_width_fraction=0.145)
    assert area_ratio < 0.15  # confirms the fixture targets below the new floor

    result = ReportAnalyzer._perspective_correct_for_ocr(image)

    assert result is None