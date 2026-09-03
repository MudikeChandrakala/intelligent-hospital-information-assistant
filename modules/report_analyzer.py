"""
modules/report_analyzer.py
=============================================================================
Computer-printed medical/laboratory report OCR and structured test-result
extraction utilities for the Intelligent Hospital Information Assistant.

This module owns the non-UI business logic for the "Medical Report
Analysis" feature:
    - OCR with EasyOCR (reusing the project's existing OCR engine)
    - OCR detection merging into reading-order text lines
    - Structured lab-test-result extraction (name / value / unit /
      reference range / status)
    - Safe, non-crashing analysis result assembly

Scope (first implementation):
    This module targets COMPUTER-PRINTED reports (CBC, lipid profile,
    glucose, liver/kidney function, etc.), not handwritten prescriptions.

    It does NOT call Gemini, RAG, or any LLM. It is an extraction /
    classification layer only and must never:
        - diagnose diseases
        - recommend medicines or dosage
        - infer treatment
        - invent a missing reference range
        - guess unclear OCR text
        - convert an uncertain OCR result into a medical fact

    If a reference range cannot be reliably extracted, the record's
    status is "Unknown" rather than being filled in from outside
    knowledge.

Relationship to prescription analysis:
    This module does not modify, and does not depend on the internal
    parsing logic of, modules/prescription_analyzer.py or
    modules/prescription_ai_service.py. Its only interaction with that
    module is optionally reusing the already-initialized EasyOCR reader
    singleton exposed by PrescriptionAnalyzer, so a second OCR model is
    not loaded into memory. If that reuse is unavailable for any reason
    (import error, model not yet initialized, etc.) this module degrades
    to a controlled failure result rather than raising.

Public API
-----------
    ReportAnalyzer
    get_report_analyzer()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from modules.reference_ranges import resolve_reference_range

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger("hospital_assistant.report_analyzer")


# =============================================================================
# CONSTANTS
# =============================================================================

# Dash-like characters different OCR engines/fonts may use to separate a
# lower and upper reference-range bound: ASCII hyphen, Unicode hyphen,
# non-breaking hyphen, figure dash, en dash, em dash, and the true minus
# sign. Used both as an explicit separator ("12-16", "12 - 16", "12–16",
# "12 − 16") and to validate a whitespace-only-separated pair (see
# _parse_lab_line).
_DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2212"
_DASH_CHAR_SET = set(_DASH_CHARS)

# Matches an explicit "lower<dash>upper" reference range, allowing any of
# the dash characters above with optional surrounding whitespace.
_EXPLICIT_RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*[" + re.escape(_DASH_CHARS) + r"]\s*(\d+(?:\.\d+)?)"
)

# Matches one-sided laboratory reference limits such as "<34", ">90",
# "≤34", or "≥90". These are kept exactly as limits rather than being
# converted into guessed two-sided ranges.
_REFERENCE_LIMIT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([<>≤≥])\s*(\d+(?:\.\d+)?)"
)

# Matches the first standalone numeric result value on a line.
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])")

# Explicit status tokens a computer-printed report may show directly next
# to a result (in any letter case OCR happens to produce, e.g. "LoW").
_STATUS_PATTERN = re.compile(r"\b(low|high|normal)\b", re.IGNORECASE)

# Strips leading list/bullet markers OCR sometimes introduces, e.g.
# "1. Hemoglobin", "- WBC".
_LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:\d+[\.\)]\s*|[\u2022\-]\s*)")

_TRIM_CHARS = " \t:,-"

_MIN_TEST_NAME_LENGTH = 2

# Generic OCR-artifact checks for candidate laboratory test names. These are
# deliberately structural rather than a whitelist of medical test names, so
# the analyzer remains report-type independent.
_TITLE_CASE_TOKEN_PATTERN = re.compile(r"^[A-Z][a-z]+$")
_LOWER_PREFIX_CAPS_TOKEN_PATTERN = re.compile(r"^[a-z]+[A-Z][A-Za-z]*$")
_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


# Administrative / header-and-footer metadata that must never be treated
# as a laboratory test result, even when the line contains digits (e.g.
# "Patient ID: 12345", "Age: 34 Y", "KMC Reg No: 445566"). This list is
# intentionally generic (not tied to any one report type) so it applies
# across CBC, lipid, glucose, LFT, KFT, thyroid, electrolyte, and urine
# reports alike.
_METADATA_KEYWORDS = (
    r"patient\s*name",
    r"patient\s*id",
    r"\bmrn\b",
    r"\buhid\b",
    r"\bage\b",
    r"\bgender\b",
    r"\bsex\b",
    r"\bdate\b",
    r"\bdoctor\b",
    r"\bdr\b",
    r"\bref\b",
    r"sample\s*type",
    r"specimen\s*type",
    r"sample\s*(?:collected|received|drawn|collection|receipt)",
    r"\bbarcode\b",
    r"lab\s*no\b",
    r"report\s*(?:no|number|reg|registration)\b",
    r"\breg(?:istration)?\s*(?:no|number)\b",
    r"kmc\s*reg\s*no",
    r"\bregistration\b",
    r"\baccession\b",
    r"\bemail\b",
    r"\bphone\b",
    r"\bmobile\b",
    r"\bfax\b",
    r"\baddress\b",
    r"\bhospital\b",
    r"diagnostics?\s*(?:centre|center|laboratory|lab)?\b",
    r"\bsamples?\s+may\s+affect\b",
    r"\bcomment\b",
    r"\binterpretation\b",
    r"\bnote\s*:",
)
_METADATA_PATTERN = re.compile("|".join(_METADATA_KEYWORDS), re.IGNORECASE)

# OCR can turn a printed address/header into a line containing short
# fragments plus a comma-separated location. Such a line can contain digits
# and dash-like tokens, but it is not a laboratory result.
_LOCATION_LINE_PATTERN = re.compile(
    r"(?:\b(?:road|street|avenue|lane|nagar|district|state|country|pin\s*code)\b|"
    r"(?:\b[A-Za-z][A-Za-z .&/-]*,\s*){2,}|"
    # OCR can fragment an address so badly that the generic comma/address
    # pattern no longer matches (for example: "D /A G N ... Hyderabad,
    # Telangana, India."). These geographic markers are administrative
    # report-header/footer content, not laboratory analytes.
    r"\b(?:hyderabad|telangana|india)\b)",
    re.IGNORECASE,
)

# Common printed measurement units. If OCR places the unit before the result
# value (for example "Indirect Bilirubin mg/dL 0.2 ..."), move the unit out
# of the test name rather than leaving it embedded in ``test_name``.
_COMMON_UNIT_PATTERN = re.compile(
    r"\b(?:mg/dL|g/dL|g/L|mEq/L|mmol/L|µL|/uL|million/uL|lakh/uL|"
    r"pg|fL|%|IU/L|U/L|ng/mL|µg/dL|mg/L)\b",
    re.IGNORECASE,
)

# Common assay/method labels which OCR frequently leaves beside a result.
# These are presentation metadata, not measurement units.
_ASSAY_METHOD_PATTERN = re.compile(
    r"\b(?:diazo|calculated|enzymatic|urease\s+gldh|uricase\s+pap|"
    r"hexokinase|hplc|ifcc|ise|biuret|bcg|pNPP\s+AMP)\b",
    re.IGNORECASE,
)

# Numeric/reference fragments that OCR may leave after a known measurement
# unit. These are not part of the unit itself.
_TRAILING_NUMERIC_REFERENCE_PATTERN = re.compile(
    r"\s+(?:[<>]=?\s*)?\d+(?:\.\d+)?(?:\s*[-–—−]\s*\d+(?:\.\d+)?)?\s*$"
)

# eGFR-style OCR often leaves a threshold such as "> 90" and an assay/method
# label after the actual unit. Keep the actual measurement unit only.
_EGFR_REFERENCE_TAIL_PATTERN = re.compile(
    r"\s*(?:[<>]=?\s*\d+(?:\.\d+)?)\s*(?:ckd[-\s]?epi)?\s*$",
    re.IGNORECASE,
)

# -----------------------------------------------------------------------
# Table-region / column reconstruction (generic - no test-name whitelist)
# -----------------------------------------------------------------------

# Header-row anchor keywords used to locate the laboratory RESULTS TABLE
# inside a full report photo/scan, and to approximate its column
# boundaries from the header row's own bounding-box positions. These are
# column/table-structure labels, not laboratory test names, so using them
# does not make the analyzer report-type-specific, and they do not need
# to appear together - whichever anchors OCR actually recognized are used.
_TABLE_HEADER_ANCHOR_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "test": re.compile(
        r"\b(?:investigation|test|examination|parameters?)\b", re.IGNORECASE
    ),
    "result": re.compile(r"\b(?:results?|values?|observed\s*value)\b", re.IGNORECASE),
    "flag": re.compile(r"\bflags?\b", re.IGNORECASE),
    "unit": re.compile(r"\bunits?\b", re.IGNORECASE),
    "reference": re.compile(
        r"\b(?:reference\s*(?:range|interval)s?|"
        r"biological\s*reference\s*intervals?|normal\s*range)\b",
        re.IGNORECASE,
    ),
}

# At least this many distinct column anchors must be found on the same
# OCR row before that row is trusted as the table header. A single stray
# keyword elsewhere in the report is not enough evidence on its own.
_MIN_TABLE_HEADER_ANCHORS = 2

# -----------------------------------------------------------------------
# Unit normalization for GENERAL reference-range lookup only (never for
# the displayed "unit" field, which keeps its existing OCR/cleanup
# behavior unchanged - see _normalize_unit_for_general_range_lookup).
# -----------------------------------------------------------------------

# Instrument/methodology descriptor words OCR frequently merges onto the
# end of a unit column (e.g. "gldL Photometric", "fL Calculated",
# "million/uL Electrical Impedance"). These are lab-methodology labels,
# never measurement units themselves, so stripping them never removes
# real unit information - it only isolates the unit that follows/precedes
# it. Deliberately a short, explicit list rather than a general phrase
# matcher, so nothing outside these exact known descriptor words is ever
# altered.
_UNIT_METHODOLOGY_SUFFIX_PATTERN = re.compile(
    r"\b(?:photometric|calculated|microscopic|electrical\s+impedance)\b",
    re.IGNORECASE,
)

# A small, explicit set of literal g/dL OCR misreadings, taken directly
# from real observed OCR output (the "/" character read as a stray "l"
# or "I" immediately before "dL", e.g. "gldL", "gmIdL"). This is NOT a
# general slash-guessing heuristic - only these exact, pre-identified
# corrupted spellings (after lowercasing and removing internal spaces)
# are recognized; any other unrecognized spelling (e.g. "IpL") is left
# unchanged and will simply fail to resolve, which is the existing safe
# Unknown behavior.
_GDL_OCR_CORRUPTION_SPELLINGS = {"gldl", "gmidl"}

# One exact OCR corruption observed in platelet reports. This is a lookup-only
# spelling correction; it does not alter the displayed OCR unit.
_LAKH_UL_OCR_CORRUPTION_SPELLINGS = {"lakhlul"}

# Matches exactly one trailing "(...)" group on a test name, e.g. the
# "(Hb)" in "Hemoglobin (Hb)" or the "(TLC)" in "Total Leucocyte Count
# (TLC)". Used only to normalize a compound name for GENERAL
# reference-range lookup (see _normalize_test_name_for_general_range_lookup)
# - never to alter the displayed test_name.
_TRAILING_PARENTHETICAL_PATTERN = re.compile(r"\s*\([^()]*\)\s*$")

# Recognizes a patient sex/gender declaration only when it directly
# follows a "Sex"/"Gender" label (allowing a short run of label
# punctuation/digits/"Y" between the label and the value, e.g.
# "Age / Gender: 36 Y / Male"). Deliberately does NOT match a bare,
# unlabeled "Male"/"F" token anywhere in the document - that would be
# guessing. When no labeled declaration is found, sex stays None.
_SEX_LABEL_PATTERN = re.compile(
    r"\b(?:gender|sex)\b(?:[^A-Za-z]|\by(?:rs?|ears?)?\b){0,20}?(male|female|m|f)\b",
    re.IGNORECASE,
)

# A flat-text record without table structure is accepted only when its OCR
# evidence is at least minimally credible. This is deliberately a low floor:
# it rejects near-zero-confidence noise while retaining valid short labels such
# as Hb, Na, and TSH when EasyOCR read the row reliably.
_MIN_FLAT_TEXT_CANDIDATE_CONFIDENCE = 0.15

# A very low mean confidence is independent evidence that the original image
# needs the existing enhanced OCR retry. The retry does not alter any content;
# candidate selection still uses OCR-derived structure and confidence only.
_OCR_RETRY_CONFIDENCE_THRESHOLD = 0.35

# Header matching remains intentionally limited to structural column labels,
# never laboratory analyte names. These terms permit a bounded edit-distance
# tolerance for ordinary printed-OCR substitutions such as "Resuit" or
# "Investigatlon" while excluding unrelated, low-confidence fragments.
_TABLE_HEADER_FUZZY_TERMS: Dict[str, Tuple[str, ...]] = {
    "test": (
        "investigation",
        "test",
        "tests",
        "examination",
        "parameter",
        "parameters",
    ),
    "result": ("result", "results", "value", "values"),
    "flag": ("flag", "flags"),
    "unit": ("unit", "units"),
    "reference": ("reference", "interval", "range"),
}

# Explicit printed flag tokens preserved as their own field, never folded
# into a laboratory value or reference range.
_FLAG_HIGH_PATTERN = re.compile(r"^h(?:igh)?$", re.IGNORECASE)
_FLAG_LOW_PATTERN = re.compile(r"^l(?:ow)?$", re.IGNORECASE)
_FLAG_ABNORMAL_PATTERN = re.compile(r"abnormal", re.IGNORECASE)



# =============================================================================
# INTERNAL DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class _OCRBlock:
    text: str
    confidence: float
    bbox: Tuple[Tuple[float, float], ...]
    left: float
    top: float
    width: float
    height: float


# =============================================================================
# REPORT ANALYZER
# =============================================================================


class ReportAnalyzer:
    """
    EasyOCR-backed analyzer for computer-printed medical/laboratory reports.

    Extraction-only: this class never diagnoses, recommends treatment, or
    invents values, units, or reference ranges that OCR did not actually
    produce.
    """

    def __init__(self, reader: Optional[Any] = None) -> None:
        """
        Args:
            reader: Optional OCR reader implementing
                ``readtext(image_array, detail=1, paragraph=False) -> list``,
                matching EasyOCR's interface. Pass a fake/mock here (as the
                unit tests do) to keep tests deterministic and avoid
                loading a real OCR model. When omitted, the shared EasyOCR
                reader already initialized by the prescription analyzer is
                reused lazily on first use, so a second OCR model is not
                loaded.
        """
        self._reader: Optional[Any] = reader
        self._reader_error: Optional[str] = None

    # -------------------------------------------------------------------------
    # OCR reader access
    # -------------------------------------------------------------------------

    @property
    def reader(self) -> Optional[Any]:
        """Return the OCR reader, lazily reusing the shared EasyOCR engine."""
        if self._reader is not None:
            return self._reader

        self._reader = self._load_shared_reader()
        return self._reader

    def _load_shared_reader(self) -> Optional[Any]:
        """
        Reuse the project's existing EasyOCR singleton (owned by
        PrescriptionAnalyzer) instead of loading a second, competing OCR
        model. Any failure here is degraded to a controlled ``None`` so
        callers can return a clean error result instead of crashing.
        """
        try:
            from modules.prescription_analyzer import PrescriptionAnalyzer

            reader = PrescriptionAnalyzer._get_reader()
            if reader is None:
                self._reader_error = "EasyOCR is unavailable."
            return reader
        except Exception as exc:  # noqa: BLE001
            self._reader_error = f"OCR engine unavailable: {exc}"
            logger.warning(self._reader_error)
            return None

    # -------------------------------------------------------------------------
    # Image handling
    # -------------------------------------------------------------------------

    @staticmethod
    def _image_to_array(image: object) -> np.ndarray:
        """Convert supported image input types into an RGB numpy array."""
        if isinstance(image, np.ndarray):
            return image

        if hasattr(image, "getvalue"):
            image_bytes = image.getvalue()  # type: ignore[call-arg]
        elif isinstance(image, (bytes, bytearray)):
            image_bytes = bytes(image)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        else:
            raise ValueError("Unsupported image input type.")

        if not image_bytes:
            raise ValueError("Uploaded image is empty.")

        with Image.open(BytesIO(image_bytes)) as pil_image:
            return np.array(pil_image.convert("RGB"))

    # -------------------------------------------------------------------------
    # OCR execution and line merging
    # -------------------------------------------------------------------------

    @staticmethod
    def _block_debug_payload(block: "_OCRBlock") -> Dict[str, Any]:
        """Return a read-only diagnostic representation of an OCR block."""
        return {
            "text": block.text,
            "confidence": block.confidence,
            "bbox": [list(point) for point in block.bbox],
            "left": block.left,
            "top": block.top,
            "width": block.width,
            "height": block.height,
        }

    @classmethod
    def _row_debug_payload(cls, row: Sequence["_OCRBlock"]) -> Dict[str, Any]:
        """Return a read-only diagnostic representation of one OCR row."""
        if not row:
            return {"text": "", "blocks": [], "coordinates": {}}

        left = min(block.left for block in row)
        top = min(block.top for block in row)
        right = max(block.left + block.width for block in row)
        bottom = max(block.top + block.height for block in row)
        ordered = sorted(row, key=lambda block: block.left)

        return {
            "text": " ".join(block.text for block in ordered).strip(),
            "blocks": [cls._block_debug_payload(block) for block in ordered],
            "coordinates": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left,
                "height": bottom - top,
            },
        }

    @staticmethod
    def _merge_ocr_results(
        results: Sequence[Any],
    ) -> Tuple[str, float, List[_OCRBlock]]:
        """Group raw OCR detections into reading-order text lines."""
        blocks: List[_OCRBlock] = []
        confidences: List[float] = []

        for item in results:
            if not item or len(item) < 3:
                continue

            bbox = item[0]
            text = str(item[1] or "").strip()
            confidence = float(item[2] or 0.0)

            if not text:
                continue

            normalized_bbox = tuple(
                (float(point[0]), float(point[1])) for point in bbox
            )
            if not normalized_bbox:
                continue

            xs = [point[0] for point in normalized_bbox]
            ys = [point[1] for point in normalized_bbox]

            blocks.append(
                _OCRBlock(
                    text=text,
                    confidence=confidence,
                    bbox=normalized_bbox,
                    left=float(min(xs)),
                    top=float(min(ys)),
                    width=float(max(xs) - min(xs)) or 1.0,
                    height=float(max(ys) - min(ys)) or 1.0,
                )
            )
            confidences.append(confidence)

        if not blocks:
            return "", 0.0, []

        blocks.sort(key=lambda block: (block.top, block.left))

        merged_lines: List[str] = []
        current_line: List[_OCRBlock] = []
        current_top: Optional[float] = None
        current_height = 0.0

        for block in blocks:
            if not current_line:
                current_line = [block]
                current_top = block.top
                current_height = block.height
                continue

            line_threshold = max(12.0, current_height * 0.6)

            if current_top is not None and abs(block.top - current_top) <= line_threshold:
                current_line.append(block)
                current_top = min(current_top, block.top)
                current_height = max(current_height, block.height)
            else:
                current_line.sort(key=lambda entry: entry.left)
                merged_lines.append(
                    " ".join(entry.text for entry in current_line).strip()
                )
                current_line = [block]
                current_top = block.top
                current_height = block.height

        if current_line:
            current_line.sort(key=lambda entry: entry.left)
            merged_lines.append(
                " ".join(entry.text for entry in current_line).strip()
            )

        merged_text = "\n".join(line for line in merged_lines if line.strip())
        average_confidence = (
            round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        )

        return merged_text, average_confidence, blocks

    @staticmethod
    def _preprocess_for_ocr(image_array: np.ndarray) -> np.ndarray:
        """
        Create a high-contrast OCR fallback image without changing the report
        content.

        This is intentionally generic: it does not crop, redraw, infer, or
        hardcode any particular laboratory test. Upscaling and contrast/
        sharpness enhancement mainly help EasyOCR retain small punctuation
        such as ``<`` and ``>`` in computer-printed reference limits.
        """
        image = Image.fromarray(image_array).convert("RGB")

        # Small report symbols are easier for OCR to distinguish after a
        # moderate upscale. Avoid extreme scaling so memory usage stays sane.
        width, height = image.size
        scale = 2 if max(width, height) < 4000 else 1
        if scale > 1:
            image = image.resize(
                (width * scale, height * scale),
                Image.Resampling.LANCZOS,
            )

        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = ImageEnhance.Contrast(gray).enhance(1.35)
        gray = ImageEnhance.Sharpness(gray).enhance(1.5)
        gray = gray.filter(ImageFilter.SHARPEN)

        # EasyOCR accepts a numpy image array. Keep it 3-channel to match the
        # normal RGB path and avoid introducing a separate OCR engine/config.
        return np.stack([np.asarray(gray)] * 3, axis=-1)

    @staticmethod
    def _adaptive_threshold_for_ocr(image_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Create a locally-adaptive thresholded OCR candidate image.

        Unlike ``_preprocess_for_ocr``'s single global contrast/sharpness
        adjustment, this targets UNEVEN lighting - partial glare, shadow,
        or low contrast confined to one region of a photographed (not
        scanned) report - which a single global enhancement cannot
        correct, since the correct threshold for a well-lit region is
        wrong for a shadowed one and vice versa.

        This is generic: it does not crop, redraw, infer, or hardcode
        anything about a particular laboratory test or report layout.

        Returns None when OpenCV is unavailable or thresholding cannot be
        computed for any reason, so this stays an OPTIONAL third
        candidate rather than a new hard dependency or a source of
        crashes - callers simply do not get this candidate and fall back
        to the two existing passes.
        """
        try:
            import cv2  # Optional dependency - see docstring above.
        except ImportError:
            return None

        try:
            image = Image.fromarray(image_array).convert("RGB")

            width, height = image.size
            scale = 2 if max(width, height) < 4000 else 1
            if scale > 1:
                image = image.resize(
                    (width * scale, height * scale),
                    Image.Resampling.LANCZOS,
                )

            gray = np.asarray(ImageOps.grayscale(image))

            # A mild blur first keeps adaptive thresholding from amplifying
            # photograph sensor noise into extra false character edges.
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresholded = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                10,
            )

            # EasyOCR accepts a numpy image array; keep it 3-channel to
            # match the normal RGB path and avoid a separate OCR config.
            return np.stack([thresholded] * 3, axis=-1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Adaptive-threshold preprocessing failed: %s", exc)
            return None

    @staticmethod
    def _order_quad_points(points: np.ndarray) -> np.ndarray:
        """Return four points in top-left, top-right, bottom-right, bottom-left order."""
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).ravel()
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(differences)]
        ordered[3] = points[np.argmax(differences)]
        return ordered

    @staticmethod
    def _perspective_correct_for_ocr(
        image_array: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Create an optional OCR candidate by rectifying a document-shaped page.

        Detection is based only on generic image geometry.  The primary path
        uses edges; when edges do not expose a usable page boundary, a
        luminance/Otsu mask is used as a conservative fallback.
        """
        # A real photographed page frequently occupies well under half of a
        # handheld photo's frame (desk/background margin around it), so the
        # minimum accepted contour area is deliberately generous rather than
        # assuming the page nearly fills the shot.
        _MIN_PAGE_CONTOUR_AREA_RATIO = 0.15
        # A candidate whose bounding box covers almost the entire detection
        # frame is not "a smaller page inside the photo" - it is the photo's
        # own outer boundary being matched by a binarization fallback, and
        # must be rejected regardless of how well it otherwise scores.
        _MAX_FRAME_COVERAGE_RATIO = 0.97

        try:
            import cv2
        except ImportError:
            return None

        try:
            rgb = np.asarray(Image.fromarray(image_array).convert("RGB"))
            height, width = rgb.shape[:2]
            if width < 80 or height < 80:
                return None

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            scale = min(1.0, 1600.0 / max(width, height))
            if scale < 1.0:
                detection = cv2.resize(
                    gray,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                detection = gray

            image_area = float(detection.shape[0] * detection.shape[1])
            detection_height, detection_width = detection.shape[:2]

            def find_page_candidates(binary: np.ndarray):
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                found = []
                for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
                    area = float(cv2.contourArea(contour))
                    if area < image_area * _MIN_PAGE_CONTOUR_AREA_RATIO:
                        continue
                    perimeter = float(cv2.arcLength(contour, True))
                    if perimeter <= 0:
                        continue
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    if len(approx) != 4 or not cv2.isContourConvex(approx):
                        continue
                    points = approx.reshape(4, 2).astype(np.float32)
                    ordered = ReportAnalyzer._order_quad_points(points)
                    edge_lengths = [
                        float(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i]))
                        for i in range(4)
                    ]
                    if min(edge_lengths) < 30:
                        continue
                    corner_cosines = []
                    for i in range(4):
                        a = ordered[(i - 1) % 4] - ordered[i]
                        b = ordered[(i + 1) % 4] - ordered[i]
                        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                        if denom <= 1e-6:
                            break
                        corner_cosines.append(abs(float(np.dot(a, b)) / denom))
                    if len(corner_cosines) != 4 or max(corner_cosines) > 0.75:
                        continue
                    # Reject a candidate whose bounding box effectively covers
                    # the whole detection frame - that is a symptom of a
                    # binarization fallback matching the photo's own outer
                    # boundary, not a smaller page inside it, and warping to
                    # it would silently do nothing useful.
                    candidate_width = float(ordered[:, 0].max() - ordered[:, 0].min())
                    candidate_height = float(ordered[:, 1].max() - ordered[:, 1].min())
                    if (
                        candidate_width >= detection_width * _MAX_FRAME_COVERAGE_RATIO
                        and candidate_height >= detection_height * _MAX_FRAME_COVERAGE_RATIO
                    ):
                        continue
                    found.append((area, ordered))
                return found

            blurred = cv2.GaussianBlur(detection, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            edges = cv2.morphologyEx(
                edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
            )
            candidates = find_page_candidates(edges)

            # Generic fallback for pages whose boundary has weak/no Canny edges
            # (for example, a bright sheet against a darker background).
            if not candidates:
                _, otsu = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                otsu = cv2.morphologyEx(
                    otsu, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
                )
                candidates = find_page_candidates(otsu)
                if not candidates:
                    candidates = find_page_candidates(cv2.bitwise_not(otsu))

            if not candidates:
                return None

            _, source = max(candidates, key=lambda item: item[0])
            output_width = max(64, int(round(max(
                np.linalg.norm(source[1] - source[0]),
                np.linalg.norm(source[2] - source[3]),
            ))))
            output_height = max(64, int(round(max(
                np.linalg.norm(source[3] - source[0]),
                np.linalg.norm(source[2] - source[1]),
            ))))

            destination = np.array(
                [[0, 0], [output_width - 1, 0],
                 [output_width - 1, output_height - 1], [0, output_height - 1]],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(source, destination)
            corrected = cv2.warpPerspective(
                rgb, matrix, (output_width, output_height),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
            )
            return corrected if corrected.ndim == 3 and corrected.shape[2] == 3 else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Perspective-correction preprocessing failed: %s", exc)
            return None


    @staticmethod
    def _needs_ocr_fallback(raw_text: str, confidence: float) -> bool:
        """
        Detect a generic OCR symptom where a one-sided reference comparator
        may have been dropped.

        Example:
            "Anti-TPO 156 IU/mL 34"
        instead of:
            "Anti-TPO 156 IU/mL <34"

        We do NOT convert the trailing number into a reference limit here.
        We only decide whether a second OCR pass is worth attempting.
        """
        if confidence < _OCR_RETRY_CONFIDENCE_THRESHOLD:
            return True

        for line in raw_text.splitlines():
            working = line.strip()
            if not working or not re.search(r"\\d", working):
                continue
            if _REFERENCE_LIMIT_PATTERN.search(working):
                continue
            if _EXPLICIT_RANGE_PATTERN.search(working):
                continue

            numbers = ReportAnalyzer._numbers_outside_parentheses(working)
            if len(numbers) != 2:
                continue

            # First number is the likely result; a single trailing number
            # immediately following a measurement/unit is a common symptom
            # of a dropped '<' or '>' comparator.
            first_end = numbers[0].end()
            gap = working[first_end : numbers[1].start()].strip()
            if gap and re.search(r"[A-Za-z%/µ]", gap):
                return True

        return False

    @classmethod
    def _ocr_quality_score(
        cls,
        raw_text: str,
        confidence: float,
        blocks: Sequence["_OCRBlock"],
    ) -> Tuple[int, float]:
        """
        Score OCR output for choosing between the normal and fallback pass.

        Explicit ranges/limits and explicit status tokens are strong evidence
        of useful structured report text. The score never creates or changes
        extracted medical values.
        """
        explicit_ranges = len(_EXPLICIT_RANGE_PATTERN.findall(raw_text))
        explicit_limits = len(_REFERENCE_LIMIT_PATTERN.findall(raw_text))
        statuses = len(_STATUS_PATTERN.findall(raw_text))
        nonempty_lines = sum(bool(line.strip()) for line in raw_text.splitlines())

        table_header_bonus = 10 if cls._detect_table_header(blocks) is not None else 0
        score = (
            explicit_ranges * 4
            + explicit_limits * 5
            + statuses * 2
            + min(nonempty_lines, 20)
            + table_header_bonus
        )
        return score, confidence

    # -------------------------------------------------------------------------
    # Table-region detection and column reconstruction
    # -------------------------------------------------------------------------

    @staticmethod
    def _group_blocks_into_rows(
        blocks: Sequence["_OCRBlock"],
    ) -> List[List["_OCRBlock"]]:
        """Group OCR blocks into reading-order rows by vertical proximity.

        Uses the same vertical-grouping idea as ``_merge_ocr_results`` but
        keeps each row's individual blocks (with their bounding boxes)
        instead of collapsing them into a single text string, so column
        position information is preserved for table reconstruction.
        """
        ordered = sorted(blocks, key=lambda block: (block.top, block.left))

        rows: List[List["_OCRBlock"]] = []
        current_row: List["_OCRBlock"] = []
        current_top: Optional[float] = None
        current_height = 0.0

        for block in ordered:
            if not current_row:
                current_row = [block]
                current_top = block.top
                current_height = block.height
                continue

            threshold = max(12.0, current_height * 0.6)
            if current_top is not None and abs(block.top - current_top) <= threshold:
                current_row.append(block)
                current_top = min(current_top, block.top)
                current_height = max(current_height, block.height)
            else:
                rows.append(current_row)
                current_row = [block]
                current_top = block.top
                current_height = block.height

        if current_row:
            rows.append(current_row)

        return rows

    @staticmethod
    def _header_token_matches(term: str, token: str) -> bool:
        """Match one structural header term with a bounded OCR tolerance."""
        if token == term:
            return True
        if len(token) < 4 or len(term) < 4:
            return False

        maximum_distance = 1 if len(term) <= 6 else 2
        if abs(len(token) - len(term)) > maximum_distance:
            return False

        previous = list(range(len(token) + 1))
        for term_index, term_character in enumerate(term, start=1):
            current = [term_index]
            for token_index, token_character in enumerate(token, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[token_index] + 1,
                        previous[token_index - 1]
                        + (term_character != token_character),
                    )
                )
            previous = current

        return previous[-1] <= maximum_distance

    @classmethod
    def _find_table_header_anchor(cls, text: str) -> Optional[str]:
        """Return a structural header column matched from one OCR block."""
        for column, pattern in _TABLE_HEADER_ANCHOR_PATTERNS.items():
            if pattern.search(text):
                return column

        tokens = re.findall(r"[A-Za-z]+", text.lower())
        for column, terms in _TABLE_HEADER_FUZZY_TERMS.items():
            if any(
                cls._header_token_matches(term, token)
                for term in terms
                for token in tokens
            ):
                return column
        return None

    @staticmethod
    def _anchors_have_table_alignment(anchors: Dict[str, float]) -> bool:
        """Require independent header signals to occupy separate columns."""
        if len(anchors) < _MIN_TABLE_HEADER_ANCHORS:
            return False

        positions = sorted(anchors.values())
        return all(
            right - left >= 20.0 for left, right in zip(positions, positions[1:])
        )

    @classmethod
    def _detect_table_header(
        cls,
        blocks: Sequence["_OCRBlock"],
        debug: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Locate the laboratory-results table header row and approximate
        column boundaries from its own bounding boxes.

        This looks only for structural column-heading words
        (Investigation/Test, Result, Flag, Units, Reference Range/
        Interval) - never a laboratory test name - so it applies
        identically across CBC, thyroid, lipid, or any other report
        layout. Returns None when no single row contains at least two
        distinct column anchors, so callers can safely fall back to the
        existing text-line parser. Rows above the detected header (e.g.
        hospital name, patient details) are never treated as table data.
        """
        if not blocks:
            return None

        rows = cls._group_blocks_into_rows(blocks)
        candidate_rows: List[Dict[str, Any]] = []

        best_row: Optional[List["_OCRBlock"]] = None
        best_anchors: Dict[str, float] = {}

        for index, row in enumerate(rows):
            anchors: Dict[str, float] = {}
            for block in row:
                column = cls._find_table_header_anchor(block.text)
                if column is not None and column not in anchors:
                    anchors[column] = block.left
            has_table_alignment = cls._anchors_have_table_alignment(anchors)
            row_debug = cls._row_debug_payload(row)
            row_debug.update(
                {
                    "row_index": index,
                    "recognized_anchors": anchors,
                    "is_header_candidate": has_table_alignment,
                }
            )
            candidate_rows.append(row_debug)

            if has_table_alignment and len(anchors) > len(
                best_anchors
            ):
                best_anchors = anchors
                best_row = row

        if debug is not None:
            debug["candidate_header_rows"] = candidate_rows

        if best_row is None:
            if debug is not None:
                debug["header_found"] = False
                debug["selected_header"] = None
            return None

        header_bottom = max(block.top + block.height for block in best_row)
        header = {"anchors": best_anchors, "header_bottom": header_bottom}
        if debug is not None:
            debug["header_found"] = True
            selected = cls._row_debug_payload(best_row)
            selected.update(
                {
                    "recognized_anchors": best_anchors,
                    "header_bottom": header_bottom,
                }
            )
            debug["selected_header"] = selected
        return header

    @classmethod
    def _reconstruct_table_rows(
        cls,
        blocks: Sequence["_OCRBlock"],
        header: Dict[str, Any],
        debug: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        Group blocks BELOW the detected header row into table rows, and
        assign each block to the nearest header-anchor column by
        horizontal position.

        Content above the header row is never included, so hospital or
        patient header text is structurally excluded regardless of its
        wording - a garbled header phrase cannot leak into a laboratory
        row just because it happens to look like plausible text. A
        neighboring column's text also cannot leak into another column
        beyond the midpoint boundary between them.
        """
        anchors: Dict[str, float] = header["anchors"]
        header_bottom: float = header["header_bottom"]

        sorted_columns = sorted(anchors.items(), key=lambda item: item[1])
        column_names = [name for name, _left in sorted_columns]
        column_lefts = [left for _name, left in sorted_columns]

        def _assign_column(left: float) -> str:
            best_index = 0
            best_distance: Optional[float] = None
            for index, anchor_left in enumerate(column_lefts):
                distance = abs(left - anchor_left)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = index
            return column_names[best_index]

        body_blocks = [block for block in blocks if block.top > header_bottom - 1.0]
        rows = cls._group_blocks_into_rows(body_blocks)

        table_rows: List[Dict[str, str]] = []
        debug_rows: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            row_sorted = sorted(row, key=lambda block: block.left)
            columns: Dict[str, List[str]] = {name: [] for name in column_names}
            for block in row_sorted:
                columns[_assign_column(block.left)].append(block.text)
            assigned_columns = {
                name: " ".join(parts).strip() for name, parts in columns.items()
            }
            table_rows.append(assigned_columns)
            if debug is not None:
                row_debug = cls._row_debug_payload(row)
                row_debug.update(
                    {
                        "row_index": index,
                        "column_assignments": assigned_columns,
                    }
                )
                debug_rows.append(row_debug)

        if debug is not None:
            debug["reconstructed_rows"] = debug_rows

        return table_rows

    @staticmethod
    def _parse_reference_column(text: str) -> str:
        """
        Extract a reference range/limit from an already column-isolated
        Reference Range cell.

        No ambiguous-structure guessing is needed here: because the
        result value and other fields are already separated by column,
        this only has to recognize the range format itself, reusing the
        exact same dash/limit conventions as the text-line parser.
        """
        cleaned = text.strip()
        if not cleaned:
            return ""

        limit_match = _REFERENCE_LIMIT_PATTERN.search(cleaned)
        if limit_match:
            return f"{limit_match.group(1)}{limit_match.group(2)}"

        range_match = _EXPLICIT_RANGE_PATTERN.search(cleaned)
        if range_match:
            return f"{range_match.group(1)}-{range_match.group(2)}"

        # A dropped dash can leave a bare "lower upper" pair; only trust
        # this when the ENTIRE cell is exactly two numbers, since the
        # column is already isolated from the result and unit text.
        numbers = _NUMBER_PATTERN.findall(cleaned)
        if len(numbers) == 2:
            return f"{numbers[0]}-{numbers[1]}"

        return ""

    @staticmethod
    def _classify_status_from_reference(
        value: Union[int, float], reference_range: str
    ) -> str:
        """Classify a value against an already-extracted reference range/limit."""
        if not reference_range:
            return "Unknown"

        if reference_range[0] in "<>≤≥":
            try:
                limit_value = float(reference_range[1:])
            except ValueError:
                return "Unknown"
            comparator = reference_range[0]
            if comparator in "<≤":
                return "High" if value > limit_value else "Normal"
            return "Low" if value < limit_value else "Normal"

        bounds = ReportAnalyzer._parse_range_bounds(reference_range)
        if not bounds:
            return "Unknown"
        lower, upper = bounds
        if value < lower:
            return "Low"
        if value > upper:
            return "High"
        return "Normal"

    @staticmethod
    def _normalize_unit_for_general_range_lookup(raw_unit: str) -> str:
        """
        Produce a unit string suitable for matching against
        reference_ranges.py, WITHOUT altering the unit that is displayed
        to the user (callers must keep using the original ``raw_unit``
        for the record's own "unit" field - this return value is only
        ever passed to resolve_reference_range).

        Two narrow, deterministic steps only:
          1. Strip known lab-methodology descriptor words that OCR
             merges onto the unit column (see
             _UNIT_METHODOLOGY_SUFFIX_PATTERN). If nothing but a
             descriptor word was present (e.g. "Microscopic" or
             "Calculated" alone), the result is "" - no unit is
             invented.
          2. If what remains exactly matches one of the small set of
             pre-identified corrupted g/dL spellings, return the
             canonical "g/dL". Otherwise the stripped text is returned
             completely unchanged - reference_ranges.py's own unit
             matching decides whether it recognizes it, and safely
             returns None (Unknown) if it does not. Nothing here ever
             guesses at an ambiguous or unrecognized unit.
        """
        if not raw_unit:
            return ""

        stripped = _UNIT_METHODOLOGY_SUFFIX_PATTERN.sub("", raw_unit)
        stripped = stripped.strip(_TRIM_CHARS + " ")
        if not stripped:
            return ""

        compact = re.sub(r"\s+", "", stripped).lower()
        if compact in _GDL_OCR_CORRUPTION_SPELLINGS:
            return "g/dL"
        if compact in _LAKH_UL_OCR_CORRUPTION_SPELLINGS:
            return "lakh/uL"

        return stripped

    @staticmethod
    def _lookup_unit_with_safe_method_inference(raw_name: str, raw_unit: str) -> str:
        """Return a lookup-only unit, with narrow test-specific method inference.

        Some OCR output contains only a methodology descriptor in the unit
        column (for example ``Microscopic`` or ``Calculated``). For a small
        set of differential/CBC analytes whose general ranges are explicitly
        percentage-based in reference_ranges.py, that descriptor is safe
        structural evidence that the printed value is a percentage. This is
        intentionally limited to the supported test names; it never changes
        the displayed unit and never infers a unit for ambiguous analytes such
        as TLC/WBC.
        """
        normalized_unit = ReportAnalyzer._normalize_unit_for_general_range_lookup(raw_unit)
        if normalized_unit:
            return normalized_unit

        canonical_name = ReportAnalyzer._normalize_test_name_for_general_range_lookup(raw_name)
        canonical_name = canonical_name.strip().lower()
        percentage_tests = {
            "neutrophils",
            "lymphocytes",
            "monocytes",
            "eosinophils",
            "basophils",
            "rdw-cv",
            "rdw cv",
            "rdw",
        }
        if canonical_name in percentage_tests:
            return "%"
        return ""

    @staticmethod
    def _normalize_test_name_for_general_range_lookup(raw_name: str) -> str:
        """
        Strip one trailing "(...)" group from a compound test name for
        GENERAL reference-range lookup only (never for the displayed
        "test_name" field, which callers must keep using the original
        ``raw_name`` for).

        e.g. "Hemoglobin (Hb)" -> "Hemoglobin",
             "Total Leucocyte Count (TLC)" -> "Total Leucocyte Count",
             "Hematocrit (PCV)" -> "Hematocrit".

        Deterministic parenthetical stripping only - no fuzzy matching,
        no substring search, and only a single trailing group is removed
        (a name with no trailing parenthetical is returned unchanged).
        reference_ranges.py's own canonical_test_name() still decides
        whether the (possibly-unchanged) result is recognized, and
        safely returns None if it is not - this never invents a name.
        """
        if not raw_name:
            return raw_name
        stripped = _TRAILING_PARENTHETICAL_PATTERN.sub("", raw_name).strip()
        return stripped or raw_name

    @staticmethod
    def _extract_patient_sex(raw_text: str) -> Optional[str]:
        """
        Extract patient sex from a "Sex"/"Gender" label elsewhere in the
        report text (e.g. "Gender: Male", "Sex: F",
        "Age / Gender: 36 Y / Male"), for use ONLY as optional context
        passed to resolve_reference_range's sex-dependent ranges.

        Returns "male", "female", or None. Never guesses: a bare,
        unlabeled "Male"/"F" token elsewhere in the document (with no
        Sex/Gender label directly attached) is intentionally NOT
        matched, since that would not be a safe, unambiguous signal.
        """
        match = _SEX_LABEL_PATTERN.search(raw_text)
        if not match:
            return None

        token = match.group(1).strip().lower()
        if token in ("male", "m"):
            return "male"
        if token in ("female", "f"):
            return "female"
        return None

    @staticmethod
    def _apply_general_reference_range(
        name: str,
        value: Union[int, float],
        unit: str,
        reference_range: str,
        status: str,
        *,
        sex: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Resolve a documented general range only when the report has none.

        The laboratory's own reference range is always authoritative. A
        general range is consulted only when the report supplied no usable
        range and the current status is still Unknown. The resolver itself
        refuses ambiguous names, units, and sex-dependent ranges without
        explicit sex context, so this helper never guesses a conversion.
        ``sex`` is optional context only (see _extract_patient_sex) - the
        resolver returns None for a sex-dependent analyte when sex is not
        supplied, rather than guessing.
        """
        if reference_range or status != "Unknown":
            return reference_range, status, {}

        resolved = resolve_reference_range(name, value, unit, sex=sex)
        if not resolved:
            return reference_range, status, {}

        return (
            str(resolved.get("reference_range") or ""),
            str(resolved.get("status") or "Unknown"),
            {
                "reference_source": resolved.get("reference_source"),
                "reference_source_label": resolved.get("reference_source_label"),
                "reference_source_url": resolved.get("reference_source_url"),
                "reference_type": "general",
                "unit_convention": resolved.get("unit_convention"),
            },
        )

    @classmethod
    def _parse_table_row(
        cls,
        columns: Dict[str, str],
        diagnostic: Optional[Dict[str, Any]] = None,
        *,
        sex: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse one column-reconstructed table row into a structured
        laboratory record.

        The test name comes ONLY from the Test/Investigation column, the
        result ONLY from the Result column, and so on - a neighboring
        column's text cannot leak into another field. Applies the same
        generic metadata/name-plausibility checks used by the text-line
        parser, plus the fact that this row was found below the detected
        table header, as independent, combined evidence against OCR
        garbage becoming a fake laboratory test.
        """
        def reject(reason: str) -> None:
            if diagnostic is not None:
                diagnostic.update({"accepted": False, "rejection_reason": reason})

        name = (columns.get("test") or "").strip()
        name = _LIST_PREFIX_PATTERN.sub("", name).strip(_TRIM_CHARS)

        if len(name) < _MIN_TEST_NAME_LENGTH or not re.search(r"[A-Za-z]", name):
            reject("Test column does not contain a plausible alphabetic test name.")
            return None
        if _METADATA_PATTERN.search(name):
            reject("Test column matches administrative metadata.")
            return None
        if _LOCATION_LINE_PATTERN.search(name):
            reject("Test column matches a location/address pattern.")
            return None

        result_text = columns.get("result") or ""
        value_match = _NUMBER_PATTERN.search(result_text)
        if value_match is None:
            # Column alignment can occasionally shift the result by one
            # bucket; as a narrow, generic tolerance, only look in an
            # adjacent column when the dedicated Result column produced
            # nothing at all.
            for fallback_column in ("flag", "unit"):
                value_match = _NUMBER_PATTERN.search(columns.get(fallback_column) or "")
                if value_match:
                    break
        if value_match is None:
            reject("Result column and permitted adjacent columns contain no numeric value.")
            return None

        try:
            value = cls._to_number(value_match.group())
        except ValueError:
            reject("Numeric result could not be converted to a number.")
            return None

        flag_text = (columns.get("flag") or "").strip()
        flag = ""
        if _FLAG_HIGH_PATTERN.fullmatch(flag_text):
            flag = "High"
        elif _FLAG_LOW_PATTERN.fullmatch(flag_text):
            flag = "Low"
        elif _FLAG_ABNORMAL_PATTERN.search(flag_text):
            flag = "Abnormal"

        unit_text = (columns.get("unit") or "").strip(_TRIM_CHARS + " ")
        reference_range = cls._parse_reference_column(columns.get("reference") or "")

        if flag in ("High", "Low"):
            status = flag
        elif reference_range:
            status = cls._classify_status_from_reference(value, reference_range)
        else:
            status = "Unknown"

        reference_range, status, reference_metadata = cls._apply_general_reference_range(
            cls._normalize_test_name_for_general_range_lookup(name),
            value,
            cls._lookup_unit_with_safe_method_inference(name, unit_text),
            reference_range,
            status,
            sex=sex,
        )

        if not cls._has_reasonable_test_name_structure(
            name, unit=unit_text, reference_range=reference_range, status=status
        ):
            reject("Test name failed generic OCR-structure validation.")
            return None

        record: Dict[str, Any] = {
            "test_name": name,
            "value": value,
            "unit": unit_text,
            "reference_range": reference_range,
            "status": status,
        }
        if reference_metadata:
            record.update(reference_metadata)
        if flag:
            record["flag"] = flag
        if diagnostic is not None:
            diagnostic.update({"accepted": True, "record": record})
        return record

    def _run_ocr(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Run OCR using the existing shared EasyOCR reader.

        The original image is always tried first. If the OCR text shows a
        generic symptom that a small comparator such as '<' or '>' may have
        been lost, one enhanced-image fallback pass is performed. The two
        passes use the same EasyOCR reader; no competing OCR architecture or
        second model is introduced.
        """
        reader = self.reader
        if reader is None:
            return {
                "success": False,
                "warnings": [self._reader_error or "OCR engine is unavailable."],
            }

        try:
            primary_results = reader.readtext(
                image_array,
                detail=1,
                paragraph=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("EasyOCR failure during report analysis.")
            return {"success": False, "warnings": [f"OCR failed: {exc}"]}

        primary_text, primary_confidence, primary_blocks = self._merge_ocr_results(
            primary_results
        )

        if not primary_text.strip():
            return {
                "success": False,
                "warnings": ["Unable to extract readable report text."],
            }

        candidates = [
            {
                "pass_name": "original",
                "raw_text": primary_text,
                "confidence": primary_confidence,
                "blocks": primary_blocks,
            }
        ]

        # Only pay the extra OCR cost when the first pass contains the
        # characteristic shape of a dropped one-sided comparator.
        if self._needs_ocr_fallback(primary_text, primary_confidence):
            try:
                enhanced_image = self._preprocess_for_ocr(image_array)
                fallback_results = reader.readtext(
                    enhanced_image,
                    detail=1,
                    paragraph=False,
                )
                fallback_text, fallback_confidence, fallback_blocks = (
                    self._merge_ocr_results(fallback_results)
                )
                if fallback_text.strip():
                    candidates.append(
                        {
                            "pass_name": "enhanced",
                            "raw_text": fallback_text,
                            "confidence": fallback_confidence,
                            "blocks": fallback_blocks,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                # The primary OCR result remains usable; a fallback failure
                # must never make an otherwise readable report fail.
                logger.warning("Enhanced OCR fallback failed: %s", exc)

            # A third candidate, tried under the same trigger condition as
            # the enhanced pass: local/adaptive thresholding targets
            # uneven lighting or glare that a single global contrast pass
            # does not correct. Skipped entirely (no error) when OpenCV
            # is unavailable.
            adaptive_image = self._adaptive_threshold_for_ocr(image_array)
            if adaptive_image is not None:
                try:
                    adaptive_results = reader.readtext(
                        adaptive_image,
                        detail=1,
                        paragraph=False,
                    )
                    adaptive_text, adaptive_confidence, adaptive_blocks = (
                        self._merge_ocr_results(adaptive_results)
                    )
                    if adaptive_text.strip():
                        candidates.append(
                            {
                                "pass_name": "adaptive_threshold",
                                "raw_text": adaptive_text,
                                "confidence": adaptive_confidence,
                                "blocks": adaptive_blocks,
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    # As with the enhanced pass, a failure here must never
                    # make an otherwise readable report fail.
                    logger.warning("Adaptive-threshold OCR pass failed: %s", exc)

            # Optional fourth candidate for photographed reports whose page
            # geometry is perspective-skewed.
            perspective_image = self._perspective_correct_for_ocr(image_array)
            if perspective_image is not None:
                try:
                    perspective_results = reader.readtext(
                        perspective_image, detail=1, paragraph=False
                    )
                    perspective_text, perspective_confidence, perspective_blocks = (
                        self._merge_ocr_results(perspective_results)
                    )
                    if perspective_text.strip():
                        candidates.append({
                            "pass_name": "perspective_corrected",
                            "raw_text": perspective_text,
                            "confidence": perspective_confidence,
                            "blocks": perspective_blocks,
                        })
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Perspective-corrected OCR pass failed: %s", exc)

        # Diagnostic-only per-pass trace. This does not affect candidate
        # generation, scoring, or selection; it simply exposes the evidence
        # that was already computed so real-image OCR failures can be compared
        # pass by pass.
        pass_diagnostics = []
        for candidate in candidates:
            candidate_text = str(candidate["raw_text"])
            candidate_confidence = float(candidate["confidence"])
            candidate_blocks = candidate.get("blocks") or []
            quality_score, _ = self._ocr_quality_score(
                candidate_text,
                candidate_confidence,
                candidate_blocks,
            )
            header = self._detect_table_header(candidate_blocks)
            pass_diagnostics.append(
                {
                    "pass_name": str(candidate["pass_name"]),
                    "confidence": candidate_confidence,
                    "quality_score": quality_score,
                    "header_found": header is not None,
                    "raw_text": candidate_text,
                    "block_count": len(candidate_blocks),
                }
            )

        best = max(
            candidates,
            key=lambda candidate: self._ocr_quality_score(
                str(candidate["raw_text"]),
                float(candidate["confidence"]),
                candidate.get("blocks") or [],
            ),
        )

        return {
            "success": True,
            "raw_text": str(best["raw_text"]),
            "confidence": float(best["confidence"]),
            "blocks": best.get("blocks") or [],
            "selected_pass": str(best["pass_name"]),
            "available_passes": [
                str(candidate["pass_name"]) for candidate in candidates
            ],
            "pass_diagnostics": pass_diagnostics,
        }

    # -------------------------------------------------------------------------
    # Structured extraction
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_number(raw: str) -> Union[int, float]:
        value = float(raw)
        if value.is_integer():
            return int(value)
        return value

    @staticmethod
    def _parse_range_bounds(reference_range: str) -> Optional[Tuple[float, float]]:
        parts = reference_range.split("-")
        if len(parts) != 2:
            return None
        try:
            lower = float(parts[0])
            upper = float(parts[1])
        except ValueError:
            return None
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper

    @staticmethod
    def _numbers_outside_parentheses(text: str) -> List[re.Match[str]]:
        """Return numeric tokens that are not part of parenthesized test names."""
        matches: List[re.Match[str]] = []
        depth = 0
        last_end = 0

        for match in _NUMBER_PATTERN.finditer(text):
            for char in text[last_end : match.start()]:
                if char == "(":
                    depth += 1
                elif char == ")" and depth:
                    depth -= 1

            if depth == 0:
                matches.append(match)

            last_end = match.end()

        return matches

    @classmethod
    def _has_reasonable_test_name_structure(
        cls,
        name: str,
        *,
        unit: str = "",
        reference_range: str = "",
        status: str = "Unknown",
    ) -> bool:
        """
        Conservatively validate a candidate laboratory test name.

        This is intentionally NOT a medical-test dictionary.  It uses only
        generic OCR/layout signals so that new report types remain supported.

        The important distinction is between legitimate short/abbreviated
        labels (``Hb``, ``TSH``, ``eGFR``, ``Na``) and isolated OCR fragments
        such as ``MLl``, ``RDt`` or ``tuarabed``.  A suspicious name is only
        rejected when the surrounding row also lacks useful laboratory
        structure.  A valid unit, reference range, or explicit status is
        therefore strong contextual evidence and prevents over-rejection.
        """
        cleaned = re.sub(r"\s+", " ", name).strip()
        if len(cleaned) < _MIN_TEST_NAME_LENGTH or not re.search(r"[A-Za-z]", cleaned):
            return False

        # A name containing multiple words is generally more structurally
        # plausible than an isolated OCR fragment.  Preserve punctuation used
        # by common analyte labels such as Anti-TPO, SGOT/AST and A/G Ratio.
        tokens = _NAME_TOKEN_PATTERN.findall(cleaned)
        if not tokens:
            return False

        # Detect classic OCR capitalization corruption without requiring a
        # list of known analytes. Examples: MLl and RDt. Legitimate labels such
        # as Hb, Na, TSH, eGFR and HbA1c do not match this pattern.
        for token in tokens:
            if not token.isalpha():
                continue

            # Accept the common generic shapes used by printed analyte labels:
            # ALL CAPS abbreviations (TSH), normal Title Case (Hemoglobin),
            # lowercase words when there is other row context, and lowercase-
            # prefix/capital-suffix forms such as eGFR or pH.  Other internal
            # case transitions (for example MLl or RDt) are a strong OCR-artifact
            # signal and are rejected without knowing the intended test name.
            if token.isupper() or _TITLE_CASE_TOKEN_PATTERN.fullmatch(token):
                continue
            if token.islower():
                continue
            if _LOWER_PREFIX_CAPS_TOKEN_PATTERN.fullmatch(token):
                continue
            return False

        has_structured_context = bool(
            unit.strip()
            or reference_range.strip()
            or status.strip().lower() != "unknown"
        )

        # A completely lowercase, single-token, multi-letter string with no
        # supporting laboratory structure is a common shape for OCR fragments
        # copied from surrounding prose/header text.  Do not reject lowercase
        # names when the row has a unit/range/status, because some laboratories
        # use non-standard capitalization and we must remain report-agnostic.
        if (
            len(tokens) == 1
            and tokens[0].isalpha()
            and tokens[0].islower()
            and len(tokens[0]) >= 5
            and not has_structured_context
        ):
            return False

        # Reject names made almost entirely from repeated characters. This is
        # generic OCR/table-border protection and does not rely on test names.
        compact = re.sub(r"[^A-Za-z0-9]", "", cleaned)
        if len(compact) >= 4 and len(set(compact.lower())) <= 2:
            return False

        return True

    @classmethod
    def _parse_lab_line(
        cls,
        line: str,
        diagnostic: Optional[Dict[str, Any]] = None,
        *,
        sex: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single OCR text line into a structured lab-test record.

        Returns None when the line does not look like a test-result line
        (e.g. patient/header/footer metadata with no numeric result).
        Units, reference ranges, and status are only ever taken from text
        OCR actually produced on that line - never invented or looked up
        from outside knowledge.

        Reference ranges/limits are recognized in three forms:
            - Explicit dash-separated, e.g. "12-16", "12 - 16", "12–16",
              "12 − 16" (see _EXPLICIT_RANGE_PATTERN for the accepted
              dash characters).
            - One-sided limits, e.g. "<34", ">90", "≤34", or "≥90".
            - OCR-corrupted whitespace-only-separated, e.g. "12 16",
              where the dash was dropped. This is only trusted when
              exactly two numbers remain after the result value on the
              line; three or more leftover numbers means the line's
              structure could not be reliably determined, and this
              method deliberately falls back to an empty reference range
              and "Unknown" status rather than guessing which numbers
              form the range.

        An explicit LOW/HIGH/NORMAL token (any letter case) found after
        the result is extracted as the status. Both the reference range
        and the status are always excluded from the returned "unit"
        field.
        """
        def reject(reason: str) -> None:
            if diagnostic is not None:
                diagnostic.update({"accepted": False, "rejection_reason": reason})

        working = line.strip()
        if not working or not re.search(r"\d", working):
            reject("Line is empty or contains no numeric token.")
            return None

        if _METADATA_PATTERN.search(working):
            reject("Line matches administrative metadata.")
            return None

        if _LOCATION_LINE_PATTERN.search(working):
            reject("Line matches a location/address pattern.")
            return None

        ambiguous_structure = False
        explicit_matches = list(_EXPLICIT_RANGE_PATTERN.finditer(working))

        if explicit_matches:
            # Explicit dash-separated range: normally reliable, but the
            # rest of the row is still validated so OCR-merged rows are not
            # mistaken for valid laboratory results.
            range_match = explicit_matches[-1]  # keep the last match on the line
            reference_range = f"{range_match.group(1)}-{range_match.group(2)}"
            working_without_range = (
                working[: range_match.start()] + " " + working[range_match.end() :]
            )

            candidate_values = cls._numbers_outside_parentheses(
                working_without_range[: range_match.start()]
            )
            if not candidate_values:
                reject("No numeric result appears before the explicit reference range.")
                return None

            # A normal row has exactly one result value before its range.
            # More than one outside-parentheses number usually means OCR
            # merged adjacent rows; do not guess which value is correct.
            if len(candidate_values) > 1:
                ambiguous_structure = True

            value_match = candidate_values[0]
            name_source = working_without_range[: value_match.start()]
            remainder = working_without_range[value_match.end() :]
        else:
            # No explicit dash range. The result value is the first standalone
            # number on the line. After it, prefer an explicitly printed
            # one-sided reference limit such as "<34" or ">90" before trying
            # the more fragile whitespace-only range recovery.
            candidate_values = cls._numbers_outside_parentheses(working)
            if not candidate_values:
                reject("Line contains no standalone numeric result token.")
                return None

            value_match = candidate_values[0]
            name_source = working[: value_match.start()]
            remainder = working[value_match.end() :]

            reference_range = ""
            limit_matches = list(_REFERENCE_LIMIT_PATTERN.finditer(remainder))
            if limit_matches:
                # A single one-sided limit is reliable because the comparator
                # is explicitly present in the OCR text. If multiple limits
                # occur, keep the line conservative rather than guessing which
                # one belongs to the current result.
                if len(limit_matches) == 1:
                    limit_match = limit_matches[0]
                    reference_range = f"{limit_match.group(1)}{limit_match.group(2)}"
                    remainder = (
                        remainder[: limit_match.start()]
                        + " "
                        + remainder[limit_match.end() :]
                    )
                else:
                    ambiguous_structure = True
            else:
                nums = cls._numbers_outside_parentheses(remainder)
                if len(nums) == 2:
                    gap = remainder[nums[0].end() : nums[1].start()].strip()
                    if gap == "" or gap in _DASH_CHAR_SET:
                        reference_range = f"{nums[0].group()}-{nums[1].group()}"
                        remainder = (
                            remainder[: nums[0].start()]
                            + " "
                            + remainder[nums[1].end() :]
                        )
                    # else: the two numbers aren't a clean pair (something
                    # other than whitespace/a dash) - leave them as
                    # unclassified text rather than guessing.
                elif len(nums) >= 3:
                    # More candidate numbers than a single range can explain
                    # (e.g. corrupted OCR merging two unrelated number
                    # groups). Do not guess which pair, if any, is real.
                    ambiguous_structure = True

        name = _LIST_PREFIX_PATTERN.sub("", name_source).strip(_TRIM_CHARS)

        # Some computer-printed/OCR rows place the measurement unit before
        # the numeric result, e.g. "Indirect Bilirubin mg/dL 0.2 ...".
        # Keep the analyte name clean and preserve the OCR-provided unit.
        unit_from_name = ""
        unit_match_in_name = list(_COMMON_UNIT_PATTERN.finditer(name))
        if unit_match_in_name:
            unit_match = unit_match_in_name[-1]
            # Only move a unit when it is at the end of the name portion;
            # otherwise a unit-like token may legitimately be part of an
            # analyte label.
            if not name[unit_match.end():].strip():
                unit_from_name = unit_match.group()
                name = name[:unit_match.start()].strip(_TRIM_CHARS)

        if len(name) < _MIN_TEST_NAME_LENGTH or not re.search(r"[A-Za-z]", name):
            reject("Text before the result is not a plausible alphabetic test name.")
            return None

        try:
            value = cls._to_number(value_match.group())
        except ValueError:
            reject("Numeric result could not be converted to a number.")
            return None

        status_matches = list(_STATUS_PATTERN.finditer(remainder))
        distinct_status_words = {match.group(1).lower() for match in status_matches}

        unit = _STATUS_PATTERN.sub("", remainder)
        # Assay methods such as "Diazo", "Calculated", "HPLC", and "ISE"
        # are useful OCR context but are not measurement units. Remove them
        # from the displayed unit field; never use them to invent a unit.
        unit = _ASSAY_METHOD_PATTERN.sub("", unit)

        # OCR may leave a second numeric token after a valid unit. For example:
        #   "PPBS 158 mg/dL 140"
        # The 140 is not safely attributable to the unit, so remove only the
        # trailing numeric fragment rather than guessing its meaning.
        unit = _TRAILING_NUMERIC_REFERENCE_PATTERN.sub("", unit)

        # eGFR reports commonly contain a threshold/method after the unit:
        #   "mL/min/1.73m² > 90 CKD-EPI"
        # Keep the measurement unit and discard only the trailing threshold
        # and method text.
        if re.search(r"\begfr\b", name, re.IGNORECASE):
            unit = _EGFR_REFERENCE_TAIL_PATTERN.sub("", unit)

        unit = re.sub(r"\s+", " ", unit).strip(_TRIM_CHARS + " ")
        if unit_from_name:
            unit = " ".join(part for part in (unit_from_name, unit) if part).strip()

        if ambiguous_structure:
            # Never invent a range, status, or unit for a line whose
            # structure could not be reliably determined.
            reference_range = ""
            unit = ""
            status = "Unknown"
        elif len(distinct_status_words) == 1:
            status = next(iter(distinct_status_words)).capitalize()
        elif len(distinct_status_words) > 1:
            # Conflicting explicit status tokens on one line (e.g. both
            # "LOW" and "HIGH" present) - untrustworthy, don't pick one.
            status = "Unknown"
        elif reference_range:
            if reference_range[0] in "<>≤≥":
                try:
                    limit_value = float(reference_range[1:])
                except ValueError:
                    status = "Unknown"
                else:
                    comparator = reference_range[0]
                    if comparator in "<≤":
                        status = "High" if value > limit_value else "Normal"
                    else:
                        status = "Low" if value < limit_value else "Normal"
            else:
                bounds = cls._parse_range_bounds(reference_range)
                if bounds:
                    lower, upper = bounds
                    if value < lower:
                        status = "Low"
                    elif value > upper:
                        status = "High"
                    else:
                        status = "Normal"
                else:
                    status = "Unknown"
        else:
            status = "Unknown"

        reference_range, status, reference_metadata = cls._apply_general_reference_range(
            cls._normalize_test_name_for_general_range_lookup(name),
            value,
            cls._lookup_unit_with_safe_method_inference(name, unit),
            reference_range,
            status,
            sex=sex,
        )

        if not cls._has_reasonable_test_name_structure(
            name,
            unit=unit,
            reference_range=reference_range,
            status=status,
        ):
            reject("Test name failed generic OCR-structure validation.")
            return None

        record = {
            "test_name": name,
            "value": value,
            "unit": unit,
            "reference_range": reference_range,
            "status": status,
        }
        if reference_metadata:
            record.update(reference_metadata)
        if diagnostic is not None:
            diagnostic.update({"accepted": True, "record": record})
        return record

    @classmethod
    def _extract_tests(
        cls,
        raw_text: str,
        blocks: Optional[Sequence["_OCRBlock"]] = None,
        debug: Optional[Dict[str, Any]] = None,
        ocr_confidence: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Parse OCR output into deduplicated, structured test records.

        When bounding-box ``blocks`` are supplied and a laboratory-table
        header can be confidently located within them (see
        ``_detect_table_header``), rows are reconstructed by column
        position so a test name, result, flag, unit, and reference range
        can never bleed into one another, and content above the header
        (hospital/patient details) is structurally excluded regardless
        of its wording. When no table header is detected - including
        every case where ``blocks`` is not supplied at all - this falls
        back to the original per-line text parser, unchanged.
        """
        table_records: Optional[List[Dict[str, Any]]] = None
        sex = cls._extract_patient_sex(raw_text)
        if debug is not None:
            debug["patient_sex"] = sex
            debug["table"] = {
                "header_found": False,
                "candidate_header_rows": [],
                "selected_header": None,
                "reconstructed_rows": [],
            }
        if blocks:
            table_debug = debug.get("table") if debug is not None else None

            header = cls._detect_table_header(blocks, debug=table_debug)
            if header is not None:
                table_rows = cls._reconstruct_table_rows(
                    blocks, header, debug=table_debug
                )
                parsed: List[Optional[Dict[str, Any]]] = []
                for index, row in enumerate(table_rows):
                    candidate: Optional[Dict[str, Any]] = None
                    if debug is not None:
                        candidate = {
                            "mode": "table",
                            "candidate_text": " | ".join(
                                f"{key}={value}" for key, value in row.items() if value
                            ),
                            "columns": row,
                        }
                    record = cls._parse_table_row(row, diagnostic=candidate, sex=sex)
                    parsed.append(record)
                    if candidate is not None and not candidate.get("accepted"):
                        debug.setdefault("rejected_candidates", []).append(candidate)
                parsed = [record for record in parsed if record is not None]
                if parsed:
                    table_records = parsed

        warnings: List[str] = []
        records: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, Any, str, str]] = set()
        duplicates_found = False

        if table_records is not None:
            if debug is not None:
                debug["parser_mode"] = "table"
            pending_records = table_records
        else:
            low_confidence_without_table = bool(
                blocks
                and ocr_confidence is not None
                and ocr_confidence < _OCR_RETRY_CONFIDENCE_THRESHOLD
            )

            if low_confidence_without_table:
                if debug is not None:
                    debug["parser_mode"] = "flat_text_blocked_low_confidence"
                    debug["flat_text_fallback_blocked"] = True
                    debug["flat_text_fallback_reason"] = (
                        "No table header was detected and overall OCR confidence "
                        "is below the minimum required for safe flat-text extraction."
                    )
                warnings = [
                    "OCR confidence is too low to safely extract structured laboratory results; manual verification is required."
                ]
                if debug is not None:
                    debug["final_structured_tests"] = []
                return [], warnings

            if debug is not None:
                debug["parser_mode"] = "flat_text"
            pending_records = []
            row_confidences: List[Optional[float]] = []
            if blocks:
                row_confidences = [
                    sum(block.confidence for block in row) / len(row)
                    if row
                    else None
                    for row in cls._group_blocks_into_rows(blocks)
                ]

            for line_index, raw_line in enumerate(raw_text.splitlines()):
                line = raw_line.strip()
                if not line:
                    continue
                candidate: Optional[Dict[str, Any]] = None
                if debug is not None:
                    candidate = {"mode": "flat_text", "candidate_text": line}

                row_confidence = (
                    row_confidences[line_index]
                    if line_index < len(row_confidences)
                    else None
                )
                if (
                    row_confidence is not None
                    and row_confidence < _MIN_FLAT_TEXT_CANDIDATE_CONFIDENCE
                ):
                    if candidate is not None:
                        candidate.update(
                            {
                                "accepted": False,
                                "ocr_confidence": row_confidence,
                                "rejection_reason": (
                                    "OCR row confidence is below the minimum "
                                    "required for flat-text extraction."
                                ),
                            }
                        )
                        debug.setdefault("rejected_candidates", []).append(candidate)
                    continue

                record = cls._parse_lab_line(line, diagnostic=candidate, sex=sex)
                if candidate is not None and not candidate.get("accepted"):
                    debug.setdefault("rejected_candidates", []).append(candidate)
                if record is not None:
                    pending_records.append(record)

        for record in pending_records:
            key = (
                record["test_name"].strip().lower(),
                record["value"],
                record["unit"].strip().lower(),
                record["reference_range"],
            )
            if key in seen:
                duplicates_found = True
                continue

            seen.add(key)
            records.append(record)

        if duplicates_found:
            warnings.append("Duplicate test results were detected and consolidated.")

        if not records:
            warnings.append(
                "No structured lab test values could be identified in the report."
            )

        if debug is not None:
            debug["final_structured_tests"] = records

        return records, warnings

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def _analyze(self, image_bytes: object, *, include_debug: bool) -> Dict[str, Any]:
        """
        Run OCR and structured extraction on a computer-printed medical
        report image.

        This is an extraction/classification layer only. It never
        diagnoses conditions, recommends medicines or dosage, or infers
        treatment, and it never fabricates a reference range, unit, or
        confidence value that OCR did not actually produce.

        Args:
            image_bytes: Raw image bytes, a file-like object exposing
                ``getvalue()``, a PIL Image, or a numpy array.

        Returns:
            A dict with keys: success, confidence, raw_text, tests, and
            warnings. The private diagnostic path additionally includes a
            read-only ``debug`` payload.
        """
        try:
            try:
                image_array = self._image_to_array(image_bytes)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unreadable or corrupted report image.")
                return {
                    "success": False,
                    "confidence": 0.0,
                    "raw_text": "",
                    "tests": [],
                    "warnings": [f"Unable to decode uploaded image: {exc}"],
                }

            ocr_result = self._run_ocr(image_array)

            if not ocr_result.get("success"):
                return {
                    "success": False,
                    "confidence": 0.0,
                    "raw_text": "",
                    "tests": [],
                    "warnings": list(ocr_result.get("warnings", [])),
                }

            raw_text = str(ocr_result.get("raw_text", ""))
            confidence = float(ocr_result.get("confidence") or 0.0)
            blocks = ocr_result.get("blocks") or []

            debug: Optional[Dict[str, Any]] = None
            if include_debug:
                debug = {
                    "ocr_detections": [
                        self._block_debug_payload(block) for block in blocks
                    ],
                    "overall_confidence": confidence,
                    "selected_ocr_pass": ocr_result.get("selected_pass", "original"),
                    "ocr_passes_considered": list(
                        ocr_result.get("available_passes", ["original"])
                    ),
                    "ocr_pass_diagnostics": list(
                        ocr_result.get("pass_diagnostics", [])
                    ),
                    "rejected_candidates": [],
                }

            tests, warnings = self._extract_tests(
                raw_text,
                blocks=blocks,
                debug=debug,
                ocr_confidence=confidence,
            )

            logger.info(
                "Report analysis: lines=%d, tests=%d, confidence=%.4f",
                len([entry for entry in raw_text.splitlines() if entry.strip()]),
                len(tests),
                confidence,
            )

            result = {
                "success": True,
                "confidence": confidence,
                "raw_text": raw_text,
                "tests": tests,
                "warnings": warnings,
            }
            if debug is not None:
                result["debug"] = debug
            return result

        except Exception as exc:  # noqa: BLE001
            logger.exception("Report analysis failed.")
            return {
                "success": False,
                "confidence": 0.0,
                "raw_text": "",
                "tests": [],
                "warnings": [f"Report analysis failed: {exc}"],
            }

    def analyze(self, image_bytes: object) -> Dict[str, Any]:
        """Analyze a report using the stable production result contract."""
        return self._analyze(image_bytes, include_debug=False)

    def analyze_debug(self, image_bytes: object) -> Dict[str, Any]:
        """
        Analyze a report and include non-authoritative diagnostic trace data.

        This method uses the same OCR and parsing decisions as ``analyze``.
        Its additional ``debug`` payload exists only to inspect those
        decisions; it is never consumed by the normal Streamlit report UI or
        by the AI explanation service.
        """
        return self._analyze(image_bytes, include_debug=True)


@lru_cache(maxsize=1)
def get_report_analyzer() -> ReportAnalyzer:
    """Return a cached ReportAnalyzer singleton."""
    return ReportAnalyzer()