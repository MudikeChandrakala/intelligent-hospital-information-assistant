"""
modules/prescription_analyzer.py
=============================================================================
Prescription OCR and structured medicine extraction utilities for the
Intelligent Hospital Information Assistant.

This module owns the non-UI business logic for prescription analysis:
    - OCR with EasyOCR
    - OCR text cleanup
    - Structured medicine extraction
    - Medicine-name correction using the REAL medicine knowledge base
    - Separation of general prescription advice from medicine directions
    - Safe, non-crashing analysis result assembly

Important:
    This module does NOT call Gemini, RAG, Retriever, or any medicine
    database service/API. It reads the local medicine JSON knowledge base
    only for OCR-name validation/correction.

Public API
-----------
    PrescriptionAnalyzer
    get_prescription_analyzer()
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None  # type: ignore[assignment]


logger = logging.getLogger("hospital_assistant.prescription_analyzer")


# =============================================================================
# CONSTANTS
# =============================================================================

_PRESERVE_DUPLICATES: set[str] = {"SOS", "OD", "BD", "TDS", "HS"}

_DOSAGE_PREFIXES: set[str] = {
    "tab", "tablet", "cap", "capsule", "syp", "syrup",
    "inj", "injection", "cream", "ointment", "drops", "drop",
    "gel", "solution", "sus", "susp", "odt",
}

_MEDICINE_START_PREFIXES: tuple[str, ...] = (
    r"tab\.?", r"tablet", r"cap\.?", r"capsule", r"syr\.?", r"syrup",
    r"inj\.?", r"injection", r"cream", r"ointment", r"drops?", r"gel",
    r"solution", r"sus\.?", r"susp\.?", r"odt",
)

_MEDICINE_NAME_STOPWORDS: set[str] = {
    "before", "after", "morning", "afternoon", "night", "bedtime",
    "food", "with", "empty", "stomach", "take", "drink", "apply",
    "use", "continue", "avoid", "od", "bd", "tds", "qid", "hs", "sos",
    "1-0-1", "1-1-1", "0-1-0", "0-0-1", "1-0-0",
    "for", "days", "day", "weeks", "week", "months", "month",
    "rest", "please", "advice", "advised", "review", "follow",
    "note", "regards", "signature", "reg", "dr",
}

_FREQUENCY_PATTERNS: tuple[str, ...] = (
    r"\b1\s*[-/]\s*0\s*[-/]\s*1\b",
    r"\b1\s*[-/]\s*1\s*[-/]\s*1\b",
    r"\b0\s*[-/]\s*1\s*[-/]\s*0\b",
    r"\b0\s*[-/]\s*0\s*[-/]\s*1\b",
    r"\b1\s*[-/]\s*0\s*[-/]\s*0\b",
    r"\bSOS\b", r"\bOD\b", r"\bBD\b", r"\bTDS\b", r"\bQID\b", r"\bHS\b",
    r"\bonce\s+daily\b", r"\btwice\s+daily\b", r"\bthrice\s+daily\b",
)

_FREQUENCY_NORMALIZATION: dict[str, str] = {
    r"\b1\s*[-/]\s*0\s*[-/]\s*1\b": "1-0-1",
    r"\b1\s*[-/]\s*1\s*[-/]\s*1\b": "1-1-1",
    r"\b0\s*[-/]\s*1\s*[-/]\s*0\b": "0-1-0",
    r"\b0\s*[-/]\s*0\s*[-/]\s*1\b": "0-0-1",
    r"\b1\s*[-/]\s*0\s*[-/]\s*0\b": "1-0-0",
    r"\bSOS\b": "SOS",
    r"\bOD\b": "OD",
    r"\bBD\b": "BD",
    r"\bTDS\b": "TDS",
    r"\bQID\b": "QID",
    r"\bHS\b": "HS",
    r"\bonce\s+daily\b": "Once Daily",
    r"\btwice\s+daily\b": "Twice Daily",
    r"\bthrice\s+daily\b": "Thrice Daily",
}

# OCR-safe numeric frequency detection.
#
# Handwritten prescriptions are commonly recognized as variations such as:
#   1-0-1, 1 / 0 / 1, 1 - 0 - 1
# and, depending on OCR, occasionally:
#   I-0-I, 1-O-1, 1-0-I, 1 0 1
#
# The I/l -> 1 and O -> 0 substitutions are intentionally restricted to a
# strict three-position frequency shape. This prevents medicine names,
# strengths, durations, and arbitrary numbers from being converted into a
# guessed schedule.
_FREQUENCY_OCR_DIGIT_MAP: dict[str, str] = {
    "0": "0",
    "1": "1",
    "I": "1",
    "L": "1",
    "O": "0",
}

_FREQUENCY_SEPARATED_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([01IloOL])"
    r"(?:\s*[-/]\s*|\s+)"
    r"([01IloOL])"
    r"(?:\s*[-/]\s*|\s+)"
    r"([01IloOL])"
    r"(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)

_FREQUENCY_SPACED_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([01IloOL])\\s+"
    r"([01IloOL])\\s+"
    r"([01IloOL])"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_TIMING_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bbefore\s+food\b", "Before Food"),
    (r"\bafter\s+food\b", "After Food"),
    (r"\bbefore\s+meals?\b", "Before Food"),
    (r"\bafter\s+meals?\b", "After Food"),
    (r"\bbefore\s+breakfast\b", "Before Breakfast"),
    (r"\bafter\s+breakfast\b", "After Breakfast"),
    (r"\bbefore\s+dinner\b", "Before Dinner"),
    (r"\bafter\s+dinner\b", "After Dinner"),
    (r"\bmorning\b", "Morning"),
    (r"\bafternoon\b", "Afternoon"),
    (r"\bnight\b", "Night"),
    (r"\bbedtime\b", "Night"),
    (r"\bat\s+night\b", "Night"),
    (r"\bempty\s+stomach\b", "Empty Stomach"),
    (r"\bwith\s+food\b", "With Food"),
)

_DURATION_PATTERN = re.compile(
    r"\b(\d+)\s*(days?|weeks?|months?)\b", re.IGNORECASE
)

_MEDICINE_NAME_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9\s\-/&().']*?)\s+"
    r"(?P<amount>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mg|mcg|g|ml|iu|units?)?\b",
    re.IGNORECASE,
)

_MEDICINE_START_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(_MEDICINE_START_PREFIXES) +
    r")\b[\s\.:,-]*(?P<remainder>.+)$",
    re.IGNORECASE,
)

_MEDICINE_LIST_PREFIX_PATTERN = re.compile(
    r"^\s*(?:(?:\d+\s*(?:[.)]\s*|-\s*)?)|[•-]\s*)"
)

_UNIT_SUFFIXED_MEDICINE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9\s\-/&().']*?)\s+"
    r"(?P<amount>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mg|mcg|g|ml|iu|units?)\b",
    re.IGNORECASE,
)

_DURATION_ONLY_LINE_PATTERN = re.compile(
    r"^\s*(?:for\s+)?\d+\s*(?:days?|weeks?|months?)\s*$",
    re.IGNORECASE,
)

_DOCTOR_LINE_PATTERN = re.compile(r"^\s*dr\.?\s+", re.IGNORECASE)

_DOCTOR_SEGMENT_PATTERN = re.compile(
    r"\s*\bdr\.?[:.,-]?\s+[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3}\s*$",
    re.IGNORECASE,
)

# General-advice markers. The colon is intentionally optional because
# clean_text() removes trailing punctuation such as ":".
_GENERAL_ADVICE_MARKER_PATTERN = re.compile(
    r"^\s*(?:adv|advice|general\s+advice|patient\s+advice|"
    r"general\s+instructions|patient\s+instructions|"
    r"advised|instructions|note)\s*:?\s*$",
    re.IGNORECASE,
)

_REGISTRATION_LINE_PATTERN = re.compile(
    r"^\s*(?:reg(?:istration)?\.?\s*(?:no|number)?|"
    r"registration\s*(?:no|number)?)\s*[:#.\-]?\s*\w+",
    re.IGNORECASE,
)

_DATE_LINE_PATTERN = re.compile(
    r"^\s*(?:date|dt)\s*[:#.\-]?\s*"
    r"\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{2,4}\s*$",
    re.IGNORECASE,
)

_INSTRUCTION_HINT_PATTERN = re.compile(
    r"\b(?:take|apply|use|continue|avoid|swallow|chew|dissolve|"
    r"with water|as directed|per day|daily|dose|tablet|capsule|"
    r"syrup|injection|before meal|after meal|before food|after food|"
    r"morning|afternoon|night|bedtime|empty stomach|with food)\b",
    re.IGNORECASE,
)

# Conservative correction threshold. A name is changed only when the
# similarity against a real KB name/alias is strong.
_MEDICINE_NAME_SIMILARITY_CUTOFF: float = 0.88
_MEDICINE_NAME_MIN_LENGTH_FOR_CORRECTION: int = 4

# Local KB filename patterns. The loader supports the project's main file
# and split/renamed copies without hardcoding medicine names.
_MEDICINE_KB_FILENAME_PATTERN = re.compile(
    r"medicine_database.*\.json$", re.IGNORECASE
)


# =============================================================================
# INTERNAL DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class _OCRBlock:
    text: str
    confidence: float
    left: float
    top: float
    height: float
    bbox: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class _MedicineEntry:
    name: str = ""
    strength: str = ""
    frequency: str = ""
    timing: str = ""
    duration: str = ""
    instructions: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "strength": self.strength,
            "frequency": self.frequency,
            "timing": self.timing,
            "duration": self.duration,
            "instructions": self.instructions,
        }


# =============================================================================
# PRESCRIPTION ANALYZER
# =============================================================================


class PrescriptionAnalyzer:
    """EasyOCR-backed prescription analyzer with local-KB validation."""

    _reader_lock: ClassVar[Lock] = Lock()
    _reader: ClassVar[Optional[Any]] = None
    _reader_error: ClassVar[Optional[str]] = None

    _medicine_kb_lock: ClassVar[Lock] = Lock()
    _medicine_names: ClassVar[Optional[Tuple[str, ...]]] = None
    _medicine_kb_error: ClassVar[Optional[str]] = None

    _MIN_SUFFICIENT_CONFIDENCE: ClassVar[float] = 55.0
    _UPSCALE_MIN_DIMENSION_PX: ClassVar[int] = 900
    _UPSCALE_FACTOR: ClassVar[float] = 2.0
    _FREQUENCY_REGION_SCALE: ClassVar[float] = 4.0
    _FREQUENCY_REGION_UPSCALE_FACTOR: ClassVar[float] = 3.0
    _FREQUENCY_ALLOWLIST: ClassVar[str] = "0123456789-/()"

    def __init__(self) -> None:
        self.reader = self._get_reader()
        self._ensure_medicine_kb_loaded()

    # -------------------------------------------------------------------------
    # EasyOCR
    # -------------------------------------------------------------------------

    @classmethod
    def _get_reader(cls) -> Optional[Any]:
        if easyocr is None:
            if cls._reader_error is None:
                cls._reader_error = "EasyOCR is not installed."
                logger.error(cls._reader_error)
            return None

        if cls._reader is not None:
            return cls._reader

        with cls._reader_lock:
            if cls._reader is not None:
                return cls._reader
            if cls._reader_error is not None:
                return None

            try:
                cls._reader = easyocr.Reader(["en"], gpu=False)
                logger.info("EasyOCR reader initialized successfully.")
            except Exception as exc:  # noqa: BLE001
                cls._reader_error = f"Failed to initialize EasyOCR: {exc}"
                logger.exception(cls._reader_error)
                cls._reader = None

        return cls._reader

    @staticmethod
    def _image_to_array(image: object) -> np.ndarray:
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

    @classmethod
    def _preprocess_for_ocr(cls, image_array: np.ndarray) -> np.ndarray:
        """Apply mild OCR preprocessing without mutating the original image."""
        try:
            pil_image = Image.fromarray(image_array.copy()).convert("L")
            width, height = pil_image.size

            if min(width, height) < cls._UPSCALE_MIN_DIMENSION_PX:
                new_size = (
                    int(width * cls._UPSCALE_FACTOR),
                    int(height * cls._UPSCALE_FACTOR),
                )
                pil_image = pil_image.resize(new_size, Image.LANCZOS)

            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
            pil_image = ImageEnhance.Contrast(pil_image).enhance(1.5)
            pil_image = ImageEnhance.Sharpness(pil_image).enhance(1.5)

            return np.array(pil_image.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR preprocessing failed; using original image: %s", exc)
            return image_array

    @staticmethod
    def _normalize_duplicate_words(text: str) -> str:
        tokens = text.split()
        collapsed: List[str] = []
        previous_normalized = ""

        for token in tokens:
            normalized = re.sub(r"[^A-Za-z0-9\-/.]+", "", token).strip()
            if not normalized:
                continue

            normalized_upper = normalized.upper()
            if (
                collapsed
                and normalized.lower() == previous_normalized.lower()
                and normalized_upper not in _PRESERVE_DUPLICATES
                and not re.fullmatch(r"\d+[-/]\d+[-/]\d+", normalized)
            ):
                continue

            collapsed.append(token)
            previous_normalized = normalized

        return " ".join(collapsed)

    @staticmethod
    def _normalize_line(line: str) -> str:
        cleaned = line.replace("\u00a0", " ")
        cleaned = re.sub(r"[•·▪●]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = PrescriptionAnalyzer._normalize_duplicate_words(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned).strip()
        return cleaned

    @staticmethod
    def _strip_dosage_prefix(name: str) -> str:
        tokens = [token for token in re.split(r"\s+", name.strip()) if token]
        while tokens and tokens[0].lower().rstrip(".") in _DOSAGE_PREFIXES:
            tokens.pop(0)
        return " ".join(tokens).strip(" ,:-.")

    @staticmethod
    def _normalize_strength(amount: str, unit: Optional[str]) -> str:
        normalized_unit = (unit or "mg").strip().lower()

        if normalized_unit == "iu":
            normalized_unit = "IU"
        elif normalized_unit.endswith("s"):
            normalized_unit = normalized_unit[:-1]

        return f"{amount} {normalized_unit}".strip()

    @staticmethod
    def _normalize_medicine_line(line: str) -> str:
        normalized = _MEDICINE_LIST_PREFIX_PATTERN.sub("", line).strip()
        normalized = re.sub(
            r"^(?:toab|tal|tat)\b",
            "Tab",
            normalized,
            flags=re.IGNORECASE,
        )
        return re.sub(
            r"\[\s*o(?=\s*mg\b)",
            "10",
            normalized,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _merge_ocr_results(
        results: Sequence[Any],
    ) -> Tuple[str, float, List[_OCRBlock]]:
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

            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]

            blocks.append(
                _OCRBlock(
                    text=text,
                    confidence=confidence,
                    left=float(min(xs)),
                    top=float(min(ys)),
                    height=float(max(ys) - min(ys)) or 1.0,
                    bbox=tuple((float(point[0]), float(point[1])) for point in bbox),
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

        merged_text = "\n".join(
            line for line in merged_lines if line.strip()
        )
        average_confidence = (
            round((sum(confidences) / len(confidences)) * 100, 1)
            if confidences
            else 0.0
        )

        return merged_text, average_confidence, blocks

    def _run_ocr_pass(self, image_array: np.ndarray) -> Dict[str, Any]:
        try:
            results = self.reader.readtext(
                image_array,
                detail=1,
                paragraph=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("EasyOCR failure during prescription analysis.")
            return {"success": False, "error": f"EasyOCR failure: {exc}"}

        if not results:
            return {
                "success": False,
                "error": "No text detected in the uploaded image.",
            }

        raw_text, average_confidence, blocks = self._merge_ocr_results(results)

        if not raw_text.strip():
            return {
                "success": False,
                "error": "No text detected in the uploaded image.",
            }

        return {
            "success": True,
            "ocr_text": raw_text,
            "confidence": average_confidence,
            "lines": len([line for line in raw_text.splitlines() if line.strip()]),
            "characters": len(raw_text),
            "ocr_detections": [
                {
                    "text": block.text,
                    "confidence": round(block.confidence, 4),
                    "bbox": [list(point) for point in block.bbox],
                }
                for block in blocks
            ],
            "_ocr_blocks": blocks,
        }

    # -------------------------------------------------------------------------
    # Medicine KB
    # -------------------------------------------------------------------------

    @staticmethod
    def _candidate_project_roots() -> List[Path]:
        roots: List[Path] = []

        try:
            module_root = Path(__file__).resolve().parents[1]
            roots.append(module_root)
        except Exception:
            pass

        roots.append(Path.cwd())

        unique: List[Path] = []
        seen: set[str] = set()

        for root in roots:
            try:
                resolved = root.resolve()
            except Exception:
                resolved = root

            key = str(resolved).lower()
            if key not in seen:
                seen.add(key)
                unique.append(resolved)

        return unique

    @classmethod
    def _find_medicine_kb_files(cls) -> List[Path]:
        """
        Locate the project's real medicine KB without assuming one exact
        filename. This supports:
            knowledge_base/structured/medicine_database.json
            medicine_database_part1.json
            medicine_database(5).json
            other medicine_database*.json files
        """
        candidates: List[Path] = []
        seen: set[str] = set()

        for root in cls._candidate_project_roots():
            structured_dir = root / "knowledge_base" / "structured"

            search_roots = [structured_dir, root]

            for search_root in search_roots:
                if not search_root.exists() or not search_root.is_dir():
                    continue

                try:
                    paths = search_root.rglob("medicine_database*.json")
                except Exception:
                    continue

                for path in paths:
                    if not path.is_file():
                        continue

                    if not _MEDICINE_KB_FILENAME_PATTERN.match(path.name):
                        continue

                    key = str(path.resolve()).lower()
                    if key in seen:
                        continue

                    seen.add(key)
                    candidates.append(path)

        # Prefer the canonical structured directory before arbitrary copies.
        candidates.sort(
            key=lambda path: (
                0
                if "knowledge_base" in str(path).lower()
                and "structured" in str(path).lower()
                else 1,
                str(path).lower(),
            )
        )

        return candidates

    @staticmethod
    def _extract_medicine_records(payload: Any) -> List[dict]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if isinstance(payload, dict):
            medicines = payload.get("medicines")
            if isinstance(medicines, list):
                return [
                    item for item in medicines
                    if isinstance(item, dict)
                ]

        return []

    @staticmethod
    def _clean_kb_name(value: Any) -> str:
        if not isinstance(value, str):
            return ""

        value = re.sub(r"\s+", " ", value).strip()
        return value

    @classmethod
    def _load_medicine_names_from_kb(cls) -> Tuple[str, ...]:
        names: List[str] = []
        seen: set[str] = set()

        kb_files = cls._find_medicine_kb_files()

        if not kb_files:
            cls._medicine_kb_error = (
                "No medicine_database*.json file was found under the project."
            )
            logger.warning(cls._medicine_kb_error)
            return ()

        for kb_file in kb_files:
            try:
                # utf-8-sig handles the BOM found in some project JSON files.
                with kb_file.open("r", encoding="utf-8-sig") as handle:
                    payload = json.load(handle)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping unreadable medicine KB file %s: %s",
                    kb_file,
                    exc,
                )
                continue

            records = cls._extract_medicine_records(payload)

            for record in records:
                values: List[Any] = [
                    record.get("generic_name"),
                    record.get("brand_names"),
                    record.get("keywords"),
                ]

                for value in values:
                    if isinstance(value, list):
                        iterable = value
                    else:
                        iterable = [value]

                    for item in iterable:
                        name = cls._clean_kb_name(item)
                        if not name:
                            continue

                        key = re.sub(r"[^a-z0-9]+", "", name.lower())
                        if not key or key in seen:
                            continue

                        seen.add(key)
                        names.append(name)

        if names:
            cls._medicine_kb_error = None
            logger.info(
                "Medicine KB loaded successfully: %d medicine names/aliases "
                "from %d file(s).",
                len(names),
                len(kb_files),
            )
        else:
            cls._medicine_kb_error = (
                "Medicine KB files were found, but no usable medicine names "
                "or aliases were present."
            )
            logger.warning(cls._medicine_kb_error)

        return tuple(names)

    @classmethod
    def _ensure_medicine_kb_loaded(cls) -> None:
        if cls._medicine_names is not None:
            return

        with cls._medicine_kb_lock:
            if cls._medicine_names is not None:
                return

            cls._medicine_names = cls._load_medicine_names_from_kb()

    @classmethod
    def reload_medicine_kb(cls) -> int:
        """
        Clear and reload the local medicine KB.

        Useful after adding/replacing medicine_database*.json files during
        development. Returns the number of loaded names/aliases.
        """
        with cls._medicine_kb_lock:
            cls._medicine_names = cls._load_medicine_names_from_kb()

        return len(cls._medicine_names)

    @staticmethod
    def _normalized_name_for_match(value: str) -> str:
        value = value.lower().strip()
        value = value.replace("&", " and ")
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @classmethod
    def _correct_common_medicine_misspelling(cls, name: str) -> str:
        """
        Correct OCR medicine names only against the real local medicine KB.

        The function deliberately does not invent medicine names. If the
        similarity is not strong enough, the OCR name is returned unchanged.
        """
        candidate = name.strip()

        if len(candidate) < _MEDICINE_NAME_MIN_LENGTH_FOR_CORRECTION:
            return name

        cls._ensure_medicine_kb_loaded()
        known_names = cls._medicine_names or ()

        if not known_names:
            return name

        normalized_candidate = cls._normalized_name_for_match(candidate)

        # Exact normalized match first.
        for known in known_names:
            if normalized_candidate == cls._normalized_name_for_match(known):
                return known

        best_name = ""
        best_ratio = 0.0

        for known in known_names:
            normalized_known = cls._normalized_name_for_match(known)
            if len(normalized_known) < 4:
                continue

            ratio = difflib.SequenceMatcher(
                None,
                normalized_candidate,
                normalized_known,
            ).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_name = known

        if best_name and best_ratio >= _MEDICINE_NAME_SIMILARITY_CUTOFF:
            logger.info(
                "OCR medicine correction applied using local KB: %r -> %r "
                "(similarity=%.3f)",
                candidate,
                best_name,
                best_ratio,
            )
            return best_name

        logger.info(
            "No sufficiently strong medicine-KB match for OCR candidate %r "
            "(best_similarity=%.3f); preserving OCR name.",
            candidate,
            best_ratio,
        )
        return name

    # -------------------------------------------------------------------------
    # OCR sufficiency / public text extraction
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_medicine_anchor_block(block: _OCRBlock) -> bool:
        """Return whether an OCR block plausibly begins a medicine line."""
        normalized = re.sub(r"^[^a-z]+", "", block.text.lower())
        return normalized.startswith(
            ("tab", "toab", "tal", "tat", "tablet", "cap", "capsule", "syr", "syp", "inj")
        )

    @staticmethod
    def _median_block_height(blocks: Sequence[_OCRBlock]) -> float:
        """Return a robust, geometry-derived OCR block height."""
        heights = sorted(block.height for block in blocks if block.height > 0)
        if not heights:
            return 1.0
        return heights[len(heights) // 2]

    def _select_frequency_regions(
        self,
        blocks: Sequence[_OCRBlock],
        image_shape: tuple[int, ...],
    ) -> List[tuple[_OCRBlock, tuple[int, int, int, int]]]:
        """Select the row directly below each detected medicine heading.

        Regions are derived entirely from EasyOCR geometry. No fixed page
        coordinates or medicine-specific dosage assumptions are used.
        """
        if len(image_shape) < 2:
            return []

        image_height, image_width = image_shape[:2]
        anchors = sorted(
            (block for block in blocks if self._is_medicine_anchor_block(block)),
            key=lambda block: (block.top, block.left),
        )
        if not anchors:
            return []

        median_height = self._median_block_height(blocks)
        regions: List[tuple[_OCRBlock, tuple[int, int, int, int]]] = []

        for index, anchor in enumerate(anchors):
            anchor_bottom = anchor.top + anchor.height
            next_anchor_top = (
                anchors[index + 1].top if index + 1 < len(anchors) else float(image_height)
            )
            region_top = int(max(0, anchor_bottom + median_height * 0.15))
            region_bottom = int(
                min(
                    image_height,
                    next_anchor_top - median_height * 0.2,
                    anchor_bottom + median_height * self._FREQUENCY_REGION_SCALE,
                )
            )

            if region_bottom <= region_top:
                continue

            regions.append((anchor, (0, region_top, image_width, region_bottom)))

        return regions

    def _preprocess_frequency_region(self, region: np.ndarray) -> np.ndarray:
        """Enhance a small dosage row without changing full-page OCR input."""
        image = Image.fromarray(region).convert("L")
        image = ImageOps.autocontrast(image)
        image = image.resize(
            (
                max(1, int(image.width * self._FREQUENCY_REGION_UPSCALE_FACTOR)),
                max(1, int(image.height * self._FREQUENCY_REGION_UPSCALE_FACTOR)),
            ),
            Image.LANCZOS,
        )
        image = ImageEnhance.Contrast(image).enhance(2.0)
        image = image.filter(ImageFilter.MedianFilter(size=3))
        return np.array(image.convert("RGB"))

    def _read_frequency_region_details(self, region: np.ndarray) -> Dict[str, Any]:
        """Read a dosage crop and retain its unmodified OCR evidence."""
        try:
            results = self.reader.readtext(
                self._preprocess_frequency_region(region),
                detail=1,
                paragraph=False,
                allowlist=self._FREQUENCY_ALLOWLIST,
            )
        except Exception as exc:  # noqa: BLE001 - supplemental OCR must not fail analysis
            logger.warning("Targeted frequency OCR failed: %s", exc)
            return {
                "ocr_text": "",
                "ocr_detections": [],
                "recognized_frequency": None,
                "error": str(exc),
            }

        text, _, blocks = self._merge_ocr_results(results)
        frequency = self._detect_frequency(text)
        if frequency:
            logger.info("Targeted OCR found explicit frequency evidence: %s", frequency)
        return {
            "ocr_text": text,
            "ocr_detections": [
                {
                    "text": block.text,
                    "confidence": round(block.confidence, 4),
                    "bbox": [list(point) for point in block.bbox],
                }
                for block in blocks
            ],
            "recognized_frequency": frequency,
        }

    def _insert_targeted_frequencies(
        self,
        ocr_text: str,
        frequencies: Sequence[Optional[str]],
    ) -> str:
        """Place accepted OCR evidence after its matching medicine line.

        The supplemental value is inserted only when the corresponding
        medicine block does not already contain an explicit frequency.
        """
        if not frequencies:
            return ocr_text

        lines = ocr_text.splitlines()
        medicine_lines = [
            index for index, line in enumerate(lines) if self._is_medicine_start_line(line)
        ]
        insertions: List[tuple[int, str]] = []

        for medicine_position, frequency in enumerate(frequencies):
            if medicine_position >= len(medicine_lines):
                break
            if not frequency:
                continue

            line_index = medicine_lines[medicine_position]
            next_line_index = (
                medicine_lines[medicine_position + 1]
                if medicine_position + 1 < len(medicine_lines)
                else len(lines)
            )
            if any(self._detect_frequency(line) for line in lines[line_index + 1:next_line_index]):
                continue
            insertions.append((line_index + 1, frequency))

        for line_index, frequency in reversed(insertions):
            lines.insert(line_index, frequency)

        return "\n".join(lines)

    def _enhance_frequency_ocr(
        self,
        image_array: np.ndarray,
        pass_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Supplement full-page OCR with validated evidence from dosage crops."""
        original_text = str(pass_result.get("ocr_text", ""))
        pass_result["raw_ocr_text"] = original_text
        blocks = pass_result.get("_ocr_blocks", [])
        if not isinstance(blocks, list):
            pass_result.pop("_ocr_blocks", None)
            return pass_result

        frequencies: List[Optional[str]] = []
        medicine_lines = [
            line for line in original_text.splitlines() if self._is_medicine_start_line(line)
        ]
        diagnostics: List[Dict[str, Any]] = []

        for index, (anchor, (left, top, right, bottom)) in enumerate(
            self._select_frequency_regions(blocks, image_array.shape)
        ):
            medicine_name = anchor.text
            if index < len(medicine_lines):
                extracted_name, _ = self._split_name_strength(medicine_lines[index])
                medicine_name = extracted_name or medicine_name

            region = image_array[top:bottom, left:right]
            if region.size == 0:
                diagnostics.append(
                    {
                        "medicine_name": medicine_name,
                        "crop_bbox": [left, top, right, bottom],
                        "executed": False,
                        "ocr_detections": [],
                        "ocr_text": "",
                        "recognized_frequency": None,
                        "final_frequency": "",
                        "error": "Selected crop was empty.",
                    }
                )
                frequencies.append(None)
                continue

            details = self._read_frequency_region_details(region)
            recognized_frequency = details.get("recognized_frequency")
            frequencies.append(
                recognized_frequency if isinstance(recognized_frequency, str) else None
            )
            diagnostics.append(
                {
                    "medicine_name": medicine_name,
                    "crop_bbox": [left, top, right, bottom],
                    "executed": True,
                    "ocr_detections": list(details.get("ocr_detections", [])),
                    "ocr_text": str(details.get("ocr_text", "")),
                    "recognized_frequency": recognized_frequency,
                    "final_frequency": "",
                    "error": details.get("error"),
                }
            )

        pass_result.pop("_ocr_blocks", None)
        pass_result["targeted_frequency_diagnostics"] = diagnostics
        if not any(frequencies):
            return pass_result

        enhanced_text = self._insert_targeted_frequencies(
            original_text,
            frequencies,
        )
        pass_result["ocr_text"] = enhanced_text
        pass_result["lines"] = len([line for line in enhanced_text.splitlines() if line.strip()])
        pass_result["characters"] = len(enhanced_text)
        return pass_result

    def _count_medicines_in_ocr_text(self, ocr_text: str) -> int:
        try:
            cleaned = self.clean_text(ocr_text)
            if not cleaned:
                return 0
            return len(self.extract_medicines(cleaned))
        except Exception:  # noqa: BLE001
            return 0

    def _is_ocr_pass_sufficient(self, pass_result: Dict[str, Any]) -> bool:
        if not pass_result.get("success"):
            return False

        confidence = float(pass_result.get("confidence") or 0.0)

        if confidence < self._MIN_SUFFICIENT_CONFIDENCE:
            return False

        return self._count_medicines_in_ocr_text(
            str(pass_result.get("ocr_text", ""))
        ) > 0

    def extract_text(self, image: object) -> Dict[str, Any]:
        """
        Conservative two-pass OCR.

        Pass 1 uses the original image. Pass 2 uses mild preprocessing only
        when pass 1 is insufficient.
        """
        if self.reader is None:
            return {
                "success": False,
                "error": self._reader_error or "EasyOCR is unavailable.",
            }

        try:
            image_array = self._image_to_array(image)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unreadable or corrupted prescription image.")
            return {
                "success": False,
                "error": f"Unreadable image upload: {exc}",
            }

        logger.info("OCR Started (pass 1: original image)")
        pass1_result = self._run_ocr_pass(image_array)

        if self._is_ocr_pass_sufficient(pass1_result):
            logger.info(
                "OCR Finished using pass 1; confidence=%.1f%%.",
                float(pass1_result.get("confidence") or 0.0),
            )
            return self._enhance_frequency_ocr(image_array, pass1_result)

        logger.info(
            "Pass 1 insufficient; running pass 2 with mild preprocessing."
        )

        preprocessed_array = self._preprocess_for_ocr(image_array)
        pass2_result = self._run_ocr_pass(preprocessed_array)

        if not pass2_result.get("success"):
            return self._enhance_frequency_ocr(image_array, pass1_result)

        if not pass1_result.get("success"):
            return self._enhance_frequency_ocr(preprocessed_array, pass2_result)

        pass1_medicines = self._count_medicines_in_ocr_text(
            str(pass1_result.get("ocr_text", ""))
        )
        pass2_medicines = self._count_medicines_in_ocr_text(
            str(pass2_result.get("ocr_text", ""))
        )

        if pass2_medicines > pass1_medicines:
            logger.info(
                "OCR Finished using pass 2: medicines pass1=%d, pass2=%d.",
                pass1_medicines,
                pass2_medicines,
            )
            return self._enhance_frequency_ocr(preprocessed_array, pass2_result)

        logger.info(
            "OCR Finished using pass 1: medicines pass1=%d, pass2=%d.",
            pass1_medicines,
            pass2_medicines,
        )
        return self._enhance_frequency_ocr(image_array, pass1_result)

    # -------------------------------------------------------------------------
    # Text cleanup / field detection
    # -------------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned_lines: List[str] = []
        previous_line = ""

        for raw_line in text.splitlines():
            line = self._normalize_line(raw_line)

            if not line:
                continue

            if line == previous_line:
                continue

            if len(re.sub(r"[^A-Za-z0-9]", "", line)) < 2:
                continue

            cleaned_lines.append(line)
            previous_line = line

        return "\n".join(cleaned_lines).strip()

    def _detect_frequency(self, line: str) -> Optional[str]:
        """
        Detect an explicitly written prescription frequency without guessing.

        Numeric schedules are normalized to canonical ``A-B-C`` form. The
        detector tolerates common OCR separator/spacing variations and a
        small set of OCR character confusions (I/l -> 1, O -> 0), but only
        when the text has the strict three-position frequency shape.

        Examples:
            1-0-1       -> 1-0-1
            1 / 0 / 1   -> 1-0-1
            1 - 0 - 1   -> 1-0-1
            1 0 1       -> 1-0-1
            I-O-I       -> 1-0-1
            0 / 0 / I   -> 0-0-1

        No frequency is inferred from medicine names, strengths, durations,
        or database information.
        """
        if not line:
            return None

        # First handle numeric schedules with explicit separators.
        match = _FREQUENCY_SEPARATED_PATTERN.search(line)
        if match:
            normalized_digits = "".join(
                _FREQUENCY_OCR_DIGIT_MAP[group.upper()]
                for group in match.groups()
            )
            return "-".join(normalized_digits)

        # Then handle OCR output where separators disappeared but the three
        # dose positions remain separated by whitespace.
        match = _FREQUENCY_SPACED_PATTERN.search(line)
        if match:
            normalized_digits = "".join(
                _FREQUENCY_OCR_DIGIT_MAP[group.upper()]
                for group in match.groups()
            )
            return "-".join(normalized_digits)

        # Preserve all existing explicit textual frequency codes/phrases.
        upper_line = line.upper()
        for pattern in _FREQUENCY_PATTERNS:
            if re.search(pattern, upper_line, flags=re.IGNORECASE):
                return _FREQUENCY_NORMALIZATION.get(pattern, line.strip())

        return None

    def _detect_timings(self, line: str) -> List[str]:
        matches: List[tuple[int, str]] = []

        for pattern, label in _TIMING_PATTERNS:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                matches.append((match.start(), label))

        ordered: List[str] = []

        for _, label in sorted(matches, key=lambda item: item[0]):
            if label not in ordered:
                ordered.append(label)

        return ordered

    def _detect_timing(self, line: str) -> Optional[str]:
        timings = self._detect_timings(line)
        return timings[0] if timings else None

    def _detect_duration(self, line: str) -> Optional[str]:
        match = _DURATION_PATTERN.search(line)

        if match:
            amount, unit = match.groups()
            return f"{amount} {unit.capitalize()}"
        return None

    @staticmethod
    def _normalize_continuation_line(line: str) -> str:
        normalized = re.sub(
            r"\baften\b", "after", line, flags=re.IGNORECASE
        )
        normalized = re.sub(
            r"\bbefone\b", "before", normalized, flags=re.IGNORECASE
        )
        normalized = re.sub(
            r"\bagten\b", "after", normalized, flags=re.IGNORECASE
        )
        normalized = re.sub(
            r"\b(?:dods|doxs|doys)\b",
            "days",
            normalized,
            flags=re.IGNORECASE,
        )
        return normalized

    def _split_name_strength(self, line: str) -> Tuple[str, str]:
        normalized_line = self._normalize_medicine_line(line)
        stripped_line = self._strip_dosage_prefix(normalized_line)

        match = _MEDICINE_NAME_PATTERN.search(stripped_line)

        if not match:
            return stripped_line, ""

        name = self._strip_dosage_prefix(match.group("name"))
        name = self._correct_common_medicine_misspelling(name)

        amount = match.group("amount")
        unit = match.group("unit")
        strength = self._normalize_strength(amount, unit)

        return name, strength

    def _looks_like_medicine_without_prefix(self, line: str) -> bool:
        return bool(_UNIT_SUFFIXED_MEDICINE_PATTERN.match(line))

    def _is_medicine_start_line(self, line: str) -> bool:
        normalized_line = self._normalize_medicine_line(line)

        if not normalized_line:
            return False

        if _DOCTOR_LINE_PATTERN.match(normalized_line):
            return False

        if _DURATION_ONLY_LINE_PATTERN.match(normalized_line):
            return False

        has_dosage_prefix = bool(
            _MEDICINE_START_PATTERN.match(normalized_line)
        )

        if (
            not has_dosage_prefix
            and not self._looks_like_medicine_without_prefix(normalized_line)
        ):
            return False

        name, _ = self._split_name_strength(normalized_line)

        if not name:
            return False

        name_tokens = [
            token for token in re.split(r"\s+", name.lower()) if token
        ]

        if not name_tokens:
            return False

        first_token = re.sub(
            r"[^a-z0-9\-/]+",
            "",
            name_tokens[0],
        ).strip()

        if first_token in _MEDICINE_NAME_STOPWORDS:
            return False

        normalized_name = re.sub(
            r"[^a-z0-9\s\-/]+",
            "",
            name.lower(),
        ).strip()

        if normalized_name in _MEDICINE_NAME_STOPWORDS:
            return False

        # An explicit dosage prefix is sufficient to preserve a medicine-line
        # candidate when OCR misses its strength. The raw name is retained;
        # no medicine identity or strength is inferred here.
        return has_dosage_prefix or bool(re.search(r"\d", normalized_line))

    def _is_instruction_line(self, line: str) -> bool:
        if not line:
            return False

        lower_line = line.lower().strip()

        if (
            self._detect_frequency(line)
            or self._detect_timing(line)
            or self._detect_duration(line)
        ):
            return False

        if _MEDICINE_START_PATTERN.match(line):
            return False

        return bool(_INSTRUCTION_HINT_PATTERN.search(lower_line))

    @staticmethod
    def _append_unique_instruction(existing: str, new_text: str) -> str:
        if not new_text:
            return existing

        if not existing:
            return new_text

        if new_text.lower() in existing.lower():
            return existing

        return f"{existing}; {new_text}"

    @staticmethod
    def _append_unique_timing(existing: str, new_text: str) -> str:
        if not new_text:
            return existing

        if not existing:
            return new_text

        existing_parts = [
            part.strip().lower()
            for part in existing.split(",")
            if part.strip()
        ]

        if new_text.lower() in existing_parts:
            return existing

        return f"{existing}, {new_text}"

    @staticmethod
    def _strip_trailing_doctor_segment(line: str) -> str:
        """Remove a trailing doctor signature from a direction/advice line."""
        return _DOCTOR_SEGMENT_PATTERN.sub("", line).strip()

    # -------------------------------------------------------------------------
    # General advice
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_general_advice_marker(line: str) -> bool:
        """
        Detect markers such as:
            Adv:
            Advice:
            General Advice:
            Instructions:

        Colon is optional because clean_text() strips trailing punctuation.
        """
        return bool(_GENERAL_ADVICE_MARKER_PATTERN.match(line.strip()))

    @staticmethod
    def _is_registration_line(line: str) -> bool:
        return bool(_REGISTRATION_LINE_PATTERN.match(line.strip()))

    @staticmethod
    def _is_date_line(line: str) -> bool:
        return bool(_DATE_LINE_PATTERN.match(line.strip()))

    @staticmethod
    def _clean_advice_line(line: str) -> str:
        """Remove doctor/signature fragments and redundant advice prefixes."""
        cleaned = line.strip()

        cleaned = re.sub(
            r"^\s*(?:adv|advice|general\s+advice|patient\s+advice)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = PrescriptionAnalyzer._strip_trailing_doctor_segment(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,")

        return cleaned

    @staticmethod
    def _is_probable_doctor_signature(line: str) -> bool:
        stripped = line.strip()

        if _DOCTOR_LINE_PATTERN.match(stripped):
            return True

        if re.search(
            r"\b(?:mbbs|md|ms|m\.d\.|reg\.?\s*no|registration\s*no)\b",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True

        return False

    @staticmethod
    def _append_unique_advice(existing: List[str], line: str) -> None:
        cleaned = PrescriptionAnalyzer._clean_advice_line(line)

        if not cleaned:
            return

        normalized = cleaned.lower()

        if any(item.lower() == normalized for item in existing):
            return

        existing.append(cleaned)

    # -------------------------------------------------------------------------
    # Medicine extraction
    # -------------------------------------------------------------------------

    def _consume_line_into_medicine(
        self,
        current: _MedicineEntry,
        line: str,
        *,
        allow_instruction: bool = True,
    ) -> _MedicineEntry:
        normalized_line = self._normalize_continuation_line(line)

        frequency = self._detect_frequency(normalized_line)
        timings = self._detect_timings(normalized_line)
        duration = self._detect_duration(normalized_line)

        if frequency and not current.frequency:
            current = _MedicineEntry(
                name=current.name,
                strength=current.strength,
                frequency=frequency,
                timing=current.timing,
                duration=current.duration,
                instructions=current.instructions,
            )

        if timings:
            timing_text = ", ".join(timings)

            current = _MedicineEntry(
                name=current.name,
                strength=current.strength,
                frequency=current.frequency,
                timing=self._append_unique_timing(
                    current.timing,
                    timing_text,
                ),
                duration=current.duration,
                instructions=current.instructions,
            )

        if duration and not current.duration:
            current = _MedicineEntry(
                name=current.name,
                strength=current.strength,
                frequency=current.frequency,
                timing=current.timing,
                duration=duration,
                instructions=current.instructions,
            )

        if allow_instruction and self._is_instruction_line(normalized_line):
            instruction_text = self._strip_trailing_doctor_segment(
                normalized_line
            )

            if instruction_text:
                current = _MedicineEntry(
                    name=current.name,
                    strength=current.strength,
                    frequency=current.frequency,
                    timing=current.timing,
                    duration=current.duration,
                    instructions=self._append_unique_instruction(
                        current.instructions,
                        instruction_text,
                    ),
                )

        return current

    @staticmethod
    def _entry_has_data(entry: Optional[_MedicineEntry]) -> bool:
        if entry is None:
            return False

        return bool(
            entry.name
            or entry.strength
            or entry.frequency
            or entry.timing
            or entry.duration
            or entry.instructions
        )

    def _extract_medicines_and_advice(
        self,
        text: str,
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Extract medicines and general prescription advice in one pass.

        General advice is kept separate from medicine-specific directions.
        A marker such as "Adv:" switches the parser into advice mode.
        Doctor/signature and registration lines are never copied into advice.
        """
        medicines: List[_MedicineEntry] = []
        general_advice: List[str] = []
        current: Optional[_MedicineEntry] = None
        advice_mode = False

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for line in lines:
            # A medicine line always starts a new medicine, even if advice
            # mode was previously active. This prevents a later prescription
            # item from being swallowed by the advice section.
            if self._is_medicine_start_line(line):
                if self._entry_has_data(current):
                    medicines.append(current)  # type: ignore[arg-type]

                name, strength = self._split_name_strength(line)

                current = _MedicineEntry(
                    name=name,
                    strength=strength,
                )

                current = self._consume_line_into_medicine(
                    current,
                    line,
                    allow_instruction=False,
                )

                advice_mode = False
                continue

            if self._is_general_advice_marker(line):
                advice_mode = True
                continue

            # Doctor/registration/date lines are document metadata, not
            # patient advice and never belong to the previous medicine.
            if self._is_probable_doctor_signature(line):
                advice_mode = False
                continue

            if self._is_registration_line(line) or self._is_date_line(line):
                continue

            if advice_mode:
                self._append_unique_advice(general_advice, line)
                continue

            if current is None:
                continue

            current = self._consume_line_into_medicine(current, line)

        if self._entry_has_data(current):
            medicines.append(current)  # type: ignore[arg-type]

        structured = [
            entry.as_dict()
            for entry in medicines
            if self._entry_has_data(entry)
        ]

        logger.info(
            "Medicines Detected: %d; General Advice Lines: %d",
            len(structured),
            len(general_advice),
        )

        return structured, general_advice

    def extract_medicines(self, text: str) -> List[Dict[str, str]]:
        """
        Backward-compatible public API.

        Existing callers receive exactly the medicine list they received
        before. General advice is intentionally not included here.
        """
        medicines, _ = self._extract_medicines_and_advice(text)
        return medicines

    # -------------------------------------------------------------------------
    # Full analysis
    # -------------------------------------------------------------------------

    def analyze(self, image: object) -> Dict[str, Any]:
        """
        Run OCR, cleanup, medicine extraction, and advice separation.

        Successful result preserves the existing fields and adds:
            general_advice: list[str]
        """
        try:
            text_result = self.extract_text(image)

            if not text_result.get("success"):
                return {
                    "success": False,
                    "error": text_result.get("error", "OCR failed."),
                }

            cleaned_text = self.clean_text(str(text_result.get("ocr_text", "")))
            raw_ocr_text = str(
                text_result.get("raw_ocr_text", text_result.get("ocr_text", ""))
            )

            logger.info(
                "Prescription trace [1/5] raw OCR: lines=%d, preview=%s",
                len([
                    line for line in raw_ocr_text.splitlines()
                    if line.strip()
                ]),
                raw_ocr_text.splitlines()[:5],
            )

            logger.info(
                "Prescription trace [2/5] cleaned OCR: lines=%d, preview=%s",
                len([
                    line for line in cleaned_text.splitlines()
                    if line.strip()
                ]),
                cleaned_text.splitlines()[:5],
            )

            if not cleaned_text:
                return {
                    "success": False,
                    "error": "No text detected in the uploaded image.",
                }

            medicines, general_advice = (
                self._extract_medicines_and_advice(cleaned_text)
            )
            targeted_diagnostics = list(
                text_result.get("targeted_frequency_diagnostics", [])
            )
            for index, diagnostic in enumerate(targeted_diagnostics):
                if not isinstance(diagnostic, dict):
                    continue
                diagnostic["final_frequency"] = (
                    medicines[index].get("frequency", "")
                    if index < len(medicines)
                    else ""
                )

            logger.info(
                "Prescription trace [3/5] medicines: count=%d, names=%s",
                len(medicines),
                [medicine.get("name") for medicine in medicines],
            )

            logger.info(
                "Prescription trace [4/5] general advice: count=%d, advice=%s",
                len(general_advice),
                general_advice,
            )

            confidence = float(text_result.get("confidence") or 0.0)
            lines = len([
                line for line in cleaned_text.splitlines()
                if line.strip()
            ])
            characters = len(cleaned_text)

            logger.info(
                "Prescription trace [5/5] OCR Finished; confidence=%.1f%%.",
                confidence,
            )

            return {
                "success": True,
                "ocr_text": cleaned_text,
                "raw_ocr_text": raw_ocr_text,
                "ocr_detections": list(text_result.get("ocr_detections", [])),
                "targeted_frequency_diagnostics": targeted_diagnostics,
                "confidence": confidence,
                "lines": lines,
                "characters": characters,
                "medicines": medicines,
                "general_advice": general_advice,
            }

        except Exception as exc:  # noqa: BLE001
            logger.exception("Prescription analysis failed.")
            return {
                "success": False,
                "error": str(exc),
            }


@lru_cache(maxsize=1)
def get_prescription_analyzer() -> PrescriptionAnalyzer:
    """Return a cached PrescriptionAnalyzer singleton."""
    return PrescriptionAnalyzer()
