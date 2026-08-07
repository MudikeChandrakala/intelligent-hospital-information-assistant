"""
modules/prescription_analyzer.py
=============================================================================
Prescription OCR and structured medicine extraction utilities for the
Intelligent Hospital Information Assistant.

This module owns the non-UI business logic for Phase 11B/11C:
    - OCR with EasyOCR
    - OCR text cleanup
    - Structured medicine extraction
    - Safe, non-crashing analysis result assembly

It deliberately does NOT call Gemini, the RAG pipeline, the Retriever,
or any medicine database. Those are reserved for later phases.

-----------------------------------------------------------------------------
Public API
-----------------------------------------------------------------------------
    PrescriptionAnalyzer -> OCR and extraction helper
    get_prescription_analyzer() -> cached singleton accessor
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import easyocr
except ImportError:  # pragma: no cover - handled gracefully at runtime
    easyocr = None  # type: ignore[assignment]

logger = logging.getLogger("hospital_assistant.prescription_analyzer")

# =============================================================================
# CONSTANTS
# =============================================================================

_PRESERVE_DUPLICATES: set[str] = {"SOS", "OD", "BD", "TDS", "HS"}
_DOSAGE_PREFIXES: set[str] = {
    "tab",
    "tablet",
    "cap",
    "capsule",
    "syp",
    "syrup",
    "inj",
    "injection",
    "cream",
    "ointment",
    "drops",
    "drop",
    "gel",
    "solution",
    "sus",
    "susp",
    "odt",
}
_MEDICINE_START_PREFIXES: tuple[str, ...] = (
    r"tab\.?",
    r"tablet",
    r"cap\.?",
    r"capsule",
    r"syr\.?",
    r"syrup",
    r"inj\.?",
    r"injection",
    r"cream",
    r"ointment",
    r"drops?",
    r"gel",
    r"solution",
    r"sus\.?",
    r"susp\.?",
    r"odt",
)
_MEDICINE_NAME_STOPWORDS: set[str] = {
    "before",
    "after",
    "morning",
    "afternoon",
    "night",
    "bedtime",
    "food",
    "with",
    "empty",
    "stomach",
    "take",
    "drink",
    "apply",
    "use",
    "continue",
    "avoid",
    "od",
    "bd",
    "tds",
    "qid",
    "hs",
    "sos",
    "1-0-1",
    "1-1-1",
    "0-1-0",
    "0-0-1",
    "1-0-0",
}
_FREQUENCY_PATTERNS: tuple[str, ...] = (
    r"\b1\s*[-/]\s*0\s*[-/]\s*1\b",
    r"\b1\s*[-/]\s*1\s*[-/]\s*1\b",
    r"\b0\s*[-/]\s*1\s*[-/]\s*0\b",
    r"\b0\s*[-/]\s*0\s*[-/]\s*1\b",
    r"\bSOS\b",
    r"\bOD\b",
    r"\bBD\b",
    r"\bTDS\b",
    r"\bQID\b",
    r"\bHS\b",
    r"\bonce\s+daily\b",
    r"\btwice\s+daily\b",
    r"\bthrice\s+daily\b",
)
_FREQUENCY_NORMALIZATION: dict[str, str] = {
    r"\b1\s*[-/]\s*0\s*[-/]\s*1\b": "1-0-1",
    r"\b1\s*[-/]\s*1\s*[-/]\s*1\b": "1-1-1",
    r"\b0\s*[-/]\s*1\s*[-/]\s*0\b": "0-1-0",
    r"\b0\s*[-/]\s*0\s*[-/]\s*1\b": "0-0-1",
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
_TIMING_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bbefore\s+food\b", "Before Food"),
    (r"\bafter\s+food\b", "After Food"),
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
_DURATION_PATTERN = re.compile(r"\b(\d+)\s*(days?|weeks?|months?)\b", re.IGNORECASE)
_DURATION_ONLY_PATTERN = re.compile(r"\b(days?|weeks?|months?)\b", re.IGNORECASE)
_MEDICINE_NAME_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9\s\-/&().']*?)\s+(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml|iu|units?)?\b",
    re.IGNORECASE,
)
_MEDICINE_START_PATTERN = re.compile(
    r"^\s*(?:"
    + "|".join(_MEDICINE_START_PREFIXES)
    + r")\b[\s\.:,-]*(?P<remainder>.+)$",
    re.IGNORECASE,
)
_MEDICINE_LIST_PREFIX_PATTERN = re.compile(
    r"^\s*(?:(?:\d+\s*(?:[.)]\s*|-\s*)?)|[•-]\s*)"
)
_INSTRUCTION_HINT_PATTERN = re.compile(
    r"\b(?:take|apply|use|continue|avoid|swallow|chew|dissolve|with water|as directed|per day|daily|dose|tablet|capsule|syrup|injection|before meal|after meal|before food|after food|morning|afternoon|night|bedtime|empty stomach|with food)\b",
    re.IGNORECASE,
)


# =============================================================================
# HELPERS
# =============================================================================


@dataclass(frozen=True)
class _OCRBlock:
    """Internal OCR block used to reconstruct readable text lines."""

    text: str
    confidence: float
    left: float
    top: float
    height: float


@dataclass(frozen=True)
class _MedicineEntry:
    """Internal structured medicine row."""

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
    """
    EasyOCR-backed prescription analyzer.

    The EasyOCR reader is initialized once per process and reused across
    runs. All public methods catch their own failures or return safe
    values so the UI never crashes.
    """

    _reader_lock: ClassVar[Lock] = Lock()
    _reader: ClassVar[Optional[Any]] = None
    _reader_error: ClassVar[Optional[str]] = None

    def __init__(self) -> None:
        self.reader = self._get_reader()

    @classmethod
    def _get_reader(cls) -> Optional[Any]:
        """Initialize EasyOCR exactly once and reuse the reader."""
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
            except Exception as exc:  # noqa: BLE001 - never let OCR init crash the app
                cls._reader_error = f"Failed to initialize EasyOCR: {exc}"
                logger.exception(cls._reader_error)
                cls._reader = None

        return cls._reader

    @staticmethod
    def _image_to_array(image: object) -> np.ndarray:
        """Convert a supported image input to a NumPy RGB array."""
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

    @staticmethod
    def _normalize_duplicate_words(text: str) -> str:
        """Collapse immediate duplicate words while preserving dosage tokens."""
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
        """Normalize a single OCR line without removing medicine data."""
        cleaned = line.replace("\u00a0", " ")
        cleaned = re.sub(r"[•·▪●]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = PrescriptionAnalyzer._normalize_duplicate_words(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned).strip()
        return cleaned

    @staticmethod
    def _strip_dosage_prefix(name: str) -> str:
        """Remove common dosage-form prefixes from a medicine name."""
        tokens = [token for token in re.split(r"\s+", name.strip()) if token]
        while tokens and tokens[0].lower().rstrip(".") in _DOSAGE_PREFIXES:
            tokens.pop(0)
        return " ".join(tokens).strip(" ,:-.")

    @staticmethod
    def _normalize_strength(amount: str, unit: Optional[str]) -> str:
        """Format a medicine strength, defaulting to mg when no unit is present."""
        normalized_unit = (unit or "mg").strip().lower()
        if normalized_unit == "iu":
            normalized_unit = "IU"
        elif normalized_unit == "g":
            normalized_unit = "g"
        elif normalized_unit == "mg":
            normalized_unit = "mg"
        elif normalized_unit == "mcg":
            normalized_unit = "mcg"
        elif normalized_unit == "ml":
            normalized_unit = "ml"
        elif normalized_unit.endswith("s"):
            normalized_unit = normalized_unit[:-1]

        return f"{amount} {normalized_unit}".strip()

    @staticmethod
    def _normalize_medicine_line(line: str) -> str:
        """Remove list markers and targeted OCR artifacts from a medicine line."""
        normalized = _MEDICINE_LIST_PREFIX_PATTERN.sub("", line).strip()
        normalized = re.sub(r"^toab\b", "Tab", normalized, flags=re.IGNORECASE)
        return re.sub(r"\[\s*o(?=\s*mg\b)", "10", normalized, flags=re.IGNORECASE)

    @staticmethod
    def _merge_ocr_results(results: Sequence[Any]) -> Tuple[str, float, List[_OCRBlock]]:
        """Merge OCR blocks into readable lines and compute confidence."""
        blocks: List[_OCRBlock] = []
        confidences: List[float] = []

        for item in results:
            if not item or len(item) < 3:
                continue
            bbox, text, confidence = item[0], str(item[1] or "").strip(), float(item[2] or 0.0)
            if not text:
                continue

            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            left = float(min(xs))
            top = float(min(ys))
            height = float(max(ys) - min(ys)) or 1.0
            blocks.append(_OCRBlock(text=text, confidence=confidence, left=left, top=top, height=height))
            confidences.append(confidence)

        if not blocks:
            return "", 0.0, []

        blocks.sort(key=lambda block: (block.top, block.left))
        merged_lines: List[str] = []
        current_line: List[_OCRBlock] = []
        current_top: Optional[float] = None
        current_height: float = 0.0

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
                merged_lines.append(" ".join(entry.text for entry in current_line).strip())
                current_line = [block]
                current_top = block.top
                current_height = block.height

        if current_line:
            current_line.sort(key=lambda entry: entry.left)
            merged_lines.append(" ".join(entry.text for entry in current_line).strip())

        merged_text = "\n".join(line for line in merged_lines if line.strip())
        average_confidence = round((sum(confidences) / len(confidences)) * 100, 1) if confidences else 0.0
        return merged_text, average_confidence, blocks

    def extract_text(self, image: object) -> Dict[str, Any]:
        """
        Extract raw OCR text and confidence from the uploaded image.

        Returns:
            A dict containing raw OCR text and metrics on success, or a
            failure payload with `success=False` and `error`.
        """
        if self.reader is None:
            return {"success": False, "error": self._reader_error or "EasyOCR is unavailable."}

        try:
            image_array = self._image_to_array(image)
        except Exception as exc:  # noqa: BLE001 - safe failure path for unreadable uploads
            logger.exception("Unreadable or corrupted prescription image.")
            return {"success": False, "error": f"Unreadable image upload: {exc}"}

        logger.info("OCR Started")
        try:
            results = self.reader.readtext(image_array, detail=1, paragraph=False)
        except Exception as exc:  # noqa: BLE001 - safe failure path for OCR runtime errors
            logger.exception("EasyOCR failure during prescription analysis.")
            return {"success": False, "error": f"EasyOCR failure: {exc}"}

        if not results:
            logger.info("OCR Finished with no detectable text.")
            return {"success": False, "error": "No text detected in the uploaded image."}

        raw_text, average_confidence, _ = self._merge_ocr_results(results)
        if not raw_text.strip():
            logger.info("OCR Finished with no readable text after merging.")
            return {"success": False, "error": "No text detected in the uploaded image."}

        logger.info("OCR Finished (average confidence=%.1f%%).", average_confidence)
        return {
            "success": True,
            "ocr_text": raw_text,
            "confidence": average_confidence,
            "lines": len([line for line in raw_text.splitlines() if line.strip()]),
            "characters": len(raw_text),
        }

    def clean_text(self, text: str) -> str:
        """
        Clean OCR text while preserving medicine names, numbers, and abbreviations.

        Removes duplicate spaces, empty lines, duplicate words, and common
        OCR artifacts while keeping dosage patterns and abbreviations intact.
        """
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

        if not matches:
            return []

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

        normalized = line.strip()
        if _DURATION_ONLY_PATTERN.fullmatch(normalized):
            return normalized.capitalize()
        return None

    @staticmethod
    def _normalize_continuation_line(line: str) -> str:
        """Correct common OCR errors in prescription direction lines."""
        normalized = re.sub(r"\baften\b", "after", line, flags=re.IGNORECASE)
        normalized = re.sub(r"\bbefone\b", "before", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bagten\b", "after", normalized, flags=re.IGNORECASE)
        return re.sub(r"\b(?:dods|doxs|doys)\b", "days", normalized, flags=re.IGNORECASE)

    def _split_name_strength(self, line: str) -> Tuple[str, str]:
        normalized_line = self._normalize_medicine_line(line)
        stripped_line = self._strip_dosage_prefix(normalized_line)
        match = _MEDICINE_NAME_PATTERN.search(stripped_line)
        if not match:
            return stripped_line, ""

        name = self._strip_dosage_prefix(match.group("name"))
        amount = match.group("amount")
        unit = match.group("unit")
        strength = self._normalize_strength(amount, unit)
        return name, strength

    def _looks_like_medicine_line(self, line: str) -> bool:
        return self._is_medicine_start_line(line)

    @staticmethod
    def _append_unique_instruction(existing: str, new_text: str) -> str:
        """Append an instruction line without duplicating identical text."""
        if not new_text:
            return existing
        if not existing:
            return new_text
        if new_text.lower() in existing.lower():
            return existing
        return f"{existing}; {new_text}"

    @staticmethod
    def _append_unique_timing(existing: str, new_text: str) -> str:
        """Append timing information without duplicating identical text."""
        if not new_text:
            return existing
        if not existing:
            return new_text

        existing_parts = [part.strip().lower() for part in existing.split(",") if part.strip()]
        if new_text.lower() in existing_parts:
            return existing
        return f"{existing}, {new_text}"

    def _is_medicine_start_line(self, line: str) -> bool:
        """Return True when a line clearly starts a new medicine entry."""
        normalized_line = self._normalize_medicine_line(line)
        if not normalized_line or not _MEDICINE_START_PATTERN.match(normalized_line):
            return False

        name, _ = self._split_name_strength(normalized_line)
        if not name:
            return False

        name_tokens = [token for token in re.split(r"\s+", name.lower()) if token]
        if not name_tokens:
            return False

        first_token = re.sub(r"[^a-z0-9\-/]+", "", name_tokens[0]).strip()
        if first_token in _MEDICINE_NAME_STOPWORDS:
            return False

        normalized_name = re.sub(r"[^a-z0-9\s\-/]+", "", name.lower()).strip()
        if normalized_name in _MEDICINE_NAME_STOPWORDS:
            return False

        return bool(re.search(r"\d", normalized_line))

    def _is_instruction_line(self, line: str) -> bool:
        """Return True when a line should be stored as instructions."""
        if not line:
            return False

        lower_line = line.lower().strip()
        if self._detect_frequency(line) or self._detect_timing(line) or self._detect_duration(line):
            return False
        if _MEDICINE_START_PATTERN.match(line):
            return False
        return bool(_INSTRUCTION_HINT_PATTERN.search(lower_line))

    def _consume_line_into_medicine(self, current: _MedicineEntry, line: str, *, allow_instruction: bool = True) -> _MedicineEntry:
        """Assign one OCR line to the current medicine entry."""
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
                timing=self._append_unique_timing(current.timing, timing_text),
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
            current = _MedicineEntry(
                name=current.name,
                strength=current.strength,
                frequency=current.frequency,
                timing=current.timing,
                duration=current.duration,
                instructions=self._append_unique_instruction(
                    current.instructions,
                    normalized_line,
                ),
            )

        return current

    def extract_medicines(self, text: str) -> List[Dict[str, str]]:
        """
        Extract structured medicine fields from cleaned OCR text.

        Returns a list of structured medicine dictionaries containing
        name, strength, frequency, timing, duration, and instructions.
        """
        medicines: List[_MedicineEntry] = []
        current: Optional[_MedicineEntry] = None

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            if self._is_medicine_start_line(line):
                name, strength = self._split_name_strength(line)
                next_entry = _MedicineEntry(name=name, strength=strength)
                next_entry = self._consume_line_into_medicine(next_entry, line, allow_instruction=False)

                if current is not None and (current.name or current.strength or current.frequency or current.timing or current.duration or current.instructions):
                    medicines.append(current)

                current = next_entry
                continue

            if current is None:
                continue

            current = self._consume_line_into_medicine(current, line)

        if current is not None and (current.name or current.strength or current.frequency or current.timing or current.duration or current.instructions):
            medicines.append(current)

        structured = [entry.as_dict() for entry in medicines if entry.name or entry.strength or entry.frequency or entry.timing or entry.duration or entry.instructions]
        logger.info("Medicines Detected: %d", len(structured))
        return structured

    def analyze(self, image: object) -> Dict[str, Any]:
        """
        Run OCR and structured medicine extraction on a prescription image.

        Returns:
            A structured analysis dict on success, or `{"success": False,
            "error": ...}` when OCR or image processing fails.
        """
        try:
            text_result = self.extract_text(image)
            if not text_result.get("success"):
                return {"success": False, "error": text_result.get("error", "OCR failed.")}

            cleaned_text = self.clean_text(str(text_result.get("ocr_text", "")))
            raw_ocr_text = str(text_result.get("ocr_text", ""))
            logger.info(
                "Prescription trace [1/4] raw OCR text: lines=%d, preview=%s",
                len([line for line in raw_ocr_text.splitlines() if line.strip()]),
                raw_ocr_text.splitlines()[:3],
            )
            logger.info(
                "Prescription trace [2/4] cleaned OCR text: lines=%d, preview=%s",
                len([line for line in cleaned_text.splitlines() if line.strip()]),
                cleaned_text.splitlines()[:3],
            )
            if not cleaned_text:
                logger.info("OCR Finished with no cleaned text.")
                return {"success": False, "error": "No text detected in the uploaded image."}

            medicines = self.extract_medicines(cleaned_text)
            logger.info(
                "Prescription trace [3/4] extract_medicines output: count=%d, names=%s, entries=%s",
                len(medicines),
                [medicine.get("name") for medicine in medicines],
                medicines[:3],
            )
            confidence = float(text_result.get("confidence") or 0.0)
            lines = len([line for line in cleaned_text.splitlines() if line.strip()])
            characters = len(cleaned_text)

            logger.info("OCR Finished (average confidence=%.1f%%).", confidence)
            logger.info("Medicines Detected: %d", len(medicines))

            return {
                "success": True,
                "ocr_text": cleaned_text,
                "confidence": confidence,
                "lines": lines,
                "characters": characters,
                "medicines": medicines,
            }
        except Exception as exc:  # noqa: BLE001 - never let analysis crash the app
            logger.exception("Prescription analysis failed.")
            return {"success": False, "error": str(exc)}


@lru_cache(maxsize=1)
def get_prescription_analyzer() -> PrescriptionAnalyzer:
    """Return a cached PrescriptionAnalyzer singleton."""
    return PrescriptionAnalyzer()
