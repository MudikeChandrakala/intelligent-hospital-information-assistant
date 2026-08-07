"""
modules/prescription_ai_service.py
=============================================================================
Prescription analysis AI service for the Intelligent Hospital Information
Assistant.

This module owns the prescription-specific retrieval and Gemini workflow:
    - medicine_database.json lookup for detected medicines
    - metadata normalization for prescription explanation
    - Gemini prompt construction for a professional prescription report
    - safe fallback report generation when Gemini is unavailable

This module deliberately does NOT:
- Touch the RAG assistant or chat system
- Change application routing or sidebar/navigation logic
- Diagnose diseases or infer conditions beyond the prescription data
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional, Sequence

from modules.gemini_client import GeminiClient

logger = logging.getLogger("hospital_assistant.prescription_ai_service")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDICINE_DATABASE_PATH = PROJECT_ROOT / "knowledge_base" / "structured" / "medicine_database.json"
LOW_CONFIDENCE_THRESHOLD = 70.0
MEDICINE_MATCH_THRESHOLD = 85.0


class PrescriptionAIService:
    """Build a prescription explanation report from OCR and medicine metadata."""

    def __init__(self) -> None:
        self._database = self._load_medicine_database()
        self._gemini_client = self._create_gemini_client()

    def _create_gemini_client(self) -> Optional[GeminiClient]:
        try:
            return GeminiClient()
        except Exception as exc:  # noqa: BLE001 - safe fallback when Gemini is unavailable
            logger.warning("Gemini client is unavailable for prescription reports: %s", exc)
            return None

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_medicine_database() -> tuple[dict[str, Any], ...]:
        try:
            with MEDICINE_DATABASE_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - safe fallback when the database cannot be read
            logger.exception("Failed to load medicine database.")
            raise RuntimeError(f"Failed to load medicine database: {exc}") from exc

        medicines = payload.get("medicines", []) if isinstance(payload, dict) else []
        if not isinstance(medicines, list):
            raise RuntimeError("Medicine database format is invalid.")

        normalized_records: list[dict[str, Any]] = []
        for entry in medicines:
            if isinstance(entry, dict):
                normalized_records.append(entry)

        return tuple(normalized_records)

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    @staticmethod
    def _normalize_medicine_name(value: str) -> str:
        text = PrescriptionAIService._normalize_text(value)
        if not text:
            return ""

        text = re.sub(
            r"\b(?:tab|tablet|cap|capsule|syr|syrup|inj|injection|cream|ointment|drops?|gel|solution|sus|susp|odt)\b",
            " ",
            text,
        )
        text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?)\b", " ", text)
        text = re.sub(
            r"\b(?:before|after|morning|afternoon|night|bedtime|food|with|empty|stomach|take|apply|use|continue|avoid|daily|dose|once|twice|thrice|qid|od|bd|tds|hs|sos)\b",
            " ",
            text,
        )
        text = re.sub(r"\b\d+\b", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        return {token for token in PrescriptionAIService._normalize_text(value).split() if token}

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _format_not_specified(label: str) -> str:
        return f"{label} not specified in the medicine database."

    def _candidate_aliases(self, record: dict[str, Any]) -> list[str]:
        aliases = [record.get("generic_name", "")]
        aliases.extend(self._as_list(record.get("brand_names", [])))
        aliases.extend(self._as_list(record.get("keywords", [])))
        return [self._normalize_medicine_name(alias) for alias in aliases if alias]

    @staticmethod
    def _similarity_score(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 100.0
        return SequenceMatcher(None, left, right).ratio() * 100.0

    def _score_record(self, detected_name: str, record: dict[str, Any], detected_strength: str = "") -> float:
        detected_normalized = self._normalize_medicine_name(detected_name)
        aliases = self._candidate_aliases(record)

        best_score = 0.0
        for alias in aliases:
            if not alias:
                continue

            score = self._similarity_score(detected_normalized, alias)

            strength_values = self._as_list(record.get("strength", []))
            if detected_strength and strength_values:
                detected_strength_normalized = self._normalize_text(detected_strength)
                strength_match = any(
                    detected_strength_normalized == self._normalize_text(strength)
                    or detected_strength_normalized in self._normalize_text(strength)
                    or self._normalize_text(strength) in detected_strength_normalized
                    for strength in strength_values
                )
                if strength_match:
                    score += 5.0

            best_score = max(best_score, min(score, 100.0))

        return best_score

    def _normalize_medicine_metadata(self, record: dict[str, Any]) -> dict[str, Any]:
        uses = self._as_list(record.get("used_for", []))
        contraindications = self._as_list(record.get("contraindications", []))
        side_effects = self._as_list(record.get("side_effects", []))
        best_time = self._as_list(record.get("best_time", []))
        dosage_strengths = self._as_list(record.get("strength", []))

        pregnancy_safe = record.get("pregnancy_safe")
        if pregnancy_safe is True:
            pregnancy_warning = "Database flags this medicine as generally pregnancy-safe, but clinical verification is still required."
        elif pregnancy_safe is False:
            pregnancy_warning = "Database flags this medicine as not pregnancy-safe; use only if specifically prescribed."
        else:
            pregnancy_warning = self._format_not_specified("Pregnancy warning")

        child_dose = str(record.get("child_dose") or "").strip()
        if child_dose:
            pediatric_warning = f"Child dose information is listed in the database: {child_dose}."
        else:
            pediatric_warning = self._format_not_specified("Pediatric warning")

        adult_dose = str(record.get("adult_dose") or "").strip()
        if adult_dose:
            elderly_warning = f"No elderly-specific dose is listed; the database adult dose is {adult_dose}."
        else:
            elderly_warning = self._format_not_specified("Elderly warning")

        precautions: list[str] = []
        if record.get("prescription_required") is True:
            precautions.append("Prescription required: Yes")
        elif record.get("prescription_required") is False:
            precautions.append("Prescription required: No")
        storage = str(record.get("storage") or "").strip()
        if storage:
            precautions.append(f"Storage: {storage}")
        food_instruction = str(record.get("food_instruction") or "").strip()
        if food_instruction:
            precautions.append(f"Food instruction: {food_instruction}")
        if contraindications:
            precautions.append(f"Contraindications: {', '.join(contraindications)}")

        dosage_information = {
            "dosage_form": str(record.get("dosage_form") or "").strip() or "Not Available",
            "strength": dosage_strengths or ["Not Available"],
            "adult_dose": adult_dose or "Not Available",
            "child_dose": child_dose or "Not Available",
            "frequency": str(record.get("frequency") or "").strip() or "Not Available",
            "maximum_daily_dose": str(record.get("maximum_daily_dose") or "").strip() or "Not Available",
        }

        return {
            "medicine_id": str(record.get("medicine_id") or "").strip() or "Not Available",
            "generic_name": str(record.get("generic_name") or "").strip() or "Not Available",
            "medicine_class": str(record.get("drug_class") or "").strip() or "Not Available",
            "uses": uses or ["Not Available"],
            "indications": uses or ["Not Available"],
            "dosage_information": dosage_information,
            "precautions": precautions or ["Not specified in the medicine database."],
            "contraindications": contraindications or ["Not Available"],
            "side_effects": side_effects or ["Not Available"],
            "storage": storage or "Not Available",
            "food_interactions": food_instruction or "Not Available",
            "pregnancy_warning": pregnancy_warning,
            "pediatric_warning": pediatric_warning,
            "elderly_warning": elderly_warning,
            "best_time": best_time or ["Not Available"],
        }

    def _match_medicine(self, detected_medicine: dict[str, str]) -> dict[str, Any]:
        raw_detected_name = str(detected_medicine.get("name") or "").strip()
        detected_name = self._normalize_medicine_name(raw_detected_name)
        detected_strength = str(detected_medicine.get("strength") or "").strip()

        best_record: Optional[dict[str, Any]] = None
        best_score = 0.0
        best_alias = ""

        for record in self._database:
            aliases = self._candidate_aliases(record)
            score = 0.0
            alias_match = ""
            for alias in aliases:
                if not alias:
                    continue
                candidate_score = self._similarity_score(detected_name, alias)
                if candidate_score > score:
                    score = candidate_score
                    alias_match = alias

            strength_values = self._as_list(record.get("strength", []))
            if detected_strength and strength_values:
                detected_strength_normalized = self._normalize_text(detected_strength)
                strength_match = any(
                    detected_strength_normalized == self._normalize_text(strength)
                    or detected_strength_normalized in self._normalize_text(strength)
                    or self._normalize_text(strength) in detected_strength_normalized
                    for strength in strength_values
                )
                if strength_match:
                    score += 5.0

            if score > best_score:
                best_score = score
                best_record = record
                best_alias = alias_match

        if best_record is None or best_score < MEDICINE_MATCH_THRESHOLD:
            return {
                "detected_name": raw_detected_name or "Not Available",
                "normalized_detected_name": detected_name or "Not Available",
                "detected_strength": detected_strength or "Not Available",
                "matched": False,
                "match_score": round(best_score, 1),
                "matched_alias": "",
                "metadata": None,
            }

        metadata = self._normalize_medicine_metadata(best_record)
        return {
            "detected_name": raw_detected_name or metadata.get("generic_name") or "Not Available",
            "normalized_detected_name": detected_name or metadata.get("generic_name") or "Not Available",
            "detected_strength": detected_strength or "Not Available",
            "matched": True,
            "match_score": round(best_score, 1),
            "matched_alias": best_alias or metadata.get("generic_name") or "Not Available",
            "metadata": metadata,
        }

    def lookup_medicines(self, detected_medicines: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
        lookup_results = [self._match_medicine(medicine) for medicine in detected_medicines]
        logger.info(
            "Prescription trace [6/6] lookup_medicines output: count=%d, names=%s, entries=%s",
            len(lookup_results),
            [item.get("detected_name") for item in lookup_results],
            lookup_results[:3],
        )
        return lookup_results

    def _build_prompt(
        self,
        matched_medicines: Sequence[dict[str, Any]],
        confidence: float,
    ) -> str:
        matched_database_entries = [item.get("metadata") for item in matched_medicines if item.get("matched") and item.get("metadata")]
        unmatched_detected_names = [
            item.get("normalized_detected_name") or item.get("detected_name") or "Not Available"
            for item in matched_medicines
            if not item.get("matched")
        ]
        payload = {
            "ocr_confidence": round(float(confidence), 1),
            "manual_verification_required": float(confidence) < LOW_CONFIDENCE_THRESHOLD,
            "matched_database_entries": matched_database_entries,
            "unmatched_detected_names": unmatched_detected_names,
            "structured_prescription": [
                {
                    "medicine_name": item.get("normalized_detected_name") or item.get("detected_name") or "Not Available",
                    "strength": item.get("detected_strength") or "Not Available",
                    "frequency": (item.get("metadata") or {}).get("dosage_information", {}).get("frequency") if item.get("matched") else "Not Available",
                    "timing": ", ".join((item.get("metadata") or {}).get("best_time", ["Not Available"])) if item.get("matched") else "Not Available",
                    "match_score": item.get("match_score", 0.0),
                }
                for item in matched_medicines
            ],
        }

        instructions = (
            "You are a hospital prescription explanation assistant.\n\n"
            "Rules:\n"
            "- Explain only the provided prescription.\n"
            "- Do NOT diagnose diseases, infer conditions, or recommend new medicines.\n"
            "- Use ONLY the matched database entries and the structured prescription context below.\n"
            "- Never invent a medicine that is not present in the matched database entries.\n"
            "- If a field is missing, write 'Not specified in the medicine database.' or 'Not Available'.\n"
            "- If OCR confidence is low, clearly warn that manual verification is recommended.\n"
            "- Keep the tone professional, hospital-style, concise, and clear.\n"
            "- Output in Markdown with exactly these section headings:\n"
            "  1. Prescription Summary\n"
            "  2. Detected Medicines\n"
            "  3. Medicine Details\n"
            "  4. Dosage Schedule\n"
            "  5. Timing (Morning / Afternoon / Night)\n"
            "  6. Before/After Food\n"
            "  7. Duration\n"
            "  8. Precautions\n"
            "  9. Possible Side Effects\n"
            "  10. Drug Interactions\n"
            "  11. Emergency Warnings\n"
            "  12. Overall Recommendations\n"
            "- Do not provide a diagnosis or claim disease certainty.\n"
            "- Use short bullet points or compact tables.\n"
            "- If no medicine metadata is matched, clearly say so and continue with the OCR-based explanation only.\n"
        )

        return (
            f"{instructions}\n\n"
            f"Prescription Context (JSON):\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
        )

    def _build_fallback_report(
        self,
        ocr_text: str,
        detected_medicines: Sequence[dict[str, str]],
        matched_medicines: Sequence[dict[str, Any]],
        confidence: float,
    ) -> str:
        lines: list[str] = []

        lines.append("# Prescription Summary")
        if detected_medicines:
            lines.append(
                f"- OCR detected {len(detected_medicines)} medicine(s) with {confidence:.1f}% confidence."
            )
        else:
            lines.append("- No structured medicines were detected from the OCR text.")
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            lines.append("- OCR confidence is low. Manual verification is recommended.")

        lines.append("")
        lines.append("# Detected Medicines")
        if detected_medicines:
            for medicine in detected_medicines:
                name = medicine.get("name") or "Not Available"
                strength = medicine.get("strength") or "Not Available"
                frequency = medicine.get("frequency") or "Not Available"
                lines.append(f"- {name} ({strength}) - {frequency}")
        else:
            lines.append("- None detected.")

        lines.append("")
        lines.append("# Medicine Details")
        if matched_medicines:
            for item in matched_medicines:
                metadata = item.get("metadata") or {}
                name = item.get("detected_name") or metadata.get("generic_name") or "Not Available"
                if item.get("matched"):
                    lines.append(f"- {name}: {metadata.get('medicine_class', 'Not Available')}")
                else:
                    lines.append(f"- {name}: No database match found.")
        else:
            lines.append("- No medicine metadata available.")

        lines.append("")
        lines.append("# Dosage Schedule")
        if matched_medicines:
            for item in matched_medicines:
                metadata = item.get("metadata") or {}
                dosage = metadata.get("dosage_information", {})
                lines.append(
                    f"- {item.get('detected_name', 'Not Available')}: {dosage.get('adult_dose', 'Not Available')}"
                )
        else:
            lines.append("- Not Available")

        lines.append("")
        lines.append("# Timing (Morning / Afternoon / Night)")
        if matched_medicines:
            for item in matched_medicines:
                metadata = item.get("metadata") or {}
                best_time = ", ".join(metadata.get("best_time", ["Not Available"]))
                lines.append(f"- {item.get('detected_name', 'Not Available')}: {best_time}")
        else:
            lines.append("- Not Available")

        lines.append("")
        lines.append("# Before/After Food")
        if matched_medicines:
            for item in matched_medicines:
                metadata = item.get("metadata") or {}
                lines.append(
                    f"- {item.get('detected_name', 'Not Available')}: {metadata.get('food_interactions', 'Not Available')}"
                )
        else:
            lines.append("- Not Available")

        lines.append("")
        lines.append("# Duration")
        if any(medicine.get("duration") for medicine in detected_medicines):
            for medicine in detected_medicines:
                if medicine.get("duration"):
                    lines.append(f"- {medicine.get('name', 'Not Available')}: {medicine.get('duration')}")
        else:
            lines.append("- Not specified in the OCR text.")

        lines.append("")
        lines.append("# Precautions")
        if matched_medicines:
            for item in matched_medicines:
                metadata = item.get("metadata") or {}
                precautions = metadata.get("precautions", ["Not Available"])
                lines.append(f"- {item.get('detected_name', 'Not Available')}: {'; '.join(precautions)}")
        else:
            lines.append("- Not Available")

        lines.append("")
        lines.append("# Possible Side Effects")
        if matched_medicines:
            for item in matched_medicines:
                metadata = item.get("metadata") or {}
                side_effects = metadata.get("side_effects", ["Not Available"])
                lines.append(f"- {item.get('detected_name', 'Not Available')}: {', '.join(side_effects)}")
        else:
            lines.append("- Not Available")

        lines.append("")
        lines.append("# Drug Interactions")
        lines.append("- No drug interaction data is stored in the medicine database for this report.")

        lines.append("")
        lines.append("# Emergency Warnings")
        lines.append("- Seek urgent medical help for breathing difficulty, severe rash, swelling, or fainting.")
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            lines.append("- Manual verification is recommended because OCR confidence is low.")

        lines.append("")
        lines.append("# Overall Recommendations")
        lines.append("- Follow the prescribing clinician's directions exactly.")
        lines.append("- Verify any uncertain medicine name, strength, or duration manually.")
        if ocr_text.strip():
            lines.append("- Use the OCR text and database lookup together to confirm the prescription details.")

        return "\n".join(lines).strip()

    def generate_report(
        self,
        ocr_text: str,
        detected_medicines: Sequence[dict[str, str]],
        confidence: float,
    ) -> dict[str, Any]:
        logger.info(
            "Prescription trace [5/6] generate_report input: count=%d, names=%s, entries=%s",
            len(detected_medicines),
            [medicine.get("name") for medicine in detected_medicines],
            list(detected_medicines)[:3],
        )
        matched_medicines = self.lookup_medicines(detected_medicines)
        prompt = self._build_prompt(matched_medicines, confidence)

        report_text: str
        used_fallback = False

        if self._gemini_client is not None:
            try:
                report_text = self._gemini_client.generate_response(prompt)
            except Exception as exc:  # noqa: BLE001 - safe fallback for Gemini failures
                logger.warning("Gemini report generation failed; using fallback report: %s", exc)
                used_fallback = True
                report_text = self._build_fallback_report(ocr_text, detected_medicines, matched_medicines, confidence)
        else:
            used_fallback = True
            report_text = self._build_fallback_report(ocr_text, detected_medicines, matched_medicines, confidence)

        return {
            "success": True,
            "report_text": report_text,
            "matched_medicines": matched_medicines,
            "low_confidence": float(confidence) < LOW_CONFIDENCE_THRESHOLD,
            "used_fallback": used_fallback,
        }


@lru_cache(maxsize=1)
def get_prescription_ai_service() -> PrescriptionAIService:
    """Return a cached PrescriptionAIService singleton."""
    return PrescriptionAIService()
