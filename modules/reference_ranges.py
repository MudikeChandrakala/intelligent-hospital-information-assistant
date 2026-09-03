"""
modules/reference_ranges.py
=============================================================================
Deterministic GENERAL reference-range resolver for common CBC laboratory
parameters.

This module is intentionally standalone. It is not imported by, and does
not import, report_analyzer.py, report_ai_service.py, gemini_client.py, or
any other part of the application. Integration is a separate, later task.

WHAT THIS MODULE IS FOR
-----------------------
Many real-world laboratory reports do not print a reference range for
every parameter. When a report's OWN range is available, that range is
always authoritative and this module must never override it - callers
should only consult this module when the uploaded report did not supply
a usable range.

When consulted, this module can supply a GENERAL adult reference range
for a small, explicitly supported list of common CBC parameters, sourced
from named reputable medical references (see _SOURCES below). Every
result is clearly tagged as a general reference range, never as if it
came from the patient's own report.

WHAT THIS MODULE WILL NOT DO
-----------------------------
- It will not guess a sex-dependent range when sex is not supplied.
- It will not guess at an ambiguous unit (e.g. it will not assume a bare
  WBC value of "7" means 7,000 - that conversion is only made when the
  unit explicitly says so, such as "x10^3/uL").
- It will not fuzzy-match an unrecognized test name.
- It will not call Gemini, the network, or any database. It is pure,
  deterministic Python with a small hardcoded, documented dataset.
- It will not diagnose, and it will not decide treatment.

Every non-None result includes reference_source / reference_source_label
so a caller can always show the reader where a GENERAL range came from
and that it is general, not the patient's own report.

Public API
-----------
    canonical_test_name(test_name: str) -> Optional[str]
    resolve_reference_range(test_name, value, unit, *, sex=None, age=None)
        -> Optional[Dict[str, Any]]
    classify_with_range(value: float, lower: float, upper: float) -> str
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

# =============================================================================
# SOURCE METADATA
# =============================================================================
#
# Every range below cites exactly one of these. Each was read directly
# (not recalled from memory) before being used here.

_SOURCES: Dict[str, Dict[str, str]] = {
    "medlineplus_cbc": {
        "label": "MedlinePlus Medical Encyclopedia - CBC blood test (NIH/NLM)",
        "url": "https://medlineplus.gov/ency/article/003642.htm",
    },
    "ccjm_rdw_mpv": {
        "label": (
            "Cleveland Clinic Journal of Medicine - "
            "\"Three neglected numbers in the CBC: The RDW, MPV, and NRBC count\""
        ),
        "url": "https://www.ccjm.org/content/86/3/167",
    },
    "ucsf_uf_differential": {
        "label": (
            "UCSF Health / UF Health patient education - "
            "Blood differential test (consistent adult ranges at both sites)"
        ),
        "url": "https://www.ucsfhealth.org/care/medical-tests/blood-differential-test",
    },
    "cleveland_clinic_anc": {
        "label": "Cleveland Clinic - Neutrophils / Absolute Neutrophil Count",
        "url": "https://my.clevelandclinic.org/health/body/22313-neutrophils",
    },
    "cleveland_clinic_alc": {
        "label": "Cleveland Clinic - Lymphopenia / Lymphocytosis (Absolute Lymphocyte Count)",
        "url": "https://my.clevelandclinic.org/health/diseases/24837-lymphopenia",
    },
}

_GENERAL_ADULT_SOURCE_LABEL = "General adult reference range"


# =============================================================================
# RANGE DATA
# =============================================================================


@dataclass(frozen=True)
class _RangeEntry:
    lower: float
    upper: float
    unit_label: str  # human-readable unit this range is expressed in
    source_key: str


# Sex-independent ranges: canonical_test -> _RangeEntry
_GENERAL_RANGES_SIMPLE: Dict[str, _RangeEntry] = {
    "wbc": _RangeEntry(4500, 11000, "cells/uL", "medlineplus_cbc"),
    "mcv": _RangeEntry(80, 100, "fL", "medlineplus_cbc"),
    "mch": _RangeEntry(27, 32, "pg", "medlineplus_cbc"),
    "mchc": _RangeEntry(32, 36, "g/dL", "medlineplus_cbc"),
    # MedlinePlus's own published text says "150,000 to 400,000/dL" for
    # platelet count on this page. Per-deciliter is not the platelet
    # convention used anywhere else in clinical practice (1 dL = 100,000
    # uL, which would be a wildly different scale); this is treated here
    # as the universally standard "/uL" convention that every other
    # source and every real lab report uses for this same 150,000-400,000
    # figure. Documented explicitly rather than silently "corrected".
    "platelets": _RangeEntry(150000, 400000, "cells/uL", "medlineplus_cbc"),
    "rdw_cv": _RangeEntry(11, 16, "%", "ccjm_rdw_mpv"),
    "mpv": _RangeEntry(8, 12, "fL", "ccjm_rdw_mpv"),
    "neutrophils_pct": _RangeEntry(40, 60, "%", "ucsf_uf_differential"),
    "lymphocytes_pct": _RangeEntry(20, 40, "%", "ucsf_uf_differential"),
    "monocytes_pct": _RangeEntry(2, 8, "%", "ucsf_uf_differential"),
    "eosinophils_pct": _RangeEntry(1, 4, "%", "ucsf_uf_differential"),
    "basophils_pct": _RangeEntry(0.5, 1, "%", "ucsf_uf_differential"),
    "anc": _RangeEntry(2500, 7000, "cells/uL", "cleveland_clinic_anc"),
    "alc": _RangeEntry(1000, 4800, "cells/uL", "cleveland_clinic_alc"),
    # amc / aec / abc: deliberately NOT included. No range for these from
    # the specifically reputable sources this module requires could be
    # confirmed with adequate confidence in this pass. canonical_test_name
    # still recognizes them; resolve_reference_range returns None.
}

# Sex-dependent ranges: canonical_test -> {"male": _RangeEntry, "female": _RangeEntry}
_GENERAL_RANGES_BY_SEX: Dict[str, Dict[str, _RangeEntry]] = {
    "hemoglobin": {
        "male": _RangeEntry(13, 18, "g/dL", "medlineplus_cbc"),
        "female": _RangeEntry(12, 16, "g/dL", "medlineplus_cbc"),
    },
    "hematocrit": {
        "male": _RangeEntry(40, 55, "%", "medlineplus_cbc"),
        "female": _RangeEntry(36, 48, "%", "medlineplus_cbc"),
    },
    "rbc": {
        "male": _RangeEntry(4.6, 6.2, "million/uL", "medlineplus_cbc"),
        "female": _RangeEntry(4.2, 5.4, "million/uL", "medlineplus_cbc"),
    },
}

_CANONICAL_TESTS = set(_GENERAL_RANGES_SIMPLE) | set(_GENERAL_RANGES_BY_SEX)


# =============================================================================
# TEST-NAME CANONICALIZATION (exact, normalized alias matching only)
# =============================================================================

_ALIAS_MAP: Dict[str, str] = {}


def _normalize_test_name_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _register_aliases(canonical: str, aliases: Sequence[str]) -> None:
    for alias in aliases:
        _ALIAS_MAP[_normalize_test_name_text(alias)] = canonical


_register_aliases("hemoglobin", ["hemoglobin", "haemoglobin", "hb", "hgb"])
_register_aliases(
    "wbc",
    [
        "wbc",
        "wbc count",
        "total wbc count",
        "tlc",
        "total leukocyte count",
        "total leucocyte count",
        "white blood cell count",
        "white blood cells",
    ],
)
_register_aliases(
    "rbc",
    ["rbc", "rbc count", "erythrocyte count", "red blood cell count", "red blood cells"],
)
_register_aliases(
    "hematocrit", ["hematocrit", "haematocrit", "hct", "pcv", "packed cell volume"]
)
_register_aliases("mcv", ["mcv", "mean corpuscular volume"])
_register_aliases("mch", ["mch", "mean corpuscular hemoglobin"])
_register_aliases(
    "mchc", ["mchc", "mean corpuscular hemoglobin concentration"]
)
_register_aliases(
    "rdw_cv", ["rdw", "rdw cv", "rdw-cv", "red cell distribution width"]
)
_register_aliases("platelets", ["platelet count", "platelets", "plt"])
_register_aliases("mpv", ["mpv", "mean platelet volume"])
_register_aliases("neutrophils_pct", ["neutrophils", "neutrophils %", "neutrophil %"])
_register_aliases("lymphocytes_pct", ["lymphocytes", "lymphocytes %", "lymphocyte %"])
_register_aliases("monocytes_pct", ["monocytes", "monocytes %", "monocyte %"])
_register_aliases("eosinophils_pct", ["eosinophils", "eosinophils %", "eosinophil %"])
_register_aliases("basophils_pct", ["basophils", "basophils %", "basophil %"])
_register_aliases("anc", ["absolute neutrophil count", "anc"])
_register_aliases("alc", ["absolute lymphocyte count", "alc"])
_register_aliases("amc", ["absolute monocyte count", "amc"])
_register_aliases("aec", ["absolute eosinophil count", "aec"])
_register_aliases("abc", ["absolute basophil count"])


def canonical_test_name(test_name: str) -> Optional[str]:
    """
    Return the canonical internal test key for an exact, normalized alias
    match, or None if the name is not recognized.

    This NEVER fuzzy-matches. "Hemglobin" (a typo) or an arbitrary OCR
    fragment will correctly return None rather than guess.
    """
    if not test_name or not isinstance(test_name, str):
        return None
    normalized = _normalize_test_name_text(test_name)
    return _ALIAS_MAP.get(normalized)


# =============================================================================
# UNIT NORMALIZATION AND CONSERVATIVE CONVERSION
# =============================================================================


def _clean_unit(unit: Optional[str]) -> str:
    """
    Normalize common representations/OCR variants of a unit string for
    comparison purposes only. This never changes the MEANING of a unit -
    only its spelling (case, the micro sign, whitespace, and the several
    equivalent ways "cubic millimeter" and "microliter" are written,
    which are the same volume: 1 mm^3 = 1 uL).
    """
    text = (unit or "").strip().lower()
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace(" ", "")
    text = text.replace("cu.mm.", "cumm")
    text = text.replace("cu.mm", "cumm")
    text = text.replace("mm^3", "cumm")
    text = text.replace("mm3", "cumm")
    return text


_PERCENT_PATTERN = re.compile(r"^%$")
_GDL_PATTERN = re.compile(r"^g(m)?/?dl$")  # g/dl, gm/dl, and a bare "gdl" OCR variant
_FL_PATTERN = re.compile(r"^fl$")
_PG_PATTERN = re.compile(r"^pg(/cell)?$")
_MILLION_PER_UL_PATTERN = re.compile(r"^(million|10\^?6|x10\^?6|10e6)/ul$")
_BARE_PER_UL_PATTERN = re.compile(r"^(cells?)?/?(ul|cumm)$")
_THOUSAND_PER_UL_PATTERN = re.compile(r"^(x?10\^?3|k|thousand)/(ul|cumm)$")
_LAKH_PER_UL_PATTERN = re.compile(r"^lakh/(ul|cumm)$")


def _resolve_percent(value: float, unit: str) -> Optional[float]:
    return float(value) if _PERCENT_PATTERN.fullmatch(_clean_unit(unit)) else None


def _resolve_gdl(value: float, unit: str) -> Optional[float]:
    return float(value) if _GDL_PATTERN.fullmatch(_clean_unit(unit)) else None


def _resolve_mchc(value: float, unit: str) -> Optional[float]:
    # MCHC is conventionally reported either as g/dL or as a numerically
    # equivalent %; both map to the same comparison scale for this analyte.
    cleaned = _clean_unit(unit)
    if _GDL_PATTERN.fullmatch(cleaned) or _PERCENT_PATTERN.fullmatch(cleaned):
        return float(value)
    return None


def _resolve_fl(value: float, unit: str) -> Optional[float]:
    return float(value) if _FL_PATTERN.fullmatch(_clean_unit(unit)) else None


def _resolve_pg(value: float, unit: str) -> Optional[float]:
    return float(value) if _PG_PATTERN.fullmatch(_clean_unit(unit)) else None


def _resolve_million_per_ul(value: float, unit: str) -> Optional[float]:
    # A bare per-uL count for RBC would be an entirely different (much
    # larger) scale - only an explicit "million" / "x10^6" unit is
    # accepted; nothing is guess-converted.
    return float(value) if _MILLION_PER_UL_PATTERN.fullmatch(_clean_unit(unit)) else None


def _make_count_per_ul_resolver(*, allow_lakh: bool) -> Callable[[float, str], Optional[float]]:
    def _resolve(value: float, unit: str) -> Optional[float]:
        cleaned = _clean_unit(unit)
        if _BARE_PER_UL_PATTERN.fullmatch(cleaned):
            return float(value)
        if _THOUSAND_PER_UL_PATTERN.fullmatch(cleaned):
            return float(value) * 1000.0
        if allow_lakh and _LAKH_PER_UL_PATTERN.fullmatch(cleaned):
            return float(value) * 100000.0
        return None

    return _resolve


_resolve_count_per_ul = _make_count_per_ul_resolver(allow_lakh=False)
_resolve_count_per_ul_with_lakh = _make_count_per_ul_resolver(allow_lakh=True)


_CONVERTERS: Dict[str, Callable[[float, str], Optional[float]]] = {
    "hemoglobin": _resolve_gdl,
    "hematocrit": _resolve_percent,
    "rbc": _resolve_million_per_ul,
    "wbc": _resolve_count_per_ul,
    "mcv": _resolve_fl,
    "mch": _resolve_pg,
    "mchc": _resolve_mchc,
    "rdw_cv": _resolve_percent,
    "platelets": _resolve_count_per_ul_with_lakh,
    "mpv": _resolve_fl,
    "neutrophils_pct": _resolve_percent,
    "lymphocytes_pct": _resolve_percent,
    "monocytes_pct": _resolve_percent,
    "eosinophils_pct": _resolve_percent,
    "basophils_pct": _resolve_percent,
    "anc": _resolve_count_per_ul,
    "alc": _resolve_count_per_ul,
}


# =============================================================================
# SEX NORMALIZATION (never guessed)
# =============================================================================


def _normalize_sex(sex: Optional[str]) -> Optional[str]:
    if not sex or not isinstance(sex, str):
        return None
    cleaned = sex.strip().lower()
    if cleaned in {"male", "m", "man"}:
        return "male"
    if cleaned in {"female", "f", "woman"}:
        return "female"
    return None


# =============================================================================
# CLASSIFICATION
# =============================================================================


def classify_with_range(value: float, lower: float, upper: float) -> str:
    """Classify a numeric value against an already-resolved [lower, upper]."""
    if value < lower:
        return "Low"
    if value > upper:
        return "High"
    return "Normal"


# =============================================================================
# PUBLIC RESOLUTION API
# =============================================================================


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def resolve_reference_range(
    test_name: str,
    value: Any,
    unit: str,
    *,
    sex: Optional[str] = None,
    age: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve a GENERAL adult reference range for a recognized CBC test.

    Returns None (never a guess) when:
      - the test name is not recognized (canonical_test_name is None),
      - the value is not numeric,
      - this analyte requires sex and sex was not supplied/recognized,
      - the analyte's data is deliberately absent from this module
        (e.g. amc/aec/abc - not yet confidently sourced),
      - the supplied unit does not unambiguously match a supported
        convention for this analyte.

    The returned dict is always tagged as a GENERAL reference range via
    reference_source / reference_source_label - never implied to be the
    patient's own laboratory range.
    """
    canonical = canonical_test_name(test_name)
    if canonical is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if canonical in _GENERAL_RANGES_BY_SEX:
        normalized_sex = _normalize_sex(sex)
        if normalized_sex is None:
            return None
        entry = _GENERAL_RANGES_BY_SEX[canonical].get(normalized_sex)
        if entry is None:
            return None
    else:
        entry = _GENERAL_RANGES_SIMPLE.get(canonical)
        if entry is None:
            return None

    converter = _CONVERTERS.get(canonical)
    if converter is None:
        return None

    comparison_value = converter(numeric_value, unit)
    if comparison_value is None:
        return None

    source = _SOURCES[entry.source_key]

    return {
        "reference_range": f"{_format_number(entry.lower)}-{_format_number(entry.upper)}",
        "reference_source": entry.source_key,
        "reference_source_label": f"{_GENERAL_ADULT_SOURCE_LABEL} ({source['label']})",
        "reference_source_url": source["url"],
        "canonical_test": canonical,
        "comparison_value": comparison_value,
        "unit_convention": entry.unit_label,
        "status": classify_with_range(comparison_value, entry.lower, entry.upper),
    }