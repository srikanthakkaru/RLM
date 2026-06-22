"""Deterministic grounding gate for RLM-generated pathology narratives.

The RLM stage occasionally hallucinates numbers (e.g. parroting a worked example's
"0.5 cm / 1.4 cm = 35.7%" into an unrelated case) or mis-attributes a computed/provisional
FIGO stage as "reported". This module re-reads the finished narrative and flags statements
that are not supported by the structured extraction or the source report.

It is intentionally conservative: it only flags the high-risk hallucination vectors
(percentages and centimetre measurements) and a couple of attribution mistakes, so a clean
narrative is never rejected for legitimate content. Selection/regeneration logic lives in the
caller; this module only reports violations.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Matching tolerances (absolute, in the measured unit). Percentages allow rounding slack
# (35.7 vs 36); lengths are matched near-exactly so a fabricated measurement cannot hide behind
# an unrelated nearby value.
_PCT_TOLERANCE = 1.0
_LEN_TOLERANCE = 0.05

# Fixed category boundaries that appear in the controlled vocabulary ("<50%", "≥50%") rather
# than as measured data — never treated as ungrounded.
_CATEGORY_PERCENTS = {50.0}

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_CM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*cm\b", re.IGNORECASE)
_MM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Endometrial FIGO stage tokens (2009/2023 forms), longest first so "IIIC2" wins over "IIIC".
_FIGO_STAGE_RE = re.compile(
    r"\b(IVB|IVA|IV|IIIC2|IIIC1|IIIC|IIIA|IIIB|III|IIC|IIB|IIA|II|IC|IB|IA1|IA2|IA|I)\b"
)

_REPORTED_ATTRIBUTION_RE = re.compile(
    r"(?:\(reported\)|\breported\b[^.\n]{0,30}\bstage\b|\bstage\b[^.\n]{0,30}\(reported\))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Violation:
    severity: str  # "error" (numeric, blocks acceptance) or "warning" (attribution/staging)
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}:{self.kind}] {self.detail}"


def _iter_numbers(text: str) -> Iterable[float]:
    for m in _NUMBER_RE.finditer(text):
        try:
            yield float(m.group())
        except ValueError:
            continue


def _collect_allowed_numbers(context: dict[str, Any], report_text: str) -> list[float]:
    """Numbers the narrative is allowed to cite: everything in the extraction context plus the
    raw report text (the report legitimately contains tumour sizes, depths, node counts, etc.).
    """
    allowed: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        elif isinstance(value, bool):
            return
        elif isinstance(value, (int, float)):
            allowed.append(float(value))
        elif isinstance(value, str):
            allowed.extend(_iter_numbers(value))

    walk(context)
    allowed.extend(_iter_numbers(report_text))

    # Derived myometrial-invasion percentage is legitimate even if never written verbatim.
    depth = _first_float(context.get("myometrial_invasion_depth_cm"))
    thick = _first_float(context.get("myometrial_thickness_cm"))
    if depth is not None and thick is not None and thick > 0:
        allowed.append(round(depth / thick * 100, 1))
    return allowed


def _first_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def _is_supported(number: float, allowed: list[float], tolerance: float) -> bool:
    return any(abs(number - a) <= tolerance for a in allowed)


def _allowed_stage_tokens(context: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("reported_figo_stage", "computed_figo_stage", "provisional_figo_stage"):
        val = context.get(key)
        if isinstance(val, str):
            for m in _FIGO_STAGE_RE.finditer(val.upper()):
                tokens.add(m.group(1))
    return tokens


def _staging_section(narrative: str) -> str:
    """Return the text under the 'Staging' header, or the whole narrative if not found."""
    m = re.search(r"(?im)^\s*staging\b[:\s]*(.+?)(?=^\s*[A-Z][A-Za-z /-]+\n|\Z)", narrative, re.S)
    return m.group(1) if m else narrative


def check_narrative(
    narrative: str,
    context: dict[str, Any],
    report_text: str = "",
) -> list[Violation]:
    """Return grounding violations for a finished narrative. Empty list == fully grounded."""
    violations: list[Violation] = []
    if not narrative or not narrative.strip():
        return [Violation("error", "empty", "narrative is empty")]

    allowed = _collect_allowed_numbers(context, report_text)

    # 1. Numeric grounding: every percentage / cm / mm figure must trace to the extraction or report.
    for m in _PERCENT_RE.finditer(narrative):
        try:
            number = float(m.group(1))
        except ValueError:
            continue
        if number in _CATEGORY_PERCENTS:
            continue  # "<50%" / "≥50%" category boundary, not measured data
        if not _is_supported(number, allowed, _PCT_TOLERANCE):
            violations.append(
                Violation(
                    "error",
                    "ungrounded_number",
                    f"'{m.group(0).strip()}' not found in extraction or report",
                )
            )

    for label, regex in (("cm", _CM_RE), ("mm", _MM_RE)):
        for m in regex.finditer(narrative):
            try:
                number = float(m.group(1))
            except ValueError:
                continue
            probe = number / 10.0 if label == "mm" else number  # also compare mm on the cm scale
            if not (
                _is_supported(number, allowed, _LEN_TOLERANCE)
                or _is_supported(probe, allowed, _LEN_TOLERANCE)
            ):
                violations.append(
                    Violation(
                        "error",
                        "ungrounded_number",
                        f"'{m.group(0).strip()}' not found in extraction or report",
                    )
                )

    # 2. Attribution: do not call a computed/provisional stage "reported".
    reported = context.get("reported_figo_stage")
    has_reported = isinstance(reported, str) and reported.strip().lower() not in (
        "",
        "not reported",
    )
    if not has_reported and _REPORTED_ATTRIBUTION_RE.search(narrative):
        violations.append(
            Violation(
                "warning",
                "false_reported_attribution",
                "narrative attributes a FIGO stage as 'reported' but the extraction has no "
                "reported_figo_stage",
            )
        )

    # 3. Invented stage: any FIGO token in the Staging section must match an allowed stage field.
    allowed_stages = _allowed_stage_tokens(context)
    if allowed_stages:
        seen = {m.group(1) for m in _FIGO_STAGE_RE.finditer(_staging_section(narrative).upper())}
        for tok in seen - allowed_stages:
            violations.append(
                Violation(
                    "warning",
                    "invented_stage",
                    f"Staging cites FIGO '{tok}' which is not among reported/computed/provisional "
                    f"stages {sorted(allowed_stages)}",
                )
            )

    return violations


def has_blocking_violation(violations: list[Violation]) -> bool:
    """True if any violation should force regeneration / fallback (numeric hallucination)."""
    return any(v.severity == "error" for v in violations)
