"""
Programmatic verification of structured pathology extraction.

Checks internal consistency of extracted JSON fields against known
medical rules (FIGO/TNM alignment, node count math, single histotype, etc.).
Returns a list of error strings.  An empty list means the extraction passed.
"""

import json
import re
from typing import Any

# ── FIGO stage ↔ TNM mapping (endometrial, 2009/8th ed) ──────────────────────
_FIGO_TO_T = {
    "IA": ["pT1a", "T1a"],
    "IB": ["pT1b", "T1b"],
    "II": ["pT2", "T2"],
    "IIIA": ["pT3a", "T3a"],
    "IIIB": ["pT3b", "T3b"],
    "IIIC1": [],  # any T, pN1/N1mi
    "IIIC2": [],  # any T, pN2/N2a
    "IVA": ["pT4", "T4"],
    "IVB": [],  # any T, M1
}


def _normalise(s: str) -> str:
    """Lower-case, strip whitespace/quotes, collapse spaces."""
    return re.sub(r"\s+", " ", str(s).strip().strip("'\"").lower())


def _get(data: dict, key: str) -> str:
    """Safely get the 'value' sub-field as a normalised string."""
    entry = data.get(key, {})
    if isinstance(entry, dict):
        return _normalise(entry.get("value", ""))
    return _normalise(str(entry))


def _get_quote(data: dict, key: str) -> str:
    """Get the 'quote' sub-field."""
    entry = data.get(key, {})
    if isinstance(entry, dict):
        return entry.get("quote", "")
    return ""


def _is_missing(val: str) -> bool:
    return val in ("", "not reported", "not available", "pending", "n/a", "none", "unknown")


# ── Public API ────────────────────────────────────────────────────────────────


def verify_extraction(data: dict[str, Any]) -> list[str]:
    """
    Run all consistency checks on extracted pathology fields.

    Args:
        data: dict whose keys are field names and values are
              {"value": "...", "quote": "..."} dicts.

    Returns:
        List of human-readable error strings (empty = all passed).
    """
    errors: list[str] = []

    # 1. FIGO ↔ TNM consistency
    figo = _get(data, "figo_stage").upper().replace("STAGE ", "").strip()
    tnm = _get(data, "tnm_stage").upper()
    if figo and not _is_missing(figo) and tnm and not _is_missing(tnm):
        expected_ts = _FIGO_TO_T.get(figo, None)
        if expected_ts is not None and expected_ts:
            t_component = tnm.split(",")[0].split(" ")[0].strip()
            match = any(t_component.upper() == et.upper() for et in expected_ts)
            if not match:
                errors.append(
                    f"FIGO/TNM mismatch: FIGO {figo} expects T-stage in {expected_ts}, "
                    f"but got '{t_component}'."
                )

    # 2. Single histologic type
    htype = _get(data, "histologic_type")
    if htype and not _is_missing(htype):
        types_found = set()
        for keyword in ["endometrioid", "serous", "clear cell", "carcinosarcoma", "mucinous",
                        "mixed", "undifferentiated", "dedifferentiated"]:
            if keyword in htype:
                types_found.add(keyword)
        if len(types_found) > 1 and "mixed" not in types_found:
            errors.append(
                f"Multiple histologic types detected: {types_found}. "
                f"If this is truly a mixed tumor, the type should say 'mixed'."
            )

    # 3. Lymph node arithmetic
    def _int_or_none(val: str) -> int | None:
        nums = re.findall(r"\d+", val)
        return int(nums[0]) if nums else None

    pelvic_ex = _int_or_none(_get(data, "pelvic_nodes_examined"))
    pelvic_pos = _int_or_none(_get(data, "pelvic_nodes_positive"))
    paraaortic_ex = _int_or_none(_get(data, "paraaortic_nodes_examined"))
    paraaortic_pos = _int_or_none(_get(data, "paraaortic_nodes_positive"))
    total_ex = _int_or_none(_get(data, "total_nodes_examined"))
    total_pos = _int_or_none(_get(data, "total_nodes_positive"))

    if pelvic_ex is not None and pelvic_pos is not None and pelvic_pos > pelvic_ex:
        errors.append(
            f"Pelvic nodes: positive ({pelvic_pos}) > examined ({pelvic_ex})."
        )
    if paraaortic_ex is not None and paraaortic_pos is not None and paraaortic_pos > paraaortic_ex:
        errors.append(
            f"Para-aortic nodes: positive ({paraaortic_pos}) > examined ({paraaortic_ex})."
        )
    if total_ex is not None and pelvic_ex is not None and paraaortic_ex is not None:
        expected_total = pelvic_ex + paraaortic_ex
        if total_ex != expected_total:
            errors.append(
                f"Total nodes examined ({total_ex}) != pelvic ({pelvic_ex}) + "
                f"para-aortic ({paraaortic_ex}) = {expected_total}."
            )

    # 4. LVSI consistency
    lvsi = _get(data, "lvsi")
    if lvsi and "not identified" in lvsi and "positive" in lvsi:
        errors.append(f"LVSI contradicts itself: '{lvsi}'.")

    # 5. Margins consistency
    margins = _get(data, "margins")
    if margins:
        if "uninvolved" in margins and "positive" in margins:
            errors.append(f"Margins contradicts itself: '{margins}'.")
        if "involved" in margins and "uninvolved" not in margins and "negative" in margins:
            errors.append(f"Margins contradicts itself: '{margins}'.")

    # 6. Grade range check
    grade = _get(data, "histologic_grade")
    if grade and not _is_missing(grade):
        grade_nums = re.findall(r"\d", grade)
        for g in grade_nums:
            if int(g) not in (1, 2, 3):
                errors.append(f"Invalid FIGO grade number: {g} (must be 1, 2, or 3).")

    # 7. Invasion depth vs myometrial thickness
    depth_str = _get(data, "myometrial_invasion_depth_cm")
    thickness_str = _get(data, "myometrial_thickness_cm")
    depth_nums = re.findall(r"[\d.]+", depth_str)
    thick_nums = re.findall(r"[\d.]+", thickness_str)
    if depth_nums and thick_nums:
        depth = float(depth_nums[0])
        thickness = float(thick_nums[0])
        if thickness > 0 and depth > thickness:
            errors.append(
                f"Invasion depth ({depth} cm) exceeds myometrial thickness ({thickness} cm)."
            )

    # 8. Every non-missing field should have a supporting quote
    for key, entry in data.items():
        if isinstance(entry, dict):
            val = _normalise(entry.get("value", ""))
            quote = entry.get("quote", "").strip()
            if val and not _is_missing(val) and not quote:
                errors.append(f"Field '{key}' has value '{val}' but no supporting quote.")

    return errors


def parse_json_from_response(text: str) -> dict | None:
    """Try to extract a JSON object from model output (possibly wrapped in markdown)."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to find JSON inside ```json ... ``` blocks
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
