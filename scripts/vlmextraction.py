from __future__ import annotations

import json
import os
import re
import sys
import time
import typer
from pathlib import Path
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
from rich.table import Table
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rlm.clients.ollama_client import OllamaClient


MEDGEMMA_MODEL = os.environ.get("MEDGEMMA_MODEL", "alibayram/medgemma:latest")
MEDGEMMA_URL = os.environ.get("MEDGEMMA_URL", "http://localhost:11434")
MEDGEMMA_OPTIONS = {
    "num_ctx": 8192,
    "num_predict": 4096,
    "temperature": 0,
}

MAX_EXTRACTION_RETRIES = 2

# Confidence at or above this is treated as reliable; below it a populated field is flagged
# as low-confidence / uncertain rather than trusted silently.
MEDGEMMA_CONFIDENCE_THRESHOLD = float(os.environ.get("MEDGEMMA_CONFIDENCE_THRESHOLD", "0.5"))

# "json" (default): structured JSON with per-field value/confidence/status/evidence. The model
#   normalizes garbled OCR and flags ambiguous fields instead of dropping them.
# "conversational": natural prompt + regex fallback for non-instruction-tuned models.
# "structured"/"legacy": legacy ANSWER 1: / ANSWER 2: numbered format.
_EXTRACTION_PROMPT_MODE = os.environ.get("MEDGEMMA_EXTRACTION_PROMPT", "json").strip().lower()

# Phrases in a field's evidence quote that mark it as ambiguous (addressed in the report but
# not assertible), so it is treated as uncertain rather than present.
_UNCERTAIN_EVIDENCE_PHRASES = (
    "cannot be excluded",
    "cannot absolutely exclude",
    "cannot exclude",
    "cannot be assessed",
    "not fully evaluated",
    "difficult to",
    "suggestive of",
    "suspicious",
    "indeterminate",
    "equivocal",
    "favor",
    "probable",
    "possible",
)

_CONFIDENCE_WORD_MAP = {
    "high": 0.9,
    "very high": 0.95,
    "medium": 0.6,
    "med": 0.6,
    "moderate": 0.6,
    "low": 0.3,
    "very low": 0.1,
    "none": 0.0,
    "unknown": 0.3,
}

# Field status values describing provenance relative to the source report.
STATUS_PRESENT = "present"
STATUS_UNCERTAIN = "uncertain"
STATUS_MISSING = "missing"
_VALID_FIELD_STATUSES = {STATUS_PRESENT, STATUS_UNCERTAIN, STATUS_MISSING}

EXTRACTION_FIELDS: dict[str, dict] = {
    "histologic_type": {"type": "string", "required": True},
    "figo_grade": {
        "type": "string",
        "required": True,
        "valid_values": ["1", "2", "3", "not reported"],
    },
    "nuclear_grade": {"type": "string", "required": False},
    "tumor_size_cm": {"type": "string", "required": True},
    "myometrial_invasion_depth_cm": {"type": "string", "required": True},
    "myometrial_thickness_cm": {"type": "string", "required": True},
    "myometrial_invasion_percentage": {"type": "number", "required": True, "min": 0, "max": 100},
    "myometrial_invasion_category": {
        "type": "string",
        "required": True,
        "valid_values": ["no invasion", "<50%", ">=50%", "not reported"],
    },
    "lymphovascular_invasion": {
        "type": "string",
        "required": True,
        "valid_values": ["identified", "not identified", "not reported"],
    },
    "cervical_stromal_involvement": {
        "type": "string",
        "required": True,
        "valid_values": ["identified", "not identified", "not reported"],
    },
    "serosal_involvement": {
        "type": "string",
        "required": True,
        "valid_values": ["identified", "not identified", "not reported"],
    },
    "adnexal_involvement": {
        "type": "string",
        "required": True,
        "valid_values": ["identified", "not identified", "not reported"],
    },
    "margin_status": {
        "type": "string",
        "required": True,
        "valid_values": ["uninvolved", "involved", "not reported"],
    },
    "closest_margin_distance_cm": {"type": "string", "required": False},
    "closest_margin_location": {"type": "string", "required": False},
    "lymph_nodes_total_examined": {"type": "integer", "required": True, "min": 0},
    "lymph_nodes_total_positive": {"type": "integer", "required": True, "min": 0},
    "lymph_node_stations": {"type": "list", "required": False},
    "extracapsular_extension": {
        "type": "string",
        "required": False,
        "valid_values": ["present", "absent", "not reported"],
    },
    "tnm_pT": {"type": "string", "required": True},
    "tnm_pN": {"type": "string", "required": True},
    "tnm_pM": {"type": "string", "required": True},
    "figo_stage": {"type": "string", "required": True},
    "procedure_type": {"type": "string", "required": False},
    "specimen_integrity": {"type": "string", "required": False},
    "additional_findings": {"type": "list", "required": False},
}


EXTRACTION_PROMPT = """Read this pathology report and answer these specific questions. Write each answer on a new line in the format "ANSWER [number]: [answer]". If the information is not in the report, write "Not found". If the report explicitly says "not identified" or "negative", write "Not identified".

QUESTIONS:

1. What is the histologic type? (e.g., endometrioid adenocarcinoma, serous carcinoma, clear cell carcinoma)

2. What is the FIGO grade? (1, 2, or 3)

3. What is the nuclear grade? (1, 2, 3, or Not found)

4. What is the tumor size in centimeters?

5. What is the depth of myometrial invasion in centimeters?

6. What is the total myometrial thickness in centimeters?

7. Is lymphovascular invasion identified? (Answer: Identified, Not identified, or Not found)

8. Is there cervical stromal involvement? (Identified, Not identified, or Not found)

9. Is there serosal involvement? (Identified, Not identified, or Not found)

10. Is there adnexal involvement? (Identified, Not identified, or Not found)

11. What is the margin status? (Uninvolved, Involved, or Not found)

12. What is the distance to the closest margin in centimeters?

13. What is the location of the closest margin?

14. How many lymph nodes were examined total?

15. How many lymph nodes were positive?

16. Is there extracapsular extension? (Present, Absent, or Not found)

17. What is the TNM pT stage? (e.g., pT1a, pT1b, pT2, pT3a)

18. What is the TNM pN stage? (e.g., pN0, pN1, pNx)

19. What is the TNM pM stage? (e.g., pM0, pM1, pMx)

20. What is the FIGO stage? (e.g., IA, IB, II, IIIA, IIIC1, IVB)

21. What procedure was performed? (e.g., hysterectomy with bilateral salpingo-oophorectomy)

22. Was the specimen intact? (Yes, No, or Not found)

23. Any additional significant findings?

PATHOLOGY REPORT:
{report_text}

ANSWERS:"""

EXTRACTION_PROMPT_CONVERSATIONAL = """You are reviewing a surgical pathology report for endometrial carcinoma. Summarize the key pathologic findings in clear prose.

Address, when present in the report: histologic type and FIGO/nuclear grade; tumor size; depth and thickness of myometrial invasion; lymphovascular invasion; cervical, serosal, and adnexal involvement; margin status and distance to closest margin; lymph nodes examined and positive; extracapsular extension; pathologic TNM (pT, pN, pM) and overall stage; procedure performed; specimen integrity; and any notable additional findings.

If something is not stated in the report, say it is not reported rather than inventing values.

PATHOLOGY REPORT:
{report_text}

Clinical summary:"""

def _format_field_catalog() -> str:
    """Render EXTRACTION_FIELDS as a prompt-ready field list with types and allowed values."""
    lines: list[str] = []
    for name, spec in EXTRACTION_FIELDS.items():
        requirement = "required" if spec.get("required") else "optional"
        valid_values = spec.get("valid_values")
        constraint = f"; allowed: {valid_values}" if valid_values else ""
        lines.append(f'- "{name}" ({spec["type"]}, {requirement}{constraint})')
    return "\n".join(lines)


# Structured-JSON extraction prompt. Braces for the JSON shape are doubled so the single
# str.format(report_text=...) call in extract_report() leaves them literal. The field catalog
# is concatenated (not a format field) and contains no curly braces.
EXTRACTION_PROMPT_STRUCTURED_JSON = (
    """You are extracting structured data from a surgical pathology report for endometrial carcinoma. The report text may be noisy or OCR-garbled (scrambled word order, broken lines, run-on sentences). Reconstruct the intended clinical meaning, but never invent findings.

Return a SINGLE JSON object and nothing else. For EVERY field listed below, output an object of the form:
  {{"value": <value>, "confidence": <number between 0.0 and 1.0>, "status": "present|uncertain|missing", "evidence": "<verbatim phrase copied from the report>"}}

Set "status" for each field:
- "present": the report clearly states it. Give the value and a high confidence (>= 0.7). Copy the exact supporting phrase into "evidence".
- "uncertain": the report addresses it but the wording is ambiguous or garbled (e.g. "cannot be excluded", "difficult to assess", scrambled syntax). Give your best-effort value, a LOWER confidence (< 0.5), and put the ambiguous phrase in "evidence".
- "missing": the report does not mention it at all. Set "value" to "not reported" (or -1 for number/integer fields, [] for list fields), "confidence" to 0.0, and "evidence" to "". Never invent a value for a "missing" field.

Normalize values to the allowed forms shown in the catalog. Worked examples:
- "High grade with serous features" -> histologic_type "serous carcinoma", figo_grade "3" (status present).
- "invasion to a depth far less than one-fourth the myometrial thickness" -> myometrial_invasion_category "<50%" (status present, evidence = that phrase).
- "neither to absolutely exclude vascular/lymphatic invasion" -> lymphovascular_invasion value "not reported", status "uncertain", evidence = that phrase.
- "11 lymph nodes ... negative for carcinoma" -> lymph_nodes_total_examined 11, lymph_nodes_total_positive 0 (status present).

For "additional_findings", list at most 5 of the MOST clinically significant findings (e.g. coexisting carcinoma, high-risk features). Do NOT enumerate benign incidental findings (cysts, leiomyomas, metaplasia, adipose tissue, individual lymph-node descriptions). Keep the entire JSON compact so it is not truncated.

FIELDS (output every one):
"""
    + _format_field_catalog()
    + """

PATHOLOGY REPORT:
{report_text}

JSON:"""
)

REEXTRACTION_PROMPT = """A first extraction pass left some REQUIRED fields empty. Re-read the pathology report carefully and extract ONLY the fields listed below. The report may be OCR-garbled — reconstruct the intended meaning, but never invent findings.

For each field, output a JSON object of the form:
  {{"value": <value>, "confidence": <number between 0.0 and 1.0>, "status": "present|uncertain|missing", "evidence": "<verbatim phrase copied from the report>"}}

If a field is genuinely absent from the report, return status "missing" with value "not reported" (or -1 for number/integer, [] for list) and confidence 0.0. Do not fabricate a value to satisfy the request.

FIELDS TO RE-EXTRACT:
{fields}

PATHOLOGY REPORT:
{report_text}

JSON:"""

CORRECTION_PROMPT = """You are an expert clinical data verifier, acting as the first human-in-the-loop (Expert 1) from a pathology AI pipeline. Your task is to compare a structured extraction against the original pathology report and correct any errors, especially the confusion between "not reported" and explicit negative findings.

**Instructions (based on clinical guidelines for endometrial cancer pathology):**

1. For each of the following fields, search the original report for explicit statements:
   - lymphovascular_invasion
   - cervical_stromal_involvement
   - serosal_involvement
   - adnexal_involvement
   - extracapsular_extension
   - margin_status
   - myometrial_invasion_category
   - figo_grade

2. If the extraction says "not reported" but the report contains a clear negative phrase (e.g., "not identified", "no evidence", "absent", "negative", "none", "not seen", "no involvement"), change it to the appropriate negative value:
   - For lymphovascular_invasion, cervical_stromal_involvement, serosal_involvement, adnexal_involvement → use "not identified"
   - For extracapsular_extension → use "absent"
   - For margin_status → if negative, use "uninvolved"

3. If the extraction says "identified" or "present" but the report says "not identified" or "absent", correct it to the negative value.

4. If the extraction says "not reported" and the report does not mention the field at all, keep "not reported".

5. For numeric fields (e.g., lymph_nodes_total_examined, myometrial_invasion_percentage), verify against the report. If the extraction has -1 but the report has a value, update it.

6. Output ONLY a JSON object containing **only the fields that you changed** (not the entire extraction). Each changed field should have its corrected value. If no changes are needed, output an empty JSON object: {{}}

**Validation issues:**
{errors}

**Original report text:**
{report_text}

**Current extraction (JSON):**
{extraction_json}

**Corrected fields (JSON only, no extra text):**"""

_NEGATIVE_FINDING_PATTERNS = (
    "not identified",
    "not seen",
    "not present",
    "negative",
    "absent",
    "no evidence",
    "none identified",
)

_POSITIVE_FINDING_PATTERNS = (
    "identified",
    "present",
    "positive",
    "involved",
)

_CATEGORICAL_FIELDS = {
    "lymphovascular_invasion",
    "cervical_stromal_involvement",
    "serosal_involvement",
    "adnexal_involvement",
    "margin_status",
    "extracapsular_extension",
    "myometrial_invasion_category",
    "figo_grade",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _canonicalize_margin_status(value: str) -> str:
    normalized = _normalize_text(value)
    if _contains_any(
        normalized,
        (
            "uninvolved",
            "margin negative",
            "negative margin",
            "free of tumor",
            "free of carcinoma",
            "clear margin",
            "margins clear",
        ),
    ):
        return "uninvolved"
    if _contains_any(
        normalized,
        (
            "involved",
            "positive margin",
            "margin positive",
            "tumor at margin",
        ),
    ):
        return "involved"
    return value


def _canonicalize_identified_field(value: str) -> str:
    normalized = _normalize_text(value)
    if _contains_any(normalized, _NEGATIVE_FINDING_PATTERNS):
        return "not identified"
    if _contains_any(normalized, _POSITIVE_FINDING_PATTERNS):
        return "identified"
    return value


def _canonicalize_extracapsular_extension(value: str) -> str:
    normalized = _normalize_text(value)
    if _contains_any(normalized, _NEGATIVE_FINDING_PATTERNS):
        return "absent"
    if _contains_any(normalized, _POSITIVE_FINDING_PATTERNS):
        return "present"
    return value


def _canonicalize_myometrial_invasion_category(value: str) -> str:
    normalized = _normalize_text(value)
    # Order matters: the >=50% / <50% patterns are checked before "no invasion" so the "0%"
    # zero-invasion cue cannot be triggered by the "0%" substring inside "50%". The zero cue is
    # also matched with a digit-boundary regex so only a standalone "0%" counts.
    if _contains_any(
        normalized,
        (
            ">=50%",
            ">50%",
            "50% or more",
            "more than 50%",
            "one-half or more",
            "more than one-half",
        ),
    ):
        return ">=50%"
    if _contains_any(
        normalized,
        (
            "<50%",
            "less than 50%",
            "less than one-half",
            "less than half",
            "invades less than one-half",
        ),
    ):
        return "<50%"
    if _contains_any(
        normalized,
        (
            "no invasion",
            "without invasion",
            "not identified",
            "not present",
        ),
    ) or re.search(r"(?<!\d)0\s*%", normalized):
        return "no invasion"
    return value


def _canonicalize_figo_grade(value: str) -> str:
    normalized = _normalize_text(value)
    if normalized in {"1", "2", "3", "not reported"}:
        return normalized
    match = re.search(r"(?:figo\s+)?grade\s*([123])\b", normalized)
    if match:
        return match.group(1)
    if normalized.startswith("grade "):
        pieces = normalized.split()
        if len(pieces) >= 2 and pieces[1] in {"1", "2", "3"}:
            return pieces[1]
    return value


def canonicalize_extraction(data: dict) -> tuple[dict, list[str]]:
    canonicalized = dict(data)
    normalizations: list[str] = []
    for field_name in _CATEGORICAL_FIELDS:
        value = canonicalized.get(field_name)
        if not isinstance(value, str) or not value.strip():
            continue
        updated_value = value
        if field_name == "margin_status":
            updated_value = _canonicalize_margin_status(value)
        elif field_name in {
            "lymphovascular_invasion",
            "cervical_stromal_involvement",
            "serosal_involvement",
            "adnexal_involvement",
        }:
            updated_value = _canonicalize_identified_field(value)
        elif field_name == "extracapsular_extension":
            updated_value = _canonicalize_extracapsular_extension(value)
        elif field_name == "myometrial_invasion_category":
            updated_value = _canonicalize_myometrial_invasion_category(value)
        elif field_name == "figo_grade":
            updated_value = _canonicalize_figo_grade(value)
        canonicalized[field_name] = updated_value
        if updated_value != value:
            normalizations.append(f"{field_name}: {value!r} -> {updated_value!r}")
    return canonicalized, normalizations


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Required fields absent from the report. Tracked separately from `errors` because absence
    # is expected for narrative / sparse reports and must NOT mark the extraction invalid — only
    # genuine data-quality problems (bad values, out-of-range, contradictions) do that.
    missing_required: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"Valid: {self.is_valid}"]
        if self.errors:
            parts.append(f"Errors ({len(self.errors)}): " + "; ".join(self.errors))
        if self.missing_required:
            parts.append(
                f"Missing required ({len(self.missing_required)}): "
                + ", ".join(self.missing_required)
            )
        if self.warnings:
            parts.append(f"Warnings ({len(self.warnings)}): " + "; ".join(self.warnings))
        return " | ".join(parts)


def validate_extraction(data: dict) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    missing_required: list[str] = []

    for field_name, spec in EXTRACTION_FIELDS.items():
        value = data.get(field_name)

        # A required field absent from the report is incompleteness, not an error: many reports
        # (older narrative dictations, OCR'd scans) do not follow the full synoptic structure.
        # It is recorded for a targeted recovery pass and surfaced to downstream reasoning, but
        # never fails validation — only malformed values below do.
        if spec.get("required") and _is_absent_value(field_name, value):
            missing_required.append(field_name)
            continue

        if value is None or value == "" or value == "not reported":
            continue

        if spec["type"] == "number" and isinstance(value, (int, float)) and value != -1:
            lo = spec.get("min")
            hi = spec.get("max")
            if lo is not None and value < lo:
                errors.append(f"{field_name}={value} below minimum {lo}")
            if hi is not None and value > hi:
                errors.append(f"{field_name}={value} above maximum {hi}")

        if spec["type"] == "integer" and isinstance(value, int) and value != -1:
            lo = spec.get("min")
            if lo is not None and value < lo:
                errors.append(f"{field_name}={value} below minimum {lo}")

        valid_values = spec.get("valid_values")
        if valid_values and isinstance(value, str):
            if value.lower() not in [v.lower() for v in valid_values]:
                errors.append(f"{field_name}='{value}' not in allowed values: {valid_values}")

    _validate_cross_field_consistency(data, errors, warnings)

    if missing_required:
        warnings.append(
            "Required fields not found in report (best-effort recovery attempted): "
            + ", ".join(missing_required)
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        missing_required=missing_required,
    )


def _validate_cross_field_consistency(data: dict, errors: list[str], warnings: list[str]) -> None:
    depth_str = data.get("myometrial_invasion_depth_cm", "not reported")
    thickness_str = data.get("myometrial_thickness_cm", "not reported")
    pct = data.get("myometrial_invasion_percentage", -1)
    category = data.get("myometrial_invasion_category", "not reported")

    depth = _try_parse_float(depth_str)
    thickness = _try_parse_float(thickness_str)

    if depth is not None and thickness is not None and thickness > 0:
        expected_pct = round(depth / thickness * 100, 1)
        if isinstance(pct, (int, float)) and pct >= 0:
            if abs(pct - expected_pct) > 10:
                warnings.append(
                    f"Invasion percentage {pct}% does not match "
                    f"depth/thickness calculation ({expected_pct}%)"
                )

        if category != "not reported":
            expected_cat = "<50%" if expected_pct < 50 else ">=50%"
            if depth == 0:
                expected_cat = "no invasion"
            if category.lower() != expected_cat.lower():
                warnings.append(
                    f"Invasion category '{category}' may conflict with "
                    f"calculated {expected_pct}% (expected '{expected_cat}')"
                )

    examined = data.get("lymph_nodes_total_examined", -1)
    positive = data.get("lymph_nodes_total_positive", -1)
    if (
        isinstance(examined, int)
        and isinstance(positive, int)
        and examined >= 0
        and positive >= 0
        and positive > examined
    ):
        errors.append(f"Positive lymph nodes ({positive}) exceeds total examined ({examined})")

    stations = data.get("lymph_node_stations", [])
    if isinstance(stations, list) and isinstance(examined, int) and examined >= 0:
        station_total = sum(s.get("examined", 0) for s in stations if isinstance(s, dict))
        if station_total > 0 and station_total != examined:
            warnings.append(
                f"Station-level examined total ({station_total}) != reported total ({examined})"
            )


def _try_parse_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.match(r"^[\d.]+", value.strip())
        if m:
            try:
                return float(m.group())
            except ValueError:
                pass
    return None


def _create_medgemma_client(
    model: str | None = None,
    base_url: str | None = None,
    options: dict | None = None,
) -> OllamaClient:
    # In the default JSON extraction mode, constrain decoding to valid JSON so malformed values
    # (e.g. an invalid FIGO stage like "IIIII") cannot escape the model. The legacy ANSWER and
    # conversational modes are free-text, so the constraint must not be applied to them.
    ollama_format = "json" if _EXTRACTION_PROMPT_MODE not in ("structured", "answer", "legacy", "conversational") else None
    return OllamaClient(
        model_name=model or MEDGEMMA_MODEL,
        base_url=base_url or MEDGEMMA_URL,
        timeout=1800,
        ollama_options=options or MEDGEMMA_OPTIONS,
        ollama_format=ollama_format,
    )


_ANSWER_LINE_PATTERN = re.compile(r"ANSWER\s*(\d+):\s*(.+)", re.IGNORECASE)


def _response_has_answer_lines(text: str) -> bool:
    return bool(_ANSWER_LINE_PATTERN.search(text))


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from model response, tolerating markdown fences and preamble."""
    cleaned = text.strip()

    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    think_pattern = re.compile(
        r"<think>[\s\S]*?</think>|<thinking>[\s\S]*?</thinking>",
        re.IGNORECASE,
    )
    cleaned = think_pattern.sub("", cleaned).strip()

    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        cleaned = cleaned[brace_start : brace_end + 1]
    elif brace_start >= 0:
        # Opening brace but no close — likely truncated mid-object (e.g. output token limit hit).
        cleaned = cleaned[brace_start:]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(cleaned)
        if repaired is not None:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None
        return None


def _repair_truncated_json(text: str) -> str | None:
    """Best-effort repair of JSON truncated mid-value (e.g. when the model hit its token cap).

    Drops the trailing incomplete element and closes any open brackets/strings. Returns the
    repaired string, or ``None`` if there is nothing salvageable. This recovers the fields that
    were emitted before truncation rather than discarding the whole extraction.
    """
    # Find the furthest position that ends a complete element (a closed string/bracket), so the
    # dangling partial token after it can be dropped.
    in_str = False
    escape = False
    last_complete = -1
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
                last_complete = i  # index of the closing quote
            continue
        if ch == '"':
            in_str = True
        elif ch in "}]":
            last_complete = i

    if last_complete < 0:
        return None

    prefix = text[: last_complete + 1].rstrip()
    prefix = prefix.rstrip(",").rstrip()

    # Recompute unclosed brackets over the salvaged prefix and append their closers.
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in prefix:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    closers = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return prefix + closers


def _coerce_confidence(raw) -> float:
    """Coerce a model-provided confidence into a float clamped to [0.0, 1.0].

    Accepts numbers, numeric strings, and categorical words (high/medium/low). Falls back to
    0.5 (neutral) when nothing usable is provided.
    """
    if isinstance(raw, bool):
        return 0.5
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    if isinstance(raw, str):
        text = raw.strip().lower()
        try:
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            if text in _CONFIDENCE_WORD_MAP:
                return _CONFIDENCE_WORD_MAP[text]
            match = re.search(r"\d?\.\d+|[01]", text)
            if match:
                try:
                    return max(0.0, min(1.0, float(match.group())))
                except ValueError:
                    pass
    return 0.5


def _derive_field_status(
    field_name: str,
    value,
    confidence: float,
    evidence: str,
    explicit: str | None,
) -> str:
    """Resolve a field's provenance status, honoring an explicit model status when valid.

    missing  -> no usable value (genuinely absent from the report)
    uncertain-> value present but ambiguous (low confidence or equivocal evidence phrasing)
    present  -> value present and asserted with adequate confidence
    """
    if isinstance(explicit, str) and explicit.strip().lower() in _VALID_FIELD_STATUSES:
        return explicit.strip().lower()
    if value is None or not _is_nondefault_field_value(field_name, value):
        return STATUS_MISSING
    evidence_text = (evidence or "").lower()
    if confidence < MEDGEMMA_CONFIDENCE_THRESHOLD or any(
        phrase in evidence_text for phrase in _UNCERTAIN_EVIDENCE_PHRASES
    ):
        return STATUS_UNCERTAIN
    return STATUS_PRESENT


def _coerce_field_value(field_name: str, value):
    """Coerce a model-supplied value to its declared numeric type when feasible.

    Models occasionally emit integer/number fields as strings (e.g. ``"11"`` lymph nodes). Left
    as strings these silently fail the ``isinstance(int)`` checks in staging (``map_nodes``),
    dropping data. This pulls the leading number out of a numeric string so any report shape maps
    cleanly. Non-numeric strings (``"not reported"``) and already-correct types pass through.
    """
    spec = EXTRACTION_FIELDS.get(field_name)
    if not spec or not isinstance(value, str):
        return value
    field_type = spec["type"]
    if field_type not in ("integer", "number"):
        return value
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return value
    number = float(match.group(0))
    if field_type == "integer":
        return int(number)
    return number


def _parse_structured_json_response(
    text: str,
) -> tuple[dict, dict[str, float], dict[str, str], dict[str, str]] | None:
    """Parse the structured-JSON extraction shape.

    Accepts both the nested ``{field: {value, confidence, status, evidence}}`` shape and the
    flat ``{field: value}`` legacy shape. Returns ``(data, confidence, status, evidence)`` or
    ``None`` if the response is not JSON containing at least one known extraction field.
    """
    parsed = _parse_json_response(text)
    if not isinstance(parsed, dict):
        return None
    if not any(key in EXTRACTION_FIELDS for key in parsed):
        return None

    data: dict = {}
    confidence: dict[str, float] = {}
    status: dict[str, str] = {}
    evidence: dict[str, str] = {}

    for key in EXTRACTION_FIELDS:
        if key not in parsed:
            continue
        raw = parsed[key]
        if isinstance(raw, dict) and ("value" in raw or "status" in raw):
            value = raw.get("value")
            conf = _coerce_confidence(raw.get("confidence"))
            ev = str(raw.get("evidence") or "")
            explicit = raw.get("status")
        else:
            value = raw
            conf = _coerce_confidence(None)
            ev = ""
            explicit = None

        value = _coerce_field_value(key, value)
        if value is not None:
            data[key] = value
        field_status = _derive_field_status(key, value, conf, ev, explicit)
        confidence[key] = conf
        status[key] = field_status
        if ev:
            evidence[key] = ev

    return data, confidence, status, evidence


# Topic keywords used to locate a field's discussion in the source report, for the equivocal
# backstop. When the model extracts a confident negative for one of these but the report hedges
# nearby, we downgrade the field to "uncertain" rather than trusting the negative.
_EQUIVOCAL_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lymphovascular_invasion": (
        "lymphovascular",
        "lymph-vascular",
        "lymphvascular",
        "vascular/lymphatic",
        "lymphatic/vascular",
        "lymphatic invasion",
        "vascular invasion",
        "lvsi",
        "lvi",
    ),
    "cervical_stromal_involvement": (
        "cervical strom",
        "endocervical",
        "cervix",
        "cervical involvement",
    ),
    "serosal_involvement": ("serosa", "serosal"),
    "adnexal_involvement": ("adnexa", "adnexal", "ovary", "ovarian", "fallopian", "tube"),
    "margin_status": ("margin",),
}

# Hedging phrases that, when found near a field's topic, mark the finding as not assertible.
_EQUIVOCAL_REPORT_PHRASES = (
    "exclud",  # "cannot exclude", "neither to absolutely exclude", "excluded"
    "difficult",
    "cannot be assess",
    "cannot assess",
    "cannot be determined",
    "cannot rule out",
    "rule out",
    "not fully evaluat",
    "equivocal",
    "indeterminate",
    "suspicious",
    "suggestive",
    "borderline",
    "questionable",
)

_POSITIVE_VALUE_TOKENS = {"identified", "present", "positive", "involved"}


def _apply_equivocal_backstop(
    data: dict,
    field_status: dict[str, str],
    field_evidence: dict[str, str],
    report_text: str,
) -> None:
    """Flag confidently-negative fields as uncertain when the report actually hedges.

    Non-instruction-tuned extractors (e.g. MedGemma) tend to flatten ambiguous findings into a
    confident negative ("not identified") instead of signaling uncertainty. This deterministic
    backstop re-reads the source: if a field was extracted as negative/absent yet the report
    discusses that topic with hedging language nearby, the field is downgraded to ``uncertain``
    and the supporting phrase captured. It only ever downgrades certainty (fails safe), and is
    skipped when the field is already a positive finding or already uncertain.
    """
    if not report_text:
        return
    low = report_text.lower()
    for field, keywords in _EQUIVOCAL_TOPIC_KEYWORDS.items():
        if field_status.get(field) == STATUS_UNCERTAIN:
            continue
        value = data.get(field)
        normalized = value.strip().lower() if isinstance(value, str) else ""
        if normalized in _POSITIVE_VALUE_TOKENS:
            continue  # a positive finding is not a false-negative risk

        flagged = False
        for keyword in keywords:
            idx = low.find(keyword)
            while idx != -1:
                window = low[max(0, idx - 160) : idx + len(keyword) + 160]
                if any(phrase in window for phrase in _EQUIVOCAL_REPORT_PHRASES):
                    snippet = report_text[max(0, idx - 60) : idx + len(keyword) + 90]
                    field_status[field] = STATUS_UNCERTAIN
                    field_evidence[field] = " ".join(snippet.split())
                    flagged = True
                    break
                idx = low.find(keyword, idx + 1)
            if flagged:
                break


def _is_nondefault_field_value(field_name: str, value) -> bool:
    spec = EXTRACTION_FIELDS[field_name]
    t = spec["type"]
    if t == "list":
        return bool(value)
    if t in ("number", "integer"):
        return value != -1
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() != "not reported"


def _is_absent_value(field_name: str, value) -> bool:
    """True when a field carries no real value (None, empty, "not reported", -1, [])."""
    return not _is_nondefault_field_value(field_name, value)


def _merge_nondefault_extraction(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if k in EXTRACTION_FIELDS and _is_nondefault_field_value(k, v):
            out[k] = v
    return out


def _extraction_has_signal(data: dict) -> bool:
    return (
        sum(1 for k in EXTRACTION_FIELDS if _is_nondefault_field_value(k, data.get(k))) >= 4
    )


def _parse_freetext_response(text: str) -> dict:
    result = _empty_extraction()
    if not text or not text.strip():
        return result

    blob = text
    low = blob.lower()
    m = re.search(
        r"invasive\s+endometrial\s+carcinoma,\s*([^\n,]+?)(?:,\s*figo|\n|$)",
        blob,
        re.IGNORECASE,
    )
    if m:
        fragment = m.group(1).strip().rstrip(".")
        if "endometrioid" in fragment.lower():
            result["histologic_type"] = "endometrioid adenocarcinoma"
        else:
            result["histologic_type"] = fragment
    elif re.search(r"endometrioid\s+(?:adenocarcinoma|carcinoma)", low):
        result["histologic_type"] = "endometrioid adenocarcinoma"
    elif re.search(r"serous\s+carcinoma", low):
        result["histologic_type"] = "serous carcinoma"
    elif re.search(r"clear\s+cell", low):
        result["histologic_type"] = "clear cell carcinoma"
    m = re.search(r"figo\s*grade\s*[:\s]*([123])\b", low)
    if m:
        result["figo_grade"] = m.group(1)
    m = re.search(r"grade\s+([123])\s*\(", low)
    if m and result["figo_grade"] == "not reported":
        result["figo_grade"] = m.group(1)
    m = re.search(r"nuclear\s*grade\s*[:\s]*([123])\b", low)
    if m:
        result["nuclear_grade"] = m.group(1)
    m = re.search(
        r"greatest\s+dimension\s*[:\s]*(\d+\.?\d*)\s*cm",
        low,
    )
    if not m:
        m = re.search(r"tumor\s+size\s*[:\s]*[^\d\n]*(\d+\.?\d*)\s*cm", low)
    if m:
        result["tumor_size_cm"] = m.group(1)

    m = re.search(r"depth\s+of\s+invasion\s*[:\s]*(\d+\.?\d*)\s*cm", low)
    if m:
        result["myometrial_invasion_depth_cm"] = m.group(1)
    m = re.search(r"myometrial\s+thickness\s*[:\s]*(\d+\.?\d*)\s*cm", low)
    if not m:
        m = re.search(r"average\s+myometrial\s+thickness\s+of\s+(\d+\.?\d*)\s*cm", low)
    if m:
        result["myometrial_thickness_cm"] = m.group(1)

    window = re.search(
        r"lymph[-\s]?vascular[^\n]{0,120}",
        low,
    )
    if window:
        seg = window.group(0)
        if "not identified" in seg or "no " in seg:
            result["lymphovascular_invasion"] = "not identified"
        elif "identified" in seg or "present" in seg:
            result["lymphovascular_invasion"] = "identified"

    m = re.search(r"endocervical\s+involvement\s*[:\s]*([^\n.]+)", low)
    if m:
        val = m.group(1).strip()
        if "not identified" in val or "negative" in val:
            result["cervical_stromal_involvement"] = "not identified"
        elif "identified" in val or "involved" in val:
            result["cervical_stromal_involvement"] = "identified"

    if re.search(r"serosal\s+involvement\s*[:\s]*not\s+identified", low):
        result["serosal_involvement"] = "not identified"
    if re.search(r"extent\s+of\s+involvement\s+of\s+other\s+organs\s*[:\s]*none", low):
        result["serosal_involvement"] = "not identified"
        result["adnexal_involvement"] = "not identified"
    if re.search(r"adnexal\s+involvement", low):
        if "not identified" in low or "none" in low:
            result["adnexal_involvement"] = "not identified"

    if re.search(r"uninvolved\s+by\s+invasive\s+carcinoma", low) or re.search(
        r"margins?\s*[:\s]*uninvolved",
        low,
    ):
        result["margin_status"] = "uninvolved"
    elif re.search(r"margin[s]?\s*[:\s]*involved", low) or "positive margin" in low:
        result["margin_status"] = "involved"

    m = re.search(
        r"distance\s+of\s+invasive\s+carcinoma\s+from\s+closest\s+margin\s*[:\s]*(\d+\.?\d*)\s*cm",
        low,
    )
    if not m:
        m = re.search(r"closest\s+margin\s*[:\s]*[^\d\n]*(\d+\.?\d*)\s*cm", low)
    if m:
        result["closest_margin_distance_cm"] = m.group(1)

    m = re.search(r"specify\s+margin\s*[:\s]*([^\n.]+)", low)
    if m:
        result["closest_margin_location"] = m.group(1).strip().title()

    # Lymph nodes — prefer summary line "Lymph Nodes: (0/28)"; else take (a/b) with largest b
    m = re.search(r"lymph\s*nodes?\s*:\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)", low)
    if m:
        pos, tot = int(m.group(1)), int(m.group(2))
        result["lymph_nodes_total_positive"] = pos
        result["lymph_nodes_total_examined"] = tot
    else:
        pairs = re.findall(r"\(\s*(\d+)\s*/\s*(\d+)\s*\)", low)
        if pairs:
            best = max(pairs, key=lambda t: int(t[1]))
            result["lymph_nodes_total_positive"] = int(best[0])
            result["lymph_nodes_total_examined"] = int(best[1])
        else:
            m = re.search(r"number\s+examined\s*[:\s]*(\d+)", low)
            if m:
                result["lymph_nodes_total_examined"] = int(m.group(1))
            m = re.search(r"number\s+involved\s*[:\s]*(\d+)", low)
            if m:
                result["lymph_nodes_total_positive"] = int(m.group(1))

    if re.search(r"extracapsular\s+extension", low):
        if re.search(r"extracapsular[^\n]{0,80}absent", low) or re.search(
            r"extracapsular[^\n]{0,80}not\s+identified",
            low,
        ):
            result["extracapsular_extension"] = "absent"
        elif re.search(r"extracapsular[^\n]{0,80}present", low):
            result["extracapsular_extension"] = "present"

    m = re.search(r"\b(p[tT]\d[a-z]?)\b", blob)
    if m:
        g = m.group(1)
        rest = g[1:]
        result["tnm_pT"] = "p" + rest[0].upper() + rest[1:].lower()

    m = re.search(r"\b(p[nN][012oxOX])\b", blob)
    if m:
        suf = m.group(1)[2:].upper()
        if suf == "O":
            suf = "0"
        result["tnm_pN"] = "pN" + suf
    elif re.search(r"p[tT]\d[a-z]?\s*,\s*no\s*,", low):
        result["tnm_pN"] = "pN0"
    elif re.search(r"no\s+regional\s+lymph\s+node\s+metastasis", low):
        result["tnm_pN"] = "pN0"

    m = re.search(r"\b(p[mM][01xX])\b", blob)
    if m:
        g = m.group(1)
        result["tnm_pM"] = "p" + g[1:].upper() if g[0] in "pP" else g
    if re.search(r"m\s*\(\s*not\s+applicable", low) or re.search(
        r"m\(not\s+applicable\)",
        low,
    ):
        result["tnm_pM"] = "pMx"

    m = re.search(
        r"\b(IA|IB|IC|IIA|IIB|IIIA|IIIB|IIIC\d?|IVA|IVB)\b",
        blob,
        re.IGNORECASE,
    )
    if m:
        result["figo_stage"] = m.group(1).upper()
    elif re.search(r"\bp[tT]1a\b", blob) and "endometrial" in low:
        result["figo_stage"] = "IA"

    if re.search(r"hysterectomy", low):
        if "salpingo" in low or "oophorectomy" in low:
            result["procedure_type"] = "hysterectomy with bilateral salpingo-oophorectomy"
        else:
            result["procedure_type"] = "hysterectomy"
    if re.search(r"specimen\s+integrity\s*[:\s]*intact", low):
        result["specimen_integrity"] = "intact"

    depth_f = _try_parse_float(result["myometrial_invasion_depth_cm"])
    thick_f = _try_parse_float(result["myometrial_thickness_cm"])
    if depth_f is not None and thick_f is not None and thick_f > 0:
        pct = (depth_f / thick_f) * 100
        result["myometrial_invasion_percentage"] = round(pct, 1)
        if depth_f == 0:
            result["myometrial_invasion_category"] = "no invasion"
        elif pct < 50:
            result["myometrial_invasion_category"] = "<50%"
        else:
            result["myometrial_invasion_category"] = ">=50%"

    return result


def _active_extraction_prompt() -> str:
    if _EXTRACTION_PROMPT_MODE in ("structured", "answer", "legacy"):
        return EXTRACTION_PROMPT
    if _EXTRACTION_PROMPT_MODE == "conversational":
        return EXTRACTION_PROMPT_CONVERSATIONAL
    return EXTRACTION_PROMPT_STRUCTURED_JSON


@dataclass
class ExtractionResult:
    data: dict
    validation: ValidationResult
    raw_response: str
    retries: int
    extraction_time: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    normalizations: list[str] = field(default_factory=list)
    field_confidence: dict[str, float] = field(default_factory=dict)
    field_status: dict[str, str] = field(default_factory=dict)
    field_evidence: dict[str, str] = field(default_factory=dict)

    @property
    def uncertain_fields(self) -> list[str]:
        """Fields the report addresses but ambiguously — must not be silently negated."""
        return [k for k, s in self.field_status.items() if s == STATUS_UNCERTAIN]

    @property
    def missing_fields(self) -> list[str]:
        """Fields genuinely absent from the report (closed-world negative inference applies)."""
        return [k for k, s in self.field_status.items() if s == STATUS_MISSING]

    @property
    def low_confidence_fields(self) -> list[str]:
        """Populated fields whose confidence is below the trust threshold (excludes missing)."""
        return [
            k
            for k, c in self.field_confidence.items()
            if c < MEDGEMMA_CONFIDENCE_THRESHOLD and self.field_status.get(k) != STATUS_MISSING
        ]

    def to_context_string(self) -> str:
        lines = ["STRUCTURED CLINICAL EXTRACTION (validated):"]
        if self.normalizations:
            lines.append("NORMALIZATIONS APPLIED:")
            for item in self.normalizations:
                lines.append(f"  - {item}")
            lines.append("")
        for key, value in self.data.items():
            if isinstance(value, list):
                if not value:
                    lines.append(f"  {key}: []")
                else:
                    lines.append(f"  {key}:")
                    for item in value:
                        if isinstance(item, dict):
                            parts = ", ".join(f"{k}={v}" for k, v in item.items())
                            lines.append(f"    - {parts}")
                        else:
                            lines.append(f"    - {item}")
            else:
                lines.append(f"  {key}: {value}")

        if self.field_confidence:
            lines.append("\nFIELD CONFIDENCE / PROVENANCE (score, status, evidence):")
            for key in self.data:
                if key not in self.field_confidence:
                    continue
                conf = self.field_confidence[key]
                status = self.field_status.get(key, STATUS_PRESENT)
                quote = self.field_evidence.get(key, "")
                suffix = f' — "{quote}"' if quote else ""
                lines.append(f"  - {key}: {conf:.2f} [{status}]{suffix}")

        uncertain = self.uncertain_fields
        low_conf = [k for k in self.low_confidence_fields if k not in uncertain]
        if uncertain or low_conf:
            lines.append("\nUNCERTAIN / LOW-CONFIDENCE FIELDS (verify before relying on them):")
            for key in uncertain:
                quote = self.field_evidence.get(key, "")
                detail = f': "{quote}"' if quote else ""
                lines.append(f"  - {key}: uncertain (addressed but ambiguous in report){detail}")
            for key in low_conf:
                conf = self.field_confidence.get(key, 0.0)
                lines.append(f"  - {key}: low confidence ({conf:.2f})")

        if self.validation.warnings:
            lines.append("\nVALIDATION WARNINGS:")
            for w in self.validation.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def extract_report(
    report_text: str,
    client: OllamaClient | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_retries: int = MAX_EXTRACTION_RETRIES,
) -> ExtractionResult:
    if client is None:
        client = _create_medgemma_client(model=model, base_url=base_url)

    t_start = time.perf_counter()
    prompt = _active_extraction_prompt().format(report_text=report_text)
    raw_response = client.completion(prompt)
    total_input = client.last_input_tokens
    total_output = client.last_output_tokens

    field_confidence: dict[str, float] = {}
    field_status: dict[str, str] = {}
    field_evidence: dict[str, str] = {}

    structured = _parse_structured_json_response(raw_response)
    json_parsed = _parse_json_response(raw_response)
    has_answers = _response_has_answer_lines(raw_response)
    if structured is not None:
        data, field_confidence, field_status, field_evidence = structured
    elif has_answers:
        data = parse_medgemma_answers(raw_response)
    elif json_parsed is not None:
        data = json_parsed
    else:
        ft_model = _parse_freetext_response(raw_response)
        ft_report = _parse_freetext_response(report_text)
        data = _merge_nondefault_extraction(ft_model, ft_report)

    retries = 0

    structured_ok = structured is not None or has_answers or json_parsed is not None
    if not structured_ok and not _extraction_has_signal(data):
        normalizations = []
        validation = ValidationResult(
            is_valid=False,
            errors=["Failed to parse extraction from model response"],
        )
    else:
        data, normalizations = canonicalize_extraction(data)
        validation = validate_extraction(data)

    # Best-effort recovery pass: re-read the report once for any required fields the first
    # extraction omitted. Their absence does NOT fail validation (reports vary in structure), so
    # this only fills fields the model overlooked — it never loops trying to satisfy a report
    # that genuinely lacks the data. A single targeted re-read is enough; a second identical one
    # would not surface fields that are not there.
    if structured_ok and validation.missing_required:
        retries += 1
        missing_required = list(validation.missing_required)
        reextraction = REEXTRACTION_PROMPT.format(
            fields="\n".join(f"- {name}" for name in missing_required),
            report_text=report_text,
        )
        raw_response = client.completion(reextraction)
        total_input += client.last_input_tokens
        total_output += client.last_output_tokens

        reparsed = _parse_structured_json_response(raw_response)
        if reparsed is not None:
            new_data, new_conf, new_status, new_evidence = reparsed
            data = _merge_nondefault_extraction(data, new_data)
            # Only adopt provenance for fields we actually asked to re-extract.
            for name in missing_required:
                if name in new_conf:
                    field_confidence[name] = new_conf[name]
                    field_status[name] = new_status.get(name, STATUS_MISSING)
                    if name in new_evidence:
                        field_evidence[name] = new_evidence[name]
        else:
            flat = _parse_json_response(raw_response)
            if flat is not None:
                data = _merge_nondefault_extraction(data, flat)

        data, normalizations = canonicalize_extraction(data)
        validation = validate_extraction(data)

    # Correction passes for genuine data-quality problems (malformed values, contradictions).
    # Missing fields are not errors, so a sparse-but-consistent report skips this entirely.
    while not validation.is_valid and retries < max_retries:
        retries += 1
        correction = CORRECTION_PROMPT.format(
            errors="\n".join(f"- {e}" for e in validation.errors),
            extraction_json=json.dumps(data, indent=2),
            report_text=report_text,
        )
        raw_response = client.completion(correction)
        total_input += client.last_input_tokens
        total_output += client.last_output_tokens

        corrected = _parse_json_response(raw_response)
        if corrected is not None:
            data = corrected

        data, normalizations = canonicalize_extraction(data)
        validation = validate_extraction(data)

    # Deterministic safety net: re-read the source for hedging the extractor may have flattened
    # into a confident negative, and downgrade those fields to uncertain.
    _apply_equivocal_backstop(data, field_status, field_evidence, report_text)

    t_end = time.perf_counter()
    return ExtractionResult(
        data=data,
        validation=validation,
        raw_response=raw_response,
        retries=retries,
        extraction_time=t_end - t_start,
        model=client.model_name,
        input_tokens=total_input,
        output_tokens=total_output,
        normalizations=normalizations,
        field_confidence=field_confidence,
        field_status=field_status,
        field_evidence=field_evidence,
    )


def _empty_extraction() -> dict:
    result = {}
    for field_name, spec in EXTRACTION_FIELDS.items():
        if spec["type"] == "list":
            result[field_name] = []
        elif spec["type"] in ("number", "integer"):
            result[field_name] = -1
        else:
            result[field_name] = "not reported"
    return result


_MEDGEMMA_ANSWER_KEY_BY_NUM: dict[int, str] = {
    1: "histologic_type",
    2: "figo_grade",
    3: "nuclear_grade",
    4: "tumor_size_cm",
    5: "myometrial_invasion_depth_cm",
    6: "myometrial_thickness_cm",
    7: "lymphovascular_invasion",
    8: "cervical_stromal_involvement",
    9: "serosal_involvement",
    10: "adnexal_involvement",
    11: "margin_status",
    12: "closest_margin_distance_cm",
    13: "closest_margin_location",
    14: "lymph_nodes_total_examined",
    15: "lymph_nodes_total_positive",
    16: "extracapsular_extension",
    17: "tnm_pT",
    18: "tnm_pN",
    19: "tnm_pM",
    20: "figo_stage",
    21: "procedure_type",
    22: "specimen_integrity",
    23: "additional_findings",
}


def parse_medgemma_answers(response_text: str) -> dict:
    """Convert MedGemma's numbered ANSWER lines to an extraction dict."""
    result = _empty_extraction()

    for line in response_text.strip().split("\n"):
        line = line.strip()
        match = _ANSWER_LINE_PATTERN.match(line)
        if not match:
            continue
        num = int(match.group(1))
        answer = match.group(2).strip()
        if answer.lower() == "not found":
            continue

        key = _MEDGEMMA_ANSWER_KEY_BY_NUM.get(num)
        if not key:
            continue

        if key in ("lymph_nodes_total_examined", "lymph_nodes_total_positive"):
            try:
                result[key] = int(answer)
            except ValueError:
                pass
        elif key == "additional_findings":
            if answer and answer.lower() != "none":
                result[key] = [answer]
        else:
            result[key] = answer

    depth_str = result.get("myometrial_invasion_depth_cm", "not reported")
    thickness_str = result.get("myometrial_thickness_cm", "not reported")
    depth = _try_parse_float(depth_str)
    thickness = _try_parse_float(thickness_str)
    if depth is not None and thickness is not None and thickness > 0:
        pct = (depth / thickness) * 100
        result["myometrial_invasion_percentage"] = round(pct, 1)
        result["myometrial_invasion_category"] = "<50%" if pct < 50 else ">=50%"
        if depth == 0:
            result["myometrial_invasion_category"] = "no invasion"

    return result


app = typer.Typer(help=" Extract structured clinical data from pathology reports using MedGemma.")
console = Console()


@app.command()
def run(
    reports: list[Path] = typer.Argument(..., help="Path(s) to pathology report text files"),
    model: str = typer.Option(MEDGEMMA_MODEL, help="Ollama model name"),
    url: str = typer.Option(MEDGEMMA_URL, help="Ollama base URL"),
    output_dir: Path = typer.Option(None, help="Directory for JSON output files"),
    retries: int = typer.Option(MAX_EXTRACTION_RETRIES, help="Max retry attempts"),
):
    """Run extraction on pathology reports"""

    console.print(Panel.fit(" Starting Extraction", style="bold green"))

    client = _create_medgemma_client(model=model, base_url=url)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for report_path in track(reports, description="Processing reports..."):
        if not report_path.exists():
            console.print(f"[red]Skipping missing file:[/red] {report_path}")
            continue

        report_text = report_path.read_text(encoding="utf-8")

        if not report_text.strip():
            console.print(f"[yellow]Skipping empty file:[/yellow] {report_path}")
            continue

        console.print(f"\n[bold cyan]Extracting:[/bold cyan] {report_path.name}")

        result = extract_report(report_text, client=client, max_retries=retries)

        status_color = "green" if result.validation.is_valid else "red"
        status = "PASS" if result.validation.is_valid else "FAIL"

        table = Table(show_header=False)
        table.add_row("Status", f"[{status_color}]{status}[/{status_color}]")
        table.add_row("Retries", str(result.retries))
        table.add_row("Time (s)", f"{result.extraction_time:.1f}")

        console.print(table)

        for err in result.validation.errors:
            console.print(f"[red]ERROR:[/red] {err}")

        for warn in result.validation.warnings:
            console.print(f"[yellow]WARN:[/yellow] {warn}")

        if output_dir:
            out_path = output_dir / f"{report_path.stem}_extraction.json"

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": str(report_path),
                        "model": result.model,
                        "extraction_time": result.extraction_time,
                        "retries": result.retries,
                        "validation": {
                            "is_valid": result.validation.is_valid,
                            "errors": result.validation.errors,
                            "warnings": result.validation.warnings,
                        },
                        "data": result.data,
                    },
                    f,
                    indent=2,
                )

            console.print(f"[green]Saved:[/green] {out_path}")
        else:
            console.print_json(data=result.data)


if __name__ == "__main__":
    app()