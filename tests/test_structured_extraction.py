"""Tests for the structured-JSON extraction path: per-field value/confidence/status/evidence,
confidence coercion, missing-vs-uncertain status, and the status-keyed retry helpers."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

from vlmextraction import (  # noqa: E402
    EXTRACTION_PROMPT_STRUCTURED_JSON,
    MEDGEMMA_CONFIDENCE_THRESHOLD,
    ExtractionResult,
    ValidationResult,
    _apply_equivocal_backstop,
    _apply_nodal_station_detection,
    _apply_peritoneal_metastasis_backstop,
    _canonicalize_myometrial_invasion_category,
    _coerce_confidence,
    _derive_field_status,
    _parse_json_response,
    _parse_structured_json_response,
    canonicalize_extraction,
    detect_nodal_stations,
    validate_extraction,
)


def test_detect_nodal_stations_para_aortic_positive() -> None:
    # Para-aortic verdict via explicit count; pelvic verdict via prose; one negative para station.
    report = (
        "C. RIGHT PARAAORTIC LYMPH NODES:\n"
        "- Two lymph nodes positive for metastatic adenocarcinoma (2/2)\n"
        "D. RIGHT PELVIC LYMPH NODE:\n"
        "- Metastatic carcinoma in one lymph node\n"
        "E. LEFT PARA-AORTIC LYMPH NODE:\n"
        "- No tumor identified (0/3)\n"
    )
    stations = detect_nodal_stations(report)
    groups = {(s["group"], s["positive"]) for s in stations}
    assert ("para_aortic", True) in groups
    assert ("pelvic", True) in groups
    assert ("para_aortic", False) in groups


def test_detect_nodal_stations_ignores_negated_metastatic() -> None:
    # "No metastatic carcinoma identified" must not register as a positive node.
    report = (
        "A. LEFT PERIAORTIC LYMPH NODE:\n"
        "- No metastatic carcinoma identified in two lymph nodes (0/2)\n"
    )
    stations = detect_nodal_stations(report)
    assert all(s["positive"] is False for s in stations)
    assert any(s["group"] == "para_aortic" for s in stations)


def test_apply_nodal_station_detection_notes_positive_group() -> None:
    data: dict = {}
    notes = _apply_nodal_station_detection(
        data, "B. PARAAORTIC LYMPH NODE:\n- Metastatic carcinoma in one node\n"
    )
    assert data["lymph_node_stations"]
    assert any("para-aortic" in note for note in notes)


def test_peritoneal_backstop_downgrades_washings_only() -> None:
    # Positive peritoneal WASHINGS/cytology alone is not IIIB2 (mirrors the A0G2 false positive):
    # a positive call with no implant in the source must be downgraded so staging stays IIIB1.
    data = {"pelvic_peritoneal_metastasis": "identified"}
    status: dict = {}
    evidence: dict = {}
    report = "Pelvic peritoneal washings: positive for malignant cells (cytology)."
    notes = _apply_peritoneal_metastasis_backstop(data, status, evidence, report)
    assert data["pelvic_peritoneal_metastasis"] == "not identified"
    assert notes


def test_peritoneal_backstop_keeps_genuine_implant() -> None:
    # A true peritoneal implant/deposit is IIIB2 and must be preserved.
    data = {"pelvic_peritoneal_metastasis": "identified"}
    status: dict = {}
    evidence: dict = {}
    report = "Cul-de-sac biopsy: metastatic adenocarcinoma forming a peritoneal implant."
    notes = _apply_peritoneal_metastasis_backstop(data, status, evidence, report)
    assert data["pelvic_peritoneal_metastasis"] == "identified"
    assert not notes
    assert evidence["pelvic_peritoneal_metastasis"]


def test_peritoneal_backstop_downgrades_benign_adhesion() -> None:
    # Benign/reactive peritoneal findings do not qualify for IIIB2.
    data = {"pelvic_peritoneal_metastasis": "identified"}
    notes = _apply_peritoneal_metastasis_backstop(
        data, {}, {}, "Pelvic peritoneal surface with benign fibrous adhesions, reactive mesothelium."
    )
    assert data["pelvic_peritoneal_metastasis"] == "not identified"
    assert notes


def test_peritoneal_backstop_ignores_node_met_on_other_line() -> None:
    # Reproduces the A0G2 false positive: a metastatic-node verdict sits on a different specimen
    # line from the peritoneum, so it must NOT be read as peritoneal disease (IIIB2). Stays IIIB1.
    data = {"pelvic_peritoneal_metastasis": "identified"}
    report = (
        "B. PELVIC LYMPH NODES, EXCISION:\n"
        "- Metastatic adenocarcinoma in two of five lymph nodes\n"
        "C. PELVIC PERITONEUM, BIOPSY:\n"
        "- Benign mesothelium; no tumor identified\n"
    )
    notes = _apply_peritoneal_metastasis_backstop(data, {}, {}, report)
    assert data["pelvic_peritoneal_metastasis"] == "not identified"
    assert notes


def test_peritoneal_backstop_leaves_negative_untouched() -> None:
    # Never invents a positive: an already-negative call is left exactly as-is.
    data = {"pelvic_peritoneal_metastasis": "not identified"}
    notes = _apply_peritoneal_metastasis_backstop(
        data, {}, {}, "Peritoneal implant of metastatic carcinoma noted."
    )
    assert data["pelvic_peritoneal_metastasis"] == "not identified"
    assert not notes

# Mirrors the garbled microscopic section of TCGA-A5-A0G1: serous histology stated, but LVSI
# only ambiguously addressed ("cannot absolutely exclude"). A capable model returns this JSON.
GARBLED_MODEL_JSON = """```json
{
  "histologic_type": {"value": "serous carcinoma", "confidence": 0.9, "status": "present",
    "evidence": "High grade with serous features"},
  "figo_grade": {"value": "3", "confidence": 0.85, "status": "present",
    "evidence": "High grade"},
  "myometrial_invasion_category": {"value": "<50%", "confidence": 0.8, "status": "present",
    "evidence": "depth far less than one-fourth the myometrial thickness"},
  "lymphovascular_invasion": {"value": "not reported", "confidence": 0.2, "status": "uncertain",
    "evidence": "neither to absolutely exclude vascular/lymphatic invasion"},
  "lymph_nodes_total_examined": {"value": 11, "confidence": 0.9, "status": "present",
    "evidence": "Lymph nodes negative for carcinoma"},
  "lymph_nodes_total_positive": {"value": 0, "confidence": 0.9, "status": "present",
    "evidence": "No metastatic carcinoma"},
  "serosal_involvement": {"value": "not reported", "confidence": 0.0, "status": "missing",
    "evidence": ""}
}
```"""


def test_parse_structured_json_populates_values_and_provenance() -> None:
    parsed = _parse_structured_json_response(GARBLED_MODEL_JSON)
    assert parsed is not None
    data, confidence, status, evidence = parsed

    assert data["histologic_type"] == "serous carcinoma"
    assert data["figo_grade"] == "3"
    assert data["myometrial_invasion_category"] == "<50%"
    assert data["lymph_nodes_total_examined"] == 11

    # Confidences are floats in [0, 1].
    assert all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in confidence.values())

    # The ambiguous LVSI is uncertain (not missing); a clearly stated field is present.
    assert status["lymphovascular_invasion"] == "uncertain"
    assert status["histologic_type"] == "present"
    assert status["serosal_involvement"] == "missing"
    assert "absolutely exclude" in evidence["lymphovascular_invasion"]


def test_structured_prompt_formats_with_report_only() -> None:
    prompt = EXTRACTION_PROMPT_STRUCTURED_JSON.format(report_text="SAMPLE REPORT BODY")
    assert "SAMPLE REPORT BODY" in prompt
    assert '"value"' in prompt and '"status"' in prompt
    # Field catalog is embedded so the model sees every field.
    assert "histologic_type" in prompt and "lymph_nodes_total_examined" in prompt


def test_confidence_coercion_paths() -> None:
    assert _coerce_confidence(0.8) == 0.8
    assert _coerce_confidence(1.5) == 1.0  # clamped
    assert _coerce_confidence(-0.2) == 0.0  # clamped
    assert _coerce_confidence("low") == 0.3  # categorical word
    assert _coerce_confidence("0.42") == 0.42  # numeric string
    assert _coerce_confidence(None) == 0.5  # neutral default
    assert _coerce_confidence(True) == 0.5  # bool is not a real score


def test_derive_status_without_explicit_label() -> None:
    # Present value, high confidence, clean evidence -> present.
    assert _derive_field_status("histologic_type", "serous carcinoma", 0.9, "serous", None) == (
        "present"
    )
    # Default/empty value -> missing regardless of confidence.
    assert _derive_field_status("histologic_type", "not reported", 0.9, "", None) == "missing"
    assert _derive_field_status("lymph_nodes_total_examined", -1, 0.9, "", None) == "missing"
    # Populated but low confidence -> uncertain.
    low = MEDGEMMA_CONFIDENCE_THRESHOLD - 0.1
    assert _derive_field_status("figo_grade", "3", low, "high grade", None) == "uncertain"
    # Populated, adequate confidence, but equivocal evidence phrasing -> uncertain.
    assert (
        _derive_field_status("figo_grade", "3", 0.9, "cannot be excluded", None) == "uncertain"
    )
    # Explicit valid status always wins.
    assert _derive_field_status("figo_grade", "3", 0.9, "clear", "uncertain") == "uncertain"


def test_numeric_strings_are_coerced() -> None:
    # A report whose model emitted node counts as strings must still yield ints so staging
    # (which checks isinstance(int)) sees them.
    parsed = _parse_structured_json_response(
        '{"lymph_nodes_total_examined": "11", "lymph_nodes_total_positive": "0",'
        ' "myometrial_invasion_percentage": "70%"}'
    )
    assert parsed is not None
    data, _c, _s, _e = parsed
    assert data["lymph_nodes_total_examined"] == 11
    assert data["lymph_nodes_total_positive"] == 0
    assert data["myometrial_invasion_percentage"] == 70.0


def test_flat_json_still_parses() -> None:
    parsed = _parse_structured_json_response('{"figo_grade": "2", "tumor_size_cm": "4.0"}')
    assert parsed is not None
    data, _confidence, status, _evidence = parsed
    assert data["figo_grade"] == "2"
    assert status["figo_grade"] == "present"


def test_non_schema_json_is_rejected() -> None:
    assert _parse_structured_json_response('{"unrelated": 1}') is None
    assert _parse_structured_json_response("not json at all") is None


def test_truncated_json_is_recovered() -> None:
    # Mirrors the real failure: a runaway additional_findings list truncated mid-array by the
    # output token cap. The fields emitted before truncation must still be recovered.
    truncated = (
        '{\n'
        '  "histologic_type": "serous carcinoma",\n'
        '  "figo_grade": "3",\n'
        '  "myometrial_invasion_category": ">=50%",\n'
        '  "lymph_nodes_total_examined": 11,\n'
        '  "additional_findings": [\n'
        '    "adenomyosis",\n'
        '    "multiple myomas",\n'
        '    "lymph node",'
    )
    recovered = _parse_json_response(truncated)
    assert recovered is not None
    assert recovered["histologic_type"] == "serous carcinoma"
    assert recovered["figo_grade"] == "3"
    assert recovered["lymph_nodes_total_examined"] == 11


def test_sparse_report_is_valid_but_incomplete() -> None:
    # A narrative report yielding only a few fields must NOT fail validation: absent required
    # fields are tracked as missing_required (for a recovery pass), not as hard errors.
    sparse = {
        "histologic_type": "endometrioid adenocarcinoma",
        "figo_grade": "2",
        "tumor_size_cm": "3.0",
    }
    result = validate_extraction(sparse)
    assert result.is_valid is True
    assert result.errors == []
    assert "myometrial_invasion_depth_cm" in result.missing_required
    assert "figo_stage" in result.missing_required
    # histologic_type was supplied, so it is not flagged missing.
    assert "histologic_type" not in result.missing_required


def test_malformed_value_still_fails_validation() -> None:
    # Genuine data-quality problems remain hard errors that gate validity (and drive correction).
    bad = {"figo_grade": "9", "lymph_nodes_total_examined": 3, "lymph_nodes_total_positive": 7}
    result = validate_extraction(bad)
    assert result.is_valid is False
    assert any("not in allowed values" in e for e in result.errors)
    assert any("exceeds total examined" in e for e in result.errors)


def test_myometrial_invasion_category_normalization() -> None:
    # Regression: ">=50%"/">50%" must not collapse to "no invasion" via the "0%" substring of
    # "50%". The zero-invasion cue only fires on a standalone "0%".
    assert _canonicalize_myometrial_invasion_category(">=50%") == ">=50%"
    assert _canonicalize_myometrial_invasion_category(">50%") == ">=50%"
    assert _canonicalize_myometrial_invasion_category("more than one-half") == ">=50%"
    assert _canonicalize_myometrial_invasion_category("<50%") == "<50%"
    assert _canonicalize_myometrial_invasion_category("less than one-half") == "<50%"
    assert _canonicalize_myometrial_invasion_category("no invasion") == "no invasion"
    assert _canonicalize_myometrial_invasion_category("0% (no invasion)") == "no invasion"

    data, norms = canonicalize_extraction({"myometrial_invasion_category": ">=50%"})
    assert data["myometrial_invasion_category"] == ">=50%"
    assert norms == []  # no spurious normalization recorded


def test_equivocal_backstop_flags_false_negative() -> None:
    # MedGemma flattened the garbled LVSI sentence into a confident negative; the backstop
    # re-reads the source and downgrades it to uncertain with the supporting quote.
    data = {"lymphovascular_invasion": "not identified"}
    status = {"lymphovascular_invasion": "present"}
    evidence: dict = {}
    report = (
        "At the base of the polypoid lesion there is dense lymphoid infiltrate, neither to "
        "absolutely exclude vascular/lymphatic invasion making it difficult."
    )
    _apply_equivocal_backstop(data, status, evidence, report)
    assert status["lymphovascular_invasion"] == "uncertain"
    assert "exclude" in evidence["lymphovascular_invasion"].lower()


def test_equivocal_backstop_leaves_clean_negative() -> None:
    # A clean negative with no hedging near the topic stays present.
    data = {"serosal_involvement": "not identified"}
    status = {"serosal_involvement": "present"}
    evidence: dict = {}
    _apply_equivocal_backstop(
        data, status, evidence, "The serosal surface is smooth and uninvolved by carcinoma."
    )
    assert status["serosal_involvement"] == "present"


def test_equivocal_backstop_never_downgrades_positive() -> None:
    # A positive finding is never downgraded, even amid hedging language.
    data = {"cervical_stromal_involvement": "identified"}
    status = {"cervical_stromal_involvement": "present"}
    evidence: dict = {}
    _apply_equivocal_backstop(
        data, status, evidence, "Cervix difficult to assess but tumor involves cervical stroma."
    )
    assert status["cervical_stromal_involvement"] == "present"


def test_extraction_result_field_partitions() -> None:
    result = ExtractionResult(
        data={"a": "x", "b": "y", "c": "z"},
        validation=ValidationResult(is_valid=True),
        raw_response="",
        retries=0,
        extraction_time=0.0,
        model="m",
        field_confidence={"a": 0.9, "b": 0.2, "c": 0.0},
        field_status={"a": "present", "b": "uncertain", "c": "missing"},
        field_evidence={"b": "cannot exclude"},
    )
    assert result.uncertain_fields == ["b"]
    assert result.missing_fields == ["c"]
    # Low-confidence excludes missing (c) but includes the uncertain low-conf field (b).
    assert result.low_confidence_fields == ["b"]
