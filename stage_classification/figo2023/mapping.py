from __future__ import annotations

import csv
import re
from importlib.resources import files
from typing import Any

from stage_classification.figo2023.schema import (
    FigoCaseFacts,
    HistologyAggressiveness,
    LvsiExtent,
    MolecularSubtype,
    NodalMetastasisSize,
)

DEFINITIONS_RESOURCE = "data/figo2023_definitions.csv"

# Spread/involvement fields that, by pathology convention, are negative when a complete
# resection specimen documents no positive finding (absence of mention = not identified).
# distant_metastasis is deliberately excluded — it is not assessable from the specimen.
NEGATIVE_INFERENCE_FIELDS = (
    "vaginal_or_parametrial_involvement",
    "pelvic_peritoneal_metastasis",
    "bladder_or_bowel_mucosa_invasion",
    "extrapelvic_peritoneal_metastasis",
    "serosal_involvement",
    "adnexal_or_fallopian_tube_involvement",
    "cervical_stromal_involvement",
)

_EQUIVOCAL_PHRASES = (
    "cannot be excluded",
    "cannot be assessed",
    "not fully evaluated",
    "indeterminate",
    "suspicious",
)

_MOLECULAR_MARKER_TOKENS = ("pole", "p53", "mmr", "mmrd", "nsmp", "molecular")
_MOLECULAR_PENDING_TOKENS = ("pending", "ordered", "awaiting", "to follow", "in progress")

# Maps a staging fact subject to the extraction field whose provenance status governs it. Used
# to decide whether closed-world negative inference may fire for that fact (see
# apply_negative_inference). Facts with no extraction counterpart are never directly addressed
# in the report, so they fall through to inference when the specimen is a complete resection.
_FACT_TO_EXTRACTION_FIELD = {
    "serosal_involvement": "serosal_involvement",
    "adnexal_or_fallopian_tube_involvement": "adnexal_involvement",
    "cervical_stromal_involvement": "cervical_stromal_involvement",
    "vaginal_or_parametrial_involvement": "vaginal_or_parametrial_involvement",
    "pelvic_peritoneal_metastasis": "pelvic_peritoneal_metastasis",
    "bladder_or_bowel_mucosa_invasion": "bladder_or_bowel_mucosa_invasion",
    "extrapelvic_peritoneal_metastasis": "extrapelvic_peritoneal_metastasis",
}

# Extraction fields whose mapped fact must NOT be trusted as a confident value when the extractor
# flagged them uncertain — they are reset to unknown so they neither rule a stage in nor out.
_UNCERTAIN_RESET_FIELDS = {
    "lymphovascular_invasion": "lvsi_extent",
    "cervical_stromal_involvement": "cervical_stromal_involvement",
    "serosal_involvement": "serosal_involvement",
    "adnexal_involvement": "adnexal_or_fallopian_tube_involvement",
    "vaginal_or_parametrial_involvement": "vaginal_or_parametrial_involvement",
    "pelvic_peritoneal_metastasis": "pelvic_peritoneal_metastasis",
    "bladder_or_bowel_mucosa_invasion": "bladder_or_bowel_mucosa_invasion",
    "extrapelvic_peritoneal_metastasis": "extrapelvic_peritoneal_metastasis",
}

# A TNM-derived staging inference (pT2 -> cervix, pN -> nodes, pM -> distant) is only trusted when
# the extraction is evidence-backed and at least this confident. Below it, the TNM value is kept
# as provenance but excluded from inference. Tracks vlmextraction.MEDGEMMA_CONFIDENCE_THRESHOLD.
TNM_INFERENCE_MIN_CONFIDENCE = 0.5


def _tnm_inference_trusted(
    field: str,
    field_status: dict[str, str] | None,
    field_confidence: dict[str, float] | None,
    field_evidence: dict[str, str] | None,
) -> bool:
    """Whether a TNM field may drive a staging inference.

    When no provenance is supplied (legacy callers / tests) the value is trusted, preserving
    prior behavior. When provenance IS available, the field must be a confident, evidence-backed
    ``present`` extraction — otherwise it is retained as provenance only and excluded here.
    """
    if not field_status:
        return True
    if field_status.get(field) != "present":
        return False
    if (field_confidence or {}).get(field, 0.0) < TNM_INFERENCE_MIN_CONFIDENCE:
        return False
    return bool((field_evidence or {}).get(field))


def facts_from_extraction(
    data: dict[str, Any],
    field_status: dict[str, str] | None = None,
    field_evidence: dict[str, str] | None = None,
    field_confidence: dict[str, float] | None = None,
) -> FigoCaseFacts:
    histologic_type = normalize_text_value(data.get("histologic_type"))
    figo_grade = normalize_grade(data.get("figo_grade"))
    aggressiveness, low_grade_endometrioid = classify_histology(histologic_type, figo_grade)
    percent = parse_percent(data.get("myometrial_invasion_percentage"))
    myometrial_present = infer_myometrial_invasion_present(
        percent,
        normalize_text_value(data.get("myometrial_invasion_category")),
    )
    lvsi_extent = map_lvsi_extent(data.get("lymphovascular_invasion"))
    tnm_pt = normalize_text_value(data.get("tnm_pT"))
    tnm_pn = normalize_text_value(data.get("tnm_pN"))
    tnm_pm = normalize_text_value(data.get("tnm_pM"))

    # TNM-derived staging inferences are only trusted when the TNM extraction is evidence-backed
    # and sufficiently confident; otherwise the value is kept as provenance but not used here.
    pt_trusted = _tnm_inference_trusted("tnm_pT", field_status, field_confidence, field_evidence)
    pn_trusted = _tnm_inference_trusted("tnm_pN", field_status, field_confidence, field_evidence)
    pm_trusted = _tnm_inference_trusted("tnm_pM", field_status, field_confidence, field_evidence)

    cervical = map_identified(data.get("cervical_stromal_involvement"))
    serosal = map_identified(data.get("serosal_involvement"))
    adnexal = map_identified(data.get("adnexal_involvement"))
    fallopian_tube = detect_fallopian_tube_involvement(
        adnexal,
        data.get("adnexal_involvement"),
        (field_evidence or {}).get("adnexal_involvement"),
    )
    vaginal_parametrial = map_identified(data.get("vaginal_or_parametrial_involvement"))
    pelvic_peritoneal = map_identified(data.get("pelvic_peritoneal_metastasis"))
    bladder_or_bowel = map_identified(data.get("bladder_or_bowel_mucosa_invasion"))
    extrapelvic_peritoneal = map_identified(data.get("extrapelvic_peritoneal_metastasis"))
    distant = map_distant_metastasis(tnm_pm) if pm_trusted else None
    regional_nodes, pelvic_nodes, para_aortic_nodes, nodal_size = map_nodes(
        data, tnm_pn if pn_trusted else None
    )

    contradictions = detect_contradictions(data, tnm_pn, tnm_pm)
    evidence = build_evidence(data)
    if field_evidence:
        # Prefer the extractor's verbatim source quotes over the generic mapping note.
        for key, quote in field_evidence.items():
            if quote:
                evidence[key] = quote

    if cervical is None and pt_trusted and tnm_pt and re.fullmatch(r"pt2[a-z]?", tnm_pt.lower()):
        cervical = True
        evidence["cervical_stromal_involvement"] = "Derived from tnm_pT"
    elif tnm_pt and not pt_trusted:
        evidence["tnm_pT"] = (
            f"Extracted {tnm_pt} retained as provenance only "
            "(not evidence-backed/confident); excluded from staging inference"
        )

    molecular_subtype = map_molecular_subtype(data)

    facts = FigoCaseFacts(
        reported_figo_stage=normalize_reported_stage(data.get("figo_stage")),
        histologic_type=histologic_type,
        figo_grade=figo_grade,
        histology_aggressiveness=aggressiveness,
        low_grade_endometrioid=low_grade_endometrioid,
        myometrial_invasion_percent=percent,
        myometrial_invasion_present=myometrial_present,
        lvsi_extent=lvsi_extent,
        cervical_stromal_involvement=cervical,
        serosal_involvement=serosal,
        adnexal_or_fallopian_tube_involvement=adnexal,
        fallopian_tube_involvement=fallopian_tube,
        vaginal_or_parametrial_involvement=vaginal_parametrial,
        pelvic_peritoneal_metastasis=pelvic_peritoneal,
        bladder_or_bowel_mucosa_invasion=bladder_or_bowel,
        extrapelvic_peritoneal_metastasis=extrapelvic_peritoneal,
        distant_metastasis=distant,
        regional_nodes_positive=regional_nodes,
        pelvic_nodes_positive=pelvic_nodes,
        para_aortic_nodes_positive=para_aortic_nodes,
        nodal_metastasis_size=nodal_size,
        molecular_subtype=molecular_subtype,
        molecular_testing_pending=detect_molecular_pending(data, molecular_subtype),
        contradictions=contradictions,
        evidence=evidence,
    )

    _reset_uncertain_findings(facts, field_status)
    apply_negative_inference(facts, data, field_status)
    return facts


def _reset_uncertain_findings(
    facts: FigoCaseFacts,
    field_status: dict[str, str] | None,
) -> None:
    """Reset findings the extractor flagged uncertain to unknown (None).

    An ``uncertain`` field is addressed in the report but not assertible (e.g. LVSI "cannot be
    excluded"). It must not be used as a confident positive OR negative: forcing it to None lets
    the stager surface it as a blocking/missing fact instead of silently ruling a stage in or
    out. lvsi_extent uses None directly (rather than positive_unknown) so it does not assert
    positivity either.
    """
    if not field_status:
        return
    for extraction_field, fact_attr in _UNCERTAIN_RESET_FIELDS.items():
        if field_status.get(extraction_field) == "uncertain":
            setattr(facts, fact_attr, None)
            facts.evidence[fact_attr] = (
                "Reset to unknown: report addresses this ambiguously (flagged uncertain)"
            )
    # The tube-vs-ovary split is derived from the adnexal finding, so it inherits its uncertainty.
    if field_status.get("adnexal_involvement") == "uncertain":
        facts.fallopian_tube_involvement = None


def is_complete_resection(data: dict[str, Any]) -> bool:
    procedure = normalize_text_value(data.get("procedure_type"))
    if procedure and "hysterectomy" in procedure.lower():
        return True
    integrity = normalize_text_value(data.get("specimen_integrity"))
    return bool(integrity and "intact" in integrity.lower())


def _has_equivocal_language(data: dict[str, Any]) -> bool:
    fragments: list[str] = []
    for key in (
        "cervical_stromal_involvement",
        "serosal_involvement",
        "adnexal_involvement",
        "lymphovascular_invasion",
        "margin_status",
    ):
        text = normalize_text_value(data.get(key))
        if text:
            fragments.append(text.lower())
    additional = data.get("additional_findings")
    if isinstance(additional, list):
        fragments.extend(str(item).lower() for item in additional)
    blob = " ".join(fragments)
    return any(phrase in blob for phrase in _EQUIVOCAL_PHRASES)


def apply_negative_inference(
    facts: FigoCaseFacts,
    data: dict[str, Any],
    field_status: dict[str, str] | None = None,
) -> FigoCaseFacts:
    """Map unmentioned spread/involvement findings to False for a complete resection.

    Pathology reports follow a closed-world convention: when an intact, fully resected
    specimen documents no positive finding, the finding is absent. This fills the binary
    spread fields the extraction schema does not capture so the stager can rule out higher
    stages instead of reporting them as indeterminate.

    A field is only inferred negative when it is genuinely *missing* from the report. A field
    the report addresses but ambiguously (extraction status ``uncertain``) is preserved and
    flagged, never silently negated — that is the difference between "not mentioned" and
    "mentioned but unclear". When per-field ``field_status`` is available it governs this
    decision; otherwise the blunt text-based ``_has_equivocal_language`` backstop disables
    inference wholesale.
    """
    if not is_complete_resection(data):
        return facts

    has_status = bool(field_status)
    if not has_status and _has_equivocal_language(data):
        return facts

    uncertain = {
        name for name, status in (field_status or {}).items() if status == "uncertain"
    }
    for field_name in NEGATIVE_INFERENCE_FIELDS:
        if getattr(facts, field_name) is not None:
            continue
        extraction_field = _FACT_TO_EXTRACTION_FIELD.get(field_name, field_name)
        if extraction_field in uncertain:
            facts.evidence[field_name] = (
                "Not inferred negative: report addresses this ambiguously (flagged uncertain)"
            )
            continue
        setattr(facts, field_name, False)
        facts.evidence[field_name] = (
            "Inferred False (complete resection, finding not mentioned)"
        )
    return facts


def detect_molecular_pending(data: dict[str, Any], molecular_subtype: str | None) -> bool:
    if molecular_subtype is not None:
        return False
    additional = data.get("additional_findings")
    candidates: list[str] = []
    if isinstance(additional, list):
        candidates.extend(str(item).lower() for item in additional)
    direct = normalize_text_value(data.get("molecular_subtype"))
    if direct:
        candidates.append(direct.lower())
    for text in candidates:
        has_marker = any(token in text for token in _MOLECULAR_MARKER_TOKENS)
        is_pending = any(token in text for token in _MOLECULAR_PENDING_TOKENS)
        if has_marker and is_pending:
            return True
    return False


def load_definitions() -> list[dict[str, str]]:
    resource = files("stage_classification.figo2023").joinpath(DEFINITIONS_RESOURCE)
    with resource.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def classify_histology(
    histologic_type: str | None,
    figo_grade: str | None,
) -> tuple[str | None, bool | None]:
    if not histologic_type:
        return None, None

    normalized = histologic_type.lower()
    if "endometrioid" in normalized:
        if figo_grade in {"1", "2"}:
            return HistologyAggressiveness.NON_AGGRESSIVE.value, True
        if figo_grade == "3":
            return HistologyAggressiveness.AGGRESSIVE.value, False
        return None, None

    for row in load_definitions():
        if row["definition_type"] != "histology_keyword":
            continue
        if row["value"] == "grade_dependent":
            continue
        if row["match"] in normalized:
            return row["value"], False

    return None, None


def normalize_text_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"not reported", "not found", "unknown"}:
        return None
    return stripped


def normalize_grade(value: Any) -> str | None:
    text = normalize_text_value(value)
    if text is None:
        return None
    match = re.search(r"\b([123])\b", text)
    if match:
        return match.group(1)
    return None


def normalize_reported_stage(value: Any) -> str | None:
    text = normalize_text_value(value)
    if text is None:
        return None
    normalized = text.upper()
    normalized = re.sub(r"\b(FIGO|STAGE)\b", "", normalized)
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    if not normalized or normalized in {"NA", "NX", "MX"}:
        return None
    return normalized


def parse_percent(value: Any) -> float | None:
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def infer_myometrial_invasion_present(
    percent: float | None,
    category: str | None,
) -> bool | None:
    if percent is not None:
        return percent > 0
    if category is None:
        return None
    normalized = category.lower()
    if normalized == "no invasion":
        return False
    if normalized in {"<50%", ">=50%"}:
        return True
    return None


def map_identified(value: Any) -> bool | None:
    text = normalize_text_value(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized in {"not identified", "absent", "negative", "none"}:
        return False
    if normalized in {"identified", "present", "positive", "involved"}:
        return True
    return None


def detect_fallopian_tube_involvement(
    adnexal_present: bool | None,
    value: Any,
    evidence_quote: str | None,
) -> bool | None:
    """Whether positive adnexal involvement is fallopian-tube (vs ovarian) disease.

    The extraction schema lumps ovary and tube into a single ``adnexal_involvement`` field, so
    tube-vs-ovary is recovered from the verbatim evidence. Fallopian tube tumours are always
    Stage IIIA1 and never meet the IA3 ovarian exception, so a True here blocks IA3. Detection is
    deliberately conservative: it fires only when the evidence names the tube WITHOUT the ovary,
    leaving an ovary-positive (potential IA3) case untouched. Returns None when adnexal is not a
    confident positive or the site is ambiguous.
    """
    if adnexal_present is not True:
        return None
    fragments = [str(value or ""), str(evidence_quote or "")]
    text = " ".join(fragments).lower()
    has_tube = "fallopian" in text or "salping" in text or re.search(r"\btube", text) is not None
    has_ovary = "ovar" in text
    if has_tube and not has_ovary:
        return True
    return None


def map_lvsi_extent(value: Any) -> str | None:
    text = normalize_text_value(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized in {"not identified", "absent", "negative", "none"}:
        return LvsiExtent.NONE.value
    if "focal" in normalized:
        return LvsiExtent.FOCAL.value
    if "substantial" in normalized or "extensive" in normalized:
        return LvsiExtent.SUBSTANTIAL.value
    if normalized in {"identified", "present", "positive"}:
        return LvsiExtent.POSITIVE_UNKNOWN.value
    return None


def map_distant_metastasis(tnm_pm: str | None) -> bool | None:
    if tnm_pm is None:
        return None
    normalized = tnm_pm.lower()
    if normalized in {"pm1", "m1"}:
        return True
    if normalized in {"pm0", "m0"}:
        return False
    return None


def _node_group_state(stations: Any, group: str) -> bool | None:
    """True if any station in ``group`` is positive, False if all present are negative, else None."""
    if not isinstance(stations, list):
        return None
    state: bool | None = None
    for station in stations:
        if not isinstance(station, dict) or station.get("group") != group:
            continue
        if station.get("positive") is True:
            return True
        if station.get("positive") is False:
            state = False
    return state


def map_nodes(
    data: dict[str, Any],
    tnm_pn: str | None,
) -> tuple[bool | None, bool | None, bool | None, str | None]:
    examined = data.get("lymph_nodes_total_examined")
    positive = data.get("lymph_nodes_total_positive")

    # Pelvic vs para-aortic station, recovered deterministically from the narrative, separates
    # IIIC1 from IIIC2. None for a group means its station was not documented (stays unknown).
    stations = data.get("lymph_node_stations")
    para_state = _node_group_state(stations, "para_aortic")
    pelvic_state = _node_group_state(stations, "pelvic")
    station_positive = para_state is True or pelvic_state is True

    if (isinstance(positive, int) and positive > 0) or station_positive:
        para = True if para_state is True else (False if para_state is False else None)
        pelvic = True if pelvic_state is True else (False if pelvic_state is False else None)
        return True, pelvic, para, NodalMetastasisSize.POSITIVE_UNKNOWN.value

    if isinstance(positive, int) and positive == 0:
        return False, False, False, None

    if tnm_pn is None:
        return None, None, None, None

    normalized = tnm_pn.lower().replace(" ", "")
    if "pna" in normalized:
        return None, None, None, None
    if normalized in {"pn0", "n0"}:
        return False, False, False, None
    if "pn0(i+)" in normalized or "n0(i+)" in normalized:
        return False, False, False, NodalMetastasisSize.ISOLATED_TUMOR_CELLS.value
    if "mi" in normalized:
        return True, None, None, NodalMetastasisSize.MICROMETASTASIS.value
    if re.search(r"pn[12]|n[12]", normalized):
        return True, None, None, NodalMetastasisSize.POSITIVE_UNKNOWN.value

    if isinstance(examined, int) and examined == 0:
        return None, None, None, None

    return None, None, None, None


def map_molecular_subtype(data: dict[str, Any]) -> str | None:
    direct = normalize_text_value(data.get("molecular_subtype"))
    candidates = [direct]
    additional = data.get("additional_findings")
    if isinstance(additional, list):
        candidates.extend(str(item) for item in additional)
    for candidate in candidates:
        if not candidate:
            continue
        normalized = candidate.lower()
        if "pole" in normalized and ("mut" in normalized or "pathogenic" in normalized):
            return MolecularSubtype.POLEMUT.value
        if "p53" in normalized and ("abnormal" in normalized or "abn" in normalized):
            return MolecularSubtype.P53ABN.value
        if "mmrd" in normalized or "mismatch repair deficient" in normalized:
            return MolecularSubtype.MMRD.value
        if "nsmp" in normalized or "no specific molecular profile" in normalized:
            return MolecularSubtype.NSMP.value
    return None


def detect_contradictions(
    data: dict[str, Any],
    tnm_pn: str | None,
    tnm_pm: str | None,
) -> list[str]:
    contradictions: list[str] = []
    examined = data.get("lymph_nodes_total_examined")
    positive = data.get("lymph_nodes_total_positive")
    if isinstance(examined, int) and isinstance(positive, int) and positive > examined:
        contradictions.append("Positive lymph node count exceeds examined lymph node count")
    if isinstance(positive, int) and positive > 0 and tnm_pn and "n0" in tnm_pn.lower():
        contradictions.append("Positive lymph nodes conflict with pN0")
    if tnm_pm and tnm_pm.lower() in {"pm1", "m1"} and normalize_reported_stage(
        data.get("figo_stage")
    ) not in {"IVC", "IV"}:
        contradictions.append("pM1 distant metastasis conflicts with non-stage-IV FIGO stage")
    return contradictions


def build_evidence(data: dict[str, Any]) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for key in (
        "histologic_type",
        "figo_grade",
        "myometrial_invasion_percentage",
        "lymphovascular_invasion",
        "cervical_stromal_involvement",
        "serosal_involvement",
        "adnexal_involvement",
        "vaginal_or_parametrial_involvement",
        "pelvic_peritoneal_metastasis",
        "bladder_or_bowel_mucosa_invasion",
        "extrapelvic_peritoneal_metastasis",
        "lymph_nodes_total_positive",
        "tnm_pT",
        "tnm_pN",
        "tnm_pM",
        "figo_stage",
    ):
        value = data.get(key)
        if value not in (None, "", "not reported", -1):
            evidence[key] = f"Mapped from extraction field {key}"
    return evidence
