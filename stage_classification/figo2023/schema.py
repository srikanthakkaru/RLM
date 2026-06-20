from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageAuditStatus(str, Enum):
    SUPPORTED = "supported"
    RECLASSIFIED = "reclassified"
    DISCREPANT = "discrepant"
    CONFLICT = "conflict"
    INDETERMINATE = "indeterminate"


class HistologyAggressiveness(str, Enum):
    NON_AGGRESSIVE = "non_aggressive"
    AGGRESSIVE = "aggressive"


class LvsiExtent(str, Enum):
    NONE = "none"
    FOCAL = "focal"
    SUBSTANTIAL = "substantial"
    POSITIVE_UNKNOWN = "positive_unknown"


class NodalMetastasisSize(str, Enum):
    ISOLATED_TUMOR_CELLS = "isolated_tumor_cells"
    MICROMETASTASIS = "micrometastasis"
    MACROMETASTASIS = "macrometastasis"
    POSITIVE_UNKNOWN = "positive_unknown"


class MolecularSubtype(str, Enum):
    POLEMUT = "POLEmut"
    MMRD = "MMRd"
    NSMP = "NSMP"
    P53ABN = "p53abn"


@dataclass
class MissingFact:
    key: str
    reason: str
    required_for: str
    source: str = "FIGO 2023"

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "reason": self.reason,
            "required_for": self.required_for,
            "source": self.source,
        }


@dataclass
class RuleMatch:
    rule_id: str
    stage: str
    description: str
    source: str
    satisfied_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "stage": self.stage,
            "description": self.description,
            "source": self.source,
            "satisfied_conditions": self.satisfied_conditions,
        }


@dataclass
class FigoCaseFacts:
    reported_figo_stage: str | None = None
    histologic_type: str | None = None
    figo_grade: str | None = None
    histology_aggressiveness: str | None = None
    low_grade_endometrioid: bool | None = None
    myometrial_invasion_percent: float | None = None
    myometrial_invasion_present: bool | None = None
    limited_to_polyp: bool | None = None
    confined_to_endometrium: bool | None = None
    lvsi_extent: str | None = None
    cervical_stromal_involvement: bool | None = None
    serosal_involvement: bool | None = None
    adnexal_or_fallopian_tube_involvement: bool | None = None
    vaginal_or_parametrial_involvement: bool | None = None
    pelvic_peritoneal_metastasis: bool | None = None
    bladder_or_bowel_mucosa_invasion: bool | None = None
    extrapelvic_peritoneal_metastasis: bool | None = None
    distant_metastasis: bool | None = None
    regional_nodes_positive: bool | None = None
    pelvic_nodes_positive: bool | None = None
    para_aortic_nodes_positive: bool | None = None
    nodal_metastasis_size: str | None = None
    additional_metastases_absent: bool | None = None
    ovarian_tumor_unilateral: bool | None = None
    ovarian_capsule_intact: bool | None = None
    molecular_subtype: str | None = None
    molecular_testing_pending: bool = False
    contradictions: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reported_figo_stage": self.reported_figo_stage,
            "histologic_type": self.histologic_type,
            "figo_grade": self.figo_grade,
            "histology_aggressiveness": self.histology_aggressiveness,
            "low_grade_endometrioid": self.low_grade_endometrioid,
            "myometrial_invasion_percent": self.myometrial_invasion_percent,
            "myometrial_invasion_present": self.myometrial_invasion_present,
            "limited_to_polyp": self.limited_to_polyp,
            "confined_to_endometrium": self.confined_to_endometrium,
            "lvsi_extent": self.lvsi_extent,
            "cervical_stromal_involvement": self.cervical_stromal_involvement,
            "serosal_involvement": self.serosal_involvement,
            "adnexal_or_fallopian_tube_involvement": (
                self.adnexal_or_fallopian_tube_involvement
            ),
            "vaginal_or_parametrial_involvement": self.vaginal_or_parametrial_involvement,
            "pelvic_peritoneal_metastasis": self.pelvic_peritoneal_metastasis,
            "bladder_or_bowel_mucosa_invasion": self.bladder_or_bowel_mucosa_invasion,
            "extrapelvic_peritoneal_metastasis": self.extrapelvic_peritoneal_metastasis,
            "distant_metastasis": self.distant_metastasis,
            "regional_nodes_positive": self.regional_nodes_positive,
            "pelvic_nodes_positive": self.pelvic_nodes_positive,
            "para_aortic_nodes_positive": self.para_aortic_nodes_positive,
            "nodal_metastasis_size": self.nodal_metastasis_size,
            "additional_metastases_absent": self.additional_metastases_absent,
            "ovarian_tumor_unilateral": self.ovarian_tumor_unilateral,
            "ovarian_capsule_intact": self.ovarian_capsule_intact,
            "molecular_subtype": self.molecular_subtype,
            "molecular_testing_pending": self.molecular_testing_pending,
            "contradictions": self.contradictions,
            "evidence": self.evidence,
        }

    def as_rule_lookup(self) -> dict[str, Any]:
        lookup = self.to_dict()
        lookup["regional_nodes_positive"] = self.derived_regional_nodes_positive()
        lookup["myometrial_invasion_present"] = self.derived_myometrial_invasion_present()
        lookup["ia3_exception_met"] = self.ia3_exception_met()
        return lookup

    def derived_myometrial_invasion_present(self) -> bool | None:
        if self.myometrial_invasion_present is not None:
            return self.myometrial_invasion_present
        if self.myometrial_invasion_percent is None:
            return None
        return self.myometrial_invasion_percent > 0

    def derived_regional_nodes_positive(self) -> bool | None:
        if self.regional_nodes_positive is not None:
            return self.regional_nodes_positive
        node_values = (self.pelvic_nodes_positive, self.para_aortic_nodes_positive)
        if any(value is True for value in node_values):
            return True
        if all(value is False for value in node_values):
            return False
        return None

    def ia3_exception_met(self) -> bool | None:
        criteria = [
            self.low_grade_endometrioid,
            self.adnexal_or_fallopian_tube_involvement,
            self._myometrial_invasion_less_than_50(),
            self._lvsi_not_substantial(),
            self.additional_metastases_absent,
            self.ovarian_tumor_unilateral,
            self.ovarian_capsule_intact,
        ]
        if any(value is False for value in criteria):
            return False
        if any(value is None for value in criteria):
            return None
        return True

    def _myometrial_invasion_less_than_50(self) -> bool | None:
        if self.myometrial_invasion_percent is None:
            return None
        return self.myometrial_invasion_percent < 50

    def _lvsi_not_substantial(self) -> bool | None:
        if self.lvsi_extent is None:
            return None
        if self.lvsi_extent == LvsiExtent.POSITIVE_UNKNOWN.value:
            return None
        return self.lvsi_extent in {LvsiExtent.NONE.value, LvsiExtent.FOCAL.value}


@dataclass
class StageAuditResult:
    computed_stage: str | None
    reported_stage: str | None
    status: StageAuditStatus
    matched_rules: list[RuleMatch] = field(default_factory=list)
    missing_facts: list[MissingFact] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rule_version: str = "FIGO 2023"
    provisional_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "computed_stage": self.computed_stage,
            "reported_stage": self.reported_stage,
            "status": self.status.value,
            "matched_rules": [rule.to_dict() for rule in self.matched_rules],
            "missing_facts": [fact.to_dict() for fact in self.missing_facts],
            "contradictions": self.contradictions,
            "notes": self.notes,
            "rule_version": self.rule_version,
            "provisional_stage": self.provisional_stage,
        }

    def to_context_string(self) -> str:
        lines = ["STAGING AUDIT (FIGO 2023 deterministic rules):"]
        lines.append(f"  reported_figo_stage: {self.reported_stage or 'not reported'}")
        lines.append(f"  computed_figo_stage: {self.computed_stage or 'indeterminate'}")
        if self.provisional_stage and not self.computed_stage:
            lines.append(f"  provisional_figo_stage: {self.provisional_stage}")
        lines.append(f"  audit_status: {self.status.value}")
        lines.append(f"  rule_version: {self.rule_version}")

        if self.matched_rules:
            lines.append("  matched_rules:")
            for rule in self.matched_rules:
                lines.append(f"    - {rule.stage}: {rule.description} ({rule.source})")
                for condition in rule.satisfied_conditions:
                    lines.append(f"      * {condition}")

        if self.missing_facts:
            lines.append("  missing_facts:")
            for fact in self.missing_facts:
                lines.append(
                    f"    - {fact.key}: {fact.reason}; needed for {fact.required_for}"
                )

        if self.contradictions:
            lines.append("  contradictions:")
            for contradiction in self.contradictions:
                lines.append(f"    - {contradiction}")

        if self.notes:
            lines.append("  notes:")
            for note in self.notes:
                lines.append(f"    - {note}")

        return "\n".join(lines)
