from __future__ import annotations

import csv
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from stage_classification.figo2023.schema import (
    FigoCaseFacts,
    MissingFact,
    MolecularSubtype,
    RuleMatch,
    StageAuditResult,
    StageAuditStatus,
)

RULE_VERSION = "FIGO 2023"
RULES_RESOURCE = "data/figo2023_rules.csv"

UNKNOWN_VALUES = {"positive_unknown", "unknown"}

# Rules with no FIGO 2009 anatomic equivalent. A reported/computed mismatch caused by one of
# these is a known FIGO 2009 -> 2023 reclassification (e.g. aggressive histology upstaging to
# IIC), not an audit failure.
RECLASSIFYING_RULE_IDS = {
    "IIC_AGGRESSIVE_INVASION",
    "IIC_AGGRESSIVE_CERVIX",
    "IIB_SUBSTANTIAL_LVSI",
    "MOLECULAR_POLEMUT",
    "MOLECULAR_P53ABN",
}

# Polyp/endometrium-confinement facts only matter when there is NO myometrial invasion; they
# are surfaced conditionally in collect_missing_facts rather than unconditionally here.
POLYP_OR_ENDOMETRIUM_FACTS: dict[str, tuple[str, str]] = {
    "limited_to_polyp": ("Distinguishes IA1/IC polyp-limited disease", "IA1/IC"),
    "confined_to_endometrium": (
        "Distinguishes IA1/IC endometrium-confined disease",
        "IA1/IC",
    ),
}

# Facts surfaced as audit notes rather than missing facts: distant metastasis is not assessable
# from a resection specimen, and molecular subtype is an annotation layer. Neither blocks a
# lower stage (their rules require eq true), so reporting them as "missing" is only noise.
NOTE_HANDLED_FACTS = {"distant_metastasis", "molecular_subtype"}

# distant_metastasis and molecular_subtype are intentionally excluded: distant metastasis is
# not assessable from a resection specimen (surfaced as a note), and molecular subtype is an
# annotation layer surfaced as a note. Neither blocks a lower stage (their rules require eq true).
STAGE_ALTERING_FACTS: dict[str, tuple[str, str]] = {
    "vaginal_or_parametrial_involvement": (
        "Rules out or supports vaginal/parametrial spread",
        "IIIB1",
    ),
    "pelvic_peritoneal_metastasis": (
        "Rules out or supports pelvic peritoneal metastasis",
        "IIIB2",
    ),
    "bladder_or_bowel_mucosa_invasion": (
        "Rules out or supports bladder/bowel mucosal invasion",
        "IVA",
    ),
    "extrapelvic_peritoneal_metastasis": (
        "Rules out or supports extrapelvic peritoneal metastasis",
        "IVB",
    ),
}


@dataclass(frozen=True)
class Condition:
    key: str
    operator: str
    value: str

    def label(self) -> str:
        return f"{self.key} {self.operator} {self.value}"


@dataclass
class StageRule:
    rule_id: str
    stage: str
    priority: int
    source: str
    description: str
    conditions: list[Condition] = field(default_factory=list)


@dataclass
class ConditionEvaluation:
    matched: bool
    missing: MissingFact | None = None


@dataclass
class RuleEvaluation:
    rule: StageRule
    matched: bool
    possible: bool
    missing_facts: list[MissingFact]
    satisfied_conditions: list[str]


def load_rules(path: Path | None = None) -> list[StageRule]:
    if path is None:
        resource = files("stage_classification.figo2023").joinpath(RULES_RESOURCE)
        with resource.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    by_id: dict[str, StageRule] = {}
    for row in rows:
        rule_id = row["rule_id"]
        rule = by_id.get(rule_id)
        if rule is None:
            rule = StageRule(
                rule_id=rule_id,
                stage=row["stage"],
                priority=int(row["priority"]),
                source=row["source"],
                description=row["description"],
            )
            by_id[rule_id] = rule
        rule.conditions.append(
            Condition(
                key=row["condition_key"],
                operator=row["operator"],
                value=row["value"],
            )
        )

    return sorted(by_id.values(), key=lambda rule: rule.priority, reverse=True)


class Figo2023EndometrialStager:
    def __init__(self, rules: list[StageRule] | None = None) -> None:
        self.rules = rules or load_rules()

    def evaluate(self, facts: FigoCaseFacts) -> StageAuditResult:
        lookup = facts.as_rule_lookup()
        evaluations = [self.evaluate_rule(rule, lookup) for rule in self.rules]
        matched = next((evaluation for evaluation in evaluations if evaluation.matched), None)

        if matched is None:
            missing = self.collect_missing_facts(facts, evaluations)
            notes = ["No FIGO 2023 rule was fully satisfied by the available facts."]
            provisional = self.select_provisional(evaluations)
            if provisional is not None:
                blocking = ", ".join(
                    fact.key for fact in provisional.missing_facts
                ) or "unresolved facts"
                notes.append(
                    f"PROVISIONAL stage {provisional.rule.stage} "
                    f"({provisional.rule.rule_id}) — all other conditions are met; depends on: "
                    f"{blocking}."
                )
            return StageAuditResult(
                computed_stage=None,
                reported_stage=facts.reported_figo_stage,
                status=StageAuditStatus.INDETERMINATE,
                missing_facts=missing,
                contradictions=facts.contradictions,
                notes=notes,
                rule_version=RULE_VERSION,
                provisional_stage=provisional.rule.stage if provisional else None,
            )

        computed_stage = matched.rule.stage
        matched_rules = [
            RuleMatch(
                rule_id=matched.rule.rule_id,
                stage=matched.rule.stage,
                description=matched.rule.description,
                source=matched.rule.source,
                satisfied_conditions=matched.satisfied_conditions,
            )
        ]

        computed_stage = self.apply_molecular_modifier(facts, computed_stage, matched_rules)

        return StageAuditResult(
            computed_stage=computed_stage,
            reported_stage=facts.reported_figo_stage,
            status=StageAuditStatus.INDETERMINATE,
            matched_rules=matched_rules,
            missing_facts=self.collect_missing_facts(facts, evaluations),
            contradictions=facts.contradictions,
            rule_version=RULE_VERSION,
        )

    def evaluate_rule(self, rule: StageRule, lookup: dict[str, Any]) -> RuleEvaluation:
        missing_facts: list[MissingFact] = []
        satisfied_conditions: list[str] = []
        possible = True

        for condition in rule.conditions:
            result = self.evaluate_condition(condition, lookup, rule)
            if result.missing:
                missing_facts.append(result.missing)
            elif result.matched:
                satisfied_conditions.append(condition.label())
            else:
                possible = False
                break

        return RuleEvaluation(
            rule=rule,
            matched=possible and not missing_facts,
            possible=possible,
            missing_facts=missing_facts if possible else [],
            satisfied_conditions=satisfied_conditions,
        )

    @staticmethod
    def select_provisional(evaluations: list[RuleEvaluation]) -> RuleEvaluation | None:
        """Best-effort stage when no rule fully matches.

        Picks the highest-priority rule that is still *possible* (no condition explicitly
        failed), is blocked only by missing facts, and has at least one condition already
        satisfied — so a rule resting entirely on an absent fact (e.g. IVC needing distant
        metastasis) is not surfaced as a provisional stage. ``evaluations`` is already ordered
        by descending rule priority, so the first qualifying entry wins.
        """
        for evaluation in evaluations:
            if (
                evaluation.possible
                and evaluation.missing_facts
                and evaluation.satisfied_conditions
            ):
                return evaluation
        return None

    def evaluate_condition(
        self,
        condition: Condition,
        lookup: dict[str, Any],
        rule: StageRule,
    ) -> ConditionEvaluation:
        actual = lookup.get(condition.key)
        if actual is None:
            return ConditionEvaluation(
                matched=False,
                missing=MissingFact(
                    key=condition.key,
                    reason="No extracted or mapped value is available",
                    required_for=f"{rule.stage} ({rule.rule_id})",
                    source=rule.source,
                ),
            )

        if isinstance(actual, str) and actual in UNKNOWN_VALUES:
            expected_values = self.parse_values(condition.value)
            if actual not in expected_values:
                return ConditionEvaluation(
                    matched=False,
                    missing=MissingFact(
                        key=condition.key,
                        reason=f"Mapped value is {actual}, which is not specific enough",
                        required_for=f"{rule.stage} ({rule.rule_id})",
                        source=rule.source,
                    ),
                )

        expected = self.parse_value(condition.value)
        if condition.operator == "eq":
            return ConditionEvaluation(matched=actual == expected)
        if condition.operator == "neq":
            return ConditionEvaluation(matched=actual != expected)
        if condition.operator == "in":
            return ConditionEvaluation(matched=actual in self.parse_values(condition.value))
        if condition.operator in {"lt", "lte", "gt", "gte"}:
            if not isinstance(actual, int | float):
                return ConditionEvaluation(
                    matched=False,
                    missing=MissingFact(
                        key=condition.key,
                        reason=f"Expected numeric value for {condition.operator} comparison",
                        required_for=f"{rule.stage} ({rule.rule_id})",
                        source=rule.source,
                    ),
                )
            numeric_expected = float(condition.value)
            if condition.operator == "lt":
                return ConditionEvaluation(matched=actual < numeric_expected)
            if condition.operator == "lte":
                return ConditionEvaluation(matched=actual <= numeric_expected)
            if condition.operator == "gt":
                return ConditionEvaluation(matched=actual > numeric_expected)
            return ConditionEvaluation(matched=actual >= numeric_expected)

        raise ValueError(f"Unsupported FIGO rule operator: {condition.operator}")

    def collect_missing_facts(
        self,
        facts: FigoCaseFacts,
        evaluations: list[RuleEvaluation],
    ) -> list[MissingFact]:
        missing: list[MissingFact] = []
        seen: set[tuple[str, str]] = set()

        for evaluation in evaluations:
            if not evaluation.possible:
                continue
            for fact in evaluation.missing_facts:
                if fact.key in NOTE_HANDLED_FACTS:
                    continue
                key = (fact.key, fact.required_for)
                if key not in seen:
                    missing.append(fact)
                    seen.add(key)

        lookup = facts.as_rule_lookup()
        for key, (reason, required_for) in STAGE_ALTERING_FACTS.items():
            if lookup.get(key) is None:
                self.add_unique_missing(
                    missing,
                    seen,
                    MissingFact(key=key, reason=reason, required_for=required_for),
                )

        # Polyp/endometrium confinement can only change the stage when invasion is absent;
        # once invasion is present they are irrelevant and would only add noise.
        if facts.derived_myometrial_invasion_present() is not True:
            for key, (reason, required_for) in POLYP_OR_ENDOMETRIUM_FACTS.items():
                if lookup.get(key) is None:
                    self.add_unique_missing(
                        missing,
                        seen,
                        MissingFact(key=key, reason=reason, required_for=required_for),
                    )

        if lookup.get("regional_nodes_positive") is True:
            if lookup.get("pelvic_nodes_positive") is None and lookup.get(
                "para_aortic_nodes_positive"
            ) is None:
                self.add_unique_missing(
                    missing,
                    seen,
                    MissingFact(
                        key="nodal_station",
                        reason="Positive regional nodes are present, but station is unknown",
                        required_for="IIIC1 vs IIIC2",
                    ),
                )
            if lookup.get("nodal_metastasis_size") in {None, "positive_unknown"}:
                self.add_unique_missing(
                    missing,
                    seen,
                    MissingFact(
                        key="nodal_metastasis_size",
                        reason="Node metastasis size is unknown",
                        required_for="IIIC1i/ii or IIIC2i/ii",
                    ),
                )

        if facts.lvsi_extent == "positive_unknown":
            self.add_unique_missing(
                missing,
                seen,
                MissingFact(
                    key="lvsi_extent",
                    reason="LVSI is positive, but focal vs substantial extent is unknown",
                    required_for="IA/IB vs IIB",
                ),
            )

        return missing

    @staticmethod
    def add_unique_missing(
        missing: list[MissingFact],
        seen: set[tuple[str, str]],
        fact: MissingFact,
    ) -> None:
        key = (fact.key, fact.required_for)
        if key in seen:
            return
        missing.append(fact)
        seen.add(key)

    @staticmethod
    def apply_molecular_modifier(
        facts: FigoCaseFacts,
        stage: str,
        matched_rules: list[RuleMatch],
    ) -> str:
        if not is_early_stage(stage):
            return stage

        if facts.molecular_subtype == MolecularSubtype.POLEMUT.value:
            matched_rules.append(
                RuleMatch(
                    rule_id="MOLECULAR_POLEMUT",
                    stage="IAmPOLEmut",
                    description=(
                        "POLEmut endometrial carcinoma in early-stage anatomic disease"
                    ),
                    source="FIGO 2023 Table 2",
                    satisfied_conditions=["molecular_subtype eq POLEmut"],
                )
            )
            return "IAmPOLEmut"

        if (
            facts.molecular_subtype == MolecularSubtype.P53ABN.value
            and facts.derived_myometrial_invasion_present() is True
        ):
            matched_rules.append(
                RuleMatch(
                    rule_id="MOLECULAR_P53ABN",
                    stage="IICmp53abn",
                    description=(
                        "p53abn endometrial carcinoma in early-stage disease with "
                        "myometrial invasion"
                    ),
                    source="FIGO 2023 Table 2",
                    satisfied_conditions=[
                        "molecular_subtype eq p53abn",
                        "myometrial_invasion_present eq true",
                    ],
                )
            )
            return "IICmp53abn"

        return stage

    @staticmethod
    def parse_value(value: str) -> Any:
        normalized = value.strip()
        if normalized.lower() == "true":
            return True
        if normalized.lower() == "false":
            return False
        try:
            if "." in normalized:
                return float(normalized)
            return int(normalized)
        except ValueError:
            return normalized

    @classmethod
    def parse_values(cls, value: str) -> set[Any]:
        return {cls.parse_value(part) for part in value.split("|")}


def is_early_stage(stage: str) -> bool:
    if stage.startswith("IV") or stage.startswith("III"):
        return False
    return stage.startswith("I")
