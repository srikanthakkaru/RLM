from __future__ import annotations

from typing import Any

from stage_classification.figo2023.mapping import facts_from_extraction, normalize_reported_stage
from stage_classification.figo2023.rules import (
    RECLASSIFYING_RULE_IDS,
    Figo2023EndometrialStager,
    is_early_stage,
)
from stage_classification.figo2023.schema import (
    FigoCaseFacts,
    StageAuditResult,
    StageAuditStatus,
)


def audit_extraction(
    data: dict[str, Any],
    stager: Figo2023EndometrialStager | None = None,
    field_status: dict[str, str] | None = None,
    field_evidence: dict[str, str] | None = None,
    field_confidence: dict[str, float] | None = None,
) -> StageAuditResult:
    return audit_facts(
        facts_from_extraction(
            data,
            field_status=field_status,
            field_evidence=field_evidence,
            field_confidence=field_confidence,
        ),
        stager=stager,
    )


def audit_facts(
    facts: FigoCaseFacts,
    stager: Figo2023EndometrialStager | None = None,
) -> StageAuditResult:
    evaluator = stager or Figo2023EndometrialStager()
    result = evaluator.evaluate(facts)

    reclassified = any(rule.rule_id in RECLASSIFYING_RULE_IDS for rule in result.matched_rules)
    result.status = compare_stages(
        reported_stage=facts.reported_figo_stage,
        computed_stage=result.computed_stage,
        contradictions=result.contradictions,
        reclassified=reclassified,
    )
    _append_audit_notes(facts, result)
    return result


def compare_stages(
    reported_stage: str | None,
    computed_stage: str | None,
    contradictions: list[str],
    *,
    reclassified: bool = False,
) -> StageAuditStatus:
    """Strict FIGO 2023 comparison: the computed 2023 stage is authoritative.

    No cross-version (FIGO 2009) reconciliation. The only granularity allowance is
    prefix consistency within 2023 notation (e.g. a reported ``IA`` is consistent with a
    computed ``IA2``). A mismatch produced by a 2023-specific upstaging rule is reported as
    ``reclassified`` rather than ``discrepant``.
    """
    if contradictions:
        return StageAuditStatus.CONFLICT
    if computed_stage is None or reported_stage is None:
        return StageAuditStatus.INDETERMINATE

    reported = normalize_reported_stage(reported_stage)
    computed = normalize_reported_stage(computed_stage)
    if reported is None or computed is None:
        return StageAuditStatus.INDETERMINATE
    if reported == computed:
        return StageAuditStatus.SUPPORTED

    # Reported stage is a coarser ancestor of the computed stage (IA vs IA2, II vs IIC):
    # fully consistent, just less specific.
    if computed.startswith(reported):
        return StageAuditStatus.SUPPORTED
    # The engine is coarser than the report because a subdividing fact is missing
    # (reported IIIC1 vs computed IIIC): cannot confirm the finer reported subdivision.
    if reported.startswith(computed):
        return StageAuditStatus.INDETERMINATE
    if reclassified:
        return StageAuditStatus.RECLASSIFIED
    return StageAuditStatus.DISCREPANT


def _append_audit_notes(facts: FigoCaseFacts, result: StageAuditResult) -> None:
    reported = normalize_reported_stage(facts.reported_figo_stage)
    computed = normalize_reported_stage(result.computed_stage) if result.computed_stage else None

    if reported and computed and computed.startswith(reported) and reported != computed:
        result.notes.append(
            f"Reported stage {reported} is a less specific ancestor of computed {computed}."
        )

    if facts.molecular_subtype is None and result.computed_stage and is_early_stage(
        result.computed_stage
    ):
        if facts.molecular_testing_pending:
            result.notes.append(
                "Molecular subtype pending; FIGO 2023 molecular substage "
                "(IAmPOLEmut / IICmp53abn) not yet assignable."
            )
        else:
            result.notes.append(
                "Molecular subtype not resolved; FIGO 2023 molecular substage "
                "(IAmPOLEmut / IICmp53abn) not applied."
            )

    if facts.distant_metastasis is None:
        result.notes.append(
            "Distant metastasis not assessed pathologically; IVC not evaluable from "
            "this specimen."
        )
