from stage_classification.figo2023 import FigoCaseFacts, audit_facts
from stage_classification.figo2023.schema import (
    HistologyAggressiveness,
    LvsiExtent,
    MolecularSubtype,
    NodalMetastasisSize,
    StageAuditStatus,
)


def base_facts(**overrides) -> FigoCaseFacts:
    values = {
        "histologic_type": "endometrioid adenocarcinoma",
        "figo_grade": "1",
        "histology_aggressiveness": HistologyAggressiveness.NON_AGGRESSIVE.value,
        "low_grade_endometrioid": True,
        "myometrial_invasion_percent": 20.0,
        "myometrial_invasion_present": True,
        "limited_to_polyp": False,
        "confined_to_endometrium": False,
        "lvsi_extent": LvsiExtent.NONE.value,
        "cervical_stromal_involvement": False,
        "serosal_involvement": False,
        "adnexal_or_fallopian_tube_involvement": False,
        "vaginal_or_parametrial_involvement": False,
        "pelvic_peritoneal_metastasis": False,
        "bladder_or_bowel_mucosa_invasion": False,
        "extrapelvic_peritoneal_metastasis": False,
        "distant_metastasis": False,
        "regional_nodes_positive": False,
        "pelvic_nodes_positive": False,
        "para_aortic_nodes_positive": False,
    }
    values.update(overrides)
    return FigoCaseFacts(**values)


def assert_stage(expected: str, **overrides) -> None:
    result = audit_facts(base_facts(**overrides))
    assert result.computed_stage == expected


def test_stage_i_substages() -> None:
    assert_stage(
        "IA1",
        myometrial_invasion_percent=0.0,
        myometrial_invasion_present=False,
        limited_to_polyp=True,
    )
    assert_stage("IA2", myometrial_invasion_percent=30.0)
    assert_stage("IB", myometrial_invasion_percent=50.0)
    assert_stage(
        "IC",
        histology_aggressiveness=HistologyAggressiveness.AGGRESSIVE.value,
        low_grade_endometrioid=False,
        myometrial_invasion_percent=0.0,
        myometrial_invasion_present=False,
        confined_to_endometrium=True,
    )


def test_stage_ia3_ovarian_exception_beats_iiia1() -> None:
    assert_stage(
        "IA3",
        adnexal_or_fallopian_tube_involvement=True,
        additional_metastases_absent=True,
        ovarian_tumor_unilateral=True,
        ovarian_capsule_intact=True,
    )
    assert_stage(
        "IIIA1",
        adnexal_or_fallopian_tube_involvement=True,
        additional_metastases_absent=True,
        ovarian_tumor_unilateral=False,
        ovarian_capsule_intact=True,
    )


def test_stage_ii_substages() -> None:
    assert_stage("IIA", cervical_stromal_involvement=True)
    assert_stage("IIB", lvsi_extent=LvsiExtent.SUBSTANTIAL.value)
    assert_stage(
        "IIC",
        histology_aggressiveness=HistologyAggressiveness.AGGRESSIVE.value,
        low_grade_endometrioid=False,
        myometrial_invasion_percent=10.0,
    )


def test_stage_iii_and_iv_precedence() -> None:
    assert_stage("IIIA2", serosal_involvement=True)
    assert_stage("IIIB1", vaginal_or_parametrial_involvement=True)
    assert_stage("IIIB2", pelvic_peritoneal_metastasis=True)
    assert_stage("IVA", bladder_or_bowel_mucosa_invasion=True)
    assert_stage("IVB", extrapelvic_peritoneal_metastasis=True)
    assert_stage(
        "IVC",
        extrapelvic_peritoneal_metastasis=True,
        distant_metastasis=True,
    )


def test_stage_iiic_node_substages_and_itc_behavior() -> None:
    assert_stage(
        "IIIC1i",
        regional_nodes_positive=None,
        pelvic_nodes_positive=True,
        para_aortic_nodes_positive=False,
        nodal_metastasis_size=NodalMetastasisSize.MICROMETASTASIS.value,
    )
    assert_stage(
        "IIIC1ii",
        regional_nodes_positive=None,
        pelvic_nodes_positive=True,
        para_aortic_nodes_positive=False,
        nodal_metastasis_size=NodalMetastasisSize.MACROMETASTASIS.value,
    )
    assert_stage(
        "IIIC2i",
        regional_nodes_positive=None,
        pelvic_nodes_positive=True,
        para_aortic_nodes_positive=True,
        nodal_metastasis_size=NodalMetastasisSize.MICROMETASTASIS.value,
    )
    assert_stage(
        "IIIC2ii",
        regional_nodes_positive=None,
        pelvic_nodes_positive=False,
        para_aortic_nodes_positive=True,
        nodal_metastasis_size=NodalMetastasisSize.MACROMETASTASIS.value,
    )
    assert_stage(
        "IA2",
        nodal_metastasis_size=NodalMetastasisSize.ISOLATED_TUMOR_CELLS.value,
    )


def test_molecular_modifiers() -> None:
    assert_stage("IAmPOLEmut", molecular_subtype=MolecularSubtype.POLEMUT.value)
    assert_stage("IICmp53abn", molecular_subtype=MolecularSubtype.P53ABN.value)
    assert_stage(
        "IIIC1i",
        molecular_subtype=MolecularSubtype.POLEMUT.value,
        regional_nodes_positive=None,
        pelvic_nodes_positive=True,
        para_aortic_nodes_positive=False,
        nodal_metastasis_size=NodalMetastasisSize.MICROMETASTASIS.value,
    )


def test_stage_comparison_statuses() -> None:
    # Exact match.
    assert audit_facts(base_facts(reported_figo_stage="IA2")).status == StageAuditStatus.SUPPORTED
    # Reported stage is a coarser ancestor of computed IA2 -> still supported.
    assert (
        audit_facts(base_facts(reported_figo_stage="IA")).status
        == StageAuditStatus.SUPPORTED
    )
    # Different branch, not explained by a reclassifying rule -> discrepant.
    assert (
        audit_facts(base_facts(reported_figo_stage="IB")).status
        == StageAuditStatus.DISCREPANT
    )
    assert (
        audit_facts(base_facts(reported_figo_stage="IA2", contradictions=["conflict"])).status
        == StageAuditStatus.CONFLICT
    )
    assert (
        audit_facts(FigoCaseFacts(reported_figo_stage="IA")).status
        == StageAuditStatus.INDETERMINATE
    )
    # Engine is coarser than the report (computed IIIC vs reported IIIC1) -> indeterminate.
    assert (
        audit_facts(
            base_facts(
                reported_figo_stage="IIIC1",
                regional_nodes_positive=True,
                pelvic_nodes_positive=None,
                para_aortic_nodes_positive=None,
                nodal_metastasis_size=NodalMetastasisSize.POSITIVE_UNKNOWN.value,
            )
        ).status
        == StageAuditStatus.INDETERMINATE
    )


def test_aggressive_histology_upstage_is_reclassified() -> None:
    # Grade-3 (aggressive) endometrioid with myometrial invasion computes IIC under FIGO 2023,
    # while the report carries a FIGO 2009 IA. That divergence is a known reclassification.
    result = audit_facts(
        base_facts(
            reported_figo_stage="IA",
            histology_aggressiveness=HistologyAggressiveness.AGGRESSIVE.value,
            low_grade_endometrioid=False,
            myometrial_invasion_percent=10.0,
        )
    )
    assert result.computed_stage == "IIC"
    assert result.status == StageAuditStatus.RECLASSIFIED


def test_provisional_stage_surfaced_when_invasion_unresolved() -> None:
    # Aggressive histology with an unresolved (None) myometrial-invasion fact: no rule fully
    # matches, but IIC is satisfied except for that one missing fact -> surfaced as provisional.
    result = audit_facts(
        base_facts(
            histologic_type="serous carcinoma",
            figo_grade="3",
            histology_aggressiveness=HistologyAggressiveness.AGGRESSIVE.value,
            low_grade_endometrioid=False,
            myometrial_invasion_percent=None,
            myometrial_invasion_present=None,
            lvsi_extent=None,
        )
    )
    assert result.status == StageAuditStatus.INDETERMINATE
    assert result.computed_stage is None
    assert result.provisional_stage is not None
    assert any("PROVISIONAL" in note for note in result.notes)


def test_no_provisional_when_only_distant_metastasis_fact_present() -> None:
    # A rule that rests entirely on an absent fact (IVC needs distant metastasis) must not be
    # surfaced as a provisional stage, since none of its conditions are actually satisfied.
    result = audit_facts(
        base_facts(
            histologic_type="serous carcinoma",
            figo_grade="3",
            histology_aggressiveness=HistologyAggressiveness.AGGRESSIVE.value,
            low_grade_endometrioid=False,
            myometrial_invasion_percent=None,
            myometrial_invasion_present=None,
            lvsi_extent=None,
        )
    )
    # Whatever provisional is chosen, it must not be the distant-metastasis-only IVC rule.
    assert result.provisional_stage != "IVC"
