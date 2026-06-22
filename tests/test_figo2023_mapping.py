from stage_classification.figo2023 import audit_extraction, facts_from_extraction
from stage_classification.figo2023.schema import (
    HistologyAggressiveness,
    LvsiExtent,
    StageAuditStatus,
)


def extraction_base(**overrides):
    data = {
        "histologic_type": "endometrioid adenocarcinoma",
        "figo_grade": "1",
        "myometrial_invasion_percentage": 35.7,
        "myometrial_invasion_category": "<50%",
        "lymphovascular_invasion": "not identified",
        "cervical_stromal_involvement": "not identified",
        "serosal_involvement": "not identified",
        "adnexal_involvement": "not identified",
        "lymph_nodes_total_examined": 28,
        "lymph_nodes_total_positive": 0,
        "tnm_pT": "pT1a",
        "tnm_pN": "pN0",
        "tnm_pM": "pM0",
        "figo_stage": "IA",
    }
    data.update(overrides)
    return data


def test_para_aortic_station_drives_iiic2() -> None:
    data = extraction_base(
        lymph_nodes_total_positive=2,
        tnm_pN="pN2",
        lymph_node_stations=[
            {"site": "right paraaortic lymph node", "group": "para_aortic", "positive": True},
            {"site": "right pelvic lymph node", "group": "pelvic", "positive": True},
        ],
    )
    facts = facts_from_extraction(data)
    assert facts.para_aortic_nodes_positive is True
    assert audit_extraction(data).computed_stage == "IIIC2"


def test_pelvic_only_station_drives_iiic1() -> None:
    data = extraction_base(
        lymph_nodes_total_positive=1,
        tnm_pN="pN1",
        lymph_node_stations=[
            {"site": "left pelvic lymph node", "group": "pelvic", "positive": True},
            {"site": "para-aortic lymph node", "group": "para_aortic", "positive": False},
        ],
    )
    facts = facts_from_extraction(data)
    assert facts.pelvic_nodes_positive is True
    assert facts.para_aortic_nodes_positive is False
    assert audit_extraction(data).computed_stage == "IIIC1"


def test_fallopian_tube_derived_from_evidence_blocks_ia3() -> None:
    data = extraction_base(adnexal_involvement="identified")
    evidence = {"adnexal_involvement": "Left fallopian tube involved by carcinoma."}
    facts = facts_from_extraction(data, field_evidence=evidence)
    assert facts.adnexal_or_fallopian_tube_involvement is True
    assert facts.fallopian_tube_involvement is True
    assert facts.ia3_exception_met() is False


def test_ovary_only_evidence_does_not_set_fallopian_tube() -> None:
    data = extraction_base(adnexal_involvement="identified")
    evidence = {"adnexal_involvement": "Right ovary involved by endometrioid carcinoma."}
    facts = facts_from_extraction(data, field_evidence=evidence)
    assert facts.adnexal_or_fallopian_tube_involvement is True
    assert facts.fallopian_tube_involvement is None


def test_mapping_current_extraction_to_ia2_audit() -> None:
    facts = facts_from_extraction(extraction_base())
    assert facts.histology_aggressiveness == HistologyAggressiveness.NON_AGGRESSIVE.value
    assert facts.lvsi_extent == LvsiExtent.NONE.value
    assert facts.myometrial_invasion_percent == 35.7

    audit = audit_extraction(extraction_base())
    assert audit.computed_stage == "IA2"
    assert audit.reported_stage == "IA"
    # Reported IA is a coarser ancestor of computed IA2 -> supported under strict 2023.
    assert audit.status == StageAuditStatus.SUPPORTED
    # Molecular subtype is now surfaced as a note, not a missing fact.
    assert not any(fact.key == "molecular_subtype" for fact in audit.missing_facts)
    assert any("Molecular subtype" in note for note in audit.notes)


def test_negative_inference_fills_unmentioned_spread_for_resection() -> None:
    facts = facts_from_extraction(
        extraction_base(procedure_type="hysterectomy with bilateral salpingo-oophorectomy")
    )
    assert facts.vaginal_or_parametrial_involvement is False
    assert facts.pelvic_peritoneal_metastasis is False
    assert facts.bladder_or_bowel_mucosa_invasion is False
    assert facts.extrapelvic_peritoneal_metastasis is False

    audit = audit_extraction(
        extraction_base(procedure_type="hysterectomy with bilateral salpingo-oophorectomy")
    )
    inferred_keys = {
        "vaginal_or_parametrial_involvement",
        "pelvic_peritoneal_metastasis",
        "bladder_or_bowel_mucosa_invasion",
        "extrapelvic_peritoneal_metastasis",
    }
    assert not any(fact.key in inferred_keys for fact in audit.missing_facts)


def test_negative_inference_suppressed_by_equivocal_language() -> None:
    facts = facts_from_extraction(
        extraction_base(
            procedure_type="hysterectomy with bilateral salpingo-oophorectomy",
            additional_findings=["Parametrial involvement cannot be excluded"],
        )
    )
    assert facts.vaginal_or_parametrial_involvement is None


def test_grade3_endometrioid_extraction_is_reclassified_to_iic() -> None:
    audit = audit_extraction(
        extraction_base(
            figo_grade="3",
            procedure_type="hysterectomy with bilateral salpingo-oophorectomy",
        )
    )
    assert audit.computed_stage == "IIC"
    assert audit.reported_stage == "IA"
    assert audit.status == StageAuditStatus.RECLASSIFIED


def test_positive_nodes_without_station_compute_generic_iiic() -> None:
    audit = audit_extraction(
        extraction_base(
            lymph_nodes_total_positive=1,
            tnm_pN="pN1",
            figo_stage="IIIC1",
        )
    )
    assert audit.computed_stage == "IIIC"
    assert audit.status == StageAuditStatus.INDETERMINATE
    assert any(fact.key == "nodal_station" for fact in audit.missing_facts)
    assert any(fact.key == "nodal_metastasis_size" for fact in audit.missing_facts)


def test_mapping_detects_contradictory_distant_metastasis_stage() -> None:
    audit = audit_extraction(extraction_base(tnm_pM="pM1", figo_stage="IA"))
    assert audit.computed_stage == "IVC"
    assert audit.status == StageAuditStatus.CONFLICT
    assert audit.contradictions


def test_uncertain_field_is_not_negative_inferred() -> None:
    # Complete resection: a genuinely-missing spread field is inferred negative (closed-world),
    # but a field the report addresses ambiguously (status uncertain) must be preserved.
    data = extraction_base(
        serosal_involvement="not reported",
        adnexal_involvement="not reported",
        procedure_type="abdominal hysterectomy",
        specimen_integrity="intact",
    )
    status = {"serosal_involvement": "missing", "adnexal_involvement": "uncertain"}
    facts = facts_from_extraction(data, field_status=status)

    assert facts.serosal_involvement is False  # missing -> closed-world negative
    assert facts.adnexal_or_fallopian_tube_involvement is None  # uncertain -> preserved


def test_missing_field_is_negative_inferred_without_status() -> None:
    # Backward compatibility: with no per-field status and no equivocal text, the legacy
    # closed-world inference still fires for a complete resection.
    data = extraction_base(
        serosal_involvement="not reported",
        procedure_type="abdominal hysterectomy",
        specimen_integrity="intact",
    )
    facts = facts_from_extraction(data)
    assert facts.serosal_involvement is False


def test_field_evidence_threads_into_facts() -> None:
    data = extraction_base()
    facts = facts_from_extraction(
        data,
        field_evidence={"histologic_type": "High grade with serous features"},
    )
    assert facts.evidence.get("histologic_type") == "High grade with serous features"


def test_tnm_pt2_inference_gated_without_evidence() -> None:
    # Hallucinated pT2 with MedGemma-style provenance (present, no evidence) must NOT be used to
    # infer cervical stromal involvement.
    data = extraction_base(
        cervical_stromal_involvement="not reported",
        tnm_pT="pT2",
    )
    status = {k: "present" for k in data}
    conf = {k: 0.9 for k in data}
    facts = facts_from_extraction(
        data, field_status=status, field_confidence=conf, field_evidence={}
    )
    assert facts.cervical_stromal_involvement is not True


def test_tnm_pt2_inference_allowed_with_evidence() -> None:
    # Evidence-backed, confident pT2 may drive the cervical inference.
    data = extraction_base(
        cervical_stromal_involvement="not reported",
        tnm_pT="pT2",
    )
    status = {k: "present" for k in data}
    conf = {k: 0.9 for k in data}
    evidence = {"tnm_pT": "tumor invades cervical stroma (pT2)"}
    facts = facts_from_extraction(
        data, field_status=status, field_confidence=conf, field_evidence=evidence
    )
    assert facts.cervical_stromal_involvement is True


def test_tnm_inference_legacy_trust_without_provenance() -> None:
    # No provenance supplied (legacy callers / tests): pT2 still drives the inference.
    data = extraction_base(cervical_stromal_involvement="not reported", tnm_pT="pT2")
    facts = facts_from_extraction(data)
    assert facts.cervical_stromal_involvement is True


def test_uncertain_lvsi_reset_to_unknown() -> None:
    # LVSI extracted as a negative but flagged uncertain must not be a confident lvsi_extent.
    data = extraction_base(lymphovascular_invasion="not identified")
    status = {"lymphovascular_invasion": "uncertain"}
    facts = facts_from_extraction(data, field_status=status)
    assert facts.lvsi_extent is None


def test_serosa_plus_vaginal_parametrial_computes_iiib1() -> None:
    # Tumor reaching the serosa AND extending to the vagina/parametrium must compute IIIB1
    # (priority 810), not be capped at IIIA2 (priority 800). This is the bug the new spread
    # extraction fields fix.
    audit = audit_extraction(
        extraction_base(
            serosal_involvement="identified",
            vaginal_or_parametrial_involvement="identified",
            procedure_type="hysterectomy with bilateral salpingo-oophorectomy",
        )
    )
    assert audit.computed_stage == "IIIB1"


def test_serosa_plus_pelvic_peritoneal_computes_iiib2() -> None:
    # Serosal involvement + pelvic peritoneal metastasis -> IIIB2, not IIIA2.
    audit = audit_extraction(
        extraction_base(
            serosal_involvement="identified",
            pelvic_peritoneal_metastasis="identified",
            procedure_type="hysterectomy with bilateral salpingo-oophorectomy",
        )
    )
    assert audit.computed_stage == "IIIB2"


def test_bladder_or_bowel_mucosa_invasion_computes_iva() -> None:
    audit = audit_extraction(
        extraction_base(
            serosal_involvement="identified",
            bladder_or_bowel_mucosa_invasion="identified",
            procedure_type="hysterectomy with bilateral salpingo-oophorectomy",
        )
    )
    assert audit.computed_stage == "IVA"


def test_extrapelvic_peritoneal_metastasis_computes_ivb() -> None:
    audit = audit_extraction(
        extraction_base(
            serosal_involvement="identified",
            extrapelvic_peritoneal_metastasis="identified",
            procedure_type="hysterectomy with bilateral salpingo-oophorectomy",
        )
    )
    assert audit.computed_stage == "IVB"


def test_uncertain_vaginal_parametrial_reset_to_unknown() -> None:
    # Per-field status "uncertain" must reset the new spread fact to None instead of trusting
    # the extracted value or letting closed-world inference overwrite it.
    data = extraction_base(
        vaginal_or_parametrial_involvement="not identified",
        procedure_type="hysterectomy with bilateral salpingo-oophorectomy",
    )
    status = {"vaginal_or_parametrial_involvement": "uncertain"}
    facts = facts_from_extraction(data, field_status=status)
    assert facts.vaginal_or_parametrial_involvement is None
