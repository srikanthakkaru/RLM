from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import grounding
from vlmextraction import ExtractionResult, extract_report

from rlm import RLM
from stage_classification.figo2023 import StageAuditResult, audit_extraction

DEFAULT_REPORT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "reports",
    "TCGA-2E-A9G8.921E6140-A03E-4FBD-9FB8-554AE96FD16C.txt",
)

OLLAMA_URL = "http://localhost:11434"
# Keep the default in sync with vlmextraction.MEDGEMMA_MODEL — the installed tag is the
# namespaced "alibayram/medgemma:latest"; a bare "medgemma:latest" 404s on /api/chat.
VLM_MODEL = os.environ.get("MEDGEMMA_MODEL", "alibayram/medgemma:latest")
RLM_MODEL = os.environ.get("RLM_MODEL", "qordmlwls/llama3.1-medical")
RLM_BACKEND = os.environ.get("RLM_BACKEND", "ollama")
RLM_BASE_URL = os.environ.get("RLM_BASE_URL")
RLM_API_KEY = os.environ.get("RLM_API_KEY")
MAX_ITERATIONS = 3
MAX_RETRIES = 2

_OLLAMA_RLM_OPTIONS: dict[str, Any] = {"num_ctx": 32768, "num_predict": 4096, "temperature": 0}

PATHOLOGY_RLM_SYSTEM_PROMPT = """You are a concise pathology narrative generator.

The upstream pipeline has already extracted, normalized, validated, and staged the report. Your job
is not to perform open-ended research. Use the provided context as the source of truth and produce
the final narrative quickly.

Rules:
- You may call llm_query or llm_query_batched when a recursive sub-call helps verify,
  decompose, or resolve the provided context.
- Use ```repl``` blocks for focused inspection, deterministic calculations, or recursive
  sub-calls that directly support the final narrative.
- If the provided context is already sufficient, answer immediately with FINAL(<full narrative>).
- Do not narrate your analysis process.
- Do not keep asking yourself what to check next; unresolved or missing facts should be stated as
  "not reported" or "requires pathologist verification".
- Finish promptly once the context and any recursive checks are sufficient.
"""

STRUCTURED_MEDICAL_PROMPT = """You are a clinical reasoning and validation engine working from a validated, structured extraction of a surgical pathology report for endometrial carcinoma.

The extracted data is available in the ```repl``` variable context. It may include:

Extracted fields (verbatim from the source report)

Normalized fields (listed under “NORMALIZATIONS APPLIED"; these indicate mapped values and should be treated as authoritative for provenance)

Computed numeric values (e.g., myometrial invasion percentage)

VALIDATION WARNINGS (flagged inconsistencies from upstream validation)

The validated structured extraction is your AUTHORITATIVE source for every clinical fact and for staging. The raw source report is ALSO provided (as `context['source_report']`, or the "SOURCE REPORT" section in single-pass mode) for ONE purpose: to let you verify wording and ground your statements so you never invent a detail. Consult it to confirm or quote a finding, but do NOT use it to re-extract, re-stage, or override the structured extraction's categorical or staging values. If the report and the extraction disagree on a staging-relevant fact, defer to the extraction and note the discrepancy rather than restating the report. Your role is final verification: the upstream pipeline has already standardized categorical data and computed derived values. You must resolve any remaining inconsistencies before generating the narrative.

Step 1 — Numeric Grounding (Scoped)
Numeric values override categorical values only for myometrial invasion:

If both myometrial_invasion_depth_cm and myometrial_thickness_cm are present and thickness > 0:

Compute: percentage = (depth / thickness) × 100

Assign category:

depth = 0 → “no invasion”

0 < percentage < 50 → “<50%”

percentage ≥ 50 → “≥50%”

If the reported myometrial_invasion_category or myometrial_invasion_percentage conflicts with the computed value:

Use the computed result in the narrative

Record the change in the “Corrections Applied” section

This numeric override rule applies only to myometrial invasion. All other fields (FIGO grade, histologic type, LVSI, margins, involvement, TNM, FIGO stage) must be used exactly as extracted.

Step 2 — Contradiction Resolution
When conflicts exist, resolve them using this priority:

Numeric values (myometrial invasion only)

Extracted values

Normalized values

Do not repeat or describe inconsistencies flagged in VALIDATION WARNINGS. Resolve them and present only the corrected result.

Step 3 — Attribution Control
For extracted values, use phrasing such as “the report states” or “the report documents.”

For computed or corrected values, use phrasing such as “derived from reported measurements” or “based on calculated values.”

Never attribute corrected or recomputed values to the original report.

Step 4 — Clinical Rules
If distant metastasis is “not reported” or “not applicable,” explicitly assign Mx.

Do not mention fertility-sparing management if procedure_type indicates hysterectomy with bilateral salpingo-oophorectomy.

Do not infer missing data; explicitly state “not reported” where applicable.

Highlight high-risk features when present, including:

FIGO grade 3

Non-endometrioid histology

LVSI positivity

Deep myometrial invasion

Cervical stromal involvement

Positive lymph nodes

Step 5 — FIGO Stage Attribution
The context provides three distinct stage fields: `reported_figo_stage` (taken from the source report), `computed_figo_stage` (derived by the staging engine), and `provisional_figo_stage` (a low-confidence computed estimate). In the Staging section:

- State a stage as "reported" ONLY if `reported_figo_stage` is present and non-empty. Otherwise never use the word "reported" for the stage.
- If the stage is computed, label it "computed"; if provisional, label it "provisional (requires pathologist verification)".
- If `reported_figo_stage` and `computed_figo_stage` disagree, present the computed stage and note the discrepancy.
- Use only stage values that appear in these three fields. Do NOT invent FIGO substages (e.g. "IIIA2") that are not present in the context.

Step 6 — Always Produce a Narrative
Even when the stage audit is "indeterminate", staging is provisional, or some fields are missing, you MUST still generate the full narrative from the data that IS available, marking unknowns as "not reported". Never refuse, never return an apology or a "cannot complete" message — narrate what is known.

Example (Numeric Override) — ILLUSTRATIVE TEMPLATE ONLY
The symbols D, T, P, and <category> below are placeholders, NOT data. NEVER copy these
symbols or any numbers from this example into your output. Use ONLY the actual numeric values
present in the `context` for THIS case. If the extraction has no myometrial depth and
thickness, do not state any invasion percentage at all.

Given depth = D cm and thickness = T cm: compute P = (D / T) × 100, then assign:
  D = 0 → “no invasion”; 0 < P < 50 → “<50%”; P ≥ 50 → “≥50%”

Narrative example (schematic):
“Derived from reported measurements (D cm depth / T cm thickness ≈ P%), myometrial invasion is classified as <category>.”

Corrections Applied (schematic):
“myometrial_invasion_category: ‘<old value>’ → ‘<new value>’ (recomputed from depth/thickness)”

Final Output Requirements
Return plain text (not JSON) with exactly these section headers:

Diagnosis

Staging

Key Findings

Expert Summary

Patient-Friendly Explanation

Next-Step Considerations

Include a “Corrections Applied” section only if any values were overridden; otherwise omit it.

Additional constraints:

Expert Summary: clinical, technical language

Patient-Friendly Explanation: plain language, no jargon, active voice

Next-Step Considerations: if the context lists any "UNCERTAIN / LOW-CONFIDENCE FIELDS" or a "provisional_figo_stage", explicitly call out those fields (with their confidence) and the provisional stage as items requiring pathologist verification. Do not treat uncertain fields as confirmed positive or negative.

Termination Rules (Critical)
Do not call FINAL(...) inside REPL blocks

Your final response must be one of:

FINAL(<full narrative>)

FINAL_VAR(final_answer) after assigning the narrative to final_answer in a REPL block

Do not output FINAL(final_answer) as plain text

Extracted data (validated):
{context}"""

STRUCTURED_COMBINED_PROMPT = f"{PATHOLOGY_RLM_SYSTEM_PROMPT}\n\n{STRUCTURED_MEDICAL_PROMPT}"

STRUCTURED_ROOT_PROMPT = """The ```repl``` variable `context` is a dict containing validated structured extraction fields (top-level keys such as `myometrial_invasion_depth_cm`), plus `normalizations`, `validation_warnings`, `field_confidence`, `stage_audit`, and `source_report` (the raw report text, for verification only).

The structured extraction fields are authoritative. `context['source_report']` is available ONLY so you can verify wording and ground statements — read it to confirm or quote a detail, but never re-extract or re-stage from it or override the extraction's staging values.

Use REPL blocks to inspect fields, recompute myometrial invasion percentage/category when both `context['myometrial_invasion_depth_cm']` and `context['myometrial_thickness_cm']` are numeric, and call llm_query or llm_query_batched when recursive review helps resolve ambiguities.

Numeric override applies only to myometrial invasion. All other fields (FIGO grade, histology, LVSI, margins, TNM, FIGO stage) must be used exactly as extracted.

Resolve any remaining inconsistencies before generating the narrative. If any values are overridden, record them in a trailing "Corrections Applied" section (omit this section if no changes were made).

Do not re-extract or reinterpret the source report.

Finalization rules:
- Do not place `FINAL(...)` inside ```repl``` blocks
- Return one of the following as plain text:
  1. FINAL(<full narrative>)
  2. FINAL_VAR(final_answer) after assigning `final_answer` in a prior ```repl``` block
- Do not output FINAL(final_answer) as plain text
"""

DIRECT_MEDICAL_PROMPT = """You are analyzing a single surgical pathology report for endometrial carcinoma.

Source of truth:
- The report is available in the ```repl``` variable `context`
- Use only information explicitly supported by `context`
- If a detail is missing or unclear, state "not reported" or "uncertain"

Workflow:
- Use the ```repl``` to inspect `context` and extract relevant findings
- You may build a small internal `state` dictionary if helpful
- Do not call `llm_query` or `llm_query_batched`; use the provided report text directly
- Do not invent or use undefined helper functions
- Keep the process efficient within iteration limits

Track the following clinical elements:
- Histologic type
- FIGO grade
- Tumor size (if reported)
- Myometrial invasion category (no invasion, <50%, >=50%)
- Lymphovascular space invasion (LVSI)
- Cervical stromal involvement
- Serosal or adnexal involvement
- Margin status
- Lymph node counts, positive nodes, nodal stations
- Extracapsular extension (if reported)
- TNM / FIGO stage with rationale
- Expert summary
- Patient-friendly explanation
- Next-step considerations (no drug-level prescribing)

Clinical rules:
- Use the explicitly reported FIGO grade
- If depth and myometrial thickness are provided, determine <50% vs >=50%
- If distant metastasis is not documented, assign Mx
- Do not mention fertility-sparing management if hysterectomy with bilateral salpingo-oophorectomy is documented

Final output requirements:
- Return plain prose (not JSON)
- Use exactly these section headers:
  Diagnosis
  Staging
  Key Findings
  Expert Summary
  Patient-Friendly Explanation
  Next-Step Considerations
- Ensure all statements are supported by `context`
- Make uncertainty explicit

Termination rules:
- Do not call `FINAL(...)` inside ```repl``` blocks
- Return one of the following as plain text:
  1. FINAL(<full narrative>)
  2. FINAL_VAR(final_answer) after defining `final_answer` in a ```repl``` block
- Do not output FINAL(final_answer) as plain text
- Do not place the final answer inside a ```repl``` block
"""

DIRECT_COMBINED_PROMPT = (
    f"{PATHOLOGY_RLM_SYSTEM_PROMPT}\n\n"
    "Additional domain-specific instructions for pathology report analysis:\n\n"
    f"{DIRECT_MEDICAL_PROMPT}"
)

DIRECT_ROOT_PROMPT = """Analyze the pathology report stored in the ```repl``` variable `context`.

Use REPL blocks to extract findings, perform focused checks, and call llm_query or llm_query_batched when recursive review helps produce a clinically grounded narrative using the required section headers. Do not invent helper functions.

Finalization rules:
- Do not call FINAL(...) inside ```repl``` blocks
- When complete, return plain text only using one of:
  FINAL(<full narrative>)
  FINAL_VAR(final_answer) after defining `final_answer` in a ```repl``` block
"""

DIRECT_NARRATIVE_PROMPT = """Write a clear, natural-language interpretation of the pathology report below.

Requirements:
- Return plain prose (not JSON)
- Use these section headers:
  Diagnosis
  Staging
  Key Findings
  Expert Summary
  Patient-Friendly Explanation
  Next-Step Considerations
- Base all statements strictly on the report text
- Explicitly state when information is "not reported"
- Do not mention fertility preservation if the report documents hysterectomy with bilateral salpingo-oophorectomy

Pathology Report:
{report_text}"""

_RLM_PLACEHOLDER_FINAL_TOKENS = frozenset(
    {"final_answer", "answer", "result", "my_answer"},
)

_RLM_REFUSAL_MARKERS = (
    "unable to complete",
    "i am unable",
    "i'm unable",
    "cannot complete your task",
    "cannot complete the task",
    "state that clearly and stop",
    "i cannot fulfill",
    "i can't fulfill",
)

_REQUIRED_SECTION_HEADERS = ("diagnosis", "staging", "key findings", "expert summary")
_MIN_HEADERS_FOR_VALID_NARRATIVE = 2

_STRUCTURAL_VIOLATION_KINDS = frozenset({"empty", "placeholder", "refusal", "missing_headers"})

_STRUCTURAL_CORRECTION = (
    "- Output the COMPLETE narrative as plain text using every required section "
    "header (Diagnosis, Staging, Key Findings, Expert Summary, "
    "Patient-Friendly Explanation, Next-Step Considerations). Do not refuse and do "
    "not return a bare token."
)

_PLAIN_TEXT_OVERRIDE = (
    "RESPONSE FORMAT (this overrides any earlier termination rules): Output the finished "
    "narrative as plain text only, using the required section headers. Do NOT write code, do NOT "
    "use a REPL, and do NOT wrap the answer in FINAL(...) or FINAL_VAR(...)."
)

_FINAL_WRAPPER_RE = re.compile(r"(?is)^\s*final(?:_var)?\s*\((.*)\)\s*$")
_SELF_CONSISTENCY_TEMPERATURE = 0.4

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "output"


def _count_violations(
    violations: list[grounding.Violation],
    severity: str,
) -> int:
    return sum(1 for v in violations if v.severity == severity)


def _severity_score(item: tuple[str, float, list[grounding.Violation]]) -> tuple[int, int]:
    viol = item[2]
    return _count_violations(viol, "error"), _count_violations(viol, "warning")


def _with_correction(prompt: str, correction: str) -> str:
    if not correction:
        return prompt
    return f"{prompt}\n\n{correction}"


def _build_rlm_backend_kwargs(
    backend: str,
    model: str,
    ollama_url: str,
    base_url: str | None = None,
    api_key: str | None = None,
    ollama_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if backend == "ollama":
        return {
            "model_name": model,
            "base_url": base_url or ollama_url,
            "timeout": 1800,
            "ollama_options": ollama_options or dict(_OLLAMA_RLM_OPTIONS),
        }
    kwargs: dict[str, Any] = {"model_name": model}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _make_rlm(
    backend: str,
    backend_kwargs: dict[str, Any],
    max_iterations: int,
    custom_system_prompt: str,
    verbose: bool,
) -> RLM:
    return RLM(
        backend=backend,
        backend_kwargs=dict(backend_kwargs),
        environment="local",
        max_iterations=max_iterations,
        max_depth=3,
        custom_system_prompt=custom_system_prompt,
        verbose=verbose,
    )


def _make_rlm_client(backend: str, backend_kwargs: dict[str, Any]):
    from rlm.clients import get_client

    return get_client(backend, dict(backend_kwargs))


def _print_stage_banner(title: str) -> None:
    rule = "=" * 80
    print(rule)
    print(title)
    print(rule)


def _print_extraction_summary(extraction: ExtractionResult) -> None:
    status = "PASSED" if extraction.validation.is_valid else "FAILED"
    print(f"  Model:      {extraction.model}")
    print(f"  Time:       {extraction.extraction_time:.1f}s")
    print(f"  Retries:    {extraction.retries}")
    print(f"  Validation: {status}")
    for err in extraction.validation.errors:
        print(f"    ERROR: {err}")
    for warn in extraction.validation.warnings:
        print(f"    WARN:  {warn}")
    if extraction.normalizations:
        print("  Normalizations:")
        for item in extraction.normalizations:
            print(f"    NOTE:  {item}")
    if not extraction.validation.is_valid:
        print(
            "\n  WARNING: Extraction failed validation after retries. "
            "Proceeding with best-effort data — review output carefully."
        )


def _print_stage_audit_summary(stage_audit: StageAuditResult) -> None:
    print(f"  Reported FIGO: {stage_audit.reported_stage or 'not reported'}")
    print(f"  Computed FIGO: {stage_audit.computed_stage or 'indeterminate'}")
    print(f"  Audit Status:  {stage_audit.status.value}")
    if stage_audit.matched_rules:
        print("  Matched Rules:")
        for rule in stage_audit.matched_rules:
            print(f"    - {rule.stage}: {rule.description}")
    if stage_audit.missing_facts:
        print(f"  Missing Facts: {len(stage_audit.missing_facts)} stage-altering facts")
    for contradiction in stage_audit.contradictions:
        print(f"    CONFLICT: {contradiction}")


def _structural_violations(text: str) -> list[grounding.Violation]:
    stripped = text.strip()
    if not stripped:
        return [grounding.Violation("error", "empty", "RLM returned no text")]
    low = stripped.lower()
    if low in _RLM_PLACEHOLDER_FINAL_TOKENS:
        return [
            grounding.Violation(
                "error", "placeholder", f"RLM returned the literal token '{stripped}'"
            )
        ]
    if any(marker in low for marker in _RLM_REFUSAL_MARKERS):
        return [grounding.Violation("error", "refusal", "RLM refused instead of narrating")]
    headers_present = sum(1 for h in _REQUIRED_SECTION_HEADERS if h in low)
    if headers_present < _MIN_HEADERS_FOR_VALID_NARRATIVE:
        return [
            grounding.Violation(
                "error",
                "missing_headers",
                "narrative is missing the required section headers "
                f"({sorted(_REQUIRED_SECTION_HEADERS)})",
            )
        ]
    return []


def _correction_line(violation: grounding.Violation) -> str:
    if violation.kind in _STRUCTURAL_VIOLATION_KINDS:
        return _STRUCTURAL_CORRECTION
    if violation.kind == "ungrounded_number":
        return (
            f"- {violation.detail}. State ONLY numbers that appear in the structured extraction or "
            "the source report; remove any other figure."
        )
    if violation.kind == "invented_stage":
        return (
            f"- {violation.detail}. Use ONLY a FIGO stage present in the reported/computed/"
            "provisional stage fields; do not invent a stage or substage."
        )
    if violation.kind == "false_reported_attribution":
        return (
            f"- {violation.detail}. Do not call a stage 'reported' unless reported_figo_stage is "
            "present; label it 'computed' or 'provisional' instead."
        )
    return f"- {violation.detail}"


def _build_correction(violations: list[grounding.Violation]) -> str:
    lines = list(dict.fromkeys(_correction_line(v) for v in violations))
    body = "\n".join(lines)
    return (
        f"YOUR PREVIOUS ATTEMPT HAD PROBLEMS. Regenerate the full narrative and fix ALL of:\n{body}"
    )


def _strip_final_wrapper(text: str) -> str:
    stripped = text.strip()
    match = _FINAL_WRAPPER_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _single_pass_narrative(
    full_prompt: str,
    backend: str,
    backend_kwargs: dict[str, Any],
) -> tuple[str, float]:
    client = _make_rlm_client(backend, backend_kwargs)
    t_start = time.perf_counter()
    response = client.completion(full_prompt)
    return _strip_final_wrapper(response), time.perf_counter() - t_start


def _sample_backend_kwargs(
    backend: str,
    backend_kwargs: dict[str, Any],
    attempt_index: int,
) -> dict[str, Any]:
    if attempt_index == 0 or backend != "ollama":
        return dict(backend_kwargs)
    bk = dict(backend_kwargs)
    options = dict(bk.get("ollama_options") or {})
    options["temperature"] = _SELF_CONSISTENCY_TEMPERATURE
    options["seed"] = attempt_index
    bk["ollama_options"] = options
    return bk


def _generate_rlm_narrative(
    *,
    context_payload: dict[str, Any],
    single_pass_prompt: str,
    report_text: str,
    backend: str,
    backend_kwargs: dict[str, Any],
    system_prompt: str,
    root_prompt: str,
    max_iterations: int,
    single_pass: bool,
    samples: int,
    max_retries: int,
    grounding_context: dict[str, Any],
    verbose: bool,
) -> tuple[str, float, list[grounding.Violation]]:
    samples = max(1, samples)
    max_attempts = max(samples, max_retries + 1)
    candidates: list[tuple[str, float, list[grounding.Violation]]] = []
    correction = ""
    best_blocking: int | None = None

    for attempt in range(max_attempts):
        bk = _sample_backend_kwargs(backend, backend_kwargs, attempt)
        attempt_single_pass = single_pass or attempt >= samples
        if attempt_single_pass:
            prompt = _with_correction(single_pass_prompt, correction)
            raw, exec_time = _single_pass_narrative(prompt, backend, bk)
        else:
            rlm = _make_rlm(backend, bk, max_iterations, system_prompt, verbose)
            result = rlm.completion(
                prompt=context_payload,
                root_prompt=_with_correction(root_prompt, correction),
            )
            raw, exec_time = result.response, result.execution_time

        text = _strip_final_wrapper(raw)
        violations = _structural_violations(text) + grounding.check_narrative(
            text, grounding_context, report_text
        )
        candidates.append((text, exec_time, violations))

        blocking = _count_violations(violations, "error")
        if attempt + 1 >= samples and blocking == 0:
            break

        in_retry_phase = attempt >= samples
        if in_retry_phase and best_blocking is not None and blocking >= best_blocking:
            print(
                f"  RLM correction retry {attempt + 1} did not reduce blocking issues "
                f"({blocking} remaining); stopping and keeping the best candidate."
            )
            break

        best_blocking = blocking if best_blocking is None else min(best_blocking, blocking)
        correction = _build_correction(violations)
        if attempt + 1 < max_attempts:
            next_mode = "single-pass correction" if attempt + 1 >= samples else "regeneration"
            print(
                f"  RLM attempt {attempt + 1} had {blocking} blocking issue(s); "
                f"running a {next_mode} with a correction note."
            )

    best_text, _, best_violations = min(candidates, key=_severity_score)
    total_time = sum(item[1] for item in candidates)

    errors = _count_violations(best_violations, "error")
    warnings = _count_violations(best_violations, "warning")
    if errors:
        print(
            f"  Grounding gate: best of {len(candidates)} RLM attempt(s) still has {errors} "
            f"error(s) and {warnings} warning(s); keeping best RLM narrative (no fallback)."
        )
    elif warnings:
        print(
            f"  Grounding gate: chosen narrative has {warnings} non-blocking warning(s) "
            "(see output footer)."
        )
    else:
        print("  Grounding gate: chosen narrative fully grounded.")

    grounding_violations = [v for v in best_violations if v.kind not in _STRUCTURAL_VIOLATION_KINDS]
    return best_text, total_time, grounding_violations


def _build_rlm_context(
    extraction: ExtractionResult,
    stage_audit: StageAuditResult,
    report_text: str,
) -> dict[str, Any]:
    return {
        **extraction.data,
        "normalizations": extraction.normalizations,
        "validation_warnings": extraction.validation.warnings,
        "field_confidence": extraction.field_confidence,
        "field_status": extraction.field_status,
        "field_evidence": extraction.field_evidence,
        "uncertain_fields": extraction.uncertain_fields,
        "missing_fields": extraction.missing_fields,
        "stage_audit": stage_audit.to_dict(),
        "reported_figo_stage": stage_audit.reported_stage,
        "computed_figo_stage": stage_audit.computed_stage,
        "provisional_figo_stage": stage_audit.provisional_stage,
        "source_report": report_text,
    }


def run_vlm_rlm_pipeline(
    report_text: str,
    vlm_model: str,
    rlm_backend: str,
    rlm_backend_kwargs: dict[str, Any],
    ollama_url: str,
    max_iterations: int,
    single_pass: bool = False,
    samples: int = 1,
    max_retries: int = MAX_RETRIES,
    verbose: bool = True,
) -> tuple[str, float, list[grounding.Violation], ExtractionResult, StageAuditResult]:
    _print_stage_banner("STAGE 1: MedGemma Structured Extraction")

    extraction = extract_report(
        report_text,
        model=vlm_model,
        base_url=ollama_url,
    )
    _print_extraction_summary(extraction)

    print()
    _print_stage_banner("STAGE 2: FIGO 2023 Staging Audit")
    stage_audit = audit_extraction(
        extraction.data,
        field_status=extraction.field_status,
        field_evidence=extraction.field_evidence,
        field_confidence=extraction.field_confidence,
    )
    _print_stage_audit_summary(stage_audit)

    context_payload = _build_rlm_context(extraction, stage_audit, report_text)
    context_str = (
        f"{extraction.to_context_string()}\n\n{stage_audit.to_context_string()}\n\n"
        "SOURCE REPORT (verification only — do not re-extract or re-stage from this; the "
        "structured extraction above is authoritative):\n"
        f"{report_text}"
    )
    print(f"\n  Context size: {len(context_str)} chars (vs {len(report_text)} raw)")
    print(f"  Reduction:    {(1 - len(context_str) / len(report_text)) * 100:.0f}%")

    print()
    _print_stage_banner("STAGE 3: RLM Clinical Reasoning")

    single_pass_prompt = (
        STRUCTURED_MEDICAL_PROMPT.replace("{context}", context_str) + "\n\n" + _PLAIN_TEXT_OVERRIDE
    )

    final_response, exec_time, violations = _generate_rlm_narrative(
        context_payload=context_payload,
        single_pass_prompt=single_pass_prompt,
        report_text=report_text,
        backend=rlm_backend,
        backend_kwargs=rlm_backend_kwargs,
        system_prompt=STRUCTURED_COMBINED_PROMPT,
        root_prompt=STRUCTURED_ROOT_PROMPT,
        max_iterations=max_iterations,
        single_pass=single_pass,
        samples=samples,
        max_retries=max_retries,
        grounding_context=context_payload,
        verbose=verbose,
    )

    total_time = extraction.extraction_time + exec_time
    return final_response, total_time, violations, extraction, stage_audit


def run_direct_rlm(
    report_text: str,
    rlm_backend: str,
    rlm_backend_kwargs: dict[str, Any],
    max_iterations: int,
    single_pass: bool = False,
    samples: int = 1,
    max_retries: int = MAX_RETRIES,
    verbose: bool = True,
) -> tuple[str, float, list[grounding.Violation]]:
    _print_stage_banner("DIRECT RLM: Pathology Report Analysis")

    final_response, exec_time, violations = _generate_rlm_narrative(
        context_payload=report_text,
        single_pass_prompt=DIRECT_NARRATIVE_PROMPT.format(report_text=report_text),
        report_text=report_text,
        backend=rlm_backend,
        backend_kwargs=rlm_backend_kwargs,
        system_prompt=DIRECT_COMBINED_PROMPT,
        root_prompt=DIRECT_ROOT_PROMPT,
        max_iterations=max_iterations,
        single_pass=single_pass,
        samples=samples,
        max_retries=max_retries,
        grounding_context={},
        verbose=verbose,
    )

    return final_response, exec_time, violations


def save_result(
    output_path: str,
    text_path: str,
    final_response: str,
    execution_time: float,
    mode: str,
    vlm_model: str | None,
    rlm_model: str,
    max_iterations: int,
    extraction: ExtractionResult | None = None,
    stage_audit: StageAuditResult | None = None,
    rlm_backend: str = "ollama",
    violations: list[grounding.Violation] | None = None,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Report: {text_path}\n")
        f.write(f"Pipeline: {mode}\n")
        if vlm_model:
            f.write(f"VLM Model: {vlm_model}\n")
        f.write(f"RLM Backend: {rlm_backend}\n")
        f.write(f"RLM Model: {rlm_model}\n")
        f.write(f"Max Iterations: {max_iterations}\n")
        f.write(f"Execution Time: {execution_time:.2f}s\n")
        if extraction:
            f.write(f"Extraction Time: {extraction.extraction_time:.2f}s\n")
            f.write(f"Extraction Retries: {extraction.retries}\n")
            f.write(f"Extraction Valid: {extraction.validation.is_valid}\n")
            if extraction.normalizations:
                f.write("Normalizations Applied:\n")
                for item in extraction.normalizations:
                    f.write(f"  - {item}\n")
            if extraction.uncertain_fields:
                f.write("Uncertain Fields (addressed but ambiguous in report):\n")
                for key in extraction.uncertain_fields:
                    conf = extraction.field_confidence.get(key, 0.0)
                    quote = extraction.field_evidence.get(key, "")
                    detail = f' — "{quote}"' if quote else ""
                    f.write(f"  - {key} ({conf:.2f}){detail}\n")
            if extraction.missing_fields:
                f.write("Missing Fields (absent from report):\n")
                for key in extraction.missing_fields:
                    f.write(f"  - {key}\n")
        if stage_audit:
            f.write(f"Stage Audit Status: {stage_audit.status.value}\n")
            f.write(f"Computed FIGO Stage: {stage_audit.computed_stage or 'indeterminate'}\n")
            if stage_audit.provisional_stage and not stage_audit.computed_stage:
                f.write(f"Provisional FIGO Stage: {stage_audit.provisional_stage}\n")
            f.write(f"Reported FIGO Stage: {stage_audit.reported_stage or 'not reported'}\n")
            if stage_audit.missing_facts:
                f.write("Stage Audit Missing Facts:\n")
                for fact in stage_audit.missing_facts:
                    f.write(f"  - {fact.key}: {fact.reason} ({fact.required_for})\n")
            if stage_audit.contradictions:
                f.write("Stage Audit Contradictions:\n")
                for contradiction in stage_audit.contradictions:
                    f.write(f"  - {contradiction}\n")
        if violations:
            f.write("Grounding Gate Violations:\n")
            for v in violations:
                f.write(f"  - {v}\n")
        else:
            f.write("Grounding Gate: passed (no violations)\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(final_response)


app = typer.Typer(help="VLM→RLM pathology report analysis pipeline.")
console = Console()


@app.command()
def run(
    reports: Annotated[list[Path], typer.Argument(help="Path(s) to pathology report text file(s)")],
    vlm_model: str = typer.Option(VLM_MODEL, help="MedGemma (Stage 1 extraction) model name"),
    rlm_model: str = typer.Option(RLM_MODEL, help="RLM (Stage 3) model name"),
    rlm_backend: str = typer.Option(
        RLM_BACKEND,
        help="RLM Stage 3 backend: ollama | openai | openrouter | vllm | vercel | anthropic | "
        "gemini | azure_openai | portkey | litellm. Lets Stage 3 run on a hosted open-weight "
        "model without a local GPU.",
    ),
    rlm_base_url: str | None = typer.Option(
        RLM_BASE_URL, help="Base URL for the RLM backend (e.g. an OpenAI-compatible endpoint)."
    ),
    rlm_api_key: str | None = typer.Option(
        RLM_API_KEY, help="API key for the RLM backend (hosted backends only)."
    ),
    ollama_url: str = typer.Option(
        OLLAMA_URL, help="Ollama base URL (Stage 1 VLM; Stage 3 if ollama)"
    ),
    max_iterations: int = typer.Option(MAX_ITERATIONS, help="Max iterations for the RLM REPL loop"),
    single_pass: bool = typer.Option(
        False,
        help="Skip the agentic REPL loop; produce the narrative in one constrained generation "
        "(Tier 3). More reliable for weak models.",
    ),
    samples: int = typer.Option(
        1,
        help="Self-consistency: generate N narratives and keep the best-grounded one (Tier 3). "
        "N>1 samples at higher temperature; only diverse on the ollama backend.",
    ),
    max_retries: int = typer.Option(
        MAX_RETRIES,
        help="Extra RLM regeneration attempts when a narrative trips the grounding gate or is "
        "structurally unusable. The RLM is re-run with a correction note; there is no fallback.",
    ),
    direct: bool = typer.Option(False, help="Skip VLM extraction; run direct RLM on raw report"),
    output_dir: Annotated[
        Path, typer.Option(help="Directory for output files")
    ] = _DEFAULT_OUTPUT_DIR,
    quiet: bool = typer.Option(False, help="Suppress RLM verbose output"),
):
    """Analyze pathology reports using VLM→RLM or direct RLM pipeline."""

    console.print(Panel.fit(" Pathology Report Analysis", style="bold green"))

    rlm_backend_kwargs = _build_rlm_backend_kwargs(
        rlm_backend,
        rlm_model,
        ollama_url,
        base_url=rlm_base_url,
        api_key=rlm_api_key,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for report_path in track(reports, description="Processing reports..."):
        if not report_path.exists():
            console.print(f"[red]Skipping missing file:[/red] {report_path}")
            continue

        report_text = report_path.read_text(encoding="utf-8")

        if not report_text.strip():
            console.print(f"[yellow]Skipping empty file:[/yellow] {report_path}")
            continue

        console.print(f"\n[bold cyan]Processing:[/bold cyan] {report_path.name}")
        console.print(f"  Mode: {'Direct RLM' if direct else 'VLM→RLM'}")
        console.print(f"  RLM: {rlm_backend} / {rlm_model}")
        if not direct:
            console.print(f"  VLM Model: {vlm_model}")
        console.print(f"  Max Iterations: {max_iterations}")
        console.print(
            f"  Single-pass: {single_pass}  Samples: {samples}  Max retries: {max_retries}"
        )

        report_name = report_path.stem
        extraction: ExtractionResult | None = None
        stage_audit: StageAuditResult | None = None
        vlm_for_save: str | None = None

        if direct:
            final_response, exec_time, violations = run_direct_rlm(
                report_text,
                rlm_backend=rlm_backend,
                rlm_backend_kwargs=rlm_backend_kwargs,
                max_iterations=max_iterations,
                single_pass=single_pass,
                samples=samples,
                max_retries=max_retries,
                verbose=not quiet,
            )
            output_file = output_dir / f"{report_name}_rlm_result.txt"
            mode = "direct_rlm"
        else:
            final_response, exec_time, violations, extraction, stage_audit = run_vlm_rlm_pipeline(
                report_text,
                vlm_model=vlm_model,
                rlm_backend=rlm_backend,
                rlm_backend_kwargs=rlm_backend_kwargs,
                ollama_url=ollama_url,
                max_iterations=max_iterations,
                single_pass=single_pass,
                samples=samples,
                max_retries=max_retries,
                verbose=not quiet,
            )
            output_file = output_dir / f"{report_name}_vlm_rlm_result.txt"
            mode = "vlm_rlm"
            vlm_for_save = vlm_model

        save_result(
            str(output_file),
            str(report_path),
            final_response,
            exec_time,
            mode=mode,
            vlm_model=vlm_for_save,
            rlm_model=rlm_model,
            max_iterations=max_iterations,
            extraction=extraction,
            stage_audit=stage_audit,
            rlm_backend=rlm_backend,
            violations=violations,
        )

        console.print(f"[green]Saved:[/green] {output_file}")
        console.print(f"[green]Execution time:[/green] {exec_time:.2f}s")


if __name__ == "__main__":
    app()
