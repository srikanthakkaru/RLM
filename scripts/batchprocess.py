#!/usr/bin/env python3
"""
Batch-run the VLM→RLM pathology pipeline on reports under data/reports.

Pipeline:
    Raw Report → MedGemma (structured extraction) → Validation Gate → RLM (clinical reasoning) → Final Output

Usage:
    # Full VLM→RLM pipeline (default)
    python scripts/batchprocess.py

    # Direct RLM only (legacy mode)
    python scripts/batchprocess.py --direct

    # Benchmark both pipelines side-by-side
    python scripts/batchprocess.py --compare

    # Custom models
    python scripts/batchprocess.py --vlm-model alibayram/medgemma:latest --rlm-model gemma4:latest

When the RLM yields unusable output it is re-run with a correction note (--max-retries); there
is no non-RLM fallback narrative.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vlmextraction import ExtractionResult, extract_report

from rlm import RLM
from rlm.core.types import RLMChatCompletion, RLMIteration
from rlm.utils.parsing import find_final_answer
from rlm.utils.prompts import RLM_SYSTEM_PROMPT
from stage_classification.figo2023 import StageAuditResult, audit_extraction

# ── Configuration ────────────────────────────────────────────────────────────

REPORT_TEXT_FILE: str | None = None
MAX_REPORTS = 25
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reports")

OLLAMA_URL = "http://localhost:11434"
VLM_MODEL = os.environ.get("MEDGEMMA_MODEL", "alibayram/medgemma:latest")
RLM_MODEL = os.environ.get("RLM_MODEL", "gemma4:latest")
MAX_ITERATIONS = 10
# Extra RLM regeneration attempts when the narrative is unusable/non-compliant. The RLM is
# re-run with a correction note; there is no non-RLM fallback.
MAX_RETRIES = 2

# ── Prompts: Structured Input (VLM→RLM mode) ────────────────────────────────

STRUCTURED_MEDICAL_PROMPT = """You are a clinical reasoning and validation engine operating on a validated structured extraction from a surgical pathology report for endometrial carcinoma.

The extraction is available in the REPL variable `context`. It may contain:
- extracted fields (from the source report, verbatim)
- normalized fields (listed under NORMALIZATIONS APPLIED — treat these as the authoritative provenance signal for what was mapped vs. quoted)
- computed numeric values (e.g., myometrial invasion percentage)
- VALIDATION WARNINGS (inconsistencies flagged by the upstream validator)

DO NOT re-extract from a raw report — the extraction is your single source of truth. You are the final verifier: the upstream pipeline has already canonicalized categorical fields and recomputed invasion category from numerics, but any remaining contradictions or warnings must be resolved here before you write the narrative.

--------------------------------------
STEP 1 — NUMERIC GROUNDING (scoped)
--------------------------------------
Numeric values OVERRIDE categorical values ONLY for myometrial invasion:

- If `myometrial_invasion_depth_cm` and `myometrial_thickness_cm` are both numeric and thickness > 0:
    percentage = depth / thickness * 100
    - depth == 0 → category = "no invasion"
    - 0 < percentage < 50 → category = "<50%"
    - percentage >= 50 → category = ">=50%"
- If the reported `myometrial_invasion_category` or `myometrial_invasion_percentage` conflicts with this derivation, use the derived values in the narrative AND record the override in the Corrections Applied section.

The numeric-override rule applies ONLY to myometrial invasion fields. For FIGO grade, histologic type, LVSI, margin status, cervical/serosal/adnexal involvement, TNM, and FIGO stage: use the extracted values verbatim — do not reclassify.

--------------------------------------
STEP 2 — CONTRADICTION RESOLUTION
--------------------------------------
Priority when fields conflict: numeric (for myometrial invasion only) > extracted > normalized.
Do not propagate values that VALIDATION WARNINGS have flagged as inconsistent. Resolve, do not merely describe.

--------------------------------------
STEP 3 — ATTRIBUTION CONTROL
--------------------------------------
- Extracted fields may be described as "the report states..." or "the report documents...".
- Computed / corrected / recategorized values must be described as "derived from reported measurements..." or "based on calculated values...". Never attribute a corrected value to the report's narrative.

--------------------------------------
STEP 4 — CLINICAL RULES
--------------------------------------
- If distant metastasis is "not reported" or "not applicable", state Mx explicitly.
- Do not mention fertility-sparing management if `procedure_type` indicates hysterectomy with bilateral salpingo-oophorectomy.
- Make uncertainty explicit — cite "not reported" fields rather than guessing.

--------------------------------------
WORKED EXAMPLE (numeric override)
--------------------------------------
context shows: depth=0.5 cm, thickness=1.4 cm, category="no invasion".
→ percentage = 35.7%, correct category = "<50%".
→ Narrative: "Derived from reported measurements (depth 0.5 cm / thickness 1.4 cm = ~35.7%), myometrial invasion is classified as <50%."
→ Corrections Applied: "myometrial_invasion_category: 'no invasion' → '<50%' (recomputed from depth/thickness)."

--------------------------------------
FINAL OUTPUT CONTRACT
--------------------------------------
Return plain prose (not JSON) with exactly these section headers:
  Diagnosis
  Staging
  Key Findings
  Expert Summary
  Patient-Friendly Explanation
  Next-Step Considerations

Append a trailing `Corrections Applied` section ONLY if you overrode one or more fields; omit it entirely otherwise.

Termination (critical):
- `FINAL` is NOT a REPL function. Never call `FINAL(...)` inside ```repl``` blocks.
- Your final assistant message must be plain text containing exactly one of:
    1. `FINAL(<full narrative>)` with the full prose inside the parentheses, OR
    2. `FINAL_VAR(final_answer)` after assigning `final_answer` to the full narrative in a prior ```repl``` block.
- Do NOT write `FINAL(final_answer)` as plain text — that returns the literal identifier, not the narrative. Use `FINAL_VAR(final_answer)`."""

STRUCTURED_COMBINED_PROMPT = (
    f"{RLM_SYSTEM_PROMPT}\n\n"
    "Additional domain-specific instructions for structured pathology data interpretation:\n\n"
    f"{STRUCTURED_MEDICAL_PROMPT}"
)

STRUCTURED_ROOT_PROMPT = (
    "The REPL variable `context` contains a validated structured extraction, plus any "
    "NORMALIZATIONS APPLIED and VALIDATION WARNINGS blocks. Use the REPL to inspect "
    "fields and to recompute myometrial invasion percentage / category from "
    "`myometrial_invasion_depth_cm` and `myometrial_thickness_cm` when both are numeric. "
    "Numeric override applies ONLY to myometrial invasion — all other fields use the "
    "extracted values verbatim. Resolve any remaining contradictions before writing the "
    "narrative; record overrides in a trailing `Corrections Applied` section (omit if none). "
    "Do not re-extract data. Do not call `FINAL` inside ```repl```. Finish with plain text: "
    "`FINAL(<full narrative>)` or `FINAL_VAR(final_answer)` after assigning `final_answer` "
    "in a ```repl``` block."
)

# ── Prompts: Direct RLM (legacy mode) ────────────────────────────────────────

DIRECT_MEDICAL_PROMPT = """You are analyzing a single surgical pathology report for endometrial carcinoma.

Source of truth:
- The full report text is already available in the REPL variable `context`.
- Use only `context` and facts directly supported by it.
- If a detail is missing or ambiguous, say "not reported" or "uncertain" rather than guessing.

Working approach:
- Use the REPL to inspect `context`, extract features, and build a small internal `state` dictionary if helpful.
- You may call `llm_query` for narrow sub-analyses, but do not invent placeholder helper functions such as FEATURE_EXTRACTOR(...).
- Keep the workflow compact enough to finish within the available iterations.

Track these clinical elements:
- histologic type
- FIGO grade
- tumor size if reported
- myometrial invasion category: no invasion, <50%, or >=50%
- lymphovascular space invasion
- cervical stromal involvement
- serosal or adnexal involvement
- margin status
- lymph node counts, positive nodes, and nodal stations
- extracapsular extension if reported
- TNM / FIGO stage with rationale
- expert summary
- patient-friendly explanation
- reasonable next-step considerations without drug-level prescribing

Clinical rules:
- Use the pathologist's reported FIGO grade when explicitly stated.
- If depth and total myometrial thickness are given, map them to <50% vs >=50%.
- If distant disease is not documented, state that metastatic status is unknown / Mx.
- Do not mention fertility-sparing management if the report documents hysterectomy with bilateral salpingo-oophorectomy.

Final response requirements:
- Return plain prose, not JSON.
- Use these section headers exactly: Diagnosis, Staging, Key Findings, Expert Summary, Patient-Friendly Explanation, Next-Step Considerations.
- Keep every substantive claim grounded in `context`.
- Make uncertainty explicit.
- Your final assistant message must contain only one of these:
  1. FINAL(<full narrative>)
  2. FINAL_VAR(final_answer) after creating `final_answer` in a prior ```repl``` block
- Do not place the final answer inside a ```repl``` block."""

DIRECT_COMBINED_PROMPT = (
    f"{RLM_SYSTEM_PROMPT}\n\n"
    "Additional domain-specific instructions for pathology report analysis:\n\n"
    f"{DIRECT_MEDICAL_PROMPT}"
)

DIRECT_ROOT_PROMPT = (
    "Analyze the pathology report stored in `context`. Use the REPL to inspect the report "
    "and assemble a clinically grounded narrative with the required section headers. "
    "Do not invent helper functions. When finished, end with either only "
    "`FINAL(<full narrative>)` or only `FINAL_VAR(final_answer)` after defining "
    "`final_answer` in the REPL."
)

# Correction note appended to the root prompt when an RLM attempt is unusable/non-compliant.
RLM_CORRECTION_NOTE = (
    "YOUR PREVIOUS ATTEMPT WAS UNUSABLE. Regenerate the COMPLETE clinical narrative as plain "
    "text using every required section header (Diagnosis, Staging, Key Findings, Expert Summary, "
    "Patient-Friendly Explanation, Next-Step Considerations). Do not refuse and do not return only "
    "code. Finish with FINAL(<full narrative>) or FINAL_VAR(final_answer) after assigning it in a "
    "```repl``` block."
)

# ── Extraction helpers (carried forward from original) ────────────────────────

_MIN_USABLE_CHARS = 80
_REPL_BLOCKS = re.compile(r"```repl\s*\n.*?\n```", re.DOTALL | re.IGNORECASE)
_THINK_BLOCKS = re.compile(
    r"<think>[\s\S]*?</think>|"
    r"<thinking>[\s\S]*?</thinking>|"
    r"<reasoning>[\s\S]*?</reasoning>",
    re.IGNORECASE,
)
_FINALISH_LOCAL_NAMES = ("final_answer", "answer", "clinical_narrative", "narrative")


def _strip_model_artifacts(text: str) -> str:
    t = _THINK_BLOCKS.sub("", text)
    t = _REPL_BLOCKS.sub("\n\n", t)
    return t.strip()


def is_seed_context_line(text: str) -> bool:
    t = text.strip()
    return t.startswith("Your context is a ") and " total characters" in t


def extract_final_relaxed(text: str) -> str | None:
    if not text or not text.strip():
        return None
    cleaned = _strip_model_artifacts(text)
    base = find_final_answer(cleaned, environment=None)
    if base and base.strip():
        return base.strip()

    m = re.search(r"(?is)\bFINAL\s*\(", cleaned)
    if not m:
        return None
    open_idx = cleaned.find("(", m.start())
    if open_idx < 0:
        return None
    depth = 0
    for j in range(open_idx, len(cleaned)):
        c = cleaned[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                inner = cleaned[open_idx + 1 : j].strip()
                return inner if inner else None
    return None


def looks_like_pathology_narrative(text: str) -> bool:
    low = text.lower()
    if "diagnosis" in low and "staging" in low:
        return True
    markers = (
        "figo",
        "tnm",
        "endometri",
        "carcinoma",
        "adenocarcinoma",
        "myometrial",
        "lymph node",
        "serous",
    )
    return sum(1 for m in markers if m in low) >= 2


def extract_narrative_heuristic(assistant_text: str) -> str | None:
    t = _strip_model_artifacts(assistant_text)
    if not t:
        return None
    if is_seed_context_line(t):
        return None
    if len(t) < _MIN_USABLE_CHARS:
        return None
    if looks_like_pathology_narrative(t):
        return t
    if len(t) > 400 and "diagnosis" in t.lower():
        return t
    return None


def iter_repl_narrative_candidates(iteration: RLMIteration) -> list[str]:
    candidates: list[str] = []
    for code_block in iteration.code_blocks:
        result = code_block.result
        for text in (result.stdout, result.stderr):
            if not text or not text.strip():
                continue
            heur = extract_narrative_heuristic(text)
            if heur:
                candidates.append(heur)
            stripped = _strip_model_artifacts(text)
            if len(stripped) >= _MIN_USABLE_CHARS and looks_like_pathology_narrative(stripped):
                candidates.append(stripped)
        for name in _FINALISH_LOCAL_NAMES:
            value = result.locals.get(name)
            if not isinstance(value, str):
                continue
            stripped = _strip_model_artifacts(value)
            if not stripped:
                continue
            if name == "final_answer":
                candidates.append(stripped)
                continue
            if len(stripped) >= _MIN_USABLE_CHARS and looks_like_pathology_narrative(stripped):
                candidates.append(stripped)
    return candidates


def resolve_clinical_output_from_rlm(
    result: RLMChatCompletion,
    iterations: list[RLMIteration],
) -> str:
    raw = (result.response or "").strip()
    candidates: list[str] = []
    for s in (raw,):
        if s:
            fin = extract_final_relaxed(s)
            if fin:
                candidates.append(fin)
            heur = extract_narrative_heuristic(s)
            if heur:
                candidates.append(heur)
            stripped = _strip_model_artifacts(s)
            if len(stripped) >= _MIN_USABLE_CHARS and not is_seed_context_line(stripped):
                candidates.append(stripped)

    for it in reversed(iterations):
        resp = it.response or ""
        fa_logged = it.final_answer
        if fa_logged and str(fa_logged).strip():
            fs = str(fa_logged).strip()
            fin = extract_final_relaxed(fs)
            if fin:
                candidates.append(fin)
            heur = extract_narrative_heuristic(fs)
            if heur:
                candidates.append(heur)
            st = _strip_model_artifacts(fs)
            if len(st) >= _MIN_USABLE_CHARS and not is_seed_context_line(st):
                candidates.append(st)
        fin = extract_final_relaxed(resp)
        if fin:
            candidates.append(fin)
        heur = extract_narrative_heuristic(resp)
        if heur:
            candidates.append(heur)
        candidates.extend(iter_repl_narrative_candidates(it))

    for c in candidates:
        if c and len(c.strip()) >= _MIN_USABLE_CHARS:
            return c.strip()

    if raw:
        return raw
    return ""


def is_medical_output_compliant(text: str) -> bool:
    normalized = text.lower().replace("\u2011", "-")
    required_sections = [
        "diagnosis",
        "staging",
        "key",
        "expert summary",
        "patient-friendly explanation",
        "next-step considerations",
    ]
    return all(section in normalized for section in required_sections)


def _is_unusable_response(final_response: str) -> bool:
    """True when an RLM response cannot serve as a clinical narrative (empty or too short and
    not pathology-like)."""
    return not final_response or (
        len(_strip_model_artifacts(final_response)) < _MIN_USABLE_CHARS
        and not looks_like_pathology_narrative(final_response)
    )


def _usage_tokens(result: RLMChatCompletion) -> tuple[int, int]:
    usage = result.usage_summary.to_dict()
    total_input = sum(
        m.get("total_input_tokens", 0) for m in usage.get("model_usage_summaries", {}).values()
    )
    total_output = sum(
        m.get("total_output_tokens", 0) for m in usage.get("model_usage_summaries", {}).values()
    )
    return total_input, total_output


def complete_with_retry(
    rlm: RLM,
    prompt: str,
    root_prompt: str,
    max_retries: int,
) -> tuple[str, float, int, int, int]:
    """Run the RLM, re-running it with a correction note while the output is unusable or
    non-compliant. The RLM itself always produces the narrative — there is no fallback. Returns
    (final_response, exec_time_total, iterations_total, input_tokens_total, output_tokens_total).
    """
    final_response = ""
    best = ""
    exec_time = 0.0
    iterations = 0
    input_tokens = 0
    output_tokens = 0
    correction = ""

    for _attempt in range(max_retries + 1):
        capture = IterationCaptureLogger()
        rlm.logger = capture
        attempt_root = root_prompt if not correction else f"{root_prompt}\n\n{correction}"
        result = rlm.completion(prompt=prompt, root_prompt=attempt_root)

        exec_time += result.execution_time
        iterations += len(capture.iterations)
        ti, to = _usage_tokens(result)
        input_tokens += ti
        output_tokens += to

        final_response = resolve_clinical_output_from_rlm(result, capture.iterations).strip()
        if not _is_unusable_response(final_response) and is_medical_output_compliant(
            final_response
        ):
            return final_response, exec_time, iterations, input_tokens, output_tokens
        if final_response and not best:
            best = final_response
        correction = RLM_CORRECTION_NOTE

    return best or final_response, exec_time, iterations, input_tokens, output_tokens


# ── Logger ───────────────────────────────────────────────────────────────────


class IterationCaptureLogger:
    def __init__(self) -> None:
        self.iterations: list[RLMIteration] = []

    def log_metadata(self, _metadata: object) -> None:
        return

    def log(self, iteration: RLMIteration) -> None:
        self.iterations.append(iteration)


# ── Pipeline processing ──────────────────────────────────────────────────────


@dataclass
class ReportMetrics:
    report_name: str
    mode: str
    execution_time: float
    extraction_time: float = 0.0
    extraction_retries: int = 0
    extraction_valid: bool = True
    rlm_iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_chars: int = 0
    raw_report_chars: int = 0
    output_chars: int = 0
    compliant: bool = False
    stage_audit_status: str = ""
    computed_figo_stage: str = ""
    reported_figo_stage: str = ""


@dataclass
class ComparisonReport:
    report_name: str
    direct: ReportMetrics | None = None
    vlm_rlm: ReportMetrics | None = None


def _make_rlm(
    model: str,
    ollama_url: str,
    max_iterations: int,
    system_prompt: str,
) -> RLM:
    return RLM(
        backend="ollama",
        backend_kwargs={
            "model_name": model,
            "base_url": ollama_url,
            "timeout": 1800,
            "ollama_options": {
                "num_ctx": 8192,
                "num_predict": 4096,
                "temperature": 0,
            },
        },
        environment="local",
        max_iterations=max_iterations,
        max_depth=1,
        custom_system_prompt=system_prompt,
        verbose=False,
    )


def process_report_vlm_rlm(
    text_path: str,
    report_text: str,
    rlm: RLM,
    vlm_model: str,
    ollama_url: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[str, ReportMetrics, ExtractionResult, StageAuditResult]:
    """VLM→Validation→Audit→RLM pipeline for a single report."""
    extraction = extract_report(report_text, model=vlm_model, base_url=ollama_url)
    stage_audit = audit_extraction(
        extraction.data,
        field_status=extraction.field_status,
        field_evidence=extraction.field_evidence,
        field_confidence=extraction.field_confidence,
    )
    # The structured extraction is authoritative; the raw report is appended for verification only
    # so the RLM can ground statements and avoid hallucinating.
    context_str = (
        f"{extraction.to_context_string()}\n\n{stage_audit.to_context_string()}\n\n"
        "SOURCE REPORT (verification only — do not re-extract or re-stage from this; the "
        "structured extraction above is authoritative):\n"
        f"{report_text}"
    )

    final_response, rlm_exec_time, rlm_iters, total_input, total_output = complete_with_retry(
        rlm, context_str, STRUCTURED_ROOT_PROMPT, max_retries
    )

    metrics = ReportMetrics(
        report_name=os.path.basename(text_path),
        mode="vlm_rlm",
        execution_time=extraction.extraction_time + rlm_exec_time,
        extraction_time=extraction.extraction_time,
        extraction_retries=extraction.retries,
        extraction_valid=extraction.validation.is_valid,
        rlm_iterations=rlm_iters,
        input_tokens=extraction.input_tokens + total_input,
        output_tokens=extraction.output_tokens + total_output,
        context_chars=len(context_str),
        raw_report_chars=len(report_text),
        output_chars=len(final_response),
        compliant=is_medical_output_compliant(final_response),
        stage_audit_status=stage_audit.status.value,
        computed_figo_stage=stage_audit.computed_stage or "",
        reported_figo_stage=stage_audit.reported_stage or "",
    )
    return final_response, metrics, extraction, stage_audit


def process_report_direct(
    text_path: str,
    report_text: str,
    rlm: RLM,
    max_retries: int = MAX_RETRIES,
) -> tuple[str, ReportMetrics]:
    """Direct RLM pipeline for a single report."""
    final_response, exec_time, rlm_iters, total_input, total_output = complete_with_retry(
        rlm, report_text, DIRECT_ROOT_PROMPT, max_retries
    )

    metrics = ReportMetrics(
        report_name=os.path.basename(text_path),
        mode="direct_rlm",
        execution_time=exec_time,
        rlm_iterations=rlm_iters,
        input_tokens=total_input,
        output_tokens=total_output,
        context_chars=len(report_text),
        raw_report_chars=len(report_text),
        output_chars=len(final_response),
        compliant=is_medical_output_compliant(final_response),
    )
    return final_response, metrics


# ── Comparison report generation ─────────────────────────────────────────────


def generate_comparison_report(
    comparisons: list[ComparisonReport],
    output_path: str,
) -> None:
    """Generate a side-by-side comparison report as text + JSON."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("PIPELINE COMPARISON REPORT: Direct RLM vs VLM→RLM")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Reports analyzed: {len(comparisons)}")
    lines.append("=" * 100)

    direct_metrics: list[ReportMetrics] = []
    vlm_metrics: list[ReportMetrics] = []

    for comp in comparisons:
        lines.append("")
        lines.append("-" * 100)
        lines.append(f"Report: {comp.report_name}")
        lines.append("-" * 100)

        if comp.direct:
            direct_metrics.append(comp.direct)
            lines.append("  Direct RLM:")
            lines.append(f"    Time:        {comp.direct.execution_time:.1f}s")
            lines.append(f"    Iterations:  {comp.direct.rlm_iterations}")
            lines.append(f"    Input tok:   {comp.direct.input_tokens}")
            lines.append(f"    Output tok:  {comp.direct.output_tokens}")
            lines.append(f"    Context:     {comp.direct.context_chars} chars")
            lines.append(f"    Compliant:   {comp.direct.compliant}")

        if comp.vlm_rlm:
            vlm_metrics.append(comp.vlm_rlm)
            lines.append("  VLM→RLM:")
            lines.append(
                f"    Time:        {comp.vlm_rlm.execution_time:.1f}s "
                f"(extraction: {comp.vlm_rlm.extraction_time:.1f}s)"
            )
            lines.append(f"    Iterations:  {comp.vlm_rlm.rlm_iterations}")
            lines.append(f"    Input tok:   {comp.vlm_rlm.input_tokens}")
            lines.append(f"    Output tok:  {comp.vlm_rlm.output_tokens}")
            lines.append(
                f"    Context:     {comp.vlm_rlm.context_chars} chars "
                f"(from {comp.vlm_rlm.raw_report_chars} raw)"
            )
            lines.append(
                f"    Extraction:  valid={comp.vlm_rlm.extraction_valid} "
                f"retries={comp.vlm_rlm.extraction_retries}"
            )
            if comp.vlm_rlm.stage_audit_status:
                lines.append(
                    f"    Stage audit: {comp.vlm_rlm.stage_audit_status} "
                    f"reported={comp.vlm_rlm.reported_figo_stage or 'not reported'} "
                    f"computed={comp.vlm_rlm.computed_figo_stage or 'indeterminate'}"
                )
            lines.append(f"    Compliant:   {comp.vlm_rlm.compliant}")

        if comp.direct and comp.vlm_rlm:
            tok_saved = comp.direct.input_tokens - comp.vlm_rlm.input_tokens
            tok_pct = (tok_saved / max(comp.direct.input_tokens, 1)) * 100
            time_diff = comp.direct.execution_time - comp.vlm_rlm.execution_time
            iter_diff = comp.direct.rlm_iterations - comp.vlm_rlm.rlm_iterations
            lines.append("  Delta:")
            lines.append(f"    Token savings:   {tok_saved:+d} ({tok_pct:+.1f}%)")
            lines.append(f"    Time delta:      {time_diff:+.1f}s")
            lines.append(f"    Iteration delta: {iter_diff:+d}")

    if direct_metrics and vlm_metrics:
        lines.append("")
        lines.append("=" * 100)
        lines.append("AGGREGATE SUMMARY")
        lines.append("=" * 100)

        n = len(comparisons)
        avg_direct_time = sum(m.execution_time for m in direct_metrics) / max(
            len(direct_metrics), 1
        )
        avg_vlm_time = sum(m.execution_time for m in vlm_metrics) / max(len(vlm_metrics), 1)
        avg_direct_tok = sum(m.input_tokens for m in direct_metrics) / max(len(direct_metrics), 1)
        avg_vlm_tok = sum(m.input_tokens for m in vlm_metrics) / max(len(vlm_metrics), 1)
        avg_direct_iter = sum(m.rlm_iterations for m in direct_metrics) / max(
            len(direct_metrics), 1
        )
        avg_vlm_iter = sum(m.rlm_iterations for m in vlm_metrics) / max(len(vlm_metrics), 1)
        direct_compliant = sum(1 for m in direct_metrics if m.compliant)
        vlm_compliant = sum(1 for m in vlm_metrics if m.compliant)
        extraction_pass = sum(1 for m in vlm_metrics if m.extraction_valid)

        lines.append(f"  Reports:                   {n}")
        lines.append(f"  Avg time (direct):         {avg_direct_time:.1f}s")
        lines.append(f"  Avg time (vlm→rlm):        {avg_vlm_time:.1f}s")
        lines.append(f"  Avg input tokens (direct):  {avg_direct_tok:.0f}")
        lines.append(f"  Avg input tokens (vlm→rlm): {avg_vlm_tok:.0f}")
        lines.append(f"  Avg iterations (direct):   {avg_direct_iter:.1f}")
        lines.append(f"  Avg iterations (vlm→rlm):  {avg_vlm_iter:.1f}")
        lines.append(f"  Compliance (direct):       {direct_compliant}/{len(direct_metrics)}")
        lines.append(f"  Compliance (vlm→rlm):      {vlm_compliant}/{len(vlm_metrics)}")
        lines.append(f"  Extraction pass rate:      {extraction_pass}/{len(vlm_metrics)}")

        if avg_direct_tok > 0:
            savings = (1 - avg_vlm_tok / avg_direct_tok) * 100
            lines.append(f"  Token reduction:           {savings:.1f}%")

    report_text = "\n".join(lines)
    txt_path = output_path + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nComparison report saved: {txt_path}")

    json_data = {
        "generated": datetime.now().isoformat(),
        "reports": [],
    }
    for comp in comparisons:
        entry: dict = {"report_name": comp.report_name}
        if comp.direct:
            entry["direct_rlm"] = {
                "execution_time": comp.direct.execution_time,
                "iterations": comp.direct.rlm_iterations,
                "input_tokens": comp.direct.input_tokens,
                "output_tokens": comp.direct.output_tokens,
                "context_chars": comp.direct.context_chars,
                "output_chars": comp.direct.output_chars,
                "compliant": comp.direct.compliant,
            }
        if comp.vlm_rlm:
            entry["vlm_rlm"] = {
                "execution_time": comp.vlm_rlm.execution_time,
                "extraction_time": comp.vlm_rlm.extraction_time,
                "extraction_retries": comp.vlm_rlm.extraction_retries,
                "extraction_valid": comp.vlm_rlm.extraction_valid,
                "iterations": comp.vlm_rlm.rlm_iterations,
                "input_tokens": comp.vlm_rlm.input_tokens,
                "output_tokens": comp.vlm_rlm.output_tokens,
                "context_chars": comp.vlm_rlm.context_chars,
                "raw_report_chars": comp.vlm_rlm.raw_report_chars,
                "output_chars": comp.vlm_rlm.output_chars,
                "compliant": comp.vlm_rlm.compliant,
                "stage_audit_status": comp.vlm_rlm.stage_audit_status,
                "computed_figo_stage": comp.vlm_rlm.computed_figo_stage,
                "reported_figo_stage": comp.vlm_rlm.reported_figo_stage,
            }
        json_data["reports"].append(entry)

    json_path = output_path + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Comparison data saved:   {json_path}")


# ── Progress bar ─────────────────────────────────────────────────────────────


def render_progress(current: int, total: int, width: int = 36) -> str:
    completed = int(width * current / max(total, 1))
    bar = "#" * completed + "-" * (width - completed)
    return f"[{bar}] {current}/{total}"


# ── Output writer ────────────────────────────────────────────────────────────


def write_output(
    output_dir: str,
    report_name: str,
    text_path: str,
    final_response: str,
    metrics: ReportMetrics,
    suffix: str = "rlm_result",
    extraction: ExtractionResult | None = None,
    stage_audit: StageAuditResult | None = None,
) -> str:
    output_file = os.path.join(output_dir, f"{report_name}_{suffix}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Report: {text_path}\n")
        f.write(f"Pipeline: {metrics.mode}\n")
        if metrics.mode == "vlm_rlm" and extraction:
            f.write(f"VLM Model: {extraction.model}\n")
        f.write(f"RLM Model: {RLM_MODEL}\n")
        f.write(f"Max Iterations: {MAX_ITERATIONS}\n")
        f.write(f"Execution Time: {metrics.execution_time:.2f}s\n")
        if metrics.mode == "vlm_rlm":
            f.write(f"Extraction Time: {metrics.extraction_time:.2f}s\n")
            f.write(f"Extraction Valid: {metrics.extraction_valid}\n")
            f.write(f"Extraction Retries: {metrics.extraction_retries}\n")
            if extraction and extraction.normalizations:
                f.write("Normalizations Applied:\n")
                for item in extraction.normalizations:
                    f.write(f"  - {item}\n")
            if stage_audit:
                f.write(f"Stage Audit Status: {stage_audit.status.value}\n")
                f.write(f"Computed FIGO Stage: {stage_audit.computed_stage or 'indeterminate'}\n")
                f.write(f"Reported FIGO Stage: {stage_audit.reported_stage or 'not reported'}\n")
                if stage_audit.missing_facts:
                    f.write("Stage Audit Missing Facts:\n")
                    for fact in stage_audit.missing_facts:
                        f.write(f"  - {fact.key}: {fact.reason} ({fact.required_for})\n")
                if stage_audit.contradictions:
                    f.write("Stage Audit Contradictions:\n")
                    for contradiction in stage_audit.contradictions:
                        f.write(f"  - {contradiction}\n")
        f.write(f"RLM Iterations: {metrics.rlm_iterations}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(final_response)
    return output_file


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Batch VLM→RLM pathology report analysis pipeline."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--direct",
        action="store_true",
        help="Run direct RLM only (legacy mode, no VLM)",
    )
    mode_group.add_argument(
        "--compare",
        action="store_true",
        help="Run both pipelines and generate comparison report",
    )
    parser.add_argument("--vlm-model", default=VLM_MODEL, help="MedGemma model name")
    parser.add_argument("--rlm-model", default=RLM_MODEL, help="RLM model name")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama base URL")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help="Extra RLM regeneration attempts when output is unusable (correction note re-run).",
    )
    parser.add_argument("--max-reports", type=int, default=MAX_REPORTS)
    parser.add_argument("--report", default=None, help="Process a single report file")
    args = parser.parse_args()

    reports_dir = os.path.abspath(REPORTS_DIR)
    if args.report:
        text_paths = [os.path.abspath(args.report)]
    elif REPORT_TEXT_FILE:
        text_paths = [os.path.abspath(REPORT_TEXT_FILE)]
    else:
        text_paths = sorted(glob(os.path.join(reports_dir, "*.txt")))[: args.max_reports]

    if not text_paths:
        print(f"Error: No report files found in {reports_dir}")
        sys.exit(1)

    mode_label = "Compare" if args.compare else ("Direct RLM" if args.direct else "VLM→RLM")
    print(f"Pipeline:  {mode_label}")
    print(f"Reports:   {len(text_paths)}")
    print(f"RLM Model: {args.rlm_model}")
    if not args.direct:
        print(f"VLM Model: {args.vlm_model}")
    print(f"Iters:     {args.max_iterations}")
    print(f"Retries:   {args.max_retries}")
    print()

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    rlm_direct = None
    rlm_structured = None

    if args.direct or args.compare:
        rlm_direct = _make_rlm(
            args.rlm_model,
            args.ollama_url,
            args.max_iterations,
            DIRECT_COMBINED_PROMPT,
        )

    if not args.direct or args.compare:
        rlm_structured = _make_rlm(
            args.rlm_model,
            args.ollama_url,
            args.max_iterations,
            STRUCTURED_COMBINED_PROMPT,
        )

    total_reports = len(text_paths)
    comparisons: list[ComparisonReport] = []
    print("Progress:", render_progress(0, total_reports), end="\r", flush=True)

    for idx, text_path in enumerate(text_paths, start=1):
        if not os.path.exists(text_path):
            print()
            print(f"Skipping missing file: {text_path}")
            print("Progress:", render_progress(idx, total_reports), end="\r", flush=True)
            continue

        with open(text_path, encoding="utf-8") as f:
            report_text = f.read()

        if not report_text.strip():
            print()
            print(f"Skipping empty file: {text_path}")
            print("Progress:", render_progress(idx, total_reports), end="\r", flush=True)
            continue

        report_name = os.path.splitext(os.path.basename(text_path))[0]
        comp = ComparisonReport(report_name=report_name)

        if args.compare or args.direct:
            try:
                direct_response, direct_metrics = process_report_direct(
                    text_path,
                    report_text,
                    rlm_direct,
                    max_retries=args.max_retries,
                )
                comp.direct = direct_metrics
                write_output(
                    output_dir,
                    report_name,
                    text_path,
                    direct_response,
                    direct_metrics,
                    suffix="rlm_result",
                )
            except Exception as exc:
                print()
                print(f"Direct RLM failed for {text_path}: {exc}")

        if args.compare or not args.direct:
            try:
                vlm_response, vlm_metrics, extraction, stage_audit = process_report_vlm_rlm(
                    text_path,
                    report_text,
                    rlm_structured,
                    vlm_model=args.vlm_model,
                    ollama_url=args.ollama_url,
                    max_retries=args.max_retries,
                )
                comp.vlm_rlm = vlm_metrics
                write_output(
                    output_dir,
                    report_name,
                    text_path,
                    vlm_response,
                    vlm_metrics,
                    suffix="vlm_rlm_result",
                    extraction=extraction,
                    stage_audit=stage_audit,
                )
            except Exception as exc:
                print()
                print(f"VLM→RLM failed for {text_path}: {exc}")

        comparisons.append(comp)
        print("Progress:", render_progress(idx, total_reports), end="\r", flush=True)

    print()

    if args.compare:
        comparison_path = os.path.join(output_dir, "pipeline_comparison")
        generate_comparison_report(comparisons, comparison_path)

    print("Completed processing all reports.")


if __name__ == "__main__":
    main()
