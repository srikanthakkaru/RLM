#!/usr/bin/env python3
"""
Pathology report analysis pipeline using RLM.

Architecture:
  1. TEXT EXTRACTION  — VLM (vision) or OCR extracts verbatim text from the PDF.
  2. STRUCTURED JSON  — RLM forces a JSON schema with source quotes for every field.
  3. VERIFICATION     — Programmatic checks (FIGO/TNM, node math, single histotype, etc.).
  4. CORRECTION LOOP  — If verification fails, errors are fed back; re-extract (max 2 retries).
  5. GROUNDED SYNTHESIS — Verified JSON is passed to medgemma to produce the 4-section report.

Usage:
    # Use existing OCR/VLM text file:
    python process_with_rlm.py

    # Run VLM extraction from the PDF first:
    python process_with_rlm.py --vlm

    # Use a different text file:
    python process_with_rlm.py --input /path/to/report.txt
"""

import json
import os
import sys
import time
from types import SimpleNamespace
from datetime import datetime

from rlm import RLM
from rlm.clients import get_client

from verify_extraction import parse_json_from_response, verify_extraction

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_PDF = "/Users/srikanth/Desktop/RLM/data/reports/TCGA-2E-A9G8.921E6140-A03E-4FBD-9FB8-554AE96FD16C.pdf"
DEFAULT_TEXT = "/Users/srikanth/Desktop/RLM/data/reports/TCGA-2E-A9G8.921E6140-A03E-4FBD-9FB8-554AE96FD16C_ocr.txt"

USE_VLM = "--vlm" in sys.argv
MAX_VERIFY_RETRIES = 2

# ── Backend config ────────────────────────────────────────────────────────────
OLLAMA_MODEL = "alibayram/medgemma:latest"
OLLAMA_URL = "http://localhost:11434"
BACKEND_KWARGS = {
    "model_name": OLLAMA_MODEL,
    "base_url": OLLAMA_URL,
    "timeout": 1800,  # 30 min — local models on CPU are slow
    "ollama_options": {
        "num_ctx": 8192,       # context window (smaller = faster prompt eval)
        "num_predict": 4096,   # max output tokens per call
        "temperature": 0,      # deterministic, faster
    },
}

# ── JSON extraction schema ────────────────────────────────────────────────────
# Each field must have "value" and "quote" (exact text from report).
EXTRACTION_FIELDS = """\
{
  "histologic_type":              {"value": "", "quote": ""},
  "histologic_grade":             {"value": "", "quote": ""},
  "tumor_size_cm":                {"value": "", "quote": ""},
  "myometrial_invasion_depth_cm": {"value": "", "quote": ""},
  "myometrial_thickness_cm":      {"value": "", "quote": ""},
  "figo_stage":                   {"value": "", "quote": ""},
  "tnm_stage":                    {"value": "", "quote": ""},
  "cervical_involvement":         {"value": "", "quote": ""},
  "lvsi":                         {"value": "", "quote": ""},
  "serosal_involvement":          {"value": "", "quote": ""},
  "adnexal_involvement":          {"value": "", "quote": ""},
  "margins":                      {"value": "", "quote": ""},
  "margin_distance_cm":           {"value": "", "quote": ""},
  "pelvic_nodes_examined":        {"value": "", "quote": ""},
  "pelvic_nodes_positive":        {"value": "", "quote": ""},
  "paraaortic_nodes_examined":    {"value": "", "quote": ""},
  "paraaortic_nodes_positive":    {"value": "", "quote": ""},
  "total_nodes_examined":         {"value": "", "quote": ""},
  "total_nodes_positive":         {"value": "", "quote": ""},
  "mmr_status":                   {"value": "", "quote": ""},
  "p53_status":                   {"value": "", "quote": ""},
  "pole_status":                  {"value": "", "quote": ""},
  "molecular_subtype":            {"value": "", "quote": ""},
  "er_pr_status":                 {"value": "", "quote": ""},
  "her2_status":                  {"value": "", "quote": ""},
  "peritoneal_cytology":          {"value": "", "quote": ""},
  "additional_findings":          {"value": "", "quote": ""}
}"""

# ── System prompts ────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = f"""You are a pathology data extractor. Your ONLY job is to fill in a JSON object with facts found in the pathology report. You must NOT guess, infer, or hallucinate any value.

RULES (critical):
1. The report text is in the REPL variable `context`.
2. For EVERY field, copy the EXACT words from the report into "quote". Then write a short normalised "value".
3. If a field is NOT mentioned in the report, set value to "not reported" and quote to "".
4. Do NOT invent values. If the report says "Ordered" or "Pending", write that — do NOT guess results.
5. There is exactly ONE histologic type in a report. Do not mix types across fields.
6. Output ONLY valid JSON — no commentary, no markdown, no explanation.

JSON SCHEMA (fill every field):
{EXTRACTION_FIELDS}

WORKFLOW in the REPL:
1. First, print the full context to see the report.
2. Then, read through it carefully and build the JSON object field by field.
3. Assign the completed JSON string to a variable: `result = json.dumps(filled_json, indent=2)`
4. Print it, then on the next step call FINAL_VAR(result).
"""

CORRECTION_PROMPT_TEMPLATE = """The previous extraction had these verification errors:

{errors}

Please re-read the report in `context` and fix ONLY the fields that caused errors.
Output the complete corrected JSON (all fields, not just changed ones).
Ensure every "quote" is copied verbatim from the report.
Output ONLY valid JSON."""

SYNTHESIS_SYSTEM_PROMPT = """You are an expert gynecologic oncology pathologist writing a clinical report.

You will receive a VERIFIED JSON extraction from a pathology report. Every field has a "value" and a supporting "quote". Using ONLY that data (do NOT add facts not present), output exactly four markdown sections. If a field says "not reported" or "pending", say so — do NOT guess.

Output format: plain markdown only. No code blocks, no FINAL(), no repl. Start with ## 1. PATIENT SUMMARY and end with ## 4. CONSULTATION RECOMMENDATIONS."""


def _get_report_text() -> tuple[str, str]:
    """Load report text, optionally running VLM extraction first."""
    # Check for --input flag
    for i, arg in enumerate(sys.argv):
        if arg == "--input" and i + 1 < len(sys.argv):
            path = sys.argv[i + 1]
            if not os.path.exists(path):
                print(f"Error: File not found: {path}")
                sys.exit(1)
            with open(path, encoding="utf-8") as f:
                return f.read(), path

    if USE_VLM:
        from extract_with_vlm import extract_pdf_with_vlm

        print("Running VLM text extraction from PDF...")
        text = extract_pdf_with_vlm(DEFAULT_PDF, model=OLLAMA_MODEL, base_url=OLLAMA_URL)
        vlm_path = DEFAULT_PDF.replace(".pdf", "_vlm.txt")
        with open(vlm_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"VLM extraction saved to: {vlm_path}\n")
        return text, vlm_path

    # Default: use existing OCR text
    if not os.path.exists(DEFAULT_TEXT):
        print(f"Error: File not found: {DEFAULT_TEXT}")
        sys.exit(1)
    with open(DEFAULT_TEXT, encoding="utf-8") as f:
        return f.read(), DEFAULT_TEXT


def _run_extraction(report_text: str) -> dict | None:
    """Phase 1: Run RLM to extract structured JSON with source quotes."""
    rlm = RLM(
        backend="ollama",
        backend_kwargs=BACKEND_KWARGS,
        environment="local",
        max_iterations=4,
        max_depth=1,
        custom_system_prompt=EXTRACTION_SYSTEM_PROMPT,
        verbose=True,
    )
    result = rlm.completion(
        prompt=report_text,
        root_prompt=(
            "The pathology report is in `context`. Read it, then output a JSON object "
            "filling every field with a value and a verbatim quote from the report. "
            "If something is not mentioned, set value to 'not reported'."
        ),
    )
    return parse_json_from_response(result.response), result


def _run_correction(report_text: str, errors: list[str]) -> dict | None:
    """Re-run extraction with error feedback."""
    error_text = "\n".join(f"- {e}" for e in errors)
    correction_prompt = CORRECTION_PROMPT_TEMPLATE.format(errors=error_text)

    rlm = RLM(
        backend="ollama",
        backend_kwargs=BACKEND_KWARGS,
        environment="local",
        max_iterations=3,
        max_depth=1,
        custom_system_prompt=EXTRACTION_SYSTEM_PROMPT + "\n\n" + correction_prompt,
        verbose=True,
    )
    result = rlm.completion(
        prompt=report_text,
        root_prompt=(
            "The previous extraction had errors. Re-read `context` and produce a corrected JSON."
        ),
    )
    return parse_json_from_response(result.response), result


def _run_synthesis(verified_json: dict):
    """Phase 3: Synthesize the 4-section report from verified JSON via a single LM call (no RLM)."""
    json_str = json.dumps(verified_json, indent=2)
    client = get_client("ollama", BACKEND_KWARGS)
    user_content = (
        "Produce the four-section clinical report from this verified extraction. "
        "Output only markdown: ## 1. PATIENT SUMMARY, ## 2. EXPERT SUMMARY, "
        "## 3. ONCOLOGY GUIDELINES, ## 4. CONSULTATION RECOMMENDATIONS.\n\n"
        "Verified JSON:\n" + json_str
    )
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    t0 = time.perf_counter()
    report = client.completion(messages)
    elapsed = time.perf_counter() - t0
    # Drop leading/trailing code fences if the model wrapped the markdown
    report = report.strip()
    if report.startswith("```") and "\n" in report:
        first_nl = report.index("\n") + 1
        end = report.rfind("```")
        if end > first_nl:
            report = report[first_nl:end].strip()
    return report, SimpleNamespace(
        response=report,
        execution_time=elapsed,
        usage_summary=client.get_usage_summary(),
    )


def main():
    print("=" * 100)
    print("RLM PATHOLOGY REPORT ANALYSIS (grounded pipeline)")
    print("=" * 100)
    print()

    # ── Step 1: Get report text ───────────────────────────────────────────
    report_text, source_path = _get_report_text()
    print(f"Source: {source_path}")
    print(f"Report length: {len(report_text)} characters")
    print()

    # ── Step 2: Structured extraction ─────────────────────────────────────
    print("=" * 80)
    print("PHASE 1: STRUCTURED JSON EXTRACTION")
    print("=" * 80)
    extraction_data, extraction_result = _run_extraction(report_text)
    total_time = extraction_result.execution_time

    if extraction_data is None:
        print("\nERROR: Model did not return valid JSON. Raw output:")
        print(extraction_result.response[:2000])
        return

    print("\nExtracted JSON:")
    print(json.dumps(extraction_data, indent=2)[:3000])
    print()

    # ── Step 3: Verification loop ─────────────────────────────────────────
    print("=" * 80)
    print("PHASE 2: VERIFICATION")
    print("=" * 80)
    errors = verify_extraction(extraction_data)

    retry = 0
    while errors and retry < MAX_VERIFY_RETRIES:
        print(f"\nVerification FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        print(f"\nRetrying extraction (attempt {retry + 2}/{MAX_VERIFY_RETRIES + 1})...")

        corrected, correction_result = _run_correction(report_text, errors)
        total_time += correction_result.execution_time

        if corrected is not None:
            extraction_data = corrected
            errors = verify_extraction(extraction_data)
        else:
            print("  Correction did not return valid JSON; keeping previous extraction.")
        retry += 1

    if errors:
        print(f"\nVerification still has {len(errors)} warnings after retries:")
        for e in errors:
            print(f"  WARNING: {e}")
        print("Proceeding with best-effort extraction.\n")
    else:
        print("\nVerification PASSED — all checks OK.\n")

    # ── Step 4: Grounded synthesis ────────────────────────────────────────
    print("=" * 80)
    print("PHASE 3: GROUNDED SYNTHESIS")
    print("=" * 80)
    final_report, synthesis_result = _run_synthesis(extraction_data)
    total_time += synthesis_result.execution_time

    print()
    print("=" * 100)
    print("FINAL REPORT")
    print("=" * 100)
    print()
    print(final_report)
    print()

    # ── Save output ───────────────────────────────────────────────────────
    base = source_path.rsplit(".", 1)[0]
    output_file = base + "_rlm_full_analysis.txt"
    json_file = base + "_extraction.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(extraction_data, f, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("ENDOMETRIAL CANCER PATHOLOGY REPORT ANALYSIS\n")
        f.write("Generated using RLM (grounded pipeline with verification)\n")
        f.write("=" * 100 + "\n\n")
        f.write(final_report)
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("\nProcessing Details:\n")
        f.write(f"- Pipeline: VLM extraction → JSON extraction → verification → synthesis\n")
        f.write(f"- Model: {OLLAMA_MODEL}\n")
        f.write(f"- Backend: Ollama ({OLLAMA_URL})\n")
        f.write(f"- Source: {source_path}\n")
        f.write(f"- Verification retries: {retry}\n")
        remaining = [e for e in errors]
        f.write(f"- Verification warnings remaining: {len(remaining)}\n")
        f.write(f"- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Total Execution Time: {total_time:.2f} seconds\n")
        f.write("=" * 100 + "\n")

    print(f"Analysis saved to: {output_file}")
    print(f"Extraction JSON saved to: {json_file}")
    print(f"Total time: {total_time:.2f}s, Retries: {retry}")


if __name__ == "__main__":
    main()
