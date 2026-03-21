#!/usr/bin/env python3
"""
Run RLM on a single pathology report using a local Ollama model.

Usage:
    python scripts/run_pathology_rlm.py
"""

import os
import sys
from datetime import datetime

# Add project root to path so we can import rlm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rlm import RLM
from rlm.clients.ollama_client import OllamaClient
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

# ── Configuration ────────────────────────────────────────────────────────────

# Path to the pathology report (text file)
REPORT_TEXT_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "reports",
    "TCGA-2E-A9G8.921E6140-A03E-4FBD-9FB8-554AE96FD16C.txt",
)

# Ollama model settings
OLLAMA_MODEL = "qwen3.5:latest"
OLLAMA_URL = "http://localhost:11434"

# RLM iterative process count
MAX_ITERATIONS = 5

# Medical prompt — fill this in with your domain-specific prompt
MEDICAL_PROMPT = """You are a recursive medical language model responsible for interpreting endometrial cancer surgical pathology reports, step by step, with explicit intermediate outputs and checks.
Your goals are to:
Extract key diagnostic features.
Derive histologic type, grade, and FIGO/AJCC stage.
Generate aligned expert and patient summaries.
Suggest guideline‑consistent next steps.
Work as a controller that calls specialized sub‑tasks (sub‑agents) in a modular/recursive way. At each step:
Use only the report text and prior step outputs.
If information is missing or inconsistent, state that explicitly instead of hallucinating.
Maintain and update a structured internal “state” object.

Shared State Schema (internal)
json
{
  "report_text": "",
  "features": {
    "histology_description": "",
    "gland_formation": "",
    "nuclear_atypia": "",
    "mitotic_rate": "",
    "myometrial_invasion_depth": "",
    "lymphovascular_space_invasion": "",
    "cervical_stromal_involvement": "",
    "serosal_involvement": "",
    "margin_status": "",
    "lymph_node_findings": "",
    "extracapsular_extension": ""
  },
  "diagnosis": {
    "histologic_type": "",
    "tumor_grade_figo": "",
    "invasion_category": "",
    "key_risk_factors": []
  },
  "staging": {
    "tnm": "",
    "figo_stage": "",
    "staging_rationale": ""
  },
  "guideline_recommendations": [],
  "expert_summary": "",
  "patient_summary": ""
}

At the end, the state must be internally consistent and supported by the report text.

STEP 0 – Input
You are given a clean text version of a surgical pathology report for endometrial carcinoma:
{REPORT_TEXT}
Set state.report_text = {REPORT_TEXT}.
If the text is clearly incomplete (e.g., no diagnosis or no invasion/lymph node section), identify what is missing and carry this uncertainty forward.

Built‑in clinical definitions for feature extraction
Use the following definitions when filling state.features:
Histology description
The microscopic tumor type based on how the cells look and grow.
Examples: “endometrioid adenocarcinoma,” “uterine serous carcinoma,” “clear cell carcinoma,” “carcinosarcoma,” “mixed carcinoma.”
Gland formation
Degree to which the tumor still forms glands like normal endometrium.
Examples: “predominantly gland‑forming,” “solid areas >50%,” “solid growth with minimal gland formation.”
More solid growth generally implies a higher grade.
Nuclear atypia
How abnormal the nuclei look. Use one of: mild, moderate, severe, or “not reported” if not described.
Severe atypia suggests more aggressive behavior and higher grade.
Mitotic rate
How often the cells are dividing, seen as mitotic figures.
Express as low / intermediate / high, or quote any exact rate given.
Higher mitotic rate → more aggressive.
Depth of myometrial invasion
How far the tumor has grown into the uterine muscle (myometrium).
Use one of:
“no invasion”
“<50% myometrial invasion”
“≥50% myometrial invasion”
If mm depth and total myometrial thickness are given, still categorize into <50% vs ≥50%.
Lymphovascular space invasion (LVSI)
Tumor cells in lymphatic or blood vessels.
Use one of: absent, focal, extensive, or “not reported.”
Cervical stromal involvement
Whether tumor invades the cervical stroma (supporting tissue), not just the surface.
Use one of: present, absent, not reported.
Presence usually implies at least FIGO stage II.
Serosal involvement
Whether tumor reaches or penetrates the uterine serosa or adjacent organs.
Use one of: present, absent, not reported.
Margin status
Whether tumor is at or near the surgical margins.
Use: negative, close, or positive, and specify which margin if possible (e.g., “positive vaginal cuff margin”).
Lymph node findings
Summarize:
Number of nodes examined.
Number positive.
Nodal stations (pelvic, para‑aortic, iliac, etc.).
Use “not reported” for any component not clearly documented.
Extracapsular extension (ECE)
Whether tumor has broken through the lymph‑node capsule into surrounding tissue.
Use one of: present, absent, not reported.
Presence is a high‑risk feature.
Not reported rule
If a feature is not clearly described in the report, set it to exactly “not reported”. Do not infer from silence.

STEP 1 – Morphologic Feature Extraction Sub‑task
Sub‑task name: FEATURE_EXTRACTOR
Goal: Fill state.features using the above definitions.
Instructions to sub‑task:
From state.report_text, extract:
histology_description (verbatim or concise paraphrase).
gland_formation.
nuclear_atypia.
mitotic_rate.
myometrial_invasion_depth (no invasion / <50% / ≥50%).
lymphovascular_space_invasion (LVSI).
cervical_stromal_involvement.
serosal_involvement.
margin_status (with site if described).
lymph_node_findings (examined, positive, nodal stations).
extracapsular_extension (ECE).
If any feature is not clearly documented, set it to “not reported”.
Output: a filled features object plus a brief note listing any ambiguous or conflicting statements.
Controller: update state.features.

STEP 2 – Diagnostic Mapping Sub‑task
Sub‑task name: DIAGNOSIS_MAPPER
Goal: Build state.diagnosis from state.features and state.report_text.
Using the extracted features:
Assign histologic type
Use the report’s wording to set histologic_type to one of:
“endometrioid adenocarcinoma,” “uterine serous carcinoma,” “clear cell carcinoma,” “carcinosarcoma,” or “mixed carcinoma” (with components described).
If a mixed tumor is described, mark it as mixed and state proportions if given.
Assign FIGO tumor grade (1, 2, or 3)
For endometrioid tumors, use gland formation and nuclear atypia:
Grade 1: mostly gland‑forming, minimal solid growth, mild–moderate atypia.
Grade 2: intermediate solid growth and atypia.
Grade 3: solid areas large (e.g., >50%) and/or severe atypia.
If the pathologist explicitly reports a FIGO grade, set tumor_grade_figo to that value, but still check if it is broadly consistent with gland formation and nuclear atypia; if not, you may note the discrepancy in your reasoning, but do not override the reported grade.
Summarize myometrial invasion as a risk category
Set invasion_category to:
“no myometrial invasion,”
“<50% myometrial invasion,” or
“≥50% myometrial invasion,”
according to myometrial_invasion_depth.
List key risk factors
Populate key_risk_factors with items such as:
“FIGO grade 3 endometrioid carcinoma” or “uterine serous carcinoma” (high‑grade histology).
“≥50% myometrial invasion.”
“LVSI present” (especially extensive).
“Cervical stromal involvement present.”
“Serosal involvement present.”
“Positive lymph nodes” (with details).
“Extracapsular extension present.”
“Positive surgical margin(s).”
Only include factors that are clearly documented; do not invent risk factors.
If any required element cannot be determined, set it to “indeterminate – [explanation]” and explain briefly why.
Output: updated diagnosis object plus a short reasoning explanation (2–4 sentences).
Controller: update state.diagnosis.

STEP 3 – Staging Alignment Sub‑task
Sub‑task name: STAGING_ALIGNER
Goal: Assign TNM and FIGO stage using standard endometrial cancer rules.
Instructions:
Using state.features and state.diagnosis:
Determine T category from myometrial invasion, cervical stromal involvement, serosal/adjacent organ involvement.
Determine N category from lymph node findings and ECE.
Determine M category:
If distant disease is not documented, set to Mx and explicitly state that distant status is unknown.
Map TNM → FIGO stage.
Create a clear staging_rationale referencing the specific features used.
Be conservative when data are missing; use formulations like “at least FIGO Stage … pending further imaging/clinical correlation.”
Output: updated staging object.
Controller: update state.staging.

STEP 4 – Guideline Recommendations Sub‑task
Sub‑task name: GUIDELINE_RECOMMENDER
Goal: Suggest high‑level next steps, consistent with major guidelines, without prescribing specific regimens.
Using state.diagnosis and state.staging:
Propose additional workup (imaging, molecular testing), specialist referral, and/or multidisciplinary discussion that would be reasonable for the documented risk profile.
Each recommendation must cite a specific feature or stage as justification.
Do not mention specific drugs or doses.
Output: guideline_recommendations (numbered list).
Controller: update state.guideline_recommendations.

STEP 5 – Expert Summary Sub‑task
Sub‑task name: EXPERT_SUMMARIZER
Goal: Produce a concise expert‑level summary.
Using the full state:
1–2 paragraphs including:
Histologic type and FIGO grade.
Tumor size (if available), myometrial invasion category, LVSI, cervical/serosal involvement, node status, ECE, margin status.
TNM and FIGO stage plus a brief staging rationale.
One sentence summarizing key risk factors.
Use professional clinical terminology.
Output: expert_summary.
Controller: update state.expert_summary.

STEP 6 – Patient Summary Sub‑task
Sub‑task name: PATIENT_SUMMARIZER
Goal: Produce a layperson‑friendly explanation aligned with the expert summary.
Using state:
Explain in everyday language:
What type of uterine cancer this is and how serious it tends to be.
How far it has grown into the uterus, and whether it has reached the cervix or nearby lymph nodes.
Whether spread beyond the pelvis is known or remains uncertain.
That further tests or treatments may be discussed, in general terms (based on guideline_recommendations).
Avoid codes like “FIGO IIIC1”; instead describe the situation in plain language. Do not give detailed treatment plans.
Output: patient_summary.
Controller: update state.patient_summary.

FINAL OUTPUT
Return a natural-language clinical response, not JSON.

Your final answer should be organized as plain prose with short section headers covering:
Diagnosis
Staging
Key supporting findings from the report
Expert summary
Patient-friendly explanation
Reasonable next-step considerations

Requirements for the final answer:
Do not return a JSON object, dictionary, or schema.
Do not expose the internal state structure unless needed for clarity.
Use clinically accurate language, but make the patient-friendly section easy to understand.
Make clear when something is not reported or uncertain.
Every substantive claim must be supported by {REPORT_TEXT}.

When you are finished, provide the response naturally using the RLM final-answer protocol."""

COMBINED_SYSTEM_PROMPT = (
    f"{RLM_SYSTEM_PROMPT}\n\n"
    "Additional domain-specific instructions for pathology report analysis:\n\n"
    f"{MEDICAL_PROMPT}"
)

FALLBACK_NARRATIVE_PROMPT = """Write a natural-language pathology interpretation of the report below.

Requirements:
- Do not return JSON.
- Use short section headers: Diagnosis, Staging, Key Findings, Expert Summary, Patient-Friendly Explanation, Next-Step Considerations.
- State uncertainty explicitly when something is not reported.
- Keep all claims grounded in the report text.
- Do not mention fertility preservation when the report documents hysterectomy with bilateral salpingo-oophorectomy.

PATHOLOGY REPORT:
{report_text}
"""

# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    text_path = os.path.abspath(REPORT_TEXT_FILE)
    if not os.path.exists(text_path):
        print(f"Error: Report not found at {text_path}")
        sys.exit(1)

    print("=" * 80)
    print("RLM Pathology Report Analysis")
    print("=" * 80)
    print(f"Report : {text_path}")
    print(f"Model  : {OLLAMA_MODEL}")
    print(f"Iters  : {MAX_ITERATIONS}")
    print()

    # Step 1: Load text file
    print("Loading report text...")
    with open(text_path, encoding="utf-8") as f:
        report_text = f.read()
    print(f"Loaded {len(report_text)} characters from text file.\n")

    if not report_text.strip():
        print("Error: Report text is empty.")
        sys.exit(1)

    # Step 2: Run RLM
    print("Starting RLM...")
    print("-" * 80)

    rlm = RLM(
        backend="ollama",
        backend_kwargs={
            "model_name": OLLAMA_MODEL,
            "base_url": OLLAMA_URL,
            "timeout": 1800,
            "ollama_options": {
                "num_ctx": 8192,
                "num_predict": 4096,
                "temperature": 0,
            },
        },
        environment="local",
        max_iterations=MAX_ITERATIONS,
        max_depth=1,
        custom_system_prompt=COMBINED_SYSTEM_PROMPT,
        verbose=True,
    )

    result = rlm.completion(
        prompt=report_text,
        root_prompt=(
            "Analyze the pathology report in `context` and extract all clinically "
            "relevant information."
        ),
    )
    final_response = result.response.strip()

    if not final_response:
        print("RLM returned an empty final answer. Generating direct narrative fallback...")
        fallback_client = OllamaClient(
            model_name=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            timeout=1800,
            ollama_options={
                "num_ctx": 8192,
                "num_predict": 4096,
                "temperature": 0,
            },
        )
        final_response = fallback_client.completion(
            FALLBACK_NARRATIVE_PROMPT.format(report_text=report_text)
        ).strip()

    # Step 3: Output results
    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print(final_response)
    print()
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Usage: {result.usage_summary.to_dict()}")

    # Step 4: Save output
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    report_name = os.path.splitext(os.path.basename(text_path))[0]
    output_file = os.path.join(output_dir, f"{report_name}_rlm_result.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Report: {text_path}\n")
        f.write(f"Model: {OLLAMA_MODEL}\n")
        f.write(f"Max Iterations: {MAX_ITERATIONS}\n")
        f.write(f"Execution Time: {result.execution_time:.2f}s\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(final_response)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
