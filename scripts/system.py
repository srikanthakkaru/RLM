from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

from rlm import RLM

OLLAMA_URL = "http://localhost:11434"
# The 27B reasoning model tag is set at runtime via RLM_MODEL; the exact digest is
# supplied by the environment, so keep only a placeholder default here.
RLM_MODEL = os.environ.get("RLM_MODEL", "gemma2:27b")
RLM_BACKEND = os.environ.get("RLM_BACKEND", "ollama")
RLM_BASE_URL = os.environ.get("RLM_BASE_URL")
RLM_API_KEY = os.environ.get("RLM_API_KEY")

# Recursion is ENABLED: this is a pure, unbounded RLM ablation. max_depth is not 1 and
# recursion is not banned.
MAX_DEPTH = int(os.environ.get("RLM_MAX_DEPTH", "5"))
# NOTE (be honest): the RLM loop is `for i in range(max_iterations)`, so a truly infinite
# value would hang. We use a large finite cap and rely on the model emitting FINAL(...) to
# terminate early. Raise via --max-iterations if the model needs more room.
MAX_ITERATIONS = int(os.environ.get("RLM_MAX_ITERATIONS", "100"))

_OLLAMA_RLM_OPTIONS: dict[str, Any] = {"num_ctx": 32768, "num_predict": 4096, "temperature": 0}

_FINAL_WRAPPER_RE = re.compile(r"(?is)^\s*final(?:_var)?\s*\((.*)\)\s*$")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "output"

SYSTEM_PROMPT = """You are a clinical reasoning agent working from a single surgical pathology report.

The raw OCR pathology report is available in the ```repl``` variable `context` (a string).
There is no upstream extraction, normalization, staging engine, or grounding gate: the
report text is the ONLY source of truth you are given.

How to work:
- You MAY use ```repl``` blocks freely to inspect, slice, and search the `context` text.
  Use as many ```repl``` blocks as you need.
- You MAY call `llm_query(prompt)` and `llm_query_batched([prompt1, prompt2, ...])` to
  recursively decompose the report into sub-questions (for example, analyze histology,
  depth of invasion, lymph nodes, and margins as separate sub-calls) and then combine the
  results. Recursion is encouraged wherever it improves the quality of your summaries.

Grounding rules (strict):
- Base every statement strictly on the wording of the report text in `context`.
- When a detail is absent from the report, say "not reported" — never invent measurements,
  findings, grades, or stages.
- Because there is no upstream staging engine, YOU must reason about the diagnosis and the
  stage yourself, but you must show that your reasoning is grounded in the report's wording
  (quote or paraphrase the specific phrases you relied on).

Termination:
- Do NOT put FINAL(...) inside ```repl``` blocks.
- Finish by returning either FINAL(<full text>) or FINAL_VAR(final_answer) after assigning
  the finished text to `final_answer` in a ```repl``` block.
"""

ROOT_PROMPT = """The ```repl``` variable `context` holds the raw pathology report text (a string).
Read it, optionally decompose it with `llm_query`/`llm_query_batched`, and produce a plain
text answer (NOT JSON) with EXACTLY these three section headers, in this order:

Patient Summary
  Plain language, no jargon, active voice, written directly for the patient.

Expert Summary
  Clinical and technical language, written for a clinician.

Diagnostic Summary
  The diagnosis, the key findings, and your staging rationale grounded in the report's wording.

Base every statement strictly on the report text; state "not reported" when a detail is
absent and never invent findings.

Termination rules:
- Do NOT place FINAL(...) inside ```repl``` blocks.
- When complete, return plain text using one of:
  FINAL(<full text>)
  FINAL_VAR(final_answer) after assigning `final_answer` in a ```repl``` block
"""


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


def _strip_final_wrapper(text: str) -> str:
    stripped = text.strip()
    match = _FINAL_WRAPPER_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def run_pure_rlm(
    report_text: str,
    rlm_backend: str,
    rlm_backend_kwargs: dict[str, Any],
    max_depth: int,
    max_iterations: int,
    verbose: bool = True,
) -> tuple[str, float]:
    rlm = RLM(
        backend=rlm_backend,
        backend_kwargs=dict(rlm_backend_kwargs),
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        custom_system_prompt=SYSTEM_PROMPT,
        verbose=verbose,
    )
    result = rlm.completion(prompt=report_text, root_prompt=ROOT_PROMPT)
    narrative = _strip_final_wrapper(result.response)
    return narrative, result.execution_time


def save_result(
    output_path: str,
    report_path: str,
    narrative: str,
    execution_time: float,
    rlm_model: str,
    rlm_backend: str,
    max_depth: int,
    max_iterations: int,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Report: {report_path}\n")
        f.write(f"RLM Model: {rlm_model}\n")
        f.write(f"RLM Backend: {rlm_backend}\n")
        f.write(f"Max Depth: {max_depth}\n")
        f.write(f"Max Iterations: {max_iterations}\n")
        f.write(f"Execution Time: {execution_time:.2f}s\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(narrative)


app = typer.Typer(help="Pure recursive RLM pathology summarization (unguarded ablation).")
console = Console()


@app.command()
def run(
    reports: Annotated[list[Path], typer.Argument(help="Path(s) to pathology report text file(s)")],
    rlm_model: str = typer.Option(RLM_MODEL, help="RLM (27B reasoning) model name"),
    rlm_backend: str = typer.Option(
        RLM_BACKEND,
        help="RLM backend: ollama | openai | openrouter | vllm | vercel | anthropic | gemini | "
        "azure_openai | portkey | litellm.",
    ),
    rlm_base_url: str | None = typer.Option(
        RLM_BASE_URL, help="Base URL for the RLM backend (e.g. an OpenAI-compatible endpoint)."
    ),
    rlm_api_key: str | None = typer.Option(
        RLM_API_KEY, help="API key for the RLM backend (hosted backends only)."
    ),
    ollama_url: str = typer.Option(OLLAMA_URL, help="Ollama base URL (if backend is ollama)"),
    max_depth: int = typer.Option(MAX_DEPTH, help="Max recursion depth for the RLM"),
    max_iterations: int = typer.Option(MAX_ITERATIONS, help="Max iterations for the RLM REPL loop"),
    output_dir: Annotated[
        Path, typer.Option(help="Directory for output files")
    ] = _DEFAULT_OUTPUT_DIR,
    quiet: bool = typer.Option(False, help="Suppress RLM verbose output"),
):
    """Summarize pathology reports with a single unguarded recursive RLM (no extraction,
    no FIGO engine, no grounding gate)."""

    console.print(Panel.fit(" Pure Recursive RLM Pathology Summaries", style="bold green"))

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
        console.print(f"  RLM: {rlm_backend} / {rlm_model}")
        console.print(f"  Max Depth: {max_depth}  Max Iterations: {max_iterations}")

        narrative, exec_time = run_pure_rlm(
            report_text,
            rlm_backend=rlm_backend,
            rlm_backend_kwargs=rlm_backend_kwargs,
            max_depth=max_depth,
            max_iterations=max_iterations,
            verbose=not quiet,
        )

        output_file = output_dir / f"{report_path.stem}_pure_rlm_summary.txt"
        save_result(
            str(output_file),
            str(report_path),
            narrative,
            exec_time,
            rlm_model=rlm_model,
            rlm_backend=rlm_backend,
            max_depth=max_depth,
            max_iterations=max_iterations,
        )

        console.print(f"[green]Saved:[/green] {output_file}")
        console.print(f"[green]Execution time:[/green] {exec_time:.2f}s")


if __name__ == "__main__":
    app()
