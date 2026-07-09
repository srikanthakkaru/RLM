from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "output"

sys.path.insert(0, str(_REPO_ROOT))

from rlm import RLM

OLLAMA_URL = "http://localhost:11434"
# The 27B reasoning model tag is set at runtime via RLM_MODEL; the exact digest is
# supplied by the environment, so keep only a placeholder default here.
RLM_MODEL = os.environ.get("RLM_MODEL", "gemma2:27b")
RLM_BACKEND = os.environ.get("RLM_BACKEND", "ollama")
RLM_BASE_URL = os.environ.get("RLM_BASE_URL")
RLM_API_KEY = os.environ.get("RLM_API_KEY")

DEFAULT_TIMEOUT = 1800

# Recursion is ENABLED and UNCAPPED: this is a pure, unbounded RLM ablation. There is no
# depth cap and no iteration cap — the model itself decides when to stop by emitting
# FINAL(...). We pass sys.maxsize for both, since the library loops with
# `for i in range(self.max_iterations)` and gates depth with
# `if self.depth >= self.max_depth: return self._fallback_answer(...)`; a huge int is lazy
# for range() (no allocation) and simply never trips the depth fallback.
# NOTE (be honest): in this library `max_depth` only gates that top-level fallback, and the
# RLM docstring states "Currently, only depth 1 is supported." `llm_query` /
# `llm_query_batched` sub-calls go through the LMHandler as FLAT `client.completion(...)`
# calls, not nested RLMs with their own REPL. So an unbounded max_depth does NOT spawn extra
# nested REPL agents today; it just guarantees the depth fallback is never what stops the
# run. Termination is entirely model-driven via FINAL(...)/FINAL_VAR(...).
UNBOUNDED_DEPTH = sys.maxsize
UNBOUNDED_ITERATIONS = sys.maxsize

_OLLAMA_RLM_OPTIONS: dict[str, Any] = {"num_ctx": 32768, "num_predict": 4096, "temperature": 0}

# Anchored on uppercase FINAL / FINAL_VAR only: that is exactly what the RLM library emits,
# so a narrative that legitimately starts with "Final (" keeps its text intact.
_FINAL_WRAPPER_RE = re.compile(r"(?s)^\s*FINAL(?:_VAR)?\s*\((.*)\)\s*$")

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
    timeout: int = DEFAULT_TIMEOUT,
    ollama_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if backend == "ollama":
        return {
            "model_name": model,
            "base_url": base_url or ollama_url,
            "timeout": timeout,
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
    verbose: bool = True,
) -> tuple[str, float]:
    rlm = RLM(
        backend=rlm_backend,
        backend_kwargs=dict(rlm_backend_kwargs),
        environment="local",
        max_depth=UNBOUNDED_DEPTH,
        max_iterations=UNBOUNDED_ITERATIONS,
        custom_system_prompt=SYSTEM_PROMPT,
        verbose=verbose,
    )
    result = rlm.completion(prompt=report_text, root_prompt=ROOT_PROMPT)
    narrative = _strip_final_wrapper(result.response)
    return narrative, result.execution_time


def save_result(
    output_path: Path,
    report_path: Path,
    narrative: str,
    execution_time: float,
    rlm_model: str,
    rlm_backend: str,
) -> None:
    header = (
        f"Report: {report_path}\n"
        f"RLM Model: {rlm_model}\n"
        f"RLM Backend: {rlm_backend}\n"
        "Max Depth: unbounded (model terminates via FINAL)\n"
        "Max Iterations: unbounded (model terminates via FINAL)\n"
        f"Execution Time: {execution_time:.2f}s\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "=" * 80
        + "\n\n"
    )
    output_path.write_text(header + narrative, encoding="utf-8")


app = typer.Typer(help="Pure recursive RLM pathology summarization (unguarded ablation).")
console = Console()


@app.command()
def run(
    reports: Annotated[list[Path], typer.Argument(help="Path(s) to pathology report text file(s)")],
    rlm_model: Annotated[str, typer.Option(help="RLM (27B reasoning) model name")] = RLM_MODEL,
    rlm_backend: Annotated[
        str,
        typer.Option(
            help="RLM backend: ollama | openai | openrouter | vllm | vercel | anthropic | gemini | "
            "azure_openai | portkey | litellm."
        ),
    ] = RLM_BACKEND,
    rlm_base_url: Annotated[
        str | None,
        typer.Option(
            help="Base URL for the RLM backend (e.g. an OpenAI-compatible endpoint). For the "
            "ollama backend this takes precedence over --ollama-url."
        ),
    ] = RLM_BASE_URL,
    rlm_api_key: Annotated[
        str | None, typer.Option(help="API key for the RLM backend (hosted backends only).")
    ] = RLM_API_KEY,
    ollama_url: Annotated[
        str,
        typer.Option(
            help="Ollama base URL (ollama backend only; ignored if --rlm-base-url is set)."
        ),
    ] = OLLAMA_URL,
    timeout: Annotated[
        int, typer.Option(help="Request timeout in seconds (ollama backend).")
    ] = DEFAULT_TIMEOUT,
    output_dir: Annotated[
        Path, typer.Option(help="Directory for output files")
    ] = _DEFAULT_OUTPUT_DIR,
    quiet: Annotated[bool, typer.Option(help="Suppress RLM verbose output")] = False,
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
        timeout=timeout,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    num_ctx = _OLLAMA_RLM_OPTIONS["num_ctx"]
    processed = 0
    skipped = 0
    failed = 0

    for report_path in track(reports, description="Processing reports..."):
        if not report_path.exists():
            console.print(f"[red]Skipping missing file:[/red] {report_path}")
            skipped += 1
            continue

        report_text = report_path.read_text(encoding="utf-8", errors="replace")

        if not report_text.strip():
            console.print(f"[yellow]Skipping empty file:[/yellow] {report_path}")
            skipped += 1
            continue

        # Rough 4-chars-per-token heuristic: warn (don't block) when a report is likely to
        # exceed the model context, since Ollama truncates silently and a partial report
        # would be summarized as if complete — dangerous for a grounding-focused ablation.
        estimated_tokens = len(report_text) // 4
        if estimated_tokens > num_ctx:
            console.print(
                f"[yellow]Warning:[/yellow] {report_path.name} is ~{estimated_tokens} tokens, "
                f"which may exceed num_ctx={num_ctx}; the report could be silently truncated."
            )

        output_file = output_dir / f"{report_path.stem}_pure_rlm_summary.txt"
        if output_file.exists():
            console.print(f"[yellow]Overwriting existing output:[/yellow] {output_file}")

        console.print(f"\n[bold cyan]Processing:[/bold cyan] {report_path.name}")
        console.print(f"  RLM: {rlm_backend} / {rlm_model}")
        console.print("  Max Depth: unbounded  Max Iterations: unbounded (model-terminated)")

        try:
            narrative, exec_time = run_pure_rlm(
                report_text,
                rlm_backend=rlm_backend,
                rlm_backend_kwargs=rlm_backend_kwargs,
                verbose=not quiet,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch alive on any single failure
            failed += 1
            console.print(f"[red]Failed:[/red] {report_path.name} -> {type(exc).__name__}: {exc}")
            error_file = output_dir / f"{report_path.stem}_pure_rlm_ERROR.txt"
            error_file.write_text(
                f"Report: {report_path}\nError: {type(exc).__name__}: {exc}\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                encoding="utf-8",
            )
            continue

        save_result(
            output_file,
            report_path,
            narrative,
            exec_time,
            rlm_model=rlm_model,
            rlm_backend=rlm_backend,
        )
        processed += 1

        console.print(f"[green]Saved:[/green] {output_file}")
        console.print(f"[green]Execution time:[/green] {exec_time:.2f}s")

    console.print(
        f"\n[bold]Done.[/bold] Processed: [green]{processed}[/green]  "
        f"Skipped: [yellow]{skipped}[/yellow]  Failed: [red]{failed}[/red]"
    )
    if processed == 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
