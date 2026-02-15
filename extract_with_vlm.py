#!/usr/bin/env python3
"""
VLM-based PDF text extraction.

Converts each PDF page to an image and sends it to a vision-language model
(e.g. medgemma via Ollama) to extract text verbatim.  This avoids OCR errors
entirely and produces clean, structured text for downstream analysis.

Usage:
    python extract_with_vlm.py <pdf_path> [--model MODEL] [--base-url URL]
"""

import argparse
import base64
import io
import os
import sys
import tempfile

import requests

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pdf2image import convert_from_path

    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

# Default VLM model (must support vision)
DEFAULT_MODEL = "alibayram/medgemma:latest"
DEFAULT_BASE_URL = "http://localhost:11434"
RENDER_DPI = 300

VLM_EXTRACTION_PROMPT = (
    "You are a medical document transcriber. "
    "Extract ALL text from this pathology report page EXACTLY as it appears. "
    "Rules:\n"
    "- Reproduce every word, number, and abbreviation verbatim.\n"
    "- Preserve the original structure (headings, bullet points, tables).\n"
    "- Do NOT interpret, summarize, or rephrase anything.\n"
    "- Do NOT add any commentary or analysis.\n"
    "- If text is unclear or partially visible, write [illegible] for that portion.\n"
    "- Include all specimen labels (A, B, C, D, E), measurements, and staging info.\n"
    "Output ONLY the transcribed text."
)


def _render_pages_pymupdf(pdf_path: str) -> list[bytes]:
    """Render PDF pages as PNG bytes using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    try:
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(dpi=RENDER_DPI)
            pages.append(pix.tobytes("png"))
    finally:
        doc.close()
    return pages


def _render_pages_pdf2image(pdf_path: str) -> list[bytes]:
    """Render PDF pages as PNG bytes using pdf2image."""
    pil_pages = convert_from_path(pdf_path, dpi=RENDER_DPI)
    pages = []
    for pil_page in pil_pages:
        buf = io.BytesIO()
        pil_page.save(buf, format="PNG")
        pages.append(buf.getvalue())
    return pages


def render_pdf_pages(pdf_path: str) -> list[bytes]:
    """Render each PDF page as PNG bytes. Uses PyMuPDF if available, else pdf2image."""
    if fitz is not None:
        return _render_pages_pymupdf(pdf_path)
    if HAS_PDF2IMAGE:
        return _render_pages_pdf2image(pdf_path)
    raise RuntimeError(
        "No PDF renderer available. Install PyMuPDF (pip install pymupdf) "
        "or pdf2image (pip install pdf2image)."
    )


def vlm_extract_page(
    image_bytes: bytes,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Send a page image to the VLM and return the extracted text."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": VLM_EXTRACTION_PROMPT,
                    "images": [b64],
                }
            ],
            "stream": False,
            "options": {
                "num_ctx": 8192,
                "num_predict": 4096,
                "temperature": 0,
            },
        },
        timeout=1800,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def extract_pdf_with_vlm(
    pdf_path: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Extract text from entire PDF using VLM, page by page."""
    pages = render_pdf_pages(pdf_path)
    full_text = ""

    for i, page_bytes in enumerate(pages):
        print(f"  Extracting page {i + 1}/{len(pages)} via VLM...")
        page_text = vlm_extract_page(page_bytes, model=model, base_url=base_url)
        full_text += f"\n--- Page {i + 1} ---\n{page_text}\n"

    return full_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Extract PDF text with a vision-language model")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama VLM model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama base URL")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: <pdf_basename>_vlm.txt)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: File not found: {args.pdf_path}")
        sys.exit(1)

    print(f"Extracting text from: {args.pdf_path}")
    print(f"Using VLM model: {args.model}")
    print()

    text = extract_pdf_with_vlm(args.pdf_path, model=args.model, base_url=args.base_url)

    output_path = args.output
    if output_path is None:
        base_name = os.path.splitext(args.pdf_path)[0]
        output_path = base_name + "_vlm.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\nExtraction complete!")
    print(f"Output saved to: {output_path}")
    print(f"Total characters: {len(text)}")


if __name__ == "__main__":
    main()
