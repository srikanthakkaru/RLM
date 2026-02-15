import os

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path

# Optional: native PDF text extraction (install with: uv pip install -e ".[ocr]")
try:
    import fitz
except ImportError:
    fitz = None

# If on Windows, uncomment and set your Tesseract path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# DPI for rendering PDF pages when OCR is used (higher = better accuracy, slower).
OCR_DPI = 300

# Minimum characters per page to accept native text; below this we run OCR for that page.
MIN_NATIVE_PAGE_CHARS = 100


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Deskew, denoise, and binarize image for better Tesseract accuracy."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    # Binarize with Otsu
    _, binary = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Deskew: detect skew angle and rotate
    coords = np.column_stack(np.where(binary > 0))
    if coords.size < 100:
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.5:
        return binary
    (h, w) = binary.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def _extract_page_text_native(doc: "fitz.Document", page_index: int) -> str:
    """Extract text from a single page using PyMuPDF (no OCR)."""
    page = doc.load_page(page_index)
    return page.get_text()


def _ocr_single_page_image(image: np.ndarray) -> str:
    """Run Tesseract OCR on a single page image (grayscale or BGR)."""
    preprocessed = _preprocess_for_ocr(image)
    return pytesseract.image_to_string(
        preprocessed, lang="eng", config="--psm 6"
    )


def perform_ocr_pdf(pdf_path: str) -> str:
    """
    Extract text from the given PDF. Uses native text extraction when the PDF
    has a text layer; falls back to OCR per page when a page has too little text.
    """
    if fitz is None:
        return _perform_ocr_pdf_fallback_only(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        try:
            for page_index in range(len(doc)):
                native_text = _extract_page_text_native(doc, page_index)
                if len(native_text.strip()) >= MIN_NATIVE_PAGE_CHARS:
                    full_text += f"\n--- Page {page_index + 1} ---\n"
                    full_text += native_text
                else:
                    # Too little text: render page and run OCR
                    page = doc.load_page(page_index)
                    pix = page.get_pixmap(dpi=OCR_DPI)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    if pix.n == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    ocr_text = _ocr_single_page_image(img)
                    full_text += f"\n--- Page {page_index + 1} ---\n"
                    full_text += ocr_text
        finally:
            doc.close()
        return full_text.strip()
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}") from e


def _perform_ocr_pdf_fallback_only(pdf_path: str) -> str:
    """OCR-only path when PyMuPDF is not installed (original behavior)."""
    try:
        pages = convert_from_path(pdf_path, dpi=OCR_DPI)
        full_text = ""
        for page_number, page in enumerate(pages):
            open_cv_image = np.array(page)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            text = _ocr_single_page_image(open_cv_image)
            full_text += f"\n--- Page {page_number + 1} ---\n"
            full_text += text
        return full_text.strip()
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}") from e


if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF file: ")

    try:
        text = perform_ocr_pdf(pdf_path)

        # Create output text file path
        base_name = os.path.splitext(pdf_path)[0]
        output_file = base_name + "_ocr.txt"

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print("\nOCR completed successfully!")
        print(f"Text saved to: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
