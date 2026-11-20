"""Render PDF pages to images with pypdfium2 (Apache-2.0/BSD; PyMuPDF is AGPL so the app avoids it).

PDFium is not thread-safe, so every call goes through one lock. Rendering stays sequential
per document (Lesson 9); OCR of the rendered pages can still run in parallel.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

_PDFIUM_LOCK = threading.Lock()


def page_count(pdf_path: Path) -> int:
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            return len(pdf)
        finally:
            pdf.close()


def render_pages(pdf_path: Path, dpi: int = 300) -> list[Image.Image]:
    """Every page as an RGB image. 300 DPI is the usual sweet spot for Tesseract (Lesson 3)."""
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            return [pdf[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(len(pdf))]
        finally:
            pdf.close()


def render_page(pdf_path: Path, page: int, dpi: int = 120) -> Image.Image:
    """One page (1-based), e.g. for the citation viewer."""
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            if not 1 <= page <= len(pdf):
                raise IndexError(f"{pdf_path.name} has {len(pdf)} pages, not {page}")
            return pdf[page - 1].render(scale=dpi / 72).to_pil().convert("RGB")
        finally:
            pdf.close()
