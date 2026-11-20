"""Page text: the PDF's own text layer when it has one, otherwise Tesseract OCR.

Rules carried over from the lessons:
- Lesson 3: a PDF whose pages average >= 100 characters has a real text layer; use it.
- Lesson 5: Tesseract `--psm 6` (one block of text) beat the default layout mode, 37 -> 44 of 50
  facts. Image cleanup (Otsu black/white) made results worse, so pages go in unmodified.
- Lesson 5: Tesseract's confidence ranked variants almost backwards. It is stored for
  information only and never used to judge quality.
- Lesson 9: OCR runs in threads, with OMP_THREAD_LIMIT=1 (one core per tesseract process; the
  threads provide the parallelism) and BMP temp images (PNG compression held the GIL for
  ~0.9 s per page; BMP took OCR from 43.5 s to 16.8 s with identical output).
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # read by each tesseract subprocess

MIN_TEXT_CHARS = 100  # Lesson 3: average characters per page for a real text layer
OCR_CONFIG = "--psm 6"
OCR_DPI = 300


def text_layer_pages(pdf_path: Path) -> list[str]:
    """The existing text of each page ('' for image-only pages)."""
    return [page.extract_text() or "" for page in PdfReader(pdf_path).pages]


def has_text_layer(pages: list[str]) -> bool:
    return sum(len(t) for t in pages) / max(len(pages), 1) >= MIN_TEXT_CHARS


def require_tesseract() -> None:
    if shutil.which("tesseract") is None:
        raise RuntimeError("Tesseract is not installed but a scanned PDF needs OCR. "
                           "Install it with `brew install tesseract` or `apt-get install tesseract-ocr`.")


def ocr_page(image: Image.Image) -> tuple[str, float, int]:
    """OCR one page. Returns (text, mean word confidence 0-100, word count)."""
    import pytesseract

    image.format = "BMP"  # Lesson 9: pytesseract's temp file is written in this format
    data = pytesseract.image_to_data(image, config=OCR_CONFIG, output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        # Tesseract numbers every word by block, paragraph and line; use that to rebuild lines.
        lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(word)
        conf = float(data["conf"][i])
        if conf >= 0:
            confs.append(conf)
    text = "\n".join(" ".join(words) for words in lines.values())
    return text, (sum(confs) / len(confs) if confs else 0.0), len(confs)


def ocr_images(images: list[Image.Image], workers: int) -> list[tuple[str, float, int]]:
    """OCR pages in parallel threads; results stay in page order."""
    require_tesseract()
    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(ocr_page, images))
