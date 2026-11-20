"""Gemini vision transcription of scanned pages (Lesson 5: 50/50 facts vs 44/50 for Tesseract).

Gemini can fall into a repetition loop (once: 41,920 characters of dashes on a form line), so:
- the prompt asks for dashed form lines to be written as `---`;
- output is capped at 8,192 tokens (a full page is ~1,000-2,000);
- suspect checks: same character x50+, same phrase x8+, > 3x the Tesseract length, or a
  finish reason other than STOP;
- one retry at temperature 0.4; if still suspect, the page is marked "suspect" and the
  Tesseract text is used instead.
"""

from __future__ import annotations

import io
import re
from typing import Any

from google.genai import types
from PIL import Image

from deedlens.ingest.gemini import Gemini, GeminiError, finish_reason

TRANSCRIBE_DPI = 200  # plenty for a vision model, and keeps uploads small
MAX_OUTPUT_TOKENS = 8192

PROMPT = """Transcribe every piece of text on this scanned page exactly as written.
Include typed text, handwriting, stamps, margin notes, and every table row.
Keep the original line breaks and reading order. Do not fix spelling, do not
summarize, and do not add commentary. If a word is unreadable, write [illegible].
Form lines made of dashes, dots or underscores: write just three of them (---),
never the whole line."""


def suspect_reason(text: str, tess_text: str, finish: str = "STOP") -> str | None:
    """Why a transcription looks broken, or None if it looks fine."""
    if finish not in ("STOP", "FinishReason.STOP"):
        return f"finish reason {finish}"
    if re.search(r"(.)\1{50,}", text):
        return "same character repeated 50+ times"
    if re.search(r"(.{4,80}?)\1{8,}", text, flags=re.S):
        return "same phrase repeated 8+ times"
    if len(text) > 3 * max(len(tess_text), 500):
        return f"{len(text):,} chars vs {len(tess_text):,} from Tesseract"
    return None


def to_jpeg_part(image: Image.Image, source_dpi: int) -> types.Part:
    """Downscale a rendered page to TRANSCRIBE_DPI and wrap it as a JPEG for the API."""
    if source_dpi > TRANSCRIBE_DPI:
        scale = TRANSCRIBE_DPI / source_dpi
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _call(gemini: Gemini, part: types.Part, temperature: float) -> tuple[str, str]:
    config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=MAX_OUTPUT_TOKENS)
    response = gemini.generate([part, PROMPT], config)
    return response.text or "", finish_reason(response)


def transcribe_page(gemini: Gemini, image: Image.Image, tess_text: str, source_dpi: int) -> dict[str, Any]:
    """Transcribe one page. Returns {"gemini": text, "status": ok|retried_ok|suspect|error, ...}.

    Status "suspect" or "error" means: use the Tesseract text for this page.
    """
    part = to_jpeg_part(image, source_dpi)
    try:
        text, finish = _call(gemini, part, temperature=0)
        reason = suspect_reason(text, tess_text, finish)
        if reason is None:
            return {"gemini": text, "status": "ok"}
        text2, finish2 = _call(gemini, part, temperature=0.4)
        reason2 = suspect_reason(text2, tess_text, finish2)
        if reason2 is None:
            return {"gemini": text2, "status": "retried_ok", "reason": reason}
        return {"gemini": text2, "status": "suspect", "reason": reason2}
    except GeminiError as e:
        if e.fatal:
            raise
        return {"gemini": None, "status": "error", "reason": str(e)[:300]}
