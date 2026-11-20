"""Small text helpers shared by ingestion, search, records and guardrails."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def squash(text: object) -> str:
    """Lowercase with all whitespace removed.

    Used for "is this value in that text" checks: OCR and LLMs break lines and
    spaces differently, so comparing squashed strings avoids false misses (Lessons 4, 13).
    """
    return _WS.sub("", "" if text is None else str(text)).lower()


def digits(value: object) -> str:
    """Only the digits of a value: '8,310' -> '8310', None -> ''."""
    return re.sub(r"\D", "", "" if value is None else str(value))


def to_int(value: object) -> int | None:
    """Integer from the digits of a value, or None if it has no digits."""
    d = digits(value)
    return int(d) if d else None


# ---------- document ids ----------

def normalize_doc_id(file_name: str) -> str:
    """Filename -> doc_id: 'Deed 1089 349.pdf' -> 'Deed_1089_349'.

    Runs of spaces/underscores become one underscore; anything outside
    letters, digits, '.', '-' and '_' also becomes '_' so the id is safe in URLs.
    """
    stem = re.sub(r"\.pdf$", "", file_name.strip(), flags=re.I)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


# ---------- parcels (tax map / SBL numbers) ----------

# Lesson 8: a tax map number like 70.15-5-11 (optionally with a sub-lot like 70.15-5-11.2).
# The section part must contain a dot, so dates such as 01-15-2004 are not taken for parcels.
# The lookarounds stop a match from starting or ending in the middle of a longer number.
PARCEL_RE = re.compile(r"(?<![\d.])\d+\.\d+-\d+-\d+(?:\.\d+)?(?![\d-])")


def normalize_parcel(value: object) -> str | None:
    """First parcel number in a value: 'Part of 70.15-5-11' -> '70.15-5-11'."""
    if value is None:
        return None
    m = PARCEL_RE.search(str(value))
    return m.group() if m else None


def find_parcels(text: str) -> list[str]:
    """Every distinct parcel number in a text, in order of first appearance."""
    seen: dict[str, None] = {}
    for m in PARCEL_RE.finditer(text or ""):
        seen.setdefault(m.group(), None)
    return list(seen)


# ---------- BM25 tokenizer ----------

# Lesson 10: keep numbers with inner . , ' - whole (48,468.96 and 70.15-5-11 stay one token);
# a whitespace split would glue punctuation onto words and miss exact matches.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,'\-][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a an and are as at be by did do does for from how in is it of on or
    that the this to was what when where which who whom with""".split()
)


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens without stopwords, for BM25."""
    return [w for w in TOKEN_RE.findall((text or "").lower()) if w not in STOPWORDS]
