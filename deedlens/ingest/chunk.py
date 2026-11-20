"""Clean page text and cut it into search chunks (Lesson 6).

1. Clean: drop lines with no letters or digits; drop lines already seen earlier in the same
   document (compared with whitespace removed). Very short lines (< 8 chars, e.g. "ss.:")
   may repeat legitimately and are kept.
2. Chunk: pack lines into chunks of about `target` tokens, counted with the EMBEDDING
   MODEL's own tokenizer, with `overlap` tokens carried into the next chunk. A chunk never
   crosses a page boundary, so every chunk cites exactly one page.
3. Label: the first line of every chunk is "{title}, page {n} of {total}".
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from deedlens.text_utils import squash

TokenCounter = Callable[[str], int]
MIN_DEDUP_CHARS = 8


def make_token_counter(model_name: str) -> TokenCounter:
    """Tokens as the embedding model counts them (no [CLS]/[SEP])."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 10**6  # we only count here, so skip the long-text warning
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def clean_document(pages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Cleaned pages as {"page": n, "lines": [...]}, plus the number of dropped lines."""
    seen: set[str] = set()
    dropped = 0
    cleaned = []
    for page in pages:
        keep = []
        for line in (page.get("text") or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if not re.search(r"[A-Za-z0-9]", line):  # e.g. "---", "☒"
                dropped += 1
                continue
            key = squash(line)
            if len(key) >= MIN_DEDUP_CHARS and key in seen:
                dropped += 1
                continue
            seen.add(key)
            keep.append(line)
        cleaned.append({"page": page["page"], "lines": keep})
    return cleaned, dropped


def split_long(line: str, limit: int, count: TokenCounter) -> list[str]:
    """Split a line that is too long into word-boundary pieces of at most `limit` tokens."""
    pieces: list[str] = []
    current: list[str] = []
    for word in line.split():
        current.append(word)
        if len(current) > 1 and count(" ".join(current)) > limit:
            current.pop()
            pieces.append(" ".join(current))
            current = [word]
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_page(lines: list[str], header: str, count: TokenCounter, target: int, overlap: int) -> list[str]:
    """Greedily pack lines into chunks that fit `target` tokens (header included), with overlap."""
    budget = target - count(header)
    unit_limit = budget - overlap  # so overlap + any one unit always fits
    if unit_limit <= 0:
        raise ValueError(f"header {header!r} leaves no room in a {target}-token chunk")
    units: list[tuple[str, int]] = []
    for line in lines:
        n = count(line)
        if n <= unit_limit:
            units.append((line, n))
        else:
            units.extend((piece, count(piece)) for piece in split_long(line, unit_limit, count))

    chunks: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    for text, n in units:
        if current and current_tokens + n > budget:
            chunks.append(current)
            carry: list[tuple[str, int]] = []  # take lines from the end until the overlap is used up
            carry_tokens = 0
            for unit in reversed(current):
                if carry_tokens + unit[1] > overlap:
                    break
                carry.insert(0, unit)
                carry_tokens += unit[1]
            current, current_tokens = carry, carry_tokens
        current.append((text, n))
        current_tokens += n
    if current:
        chunks.append(current)
    return [header + "\n" + "\n".join(t for t, _ in c) for c in chunks]


def chunk_document(pages: list[dict[str, Any]], meta: dict[str, Any], count: TokenCounter,
                   target: int = 200, overlap: int = 40) -> list[dict[str, Any]]:
    """All chunks of one document with their metadata.

    `meta` needs doc_id and title; instrument_type, year, recorded_at and parcels are copied
    into every chunk so search can filter on them.
    """
    cleaned, _ = clean_document(pages)
    total = len(pages)
    out = []
    for page in cleaned:
        if not page["lines"]:
            continue
        header = f"{meta['title']}, page {page['page']} of {total}"
        for i, text in enumerate(chunk_page(page["lines"], header, count, target, overlap)):
            out.append({
                "id": f"{meta['doc_id']}-p{page['page']}-c{i}",
                "doc_id": meta["doc_id"],
                "page": page["page"],
                "chunk": i,
                "tokens": count(text),
                "title": meta["title"],
                "instrument_type": meta.get("instrument_type"),
                "year": meta.get("year"),
                "recorded_at": meta.get("recorded_at"),
                "parcels": list(meta.get("parcels") or []),
                "text": text,
            })
    return out
