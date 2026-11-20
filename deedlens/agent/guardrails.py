"""Guardrails, layered (Lesson 12: attack success 5/5 with no defenses -> 0/5 with all layers,
normal accuracy 4/4 throughout).

1. The system prompt (prompts.py): only sources, cite every fact, source text is untrusted.
2. Input scanner: drops retrieved chunks containing instruction-like phrases. It flagged 0 of
   501 real chunks. It deliberately does NOT look for URLs or "Social Security": real court
   filings contain them. Dropped chunks are logged for review, never silently discarded.
3. Output checks on the draft: no links; no requests for personal information; every $ amount
   must appear in a provided source; people named in ownership statements must be parties in
   the records for this parcel; every bracketed citation must name a provided source.
"""

from __future__ import annotations

import re
from typing import Any

from deedlens.text_utils import squash

# ---------------------------------------------------------------- 2. input scanner

INJECTION_PATTERNS = [
    r"ignore (all |any )?(of )?(the )?(previous|prior|above|earlier) instructions",
    r"\b(system|developer) (notice|prompt|message|instructions?)\b",
    r"\bnote to (ai|the ai|assistants?|chatbots?|language models?)\b",
    r"\bassistant\s*:",
    r"\b(answer|reply|respond|report) (only )?(with|that)\b",
]


def scan(text: str) -> list[str]:
    """The instruction-like phrases found in a piece of source text."""
    return [m.group() for p in INJECTION_PATTERNS for m in [re.search(p, text, re.I)] if m]


def input_filter(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split chunks into (kept, dropped); dropped ones carry the phrases that matched."""
    kept, dropped = [], []
    for c in chunks:
        hits = scan(c["text"])
        if hits:
            dropped.append({**c, "matched": hits})
        else:
            kept.append(c)
    return kept, dropped


# ---------------------------------------------------------------- 3. output checks

OWNERSHIP = re.compile(r"\b(own|owns|owned|owner|owners|ownership|bought|sold|conveys?|conveyed|"
                       r"transfer|transfers|transferred|grantee|grantor|recipient)\b", re.I)
NAME = re.compile(r"\b([A-Z][a-z]+(?: [A-Z]\.)?(?: [A-Z][a-z]+)+)\b")
# Capitalised phrases that are places or document words, not people.
PLACE_WORDS = {"road", "street", "avenue", "drive", "lane", "court", "terrace", "square", "place", "way",
               "county", "town", "city", "village", "york", "state", "map", "book", "page", "reel", "frame",
               "deed", "deeds", "judgment", "clerk", "office", "treasurer", "public", "notary", "record",
               "records", "filing", "foreclosure", "index", "subdivision", "hollow", "falls", "point",
               "warranty", "quitclaim", "tax", "lot", "section", "block", "exhibit", "law", "attorney"}
LINK = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|net|org|io|gov|info)\b", re.I)
PERSONAL = re.compile(r"social security|\bssn\b|password|bank account|credit card", re.I)
MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d\d)?")
CITATION = re.compile(r"\[([^\[\]]+)\]")


def citation_parts(answer: str) -> list[str]:
    """Every cited label; '[A; B]' counts as two citations."""
    return [p.strip() for c in CITATION.findall(answer) for p in c.split(";") if p.strip()]


def match_citation(label: str, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve a cited label to a provided source: {label, doc_id, page, kind}, or None.

    Sources are chunks (label "{title}, page {n}") and records lines (label "records: {title}").
    The longest title contained in the label wins, so a short title can't shadow a longer one.
    """
    s = squash(label)
    is_records = s.startswith("records:")
    found = [(len(squash(src["title"])), s.find(squash(src["title"])), src) for src in sources]
    found = [f for f in found if f[1] >= 0]
    if not found:
        return None
    # Longest title first; among equal titles prefer the source kind the label asks for.
    found.sort(key=lambda f: (-f[0], (f[2]["kind"] == "records") != is_records))
    length, pos, src = found[0]
    if is_records:
        return {"label": label, "doc_id": src["doc_id"], "page": 1, "kind": "records"}
    m = re.search(r"page(\d+)", s[pos + length:])
    page = int(m.group(1)) if m else int(src.get("page") or 1)
    return {"label": label, "doc_id": src["doc_id"], "page": page, "kind": "chunk"}


def resolve_citations(answer: str, sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """(citations that name a provided source, labels that don't)."""
    good, bad, seen = [], [], set()
    for label in citation_parts(answer):
        m = match_citation(label, sources)
        if m is None:
            bad.append(label)
        elif label not in seen:
            seen.add(label)
            good.append(m)
    return good, bad


def output_issues(answer: str, sources: list[dict[str, Any]], source_text: str,
                  known_names: set[str] | None) -> list[str]:
    """Problems with a draft. `known_names=None` skips the owner check (no records available)."""
    _, bad = resolve_citations(answer, sources)
    issues = [f"cites a source that wasn't provided: [{c}]" for c in bad]
    if m := LINK.search(answer):
        issues.append(f"contains a link: {m.group()}")
    if m := PERSONAL.search(answer):
        issues.append(f"asks about personal info: {m.group()}")
    src = squash(source_text)
    for amount in MONEY.findall(answer):
        if squash(amount.lstrip("$").rstrip(".,")) not in src:
            issues.append(f"amount not in any source: {amount}")
    if known_names is not None and OWNERSHIP.search(answer):
        for name in NAME.findall(CITATION.sub(" ", answer)):  # names inside citations are document titles
            words = [w.lower() for w in name.split()]
            if words[-1] in PLACE_WORDS or any(w in PLACE_WORDS for w in words):
                continue
            if words[-1] not in known_names:
                issues.append(f"named as a party but not in the records for this parcel: {name}")
    return list(dict.fromkeys(issues))  # de-duplicate, keep order


def weak_values_used(draft: str, weak: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The weakly-evidenced values (records.weak_values) that appear in the draft."""
    d = squash(draft)
    seen, out = set(), []
    for w in weak:
        key = (w["doc_id"], w["field"], w["value"])
        if key not in seen and any(squash(f) in d for f in w["forms"]):
            seen.add(key)
            out.append({k: v for k, v in w.items() if k != "forms"})
    return out
