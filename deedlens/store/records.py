"""SQLite records: documents, instruments, parties, parcels, the chain of title, evidence.

Rules carried over from Lesson 8:
- Documents are linked by recording PAGE RANGE: a deed citing Book 8310 Page 337 is the
  document recorded at 8310/336 with 3 pages (cover page 336, deed text from 337). Exact
  page matching missed these links.
- Multi-parcel documents (tax foreclosure judgments) are linked by the parcel numbers found
  in their text.
- Deed amounts <= $10 are nominal and are labelled so the LLM doesn't present them as a price.
- Question-time code opens the database read-only.

Ingestion rebuilds every record table from scratch on each run (cheap, and never leaves stale
rows behind); `review_log` is never cleared.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deedlens.text_utils import digits, normalize_parcel

NOMINAL_MAX = 10.0  # Lesson 8: $1 / $10 "and other good and valuable consideration"
JUDGMENT = "foreclosure judgment"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, file_name TEXT, num_pages INTEGER, sha256 TEXT, title TEXT,
    instrument_type TEXT, year INTEGER);
CREATE TABLE IF NOT EXISTS instruments (
    doc_id TEXT PRIMARY KEY, volume TEXT, page INTEGER, num_pages INTEGER,
    date_signed TEXT, date_recorded TEXT,
    consideration_in_deed REAL, consideration_reported REAL, transfer_tax REAL,
    parcel_id TEXT, property_address TEXT, town TEXT, acreage REAL,
    prior_volume TEXT, prior_page INTEGER);
-- source: 'extracted' (the record's parcel_id), 'text' (found in the text), 'chain' (inherited)
CREATE TABLE IF NOT EXISTS parcels (doc_id TEXT, parcel_id TEXT, source TEXT,
    UNIQUE (doc_id, parcel_id));
CREATE TABLE IF NOT EXISTS parties (doc_id TEXT, role TEXT, name TEXT);
CREATE TABLE IF NOT EXISTS chain (doc_id TEXT, previous_doc_id TEXT, how TEXT,
    UNIQUE (doc_id, previous_doc_id));
CREATE TABLE IF NOT EXISTS evidence (doc_id TEXT, field TEXT, page INTEGER, quote TEXT, verified TEXT);
CREATE TABLE IF NOT EXISTS review_log (id INTEGER PRIMARY KEY, created_at TEXT, kind TEXT,
    doc_id TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS parcels_by_parcel ON parcels (parcel_id);
CREATE INDEX IF NOT EXISTS parties_by_doc ON parties (doc_id);
"""
RECORD_TABLES = ("documents", "instruments", "parcels", "parties", "chain", "evidence")


# ---------------------------------------------------------------- connections

def connect(path: Path, read_only: bool = False) -> sqlite3.Connection:
    """Read-only uses SQLite's URI mode (Lesson 8): the agent can't change records by mistake."""
    if read_only:
        db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, check_same_thread=False)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        db.executescript(SCHEMA)
    db.row_factory = sqlite3.Row
    return db


# ---------------------------------------------------------------- record helpers

def year_of(record: dict[str, Any] | None) -> int | None:
    if not record:
        return None
    for key in ("date_recorded", "date_signed"):
        m = re.match(r"(\d{4})", str(record.get(key) or ""))
        if m:
            return int(m.group(1))
    return None


def recorded_ref(record: dict[str, Any] | None) -> tuple[str | None, int | None, str]:
    """(volume digits, page number, kind) of where the document itself is recorded."""
    ref = (record or {}).get("recorded_at") or {}
    volume = digits(ref.get("volume")) or None
    page = int(digits(ref.get("page"))) if digits(ref.get("page")) else None
    return volume, page, ref.get("kind") or "book_page"


def ref_label(volume: str | None, page: int | None, kind: str) -> str | None:
    if not volume or page is None:
        return None
    a, b = ("Reel", "Frame") if kind == "reel_frame" else ("Book", "Page")
    return f"{a} {volume}, {b} {page}"


# ---------------------------------------------------------------- loading

@dataclass
class DocInput:
    """Everything ingestion knows about one document. `record` is None without extraction."""
    doc_id: str
    file_name: str
    num_pages: int
    sha256: str
    title: str
    record: dict[str, Any] | None = None
    parcels: list[str] = field(default_factory=list)  # own parcels: extracted first, then text


@dataclass
class LinkReport:
    links: list[tuple[str, str, str]] = field(default_factory=list)  # (doc, previous, how)
    unresolved: list[str] = field(default_factory=list)  # citations of documents not loaded
    mismatches: list[str] = field(default_factory=list)  # sellers who weren't previous buyers
    unscoped: list[str] = field(default_factory=list)  # documents with no parcel at all


def _float(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def load_document(db: sqlite3.Connection, doc: DocInput) -> None:
    r = doc.record
    db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
               (doc.doc_id, doc.file_name, doc.num_pages, doc.sha256, doc.title,
                r["instrument_type"] if r else None, year_of(r)))
    extracted = normalize_parcel(r.get("parcel_id")) if r else None
    for p in doc.parcels:
        db.execute("INSERT OR IGNORE INTO parcels VALUES (?,?,?)",
                   (doc.doc_id, p, "extracted" if p == extracted else "text"))
    if not r:
        return
    volume, page, _ = recorded_ref(r)
    prior = r.get("prior_deed") or {}
    prior_page = digits(prior.get("page"))
    db.execute("INSERT INTO instruments VALUES (?,?,?,?, ?,?, ?,?,?, ?,?,?,?, ?,?)", (
        doc.doc_id, volume, page, doc.num_pages,
        r.get("date_signed"), r.get("date_recorded"),
        _float(r.get("consideration_in_deed")), _float(r.get("consideration_reported")),
        _float(r.get("transfer_tax")),
        extracted, r.get("property_address"), r.get("town"), _float(r.get("acreage")),
        digits(prior.get("volume")) or None, int(prior_page) if prior_page else None,
    ))
    for role in ("grantor", "grantee"):
        for name in r.get(role + "s") or []:
            db.execute("INSERT INTO parties VALUES (?,?,?)", (doc.doc_id, role, name))
    for ev in r.get("evidence") or []:
        db.execute("INSERT INTO evidence VALUES (?,?,?,?,?)",
                   (doc.doc_id, ev["field"], ev.get("page"), ev.get("quote"), ev.get("verified")))


def _sort_date(row: sqlite3.Row | dict) -> str:
    return row["date_recorded"] or row["date_signed"] or ""


def link_documents(db: sqlite3.Connection) -> LinkReport:
    report = LinkReport()

    # 1. "Being the same premises ..." citations, matched against each document's page RANGE.
    for row in db.execute("SELECT doc_id, prior_volume, prior_page FROM instruments "
                          "WHERE prior_volume IS NOT NULL AND prior_page IS NOT NULL").fetchall():
        match = db.execute(
            "SELECT doc_id FROM instruments WHERE volume = ? AND doc_id != ? "
            "AND ? BETWEEN page AND page + num_pages - 1",
            (row["prior_volume"], row["doc_id"], row["prior_page"])).fetchone()
        if match:
            how = f"cites {row['prior_volume']}/{row['prior_page']}"
            db.execute("INSERT OR IGNORE INTO chain VALUES (?,?,?)", (row["doc_id"], match[0], how))
        else:
            report.unresolved.append(
                f"{row['doc_id']} cites {row['prior_volume']}/{row['prior_page']}, which isn't loaded")

    # 2. Multi-parcel judgments: link to the latest earlier document of each parcel they list.
    judgments = db.execute(
        "SELECT i.doc_id, i.date_recorded, i.date_signed FROM instruments i "
        "JOIN documents d USING (doc_id) WHERE d.instrument_type = ?", (JUDGMENT,)).fetchall()
    for j in judgments:
        when = _sort_date(j)
        for (parcel,) in db.execute("SELECT parcel_id FROM parcels WHERE doc_id = ? AND source != 'chain'",
                                    (j["doc_id"],)).fetchall():
            previous = db.execute(
                "SELECT i.doc_id FROM instruments i JOIN parcels p USING (doc_id) "
                "WHERE p.parcel_id = ? AND p.source != 'chain' AND i.doc_id != ? "
                "AND COALESCE(i.date_recorded, i.date_signed) < ? "
                "ORDER BY COALESCE(i.date_recorded, i.date_signed) DESC LIMIT 1",
                (parcel, j["doc_id"], when)).fetchone()
            if previous:
                db.execute("INSERT OR IGNORE INTO chain VALUES (?,?,?)",
                           (j["doc_id"], previous[0], f"lists parcel {parcel}"))

    report.links = [tuple(r) for r in db.execute("SELECT doc_id, previous_doc_id, how FROM chain")]
    return report


def propagate_parcels(db: sqlite3.Connection) -> None:
    """Documents with no parcel of their own inherit the parcels of the documents they link to.

    Older deeds often lack a tax map number but belong to the same land as the deeds before
    and after them. Judgments don't donate: they list many parcels, and a deed linked to one
    belongs to only one of them. Repeats until nothing changes (chains of parcel-less deeds).
    """
    while True:
        added = 0
        empty = [r[0] for r in db.execute(
            "SELECT doc_id FROM documents WHERE doc_id NOT IN "
            "(SELECT doc_id FROM parcels WHERE source != 'chain')")]
        for doc in empty:
            neighbours = db.execute(
                "SELECT previous_doc_id FROM chain WHERE doc_id = ? UNION "
                "SELECT doc_id FROM chain WHERE previous_doc_id = ?", (doc, doc)).fetchall()
            for (n,) in neighbours:
                kind = db.execute("SELECT instrument_type FROM documents WHERE doc_id = ?", (n,)).fetchone()
                if kind and kind[0] == JUDGMENT:
                    continue
                for (parcel,) in db.execute("SELECT parcel_id FROM parcels WHERE doc_id = ?", (n,)).fetchall():
                    added += db.execute("INSERT OR IGNORE INTO parcels VALUES (?,?,'chain')",
                                        (doc, parcel)).rowcount
        if not added:
            return


# Words that say nothing about who a party is ("WALTER and EDITH SMITH, his wife").
_NAME_NOISE = {"and", "the", "his", "her", "their", "its", "wife", "husband", "acting", "through",
               "by", "of", "as", "jr", "sr", "inc", "llc", "estate", "trustee", "treasurer"}


def _surnames(names: list[str]) -> set[str]:
    """Significant words of the party names. Lesson 8 compared last words only, but extraction
    sometimes returns "A and B, his wife" or "COUNTY OF X, acting by ..." as one name."""
    return {w for n in names for w in re.findall(r"[a-z]+", n.lower())
            if len(w) >= 3 and w not in _NAME_NOISE}


def check_links(db: sqlite3.Connection) -> list[str]:
    """Sanity check from Lesson 8: sellers should have been buyers in the previous document."""
    problems = []
    for doc, previous, how in db.execute("SELECT doc_id, previous_doc_id, how FROM chain").fetchall():
        if how.startswith("lists parcel"):
            continue
        sellers = party_names(db, doc, "grantor")
        buyers = party_names(db, previous, "grantee")
        if sellers and buyers and not (_surnames(sellers) & _surnames(buyers)):
            problems.append(f"{doc}: sellers {sellers} were not buyers in {previous} {buyers}")
    return problems


def rebuild(db: sqlite3.Connection, docs: list[DocInput]) -> LinkReport:
    """Replace all record tables with `docs`, then link, propagate parcels and check links."""
    with db:
        for table in RECORD_TABLES:
            db.execute(f"DELETE FROM {table}")
        for doc in docs:
            load_document(db, doc)
        report = link_documents(db)
        propagate_parcels(db)
        report.mismatches = check_links(db)
        report.unscoped = [r[0] for r in db.execute(
            "SELECT doc_id FROM documents WHERE doc_id NOT IN (SELECT doc_id FROM parcels)")]
    return report


# ---------------------------------------------------------------- queries

def party_names(db: sqlite3.Connection, doc_id: str, role: str) -> list[str]:
    return [r[0] for r in db.execute("SELECT name FROM parties WHERE doc_id = ? AND role = ?",
                                     (doc_id, role))]


def doc_parcels(db: sqlite3.Connection, doc_id: str) -> list[str]:
    """Own parcels first (extracted, then text), then inherited ones."""
    order = "CASE source WHEN 'extracted' THEN 0 WHEN 'text' THEN 1 ELSE 2 END, rowid"
    return [r[0] for r in db.execute(f"SELECT parcel_id FROM parcels WHERE doc_id = ? ORDER BY {order}",
                                     (doc_id,))]


def all_parcels(db: sqlite3.Connection) -> list[str]:
    return [r[0] for r in db.execute("SELECT DISTINCT parcel_id FROM parcels ORDER BY parcel_id")]


def parcel_documents(db: sqlite3.Connection, parcel: str) -> list[sqlite3.Row]:
    """Every document of a parcel, newest first, with its instrument fields (if extracted)."""
    return db.execute(
        "SELECT d.*, i.volume, i.page, i.date_signed, i.date_recorded, i.consideration_in_deed, "
        "i.consideration_reported, i.transfer_tax, i.property_address, i.town, i.acreage, "
        "i.prior_volume, i.prior_page "
        "FROM documents d JOIN parcels p USING (doc_id) LEFT JOIN instruments i USING (doc_id) "
        "WHERE p.parcel_id = ? "
        "ORDER BY COALESCE(i.date_recorded, i.date_signed, CAST(d.year AS TEXT), '') DESC", (parcel,)).fetchall()


def is_nominal(amount: float | None) -> bool:
    return amount is not None and 0 < amount <= NOMINAL_MAX


def price_label(amount: float | None) -> str | None:
    """'$10.00 (nominal: a token amount, not the real price)' for deed amounts <= $10."""
    if amount is None:
        return None
    text = f"${amount:,.2f}"
    return f"{text} (nominal: a token amount, not the real price)" if is_nominal(amount) else text


def log_review(db: sqlite3.Connection, kind: str, doc_id: str | None, detail: str) -> None:
    """Things a person should look at later (e.g. a chunk the injection scanner dropped)."""
    with db:
        db.execute("INSERT INTO review_log (created_at, kind, doc_id, detail) VALUES (?,?,?,?)",
                   (datetime.now(UTC).isoformat(timespec="seconds"), kind, doc_id, detail))
