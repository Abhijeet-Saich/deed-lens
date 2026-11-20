"""Ingestion: PDFs in data/raw -> page text -> records -> chunks -> search index.

Caching (so re-runs only pay for what changed):
- page text is cached per document together with the PDF's SHA-256;
- Gemini work is never thrown away: a page transcribed earlier is reused, pages marked
  "suspect" are not retried (Lesson 5: don't keep paying for a page that loops), pages that
  failed with an API error are retried on the next run;
- extraction is cached with a hash of the text it was run on;
- record tables are rebuilt from scratch every run (milliseconds, never stale);
- the vector index is re-embedded only when chunks.jsonl changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deedlens.config import Settings
from deedlens.ingest.chunk import chunk_document, make_token_counter
from deedlens.ingest.extract import document_text, extract_record, make_title, own_parcels, verify_evidence
from deedlens.ingest.gemini import Gemini, GeminiError
from deedlens.ingest.index import build_index, index_is_current, write_chunks
from deedlens.ingest.ocr import OCR_DPI, has_text_layer, ocr_images, text_layer_pages
from deedlens.ingest.render import render_pages
from deedlens.ingest.transcribe import suspect_reason, transcribe_page
from deedlens.store import records
from deedlens.store.search import make_embeddings, open_qdrant
from deedlens.text_utils import normalize_doc_id

GOOD = ("ok", "retried_ok")  # Gemini text is used for these pages
FINAL = ("ok", "retried_ok", "suspect")  # no further Gemini calls for these pages


# ---------------------------------------------------------------- small helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def copy_sources(source: Path, raw_dir: Path) -> int:
    """Copy new or changed PDFs from `source` into data/raw. Returns how many were copied."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pdf in sorted(source.glob("*.pdf")):
        dest = raw_dir / pdf.name
        if not dest.exists() or dest.read_bytes() != pdf.read_bytes():
            shutil.copy2(pdf, dest)
            copied += 1
    return copied


# ---------------------------------------------------------------- reports

@dataclass
class DocReport:
    doc_id: str
    file: str
    pages: int = 0
    method: str = ""
    gemini: str = ""
    record: str = ""
    chunks: int = 0
    seconds: float = 0.0
    error: str | None = None


@dataclass
class Summary:
    docs: list[DocReport] = field(default_factory=list)
    chunks: int = 0
    index: str = ""
    gemini_calls: int = 0
    seconds: float = 0.0
    links: records.LinkReport = field(default_factory=records.LinkReport)
    chains: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def search_only(self) -> bool:
        return bool(self.docs) and not any(d.record in ("new", "cached") for d in self.docs)


# ---------------------------------------------------------------- the pipeline

class Pipeline:
    def __init__(self, settings: Settings, *, use_gemini: bool = True, redo: bool = False,
                 log: Callable[[str], None] = print):
        self.s = settings
        self.redo = redo
        self.log = log
        self.gemini: Gemini | None = None
        if use_gemini and settings.has_gemini:
            self.gemini = Gemini(settings.gemini_api_key, settings.gemini_model,
                                 settings.gemini_seconds_between_calls)

    # ----- step 1 + 2: page text (text layer | Tesseract | Gemini) -----

    def _disable_gemini(self, err: GeminiError, summary: Summary) -> None:
        summary.warnings.append(f"Gemini switched off for this run: {err}")
        self.log(f"   ! Gemini switched off for this run: {err}")
        if self.gemini:
            summary.gemini_calls += self.gemini.calls
        self.gemini = None

    def page_texts(self, pdf: Path, doc_id: str, sha: str) -> dict[str, Any]:
        path = self.s.text_dir / f"{doc_id}.json"
        prior = load_json(path)
        if prior and (prior.get("sha256") != sha or self.redo):
            prior = None
        want_gemini = self.gemini is not None and self.s.use_gemini_transcription
        if prior and (prior["kind"] == "text_layer" or prior["gemini_complete"] or not want_gemini):
            return prior

        layer = text_layer_pages(pdf)
        if has_text_layer(layer):
            doc = {"doc_id": doc_id, "file": pdf.name, "sha256": sha, "kind": "text_layer",
                   "gemini_complete": True,
                   "pages": [{"page": i, "text": t, "method": "text_layer", "confidence": None,
                              "status": "ok"} for i, t in enumerate(layer, 1)]}
            save_json(path, doc)
            return doc

        images = None
        if prior:  # Tesseract already ran on this exact file: reuse it
            tess = [(p["tesseract"], p["confidence"]) for p in prior["pages"]]
        else:
            images = render_pages(pdf, OCR_DPI)  # sequential: PDFium isn't thread-safe
            tess = [(t, round(c, 1)) for t, c, _ in ocr_images(images, self.s.ocr_workers)]

        pages = []
        for i, (tess_text, conf) in enumerate(tess):
            page: dict[str, Any] = {"page": i + 1, "tesseract": tess_text, "confidence": conf,
                                    "gemini": None, "status": "not_run"}
            before = prior["pages"][i] if prior else None
            reusable = before and before.get("status") in FINAL and (
                before["status"] == "suspect" or suspect_reason(before["gemini"] or "", tess_text) is None)
            if reusable or (before and not want_gemini):
                page.update({k: before.get(k) for k in ("gemini", "status", "reason") if k in before})
            elif want_gemini:
                if images is None:
                    images = render_pages(pdf, OCR_DPI)
                page.update(transcribe_page(self.gemini, images[i], tess_text, OCR_DPI))
            good = page["status"] in GOOD
            page["text"] = page["gemini"] if good else tess_text
            page["method"] = "gemini" if good else "tesseract"
            pages.append(page)
        doc = {"doc_id": doc_id, "file": pdf.name, "sha256": sha, "kind": "scanned",
               "gemini_complete": all(p["status"] in FINAL for p in pages), "pages": pages}
        save_json(path, doc)
        return doc

    # ----- step 3: extraction -----

    def record(self, doc_id: str, text: str) -> tuple[dict[str, Any] | None, str]:
        path = self.s.extracted_dir / f"{doc_id}.json"
        text_sha = sha256_bytes(text.encode())
        prior = load_json(path)
        if prior and prior.get("text_sha256") == text_sha and not self.redo:
            return verify_evidence(prior["record"], text), "cached"
        if self.gemini is None:
            return None, "skipped"
        try:
            rec = verify_evidence(extract_record(self.gemini, text), text)
        except GeminiError as e:
            if e.fatal:
                raise
            return None, f"failed: {str(e)[:80]}"
        save_json(path, {"doc_id": doc_id, "text_sha256": text_sha, "model": self.s.gemini_model,
                         "record": rec})
        return rec, "new"

    # ----- the whole run -----

    def run(self, source: Path | None = None) -> Summary:
        start = time.time()
        summary = Summary()
        s = self.s
        if source:
            n = copy_sources(source, s.raw_dir)
            self.log(f"Copied {n} new or changed PDF(s) from {source} into {s.raw_dir}")
        pdfs = sorted(s.raw_dir.glob("*.pdf")) if s.raw_dir.exists() else []
        ids = Counter(normalize_doc_id(p.name) for p in pdfs)
        if clash := [d for d, n in ids.items() if n > 1]:
            raise RuntimeError(f"Different files map to the same doc_id: {clash}. Rename one of them.")

        manifest = load_json(s.manifest_path) or {}
        inputs: list[records.DocInput] = []
        texts: dict[str, dict[str, Any]] = {}
        recs: dict[str, dict[str, Any] | None] = {}
        for pdf in pdfs:
            doc_id = normalize_doc_id(pdf.name)
            report = DocReport(doc_id, pdf.name)
            summary.docs.append(report)
            t0 = time.time()
            self.log(f"{pdf.name} ...")
            try:
                sha = sha256_bytes(pdf.read_bytes())
                try:
                    doc = self.page_texts(pdf, doc_id, sha)
                except GeminiError as e:
                    self._disable_gemini(e, summary)
                    doc = self.page_texts(pdf, doc_id, sha)
                text = document_text(doc["pages"])
                try:
                    rec, report.record = self.record(doc_id, text)
                except GeminiError as e:
                    self._disable_gemini(e, summary)
                    rec, report.record = self.record(doc_id, text)
            except Exception as e:  # one bad file must not stop the others
                report.error = f"{type(e).__name__}: {e}"
                summary.warnings.append(f"{pdf.name}: {report.error}")
                self.log(f"   ! {report.error}")
                continue
            report.pages = len(doc["pages"])
            methods = Counter(p["method"] for p in doc["pages"])
            report.method = "+".join(sorted(methods))
            statuses = Counter(p["status"] for p in doc["pages"])
            report.gemini = "" if doc["kind"] == "text_layer" else ", ".join(
                f"{n} {k}" for k, n in sorted(statuses.items()))
            texts[doc_id], recs[doc_id] = doc, rec
            inputs.append(records.DocInput(
                doc_id=doc_id, file_name=pdf.name, num_pages=len(doc["pages"]), sha256=sha,
                title=make_title(rec, pdf.name), record=rec, parcels=own_parcels(rec, text)))
            manifest[doc_id] = {"file": pdf.name, "sha256": sha, "num_pages": len(doc["pages"]),
                                "text": report.method, "gemini_complete": doc["gemini_complete"],
                                "extracted": rec is not None}
            save_json(s.manifest_path, manifest)
            report.seconds = time.time() - t0
        for gone in set(manifest) - {d.doc_id for d in inputs}:
            manifest.pop(gone)
        save_json(s.manifest_path, manifest)

        # ----- step 4: records -----
        db = records.connect(s.db_path)
        try:
            summary.links = records.rebuild(db, inputs)
            for parcel in records.all_parcels(db):
                summary.chains[parcel] = [r["title"] for r in records.parcel_documents(db, parcel)]

            # ----- step 5: chunks -----
            count = make_token_counter(s.embed_model)
            chunks: list[dict[str, Any]] = []
            by_doc = {r.doc_id: r for r in summary.docs}
            for d in inputs:
                row = db.execute("SELECT title, instrument_type, year FROM documents WHERE doc_id = ?",
                                 (d.doc_id,)).fetchone()
                meta = {"doc_id": d.doc_id, "title": row["title"], "instrument_type": row["instrument_type"],
                        "year": row["year"], "recorded_at": records.ref_label(*records.recorded_ref(d.record)),
                        "parcels": records.doc_parcels(db, d.doc_id)}
                doc_chunks = chunk_document(texts[d.doc_id]["pages"], meta, count,
                                            s.chunk_target_tokens, s.chunk_overlap_tokens)
                by_doc[d.doc_id].chunks = len(doc_chunks)
                chunks.extend(doc_chunks)
        finally:
            db.close()
        summary.chunks = len(chunks)

        # ----- step 6: index -----
        changed = write_chunks(s.chunks_file, chunks)
        client = open_qdrant(s.qdrant_dir)
        try:
            if changed or self.redo or not index_is_current(client, len(chunks)):
                self.log(f"Embedding {len(chunks)} chunks with {s.embed_model} ...")
                build_index(client, make_embeddings(s.embed_model), chunks)
                summary.index = "rebuilt"
            else:
                summary.index = "unchanged"
        finally:
            client.close()

        if self.gemini:
            summary.gemini_calls += self.gemini.calls
        summary.warnings += summary.links.unresolved + summary.links.mismatches
        summary.warnings += [f"{d}: no parcel found, so no parcel-scoped search will see it"
                             for d in summary.links.unscoped]
        summary.seconds = time.time() - start
        return summary


# ---------------------------------------------------------------- printing

def format_summary(summary: Summary) -> str:
    lines = [f"\n{'document':<22}{'pages':>6}  {'text from':<20}{'gemini pages':<22}{'record':<10}{'chunks':>7}"
             f"{'secs':>7}"]
    for d in summary.docs:
        if d.error:
            lines.append(f"{d.doc_id:<22}  ERROR {d.error}")
            continue
        lines.append(f"{d.doc_id:<22}{d.pages:>6}  {d.method:<20}{d.gemini or '-':<22}{d.record:<10}"
                     f"{d.chunks:>7}{d.seconds:>7.1f}")
    pages = sum(d.pages for d in summary.docs)
    lines.append(f"\n{len(summary.docs)} documents, {pages} pages, {summary.chunks} chunks; "
                 f"index {summary.index}; Gemini calls {summary.gemini_calls}; {summary.seconds:.1f} s")
    lines.append(f"Links in the chain of title: {len(summary.links.links)}")
    for parcel, titles in summary.chains.items():
        lines.append(f"  {parcel}: " + "  <-  ".join(titles))
    if summary.search_only:
        lines.append("\nWARNING: no records were extracted (no Gemini key, --no-gemini, or Gemini failed). "
                     "The app will run in search-only mode: no chain of title, no records answers.")
    if summary.warnings:
        lines.append("\nNotes:")
        lines += [f"  - {w}" for w in summary.warnings]
    return "\n".join(lines)
