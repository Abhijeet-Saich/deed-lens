"""End-to-end ingestion on text-layer samples with fakes: no key, no network, no Tesseract."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings

from deedlens.config import Settings
from deedlens.ingest import pipeline as pl
from deedlens.store import records

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
FILES = ["Deed 1580 233.pdf", "Deed 2874 610.pdf", "Deed 3307 52.pdf"]


class FakeEmbeddings(Embeddings):
    """Deterministic 8-dim unit vectors from a hash: enough for Qdrant to store and search."""

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        v = [b - 127.5 for b in h[:8]]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def fake_record(text: str) -> dict:
    """What Gemini would return, keyed off the book number in the cover page / text."""
    def r(volume, page, date, grantors, grantees, prior, parcel, in_deed, reported):
        return {"instrument_type": "warranty deed",
                "recorded_at": {"volume": volume, "page": page, "kind": "book_page"},
                "instrument_number": None, "date_signed": None, "date_recorded": date,
                "grantors": grantors, "grantees": grantees, "consideration_in_deed": in_deed,
                "consideration_reported": reported, "transfer_tax": None,
                "prior_deed": {"volume": prior[0], "page": prior[1], "kind": "book_page"},
                "parcel_id": parcel, "property_address": None, "town": None, "acreage": None,
                "evidence": [{"field": "grantees", "page": 1, "quote": grantees[0]}]}
    if "BK 1580" in text:
        return r("1580", "233", "1996-09-16", ["GORDON FAYE"], ["RUTH ABERNATHY", "SAMUEL ABERNATHY"],
                 ("905", "340"), "12.04-3-8", 71500.0, None)
    if "BK 2874" in text:
        return r("2874", "610", "2012-03-09", ["RUTH ABERNATHY", "SAMUEL ABERNATHY"], ["CALVIN OSTROWSKI"],
                 ("1580", "233"), "12.04-3-8", 10.0, 189900.0)
    return r("3307", "52", "2019-05-22", ["DESMOND OKAFOR"], ["LENA MARSH", "TOBIAS RENN"],
             ("2218", "402"), "12.04-3-7", 1.0, 236000.0)


@pytest.fixture
def run(tmp_path, monkeypatch):
    raw = tmp_path / "src"
    raw.mkdir()
    for f in FILES:
        shutil.copy(SAMPLES / f, raw / f)
    calls = {"extract": 0, "embed": 0}

    def extract(_gemini, text):
        calls["extract"] += 1
        return fake_record(text)

    def embeddings(_model):
        calls["embed"] += 1
        return FakeEmbeddings()

    monkeypatch.setattr(pl, "extract_record", extract)
    monkeypatch.setattr(pl, "make_embeddings", embeddings)
    monkeypatch.setattr(pl, "make_token_counter", lambda _m: lambda t: len(t.split()))
    settings = Settings.from_env({"DATA_DIR": str(tmp_path / "data")})

    def go(**kw):
        p = pl.Pipeline(settings, use_gemini=False, log=lambda _m: None, **kw)
        p.gemini = SimpleNamespace(calls=0)  # pretend a client exists; extraction itself is faked
        return p.run(raw)

    return go, settings, calls


def test_ingest_builds_records_chunks_and_index_then_caches(run):
    go, s, calls = run
    summary = go()
    assert [d.record for d in summary.docs] == ["new"] * 3
    assert all(d.method == "text_layer" and d.chunks > 0 for d in summary.docs)
    assert summary.index == "rebuilt" and calls == {"extract": 3, "embed": 1}
    assert summary.chains["12.04-3-8"] == ["2012 warranty deed (Book 2874, Page 610)",
                                          "1996 warranty deed (Book 1580, Page 233)"]
    assert "Deed_3307_52 cites 2218/402, which isn't loaded" in summary.warnings

    chunks = [json.loads(line) for line in s.chunks_file.read_text().splitlines()]
    assert len(chunks) == summary.chunks
    assert {tuple(c["parcels"]) for c in chunks if c["doc_id"] == "Deed_2874_610"} == {("12.04-3-8",)}
    assert chunks[0]["text"].startswith(chunks[0]["title"] + ", page 1 of ")

    manifest = json.loads(s.manifest_path.read_text())
    assert set(manifest) == {"Deed_1580_233", "Deed_2874_610", "Deed_3307_52"}

    again = go()  # nothing changed: no extraction, no embedding
    assert [d.record for d in again.docs] == ["cached"] * 3
    assert again.index == "unchanged" and calls == {"extract": 3, "embed": 1}

    db = records.connect(s.db_path, read_only=True)
    assert db.execute("SELECT COUNT(*) FROM chain").fetchone()[0] == 1
    db.close()


def test_redo_ignores_caches(run):
    go, _, calls = run
    go()
    go(redo=True)
    assert calls == {"extract": 6, "embed": 2}
