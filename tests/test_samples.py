"""The committed fictional samples and the eval files that describe them stay consistent."""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import pytest
from pypdf import PdfReader

from deedlens.text_utils import normalize_doc_id, squash

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
EVAL = ROOT / "eval"
BANNER_PARTS = ("fictionalsampledocument", "notarealrecord")

# file -> (pages, image only)
EXPECTED = {
    "Fore Jud 598 12.pdf": (3, False),
    "Tax Deed 612 118.pdf": (2, True),
    "Deed 1043 77.pdf": (2, True),
    "Deed 2218 401.pdf": (3, False),
    "Deed 3307 52.pdf": (3, False),
    "Deed 1580 233.pdf": (2, False),
    "Deed 2874 610.pdf": (3, False),
    "Deed 3105 288.pdf": (3, False),
}


def _page_texts(source) -> list[str]:
    return [p.extract_text() or "" for p in PdfReader(source).pages]


@lru_cache(maxsize=1)
def _generator():
    spec = importlib.util.spec_from_file_location("make_samples", SAMPLES / "make_samples.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_samples"] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _source_texts() -> dict[str, str]:
    """doc_id -> full text of each sample as generated *before* scanning (ground truth)."""
    gen = _generator()
    return {normalize_doc_id(s.file): "\n".join(_page_texts(io.BytesIO(gen.build_pdf(s))))
            for s in gen.all_samples()}


def test_committed_pdfs_match_expected_set():
    assert sorted(p.name for p in SAMPLES.glob("*.pdf")) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_page_count_and_text_layer(name):
    pages, image_only = EXPECTED[name]
    texts = _page_texts(SAMPLES / name)
    assert len(texts) == pages
    if image_only:
        assert all(t.strip() == "" for t in texts), "scanned samples must have no text layer"
    else:
        for i, t in enumerate(texts, 1):
            assert all(part in squash(t) for part in BANNER_PARTS), f"page {i} lacks the banner"
            assert len(t) >= 100  # Lesson 3 rule: >= 100 chars/page counts as a real text layer


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
@pytest.mark.parametrize("name", [n for n, (_, img) in EXPECTED.items() if img])
def test_scanned_pages_show_banner_to_ocr(name):
    import pypdfium2 as pdfium
    import pytesseract

    pdf = pdfium.PdfDocument(SAMPLES / name)
    for i in range(len(pdf)):
        img = pdf[i].render(scale=300 / 72).to_pil().convert("L")
        h = img.height
        for band in (img.crop((0, 0, img.width, h // 12)), img.crop((0, h - h // 12, img.width, h))):
            text = squash(pytesseract.image_to_string(band, config="--psm 6"))
            assert "fictionalsample" in text and "realrecord" in text, f"{name} page {i + 1}"
    pdf.close()


def test_every_fact_is_in_its_document():
    facts = json.loads((EVAL / "facts_sample.json").read_text())
    texts = _source_texts()
    assert set(facts) == set(texts)
    for doc_id, items in facts.items():
        for f in items:
            assert f["kind"] in {"id", "money", "name", "place"}
            assert squash(f["value"]) in squash(texts[doc_id]), f"{doc_id}: {f['label']}"


def test_answer_file_is_well_formed():
    items = json.loads((EVAL / "answers_sample.json").read_text())
    corpus = squash("\n".join(_source_texts().values()))
    parcels = {"12.04-3-7", "12.04-3-8", "12.04-3-9"}
    assert len({i["id"] for i in items}) == len(items)
    for item in items:
        assert item["parcel"] in parcels
        assert item["question"] and len(item["question"]) <= 500
        assert item["must_include"] and all(isinstance(g, list) and g for g in item["must_include"])
        # A contamination value that appears in no document could never be caught, so it tests nothing.
        for value in item["must_not_include"]:
            assert squash(value) in corpus, f"{item['id']}: {value!r} not in any sample"
