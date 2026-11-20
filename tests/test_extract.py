import pytest

from deedlens.ingest.extract import Instrument, document_text, make_title, own_parcels, verify_evidence
from deedlens.ingest.transcribe import suspect_reason

PAGES = [
    {"page": 1, "text": "RECORDING COVER PAGE\nGRANTOR(S) | GRANTEE(S)\nQUILL, MARTA | OKAFOR, DESMOND\n"
                        "Consideration: $142,500.00"},
    {"page": 2, "text": "THIS INDENTURE, made the 20th day of August, 2004, BETWEEN MARTA QUILL ...\n"
                        "Tax Map No. 12.04-3-7. Lots 6 and 7, see also 12.04-3-8."},
]


def test_evidence_three_levels():
    text = document_text(PAGES)
    record = {"evidence": [
        {"field": "grantors", "page": 2, "quote": "BETWEEN  MARTA\nQUILL"},              # spacing differs
        {"field": "consideration_reported", "page": 1, "quote": "Consideration $142,500.00"},  # no colon
        {"field": "grantees", "page": 1, "quote": "GRANTEE(S) OKAFOR, DESMOND"},       # table cells
        {"field": "parcel_id", "page": 2, "quote": "Tax Map No. 99.99-9-9"},
    ]}
    levels = [e["verified"] for e in verify_evidence(record, text)["evidence"]]
    assert levels == ["exact", "words on page", "words on page", "not found"]


def test_words_on_page_uses_the_cited_page_only():
    record = {"evidence": [{"field": "grantees", "page": 2, "quote": "OKAFOR GRANTEE(S)"}]}
    assert verify_evidence(record, document_text(PAGES))["evidence"][0]["verified"] == "not found"


def test_title_from_record_or_file_name():
    record = {"instrument_type": "warranty deed", "date_recorded": "2004-08-27", "date_signed": None,
              "recorded_at": {"volume": "2218", "page": "401", "kind": "book_page"}}
    assert make_title(record, "Deed 2218 401.pdf") == "2004 warranty deed (Book 2218, Page 401)"
    assert make_title(None, "Deed 2218 401.pdf") == "Deed 2218 401"
    record["recorded_at"] = {"volume": None, "page": None, "kind": None}
    assert make_title(record, "Scan.PDF") == "Scan"


def test_own_parcels_extracted_first_then_text():
    text = document_text(PAGES)
    assert own_parcels({"parcel_id": "Part of 12.04-3-7"}, text) == ["12.04-3-7", "12.04-3-8"]
    assert own_parcels(None, "nothing here") == []


@pytest.mark.parametrize("text, finish, expected", [
    ("A normal page of deed text.", "STOP", None),
    ("x", "MAX_TOKENS", "finish reason MAX_TOKENS"),
    ("Grantor " + "-" * 60, "STOP", "same character repeated 50+ times"),
    ("and the said " * 12, "STOP", "same phrase repeated 8+ times"),
])
def test_suspect_checks(text, finish, expected):
    assert suspect_reason(text, "tesseract text", finish) == expected


def test_suspect_when_far_longer_than_tesseract():
    assert suspect_reason(" ".join(f"w{i:04d}" for i in range(400)), "short", "STOP").startswith("2,399 chars")


def test_schema_fields_are_required_but_nullable():
    """Lesson 8: Gemini response schemas reject defaults, so no field may have one."""
    for name, f in Instrument.model_fields.items():
        assert f.is_required(), name
