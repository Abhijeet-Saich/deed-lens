import sqlite3

import pytest

from deedlens.store import records
from deedlens.store.records import DocInput


def rec(kind, volume, page, date, grantors, grantees, prior=None, parcel=None, in_deed=None):
    return {
        "instrument_type": kind,
        "recorded_at": {"volume": str(volume), "page": str(page), "kind": "book_page"},
        "date_signed": None, "date_recorded": date,
        "grantors": grantors, "grantees": grantees,
        "consideration_in_deed": in_deed, "consideration_reported": None, "transfer_tax": None,
        "prior_deed": {"volume": str(prior[0]), "page": str(prior[1]), "kind": "book_page"} if prior else None,
        "parcel_id": parcel, "property_address": None, "town": None, "acreage": None,
        "evidence": [{"field": "grantors", "page": 1, "quote": "x", "verified": "exact"}],
    }


def doc(doc_id, num_pages, record, parcels=()):
    return DocInput(doc_id, f"{doc_id}.pdf", num_pages, "sha", doc_id, record, list(parcels))


@pytest.fixture
def db(tmp_path):
    conn = records.connect(tmp_path / "t.db")
    yield conn
    conn.close()


def test_page_range_link_matches_deed_text_page(db):
    # Book 8310, pages 336-338: cover page 336, deed text from 337. The later deed cites 337.
    older = doc("Deed_8310_336", 3, rec("warranty deed", 8310, 336, "2017-03-01", ["A Seller"], ["B Buyer"],
                                        parcel="70.15-5-11"), ["70.15-5-11"])
    newer = doc("Deed_8721_169", 2, rec("warranty deed", 8721, 169, "2021-06-01", ["B Buyer"], ["C Next"],
                                        prior=(8310, 337), parcel="70.15-5-11"), ["70.15-5-11"])
    report = records.rebuild(db, [older, newer])
    assert report.links == [("Deed_8721_169", "Deed_8310_336", "cites 8310/337")]
    assert report.unresolved == [] and report.mismatches == []


def test_citation_outside_range_is_unresolved(db):
    older = doc("D1", 3, rec("warranty deed", 8310, 336, "2017-03-01", [], []))
    newer = doc("D2", 2, rec("warranty deed", 8721, 169, "2021-06-01", [], [], prior=(8310, 339)))
    report = records.rebuild(db, [older, newer])
    assert report.links == [] and report.unresolved == ["D2 cites 8310/339, which isn't loaded"]


def test_parcel_propagates_along_chain_but_not_from_judgments(db):
    judgment = doc("Jud", 3, rec("foreclosure judgment", 598, 12, "1974-11-18", [], ["COUNTY"]),
                   ["12.04-3-7", "15.02-1-4"])
    tax_deed = doc("Tax", 2, rec("tax deed", 612, 118, "1975-04-22", ["COUNTY"], ["W P"], prior=(598, 14),
                                 parcel="12.04-3-7"), ["12.04-3-7"])
    quitclaim = doc("QC", 2, rec("quitclaim deed", 1043, 77, "1988-06-10", ["W P"], ["M Q"], prior=(612, 118)))
    orphan = doc("Orphan", 1, rec("quitclaim deed", 9, 9, "1990-01-01", ["X"], ["Y"], prior=(598, 13)))
    report = records.rebuild(db, [judgment, tax_deed, quitclaim, orphan])
    assert ("Tax", "Jud", "cites 598/14") in report.links
    assert records.doc_parcels(db, "QC") == ["12.04-3-7"]  # inherited from the tax deed
    assert records.doc_parcels(db, "Orphan") == []  # linked only to a judgment: not guessed
    assert report.unscoped == ["Orphan"]
    assert [r["doc_id"] for r in records.parcel_documents(db, "12.04-3-7")] == ["QC", "Tax", "Jud"]


def test_judgment_links_to_latest_earlier_document_of_listed_parcel(db):
    d1 = doc("D1", 1, rec("warranty deed", 1, 1, "1990-01-01", ["A"], ["B"], parcel="1.1-1-1"), ["1.1-1-1"])
    d2 = doc("D2", 1, rec("warranty deed", 2, 1, "2000-01-01", ["B"], ["C"], prior=(1, 1), parcel="1.1-1-1"),
             ["1.1-1-1"])
    jud = doc("J", 2, rec("foreclosure judgment", 3, 1, "2010-01-01", [], []), ["1.1-1-1", "2.2-2-2"])
    report = records.rebuild(db, [d1, d2, jud])
    assert ("J", "D2", "lists parcel 1.1-1-1") in report.links


def test_link_check_flags_sellers_who_were_not_buyers(db):
    a = doc("A", 1, rec("warranty deed", 1, 1, "1990-01-01", ["X"], ["WALTER SMITH and EDITH SMITH, his wife"]))
    b = doc("B", 1, rec("warranty deed", 2, 1, "2000-01-01", ["Edith Smith"], ["Z"], prior=(1, 1)))
    c = doc("C", 1, rec("warranty deed", 3, 1, "2010-01-01", ["Nobody Else"], ["Q"], prior=(2, 1)))
    report = records.rebuild(db, [a, b, c])
    assert len(report.mismatches) == 1 and report.mismatches[0].startswith("C:")


def test_rebuild_replaces_rows_but_keeps_review_log(db):
    d = doc("D", 1, rec("warranty deed", 1, 1, "1990-01-01", ["A"], ["B"]), ["1.1-1-1"])
    records.rebuild(db, [d])
    records.log_review(db, "dropped_chunk", "D", "looked like an instruction")
    records.rebuild(db, [d])
    assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM parties").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM review_log").fetchone()[0] == 1


def test_document_without_record_is_kept_for_search(db):
    records.rebuild(db, [doc("Scan", 2, None, ["9.9-9-9"])])
    assert records.doc_parcels(db, "Scan") == ["9.9-9-9"]
    assert db.execute("SELECT COUNT(*) FROM instruments").fetchone()[0] == 0


@pytest.mark.parametrize("amount, label", [
    (10.0, "$10.00 (nominal: a token amount, not the real price)"),
    (1.0, "$1.00 (nominal: a token amount, not the real price)"),
    (850.0, "$850.00"),
    (236000.0, "$236,000.00"),
    (None, None),
])
def test_price_label_marks_nominal_amounts(amount, label):
    assert records.price_label(amount) == label


def test_read_only_connection_cannot_write(tmp_path):
    path = tmp_path / "r.db"
    records.connect(path).close()
    ro = records.connect(path, read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("INSERT INTO documents (doc_id) VALUES ('x')")
    ro.close()


def test_title_helpers():
    r = rec("warranty deed", 3307, 52, "2019-05-22", [], [])
    assert records.ref_label(*records.recorded_ref(r)) == "Book 3307, Page 52"
    r["recorded_at"]["kind"] = "reel_frame"
    assert records.ref_label(*records.recorded_ref(r)) == "Reel 3307, Frame 52"
    assert records.year_of(r) == 2019
    assert records.year_of({"date_recorded": None, "date_signed": "1988-06-02"}) == 1988
