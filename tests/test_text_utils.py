import pytest

from deedlens.text_utils import (
    bm25_tokenize,
    digits,
    find_parcels,
    normalize_doc_id,
    normalize_parcel,
    squash,
    to_int,
)


def test_squash_removes_all_whitespace_and_lowercases():
    assert squash("  Book 8310,\n Page  336 ") == "book8310,page336"
    assert squash(None) == ""
    assert squash(12) == "12"


def test_digits_and_to_int():
    assert digits("8,310") == "8310"
    assert digits(None) == ""
    assert to_int("Page 337") == 337
    assert to_int("none") is None


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Deed 1089 349.pdf", "Deed_1089_349"),
        ("Deed 1089 349", "Deed_1089_349"),
        ("Fore  Jud__411 305.PDF", "Fore_Jud_411_305"),
        ("Jud EF2025-280134.pdf", "Jud_EF2025-280134"),
        ("weird/name?.pdf", "weird_name"),
    ],
)
def test_normalize_doc_id(name, expected):
    assert normalize_doc_id(name) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Part of 70.15-5-11", "70.15-5-11"),
        ("70.15-5-11", "70.15-5-11"),
        ("Tax Map No. 12.04-3-7.2", "12.04-3-7.2"),
        ("SBL 12.04-3-8.", "12.04-3-8"),
        ("no parcel here", None),
        (None, None),
        ("recorded 01-15-2004", None),  # a date is not a parcel
    ],
)
def test_normalize_parcel(value, expected):
    assert normalize_parcel(value) == expected


def test_find_parcels_distinct_in_order():
    text = "Exhibit A: 12.04-3-7, 12.04-3-8 and again 12.04-3-7; also 9.1-2-3. Dated 03-04-2019."
    assert find_parcels(text) == ["12.04-3-7", "12.04-3-8", "9.1-2-3"]


def test_find_parcels_not_inside_longer_numbers():
    assert find_parcels("ref 112.04-3-7-9") == []


def test_bm25_tokenizer_keeps_amounts_and_parcels_whole():
    tokens = bm25_tokenize("The amount owed was $48,468.96 on parcel 70.15-5-11, O'Brien's lot.")
    assert "48,468.96" in tokens
    assert "70.15-5-11" in tokens
    assert "o'brien's" in tokens
    assert "the" not in tokens and "on" not in tokens  # stopwords dropped
    assert all(t == t.lower() for t in tokens)


def test_bm25_tokenizer_trailing_punctuation_not_kept():
    assert bm25_tokenize("Book 8310.") == ["book", "8310"]
