from deedlens.ingest.chunk import chunk_document, chunk_page, clean_document


def words(text: str) -> int:  # fake tokenizer: one token per word
    return len(text.split())


META = {"doc_id": "Deed_1", "title": "2019 warranty deed (Book 1, Page 2)", "instrument_type": "warranty deed",
        "year": 2019, "recorded_at": "Book 1, Page 2", "parcels": ["12.04-3-7"]}


def long_page(n_lines: int, page: int) -> dict:
    return {"page": page, "text": "\n".join(f"line {page}-{i} " + "word " * 12 for i in range(n_lines))}


def test_clean_drops_symbol_lines_and_repeats_but_keeps_short_lines():
    pages = [{"page": 1, "text": "FICTIONAL SAMPLE BANNER\n---\nss.:\nThe grantor conveys"},
             {"page": 2, "text": "fictional  sample  banner\nss.:\n☒ ☐\nThe grantee accepts"}]
    cleaned, dropped = clean_document(pages)
    assert cleaned[0]["lines"] == ["FICTIONAL SAMPLE BANNER", "ss.:", "The grantor conveys"]
    assert cleaned[1]["lines"] == ["ss.:", "The grantee accepts"]  # repeat dropped (spacing/case ignored)
    assert dropped == 3


def test_no_chunk_exceeds_target_and_overlap_is_carried():
    lines = [f"line {i} " + "word " * 12 for i in range(40)]
    chunks = chunk_page(lines, "Title, page 1 of 1", words, target=60, overlap=15)
    assert len(chunks) > 1
    assert all(words(c) <= 60 for c in chunks)
    first_lines, second_lines = chunks[0].splitlines()[1:], chunks[1].splitlines()[1:]
    assert second_lines[0] in first_lines  # overlap: the next chunk starts with a carried line


def test_long_single_line_is_split_on_words():
    chunks = chunk_page(["word " * 200], "H", words, target=50, overlap=10)
    assert all(words(c) <= 50 for c in chunks)
    assert sum(words(c) - 1 for c in chunks) == 200  # nothing lost (overlap can't carry a whole piece)


def test_chunks_never_cross_pages_and_have_header_and_metadata():
    pages = [long_page(20, 1), long_page(20, 2), {"page": 3, "text": "---"}]
    chunks = chunk_document(pages, META, words, target=60, overlap=15)
    assert {c["page"] for c in chunks} == {1, 2}  # page 3 had nothing left after cleaning
    for c in chunks:
        header, *body = c["text"].splitlines()
        assert header == f"{META['title']}, page {c['page']} of 3"
        assert all(line.startswith(f"line {c['page']}-") for line in body)
        assert c["parcels"] == ["12.04-3-7"] and c["doc_id"] == "Deed_1"
        assert c["id"] == f"Deed_1-p{c['page']}-c{c['chunk']}"
        assert c["tokens"] <= 60
