"""Lesson 13: every search is scoped to one parcel, for vector search AND BM25."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from deedlens.ingest.index import build_index
from deedlens.store.search import COLLECTION, BM25Retriever, SearchIndex, open_qdrant
from tests.test_pipeline import FakeEmbeddings

A, B = "12.04-3-7", "12.04-3-8"
CHUNKS = [  # near-identical neighbour deeds, as in the Lesson 13 drill
    {"id": "a1", "doc_id": "DA", "page": 2, "title": "A deed", "parcels": [A],
     "text": "containing 0.62 acres, more or less, known as 14 Juniper Hollow Road"},
    {"id": "b1", "doc_id": "DB", "page": 2, "title": "B deed", "parcels": [B],
     "text": "containing 0.71 acres, more or less, known as 16 Juniper Hollow Road"},
    {"id": "j1", "doc_id": "DJ", "page": 3, "title": "Judgment", "parcels": [A, B, "15.02-1-4"],
     "text": "Exhibit A: 12.04-3-7 $412.37; 12.04-3-8 $198.05"},
]


class FakeReranker:
    """Scores by word overlap with the query, so ranking is deterministic."""

    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]


def test_bm25_keeps_numbers_whole():
    docs = [Document(page_content=c["text"], metadata=c) for c in CHUNKS]
    hits = BM25Retriever(docs=docs, k=5).invoke("how many acres 0.71")
    assert hits[0].metadata["id"] == "b1"


def test_parcel_filter_applies_to_both_methods(tmp_path):
    client = open_qdrant(tmp_path / "qdrant")
    try:
        emb = FakeEmbeddings()
        build_index(client, emb, CHUNKS)
        store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=emb)
        docs = [Document(page_content=c["text"], metadata={k: v for k, v in c.items() if k != "text"})
                for c in CHUNKS]
        index = SearchIndex(store, docs, FakeReranker(), per_method_k=25, context_k=5)

        for parcel, own, other in [(A, "a1", "b1"), (B, "b1", "a1")]:
            retriever = index.retriever(parcel)
            vector_ids = {d.metadata["id"] for d in retriever.first.invoke("acres Juniper Hollow Road")}
            bm25_ids = {d.metadata["id"] for d in retriever.second.invoke("acres Juniper Hollow Road")}
            assert own in vector_ids and other not in vector_ids
            assert own in bm25_ids and other not in bm25_ids
            final = [d.metadata["id"] for d in index.search(parcel, "acres Juniper Hollow Road")]
            assert final[0] == own and other not in final
            assert "j1" in [d.metadata["id"] for d in index.search(parcel, "Exhibit amount 12.04-3-7")]

        assert index.search("99.9-9-9", "acres") == []  # unknown parcel sees nothing
    finally:
        client.close()
