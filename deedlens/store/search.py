"""Search at question time: two-stage retrieval, always scoped to one parcel.

Rules carried over from the lessons:
- Lesson 7: embeddings are BAAI/bge-small-en-v1.5, normalized, with a prefix on QUERIES only.
- Lesson 7: stage 1 = vector top 25 ∪ BM25 top 25 (merged, deduplicated); stage 2 = the
  ms-marco cross-encoder reranks every candidate. MRR went 0.58 (vector) -> 0.79. Reciprocal
  Rank Fusion is NOT used as the final ranking: it buried answers only one method found.
- Lesson 10: BM25 tokenizer keeps numbers like 48,468.96 and 70.15-5-11 whole.
- Lesson 13: every search is filtered to the selected parcel, for BOTH methods. Without it,
  look-alike neighbour deeds dropped true accuracy from 9/12 to 4/12.
- Lesson 13: BM25 scores depend on the corpus (IDF, average length). BM25 is built ONCE over all
  chunks and its results are then filtered to the parcel. (Building it over one parcel's few
  chunks gave degenerate IDF: a word in half of them scored 0, so BM25 returned nothing.)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from deedlens.text_utils import bm25_tokenize

COLLECTION = "deedlens_chunks"
# Lesson 7: bge models expect this prefix on QUERIES only; passages are embedded as they are.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------- building blocks

def make_embeddings(model_name: str) -> Any:
    """HuggingFaceEmbeddings with normalized vectors and the query-only prefix.

    `query_encode_kwargs` REPLACES `encode_kwargs` for queries (it is not merged), so the
    normalize flag must be repeated there; without it queries would not be normalized.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": prefix},
    )


def make_reranker(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=512)


def open_qdrant(path: Path) -> Any:
    """Qdrant local mode allows ONE client per storage folder: open it once and share it."""
    from qdrant_client import QdrantClient

    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def parcel_filter(parcel: str) -> Any:
    """Matches chunks whose `parcels` list contains this parcel."""
    from qdrant_client import models

    return models.Filter(must=[models.FieldCondition(key="metadata.parcels",
                                                     match=models.MatchValue(value=parcel))])


def load_chunk_documents(chunks_file: Path) -> list[Document]:
    """chunks.jsonl as LangChain Documents: text -> page_content, everything else -> metadata."""
    if not chunks_file.exists():
        return []
    docs = []
    for line in chunks_file.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            docs.append(Document(page_content=c.pop("text"), metadata=c))
    return docs


# ---------------------------------------------------------------- retrievers

class BM25Retriever(BaseRetriever):
    """Keyword search (our tokenizer, numbers kept whole), optionally limited to one parcel.

    Scores come from the whole `docs` corpus; `parcel` only filters which results come back.
    Pass a prebuilt `bm25` to share one index between per-parcel retrievers.
    """

    docs: list[Document]
    k: int = 25
    parcel: str | None = None
    bm25: Any = None

    def model_post_init(self, _context: Any) -> None:
        from rank_bm25 import BM25Okapi

        if self.bm25 is None and self.docs:
            self.bm25 = BM25Okapi([bm25_tokenize(d.page_content) for d in self.docs])

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if self.bm25 is None or self.k <= 0:
            return []
        scores = self.bm25.get_scores(bm25_tokenize(query))
        order = sorted(range(len(self.docs)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if scores[i] <= 0 or len(out) >= self.k:
                break
            if self.parcel is None or self.parcel in (self.docs[i].metadata.get("parcels") or []):
                out.append(self.docs[i])
        return out


class TwoStageRetriever(BaseRetriever):
    """Stage 1: union of two retrievers. Stage 2: a cross-encoder reranks every candidate."""

    first: BaseRetriever
    second: BaseRetriever
    reranker: Any
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        pool: dict[str, Document] = {}
        for d in self.first.invoke(query) + self.second.invoke(query):
            pool.setdefault(d.metadata["id"], d)  # merge, dropping duplicates
        docs = list(pool.values())
        if not docs:
            return []
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores, strict=True), key=lambda pair: -float(pair[1]))
        out = []
        for d, s in ranked[: self.k]:
            d.metadata["rerank_score"] = float(s)
            out.append(d)
        return out


class SearchIndex:
    """Everything search needs, loaded once: Qdrant store, chunk list (for BM25), reranker."""

    def __init__(self, store: Any, docs: list[Document], reranker: Any, per_method_k: int = 25,
                 context_k: int = 5):
        self.store = store
        self.docs = docs
        self.reranker = reranker
        self.per_method_k = per_method_k
        self.context_k = context_k
        self._bm25 = BM25Retriever(docs=docs).bm25  # one keyword index over every chunk
        self._lock = threading.Lock()  # one search at a time: local Qdrant + models aren't built for concurrency

    @classmethod
    def from_settings(cls, settings: Any, client: Any) -> SearchIndex:
        from langchain_qdrant import QdrantVectorStore

        embeddings = make_embeddings(settings.embed_model)
        store = None
        if client.collection_exists(COLLECTION):
            store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
        return cls(store, load_chunk_documents(settings.chunks_file), make_reranker(settings.rerank_model),
                   settings.per_method_k, settings.context_k)

    def retriever(self, parcel: str) -> TwoStageRetriever:
        """A two-stage retriever that can only see chunks of `parcel`."""
        if self.store is not None:
            first: BaseRetriever = self.store.as_retriever(
                search_kwargs={"k": self.per_method_k, "filter": parcel_filter(parcel)})
        else:
            first = BM25Retriever(docs=[], k=0)
        return TwoStageRetriever(first=first, second=BM25Retriever(docs=self.docs, k=self.per_method_k, parcel=parcel,
                                                        bm25=self._bm25),
                                 reranker=self.reranker, k=self.context_k)

    def search(self, parcel: str, question: str) -> list[Document]:
        with self._lock:
            return self.retriever(parcel).invoke(question)
