"""chunks.jsonl (read by BM25) and the Qdrant collection (vector search) built from it."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from deedlens.store.search import COLLECTION


def read_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_chunks(path: Path, chunks: list[dict[str, Any]]) -> bool:
    """Write chunks.jsonl atomically. Returns False (and writes nothing) if it is unchanged."""
    content = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks)
    if path.exists() and path.read_text() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text(content)
    os.replace(tmp, path)
    return True


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be UUIDs or integers; derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"deedlens:{chunk_id}"))


def index_is_current(client: Any, n_chunks: int) -> bool:
    if not client.collection_exists(COLLECTION):
        return False
    return client.count(COLLECTION, exact=True).count == n_chunks


def build_index(client: Any, embeddings: Any, chunks: list[dict[str, Any]]) -> None:
    """(Re)create the collection and embed every chunk. Metadata goes under `metadata`."""
    from langchain_core.documents import Document
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import models

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    dim = len(embeddings.embed_query("dimension probe"))
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=dim,
                                                                            distance=models.Distance.COSINE))
    if not chunks:
        return
    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    docs = [Document(page_content=c["text"], metadata={k: v for k, v in c.items() if k != "text"})
            for c in chunks]
    store.add_documents(docs, ids=[point_id(c["id"]) for c in chunks], batch_size=64)
