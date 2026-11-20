"""Search at question time: embeddings, the Qdrant store and the parcel filter.

(The BM25 and two-stage retrievers are added with the agent.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

COLLECTION = "deedlens_chunks"
# Lesson 7: bge models expect this prefix on QUERIES only; passages are embedded as they are.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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


def open_qdrant(path: Path) -> Any:
    """Qdrant local mode allows ONE client per storage folder: open it once and share it."""
    from qdrant_client import QdrantClient

    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def parcel_filter(parcel: str) -> Any:
    """Lesson 13: every search is scoped to one parcel. Matches if `parcels` contains it."""
    from qdrant_client import models

    return models.Filter(must=[models.FieldCondition(key="metadata.parcels",
                                                     match=models.MatchValue(value=parcel))])
