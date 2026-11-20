"""Settings read once from the environment (and an optional `.env` file).

Every path the app writes derives from DATA_DIR, so pointing DATA_DIR somewhere else
(a test's tmp folder, a Docker volume) moves all state at once.
Variables already set in the shell win over the `.env` file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PLACEHOLDER_KEYS = {"", "your-gemini-api-key-here"}


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    return int(value) if value and value.strip() else default


def _float(value: str | None, default: float) -> float:
    return float(value) if value and value.strip() else default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    data_dir: Path = Path("data")
    embed_model: str = "BAAI/bge-small-en-v1.5"  # Lesson 7: best of 4 models compared
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lesson 7: two-stage search
    chunk_target_tokens: int = 200  # Lesson 6: model reads 512, chunks stay <= 200
    chunk_overlap_tokens: int = 40
    per_method_k: int = 25  # Lesson 7: vector top 25 ∪ BM25 top 25 before reranking
    context_k: int = 5
    ocr_workers: int = field(default_factory=lambda: min(8, os.cpu_count() or 1))  # Lesson 9
    use_gemini_transcription: bool = True  # Lesson 5: Gemini reached 50/50 facts
    gemini_seconds_between_calls: float = 4.0
    hf_home: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        """Build settings from a mapping (default: os.environ). Tests pass their own dict."""
        e = os.environ if env is None else env
        d = cls()
        return cls(
            gemini_api_key=e.get("GEMINI_API_KEY", "").strip(),
            gemini_model=e.get("GEMINI_MODEL") or d.gemini_model,
            data_dir=Path(e.get("DATA_DIR") or d.data_dir),
            embed_model=e.get("EMBED_MODEL") or d.embed_model,
            rerank_model=e.get("RERANK_MODEL") or d.rerank_model,
            chunk_target_tokens=_int(e.get("CHUNK_TARGET_TOKENS"), d.chunk_target_tokens),
            chunk_overlap_tokens=_int(e.get("CHUNK_OVERLAP_TOKENS"), d.chunk_overlap_tokens),
            per_method_k=_int(e.get("PER_METHOD_K"), d.per_method_k),
            context_k=_int(e.get("CONTEXT_K"), d.context_k),
            ocr_workers=_int(e.get("OCR_WORKERS"), d.ocr_workers),
            use_gemini_transcription=_bool(e.get("USE_GEMINI_TRANSCRIPTION"), d.use_gemini_transcription),
            gemini_seconds_between_calls=_float(
                e.get("GEMINI_SECONDS_BETWEEN_CALLS"), d.gemini_seconds_between_calls
            ),
            hf_home=e.get("HF_HOME") or None,
            host=e.get("HOST") or d.host,
            port=_int(e.get("PORT"), d.port),
        )

    # ---- derived ----
    @property
    def has_gemini(self) -> bool:
        return self.gemini_api_key not in PLACEHOLDER_KEYS

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def text_dir(self) -> Path:
        return self.data_dir / "text"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def chunks_file(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def qdrant_dir(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "deedlens.db"

    @property
    def checkpoints_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def page_cache_dir(self) -> Path:
        return self.data_dir / "page_cache"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings. Loads `.env` from the current directory first."""
    load_dotenv(Path.cwd() / ".env", override=False)
    return Settings.from_env()
