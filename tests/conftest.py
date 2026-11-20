"""Shared test fixtures. Unit tests never need an API key, network or model downloads.

Fakes for embeddings, the reranker and the LLM are added here as later phases need them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deedlens.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with all state under a temporary DATA_DIR and no Gemini key."""
    return Settings.from_env({"DATA_DIR": str(tmp_path / "data")})
