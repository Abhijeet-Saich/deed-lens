from pathlib import Path

from deedlens.config import Settings


def test_defaults_without_key(settings):
    assert settings.has_gemini is False
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.embed_model == "BAAI/bge-small-en-v1.5"
    assert settings.chunk_target_tokens == 200
    assert settings.chunk_overlap_tokens == 40
    assert settings.per_method_k == 25
    assert settings.context_k == 5
    assert 1 <= settings.ocr_workers <= 8


def test_all_paths_derive_from_data_dir(tmp_path):
    s = Settings.from_env({"DATA_DIR": str(tmp_path / "x")})
    base = tmp_path / "x"
    for p in (s.raw_dir, s.text_dir, s.extracted_dir, s.chunks_file, s.qdrant_dir,
              s.db_path, s.checkpoints_path, s.manifest_path, s.page_cache_dir, s.results_dir):
        assert Path(p).is_relative_to(base)


def test_env_overrides_and_placeholder_key():
    s = Settings.from_env({
        "GEMINI_API_KEY": "your-gemini-api-key-here",
        "USE_GEMINI_TRANSCRIPTION": "false",
        "CONTEXT_K": "8",
        "GEMINI_SECONDS_BETWEEN_CALLS": "0.5",
    })
    assert s.has_gemini is False
    assert s.use_gemini_transcription is False
    assert s.context_k == 8
    assert s.gemini_seconds_between_calls == 0.5
    assert Settings.from_env({"GEMINI_API_KEY": "abc"}).has_gemini is True
