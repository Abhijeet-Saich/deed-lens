# DeedLens

Question answering over land records (deeds, mortgages, court judgments) that answers with
citations to the exact scanned page.

> Work in progress. The full README (quick start, architecture, measured design decisions)
> arrives with packaging; see `docs/BUILD_PROMPT.md`.

## Development

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # Intel Mac: pip install torch==2.2.2 "numpy<2" first
pytest -q
```

License: MIT.
