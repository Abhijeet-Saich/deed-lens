"""python -m deedlens.ingest [--source DIR] [--redo] [--no-gemini]"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from deedlens.config import get_settings
from deedlens.ingest.pipeline import Pipeline, format_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deedlens.ingest",
                                     description="Ingest the PDFs in DATA_DIR/raw.")
    parser.add_argument("--source", type=Path, help="copy PDFs from this folder into DATA_DIR/raw first")
    parser.add_argument("--redo", action="store_true", help="ignore every cache and redo all steps")
    parser.add_argument("--no-gemini", action="store_true",
                        help="Tesseract only, no extraction (search-only mode); needs no API key")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.source and not args.source.is_dir():
        parser.error(f"--source {args.source} is not a folder")
    use_gemini = not args.no_gemini
    if use_gemini and not settings.has_gemini:
        print("GEMINI_API_KEY is not set: running Tesseract only, without extraction (search-only mode).")
        use_gemini = False
    if not use_gemini:
        settings = replace(settings, use_gemini_transcription=False)

    summary = Pipeline(settings, use_gemini=use_gemini, redo=args.redo).run(args.source)
    print(format_summary(summary))
    return 1 if any(d.error for d in summary.docs) else 0


if __name__ == "__main__":
    sys.exit(main())
