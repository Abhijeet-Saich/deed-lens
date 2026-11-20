"""python -m deedlens.agent "question" --parcel 12.04-3-7 [--auto-approve]

A small CLI for trying the agent by hand. Pauses for review in the terminal.
"""

from __future__ import annotations

import argparse
import sys

from deedlens.agent.runtime import AgentRuntime
from deedlens.config import get_settings
from deedlens.store import records


def ask_reviewer(result: dict, auto: bool) -> tuple[str, str | None]:
    print("\n   PAUSED FOR HUMAN REVIEW")
    print(f"   Draft: {result['draft'].strip()}")
    for f in result["flagged"]:
        print(f"   * {f['document']} / {f['field']} = {f['value']}  "
              f"(page {f['page']}, evidence: {f['evidence']}, quote: \"{f['quote']}\")")
    for issue in result["issues"]:
        print(f"   ! {issue}")
    if auto:
        print("   --auto-approve: approving")
        return "approve", None
    choice = input("   [a]pprove / [e]dit / [r]eject: ").strip().lower()[:1]
    if choice == "e":
        return "edit", input("   Corrected answer: ")
    return ("reject", None) if choice == "r" else ("approve", None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deedlens.agent")
    parser.add_argument("question")
    parser.add_argument("--parcel", required=True, help="tax parcel, e.g. 12.04-3-7")
    parser.add_argument("--auto-approve", action="store_true", help="approve every review without asking")
    args = parser.parse_args(argv)

    settings = get_settings()
    db = records.connect(settings.db_path, read_only=True)
    parcels = records.all_parcels(db)
    db.close()
    if args.parcel not in parcels:
        parser.error(f"unknown parcel {args.parcel}; known: {', '.join(parcels)}")

    runtime = AgentRuntime(settings)
    try:
        result = runtime.ask(args.parcel, args.question)
        while result["status"] == "needs_review":
            action, text = ask_reviewer(result, args.auto_approve)
            result = runtime.review(result["thread_id"], action, text)
    finally:
        runtime.close()
    print(f"\n   route:  {result['route']} ({result['route_reason']})")
    print(f"   path:   {' -> '.join(result['path'])}")
    print(f"   review: {result['review']}")
    print(f"   ANSWER: {result['answer'].strip()}")
    for c in result["citations"]:
        print(f"   cite:   [{c['label']}] -> {c['doc_id']} page {c['page']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
