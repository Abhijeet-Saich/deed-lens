"""Answer-level evaluation of the DeedLens agent (generalised from Lesson 13).

    python -m eval.run_eval                       # eval/answers_sample.json, with the judge
    python -m eval.run_eval --no-judge            # fewer Gemini calls
    python -m eval.run_eval --answers my.json --repeat 3

Each item: {id, parcel, question, must_include: [[alternatives, ...], ...], must_not_include: [...]}.
For every item the agent answers (reviews are auto-approved, but counted). Then:
  correct       share of must_include groups present in the answer (any alternative counts)
  contaminated  a must_not_include value appears in the answer. Lesson 13: without this, answers
                mixing in a look-alike neighbour's values still looked "correct".
  in_context    share of must_include groups present in what the model was given
  stage         where a failure happened: routing/retrieval (the fact never reached the model),
                generation (it was there but not used), or contamination
  faithful      a Gemini judge splits the answer into claims and checks each against its cited
                source (supported 1, partly 0.5, unsupported 0). Lesson 13: the judge must pass
                three known-verdict checks first, or its scores are reported as untrusted.
Lesson 13: answers vary between runs even at temperature 0, so --repeat reports rates over runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deedlens.text_utils import squash

DEFAULT_ANSWERS = Path(__file__).resolve().parent / "answers_sample.json"


# ---------------------------------------------------------------- scoring (pure functions)

def share_present(groups: list[list[str]], text: str) -> float:
    """Share of groups with at least one alternative in `text` (whitespace and case ignored)."""
    if not groups:
        return 1.0
    t = squash(text)
    return sum(any(squash(alt) in t for alt in group) for group in groups) / len(groups)


def contamination(values: list[str], text: str) -> list[str]:
    """The must_not_include values that appear in `text`."""
    t = squash(text)
    return [v for v in values if squash(v) in t]


def failing_stage(correct: float, in_context: float, contaminated: list[str], context: str) -> str:
    if contaminated:
        from_context = any(squash(v) in squash(context) for v in contaminated)
        return "contamination (retrieval)" if from_context else "contamination (generation)"
    if correct >= 1:
        return ""
    return "generation" if in_context > correct else "routing/retrieval"


def context_text(state: dict[str, Any]) -> str:
    """What the answer step saw, with the same labels it saw."""
    parts = [r["text"] for r in state.get("records") or []]
    parts += [f"[{c['label']}]\n{c['text']}" for c in state.get("chunks") or []]
    return "\n\n".join(parts)


def score_item(item: dict[str, Any], answer: str, context: str) -> dict[str, Any]:
    correct = share_present(item["must_include"], answer)
    in_ctx = share_present(item["must_include"], context)
    bad = contamination(item.get("must_not_include", []), answer)
    return {"correct": correct, "in_context": in_ctx, "contaminated": bad,
            "stage": failing_stage(correct, in_ctx, bad, context)}


# ---------------------------------------------------------------- the judge

class Claim(BaseModel):
    claim: str = Field(description="One factual statement from the answer")
    cited: str = Field(description="The source label(s) the answer cites for it; empty if none")
    verdict: Literal["supported", "partly", "unsupported"]
    reason: str = Field(description="Short reason, naming what is or isn't in the cited source")


class Judgement(BaseModel):
    claims: list[Claim]


JUDGE_PROMPT = """You check whether an ANSWER is faithful to its SOURCES.
List each factual statement in the answer as a separate claim. For each claim, give the source
label(s) it cites and a verdict, judged ONLY against the text of the cited source(s):
  supported   - every fact in the claim is stated in the cited source text
  partly      - some facts in the claim are stated there, others are not
  unsupported - the cited source text doesn't state it, or the claim cites nothing
Do not use your own knowledge. Statements that the records don't show something are claims
about absence: mark them supported if the cited sources indeed don't show it.

SOURCES:
{sources}

ANSWER:
{answer}"""

VERDICT_SCORE = {"supported": 1.0, "partly": 0.5, "unsupported": 0.0}


def judge(judge_llm: Any, answer: str, sources: str) -> tuple[float, list[Claim]]:
    result = judge_llm.invoke(JUDGE_PROMPT.format(sources=sources, answer=answer))
    claims = result.claims
    return (sum(VERDICT_SCORE[c.verdict] for c in claims) / len(claims) if claims else 0.0), claims


def judge_cases(chunks_file: Path) -> list[tuple[str, str, str, str]]:
    """Three known-verdict cases built from the sample chunks: (id, answer, sources, expected)."""
    chunks = [json.loads(line) for line in chunks_file.read_text().splitlines() if line.strip()]

    def find(doc_id: str, needle: str) -> dict[str, Any] | None:
        return next((c for c in chunks if c["doc_id"] == doc_id and squash(needle) in squash(c["text"])), None)

    owed = find("Fore_Jud_598_12", "$412.37")
    cover = find("Deed_3307_52", "$236,000.00")
    acres = find("Deed_3307_52", "0.62 acres")
    if not (owed and cover and acres):
        return []
    label = lambda c: f"{c['title']}, page {c['page']}"  # noqa: E731
    src = lambda c: f"[{label(c)}]\n{c['text']}"  # noqa: E731
    return [
        ("J1", f"The amount due on parcel 12.04-3-7 was $412.37 [{label(owed)}].", src(owed), "supported"),
        ("J2", f"The 2019 sale price was $500,000 [{label(cover)}].", src(cover), "unsupported"),
        ("J3", f"The lot is 0.62 acres and was sold for $75,000 [{label(acres)}].", src(acres), "partly"),
    ]


def check_the_judge(judge_llm: Any, chunks_file: Path) -> tuple[int, int]:
    """(passed, total). The judge's scores are only trusted if it passes all of them."""
    cases = judge_cases(chunks_file)
    if not cases:
        print("Judge self-check: the sample chunks aren't loaded, so the judge can't be checked.")
        return 0, 0
    passed = 0
    print("Checking the judge on 3 known cases:")
    for cid, answer, sources, expected in cases:
        _, claims = judge(judge_llm, answer, sources)
        verdicts = [c.verdict for c in claims]
        ok = verdicts == [expected] or (expected == "partly" and set(verdicts) == {"supported", "unsupported"})
        passed += ok
        print(f"   {cid} expected {expected:<11} got {'/'.join(verdicts) or '-':<24} {'ok' if ok else 'FAIL'}")
    return passed, len(cases)


# ---------------------------------------------------------------- running

def run_item(runtime: Any, item: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    result = runtime.ask(item["parcel"], item["question"])
    paused, flagged, issues = False, [], []
    if result["status"] == "needs_review":  # auto-approve, but remember it paused and why
        paused, flagged, issues = True, [f["value"] for f in result["flagged"]], result["issues"]
        result = runtime.review(result["thread_id"], "approve")
    state = runtime.values(result["thread_id"])
    answer = result.get("answer", "")
    path = result.get("path", [])
    return {"id": item["id"], "parcel": item["parcel"], "question": item["question"], "answer": answer,
            "route": result.get("route"), "path": path, "context": context_text(state),
            "retried": path.count("answer") > 1, "paused": paused, "flagged": flagged, "issues": issues,
            "dropped_chunks": len(state.get("dropped_chunks") or []), "seconds": time.time() - start}


def print_table(rows: list[dict[str, Any]], with_judge: bool) -> None:
    head = f"{'id':<22}{'route':<8}{'correct':>8}{'in ctx':>8}"
    head += f"{'faithful':>9}" if with_judge else ""
    print("\n" + head + f"{'retry':>6}{'review':>7}{'secs':>6}  stage")
    for r in rows:
        line = f"{r['id']:<22}{r['route'] or '-':<8}{r['correct']:>8.0%}{r['in_context']:>8.0%}"
        line += f"{r['faithful']:>9.2f}" if with_judge else ""
        line += f"{'yes' if r['retried'] else '':>6}{'yes' if r['paused'] else '':>7}{r['seconds']:>6.0f}"
        stage = r["stage"] + (f": {', '.join(r['contaminated'])}" if r["contaminated"] else "")
        print(line + (f"  <- {stage}" if stage else ""))


def summarise(rows: list[dict[str, Any]], judge_ok: tuple[int, int] | None, runs: int) -> dict[str, Any]:
    n = len(rows)
    s = {
        "items": n // runs, "runs": runs,
        "fully_correct": sum(r["correct"] == 1 and not r["contaminated"] for r in rows) / n,
        "average_correctness": sum(r["correct"] for r in rows) / n,
        "contaminated": sum(bool(r["contaminated"]) for r in rows) / n,
        "retried": sum(r["retried"] for r in rows) / n,
        "paused": sum(r["paused"] for r in rows) / n,
        "seconds_per_question": sum(r["seconds"] for r in rows) / n,
    }
    if judge_ok is not None:
        s["faithfulness"] = sum(r["faithful"] for r in rows) / n
        s["judge_checks"] = f"{judge_ok[0]}/{judge_ok[1]}"
        s["judge_trusted"] = judge_ok[1] > 0 and judge_ok[0] == judge_ok[1]
    return s


def print_summary(s: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    runs = f" over {s['runs']} runs" if s["runs"] > 1 else ""
    print(f"\nFully correct (all facts, no contamination): {s['fully_correct']:.0%}{runs}"
          f"   (average correctness {s['average_correctness']:.0%})")
    print(f"Contaminated with another parcel's values:   {s['contaminated']:.0%}")
    if "faithfulness" in s:
        trust = "" if s["judge_trusted"] else "   <- judge failed its checks: don't trust this number"
        print(f"Average faithfulness:                        {s['faithfulness']:.2f}"
              f"   (judge passed {s['judge_checks']} checks){trust}")
    print(f"Retry loop fired:                            {s['retried']:.0%}")
    print(f"Paused for review (auto-approved here):      {s['paused']:.0%}")
    print(f"Seconds per question:                        {s['seconds_per_question']:.1f}")
    if "faithfulness" in s:
        weak = [(r["id"], c) for r in rows for c in r["claims"] if c["verdict"] != "supported"]
        if weak:
            print("\nClaims the judge did not fully support:")
            for rid, c in weak:
                print(f"   {rid} [{c['verdict']}] {c['claim'][:90]}\n        reason: {c['reason'][:120]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS, help="answers file to evaluate")
    parser.add_argument("--no-judge", action="store_true", help="skip the faithfulness judge (fewer API calls)")
    parser.add_argument("--repeat", type=int, default=1, help="run every question N times (Lesson 13)")
    parser.add_argument("--only", nargs="*", help="only these item ids")
    args = parser.parse_args(argv)

    from deedlens.agent.runtime import AgentRuntime, make_llm
    from deedlens.config import get_settings

    settings = get_settings()
    if not settings.has_gemini:
        print("GEMINI_API_KEY is not set: the evaluation needs Gemini to answer questions.")
        return 2
    items = json.loads(args.answers.read_text())
    if args.only:
        items = [i for i in items if i["id"] in set(args.only)]

    runtime = AgentRuntime(settings)
    judge_llm, judge_ok = None, None
    try:
        if not args.no_judge:
            judge_llm = make_llm(settings).with_structured_output(Judgement)
            judge_ok = check_the_judge(judge_llm, settings.chunks_file)
        rows = []
        for run in range(args.repeat):
            for item in items:
                r = run_item(runtime, item)
                r.update(score_item(item, r["answer"], r["context"]))
                r["run"] = run + 1
                if judge_llm is not None:
                    r["faithful"], claims = judge(judge_llm, r["answer"], r["context"])
                    r["claims"] = [c.model_dump() for c in claims]
                rows.append(r)
                print(f"   run {run + 1} {r['id']}: correct {r['correct']:.0%}"
                      f"{' CONTAMINATED' if r['contaminated'] else ''}, {r['seconds']:.0f}s")
    finally:
        runtime.close()

    print_table(rows, judge_llm is not None)
    summary = summarise(rows, judge_ok, args.repeat)
    print_summary(summary, rows)

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out = settings.results_dir / f"answers_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"answers_file": str(args.answers), "model": settings.gemini_model,
                               "summary": summary, "rows": rows}, indent=2))
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
