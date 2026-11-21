"""Scoring logic of eval/run_eval.py (pure functions: no agent, no key)."""

import json
from pathlib import Path

from eval.run_eval import context_text, failing_stage, judge_cases, score_item, share_present, summarise

ITEM = {"id": "A4", "parcel": "12.04-3-7", "question": "How large is the lot?",
        "must_include": [["0.62"], ["14 Juniper Hollow"]],
        "must_not_include": ["0.71", "16 Juniper Hollow"]}


def test_share_present_counts_groups_with_any_alternative():
    groups = [["$10", "ten dollars"], ["nominal", "token"], ["missing"]]
    assert share_present(groups, "The deed states TEN  dollars, a token amount.") == 2 / 3
    assert share_present([], "anything") == 1.0


def test_correct_answer_scores_clean():
    s = score_item(ITEM, "The lot is 0.62 acres at 14 Juniper Hollow Road.", "0.62 ... 14 Juniper Hollow")
    assert s == {"correct": 1.0, "in_context": 1.0, "contaminated": [], "stage": ""}


def test_contamination_catches_parcel_b_value_even_when_answer_looks_correct():
    answer = "The lot is 0.62 acres (or 0.71 acres) at 14 Juniper Hollow Road."
    context = "[records: 2019 deed] acreage 0.62; 14 Juniper Hollow Road\n[1996 deed] 0.71 acres"
    s = score_item(ITEM, answer, context)
    assert s["correct"] == 1.0  # the simple metric alone would call this right
    assert s["contaminated"] == ["0.71"]
    assert s["stage"] == "contamination (retrieval)"  # the neighbour's chunk reached the model


def test_failing_stage():
    assert failing_stage(0.5, 1.0, [], "") == "generation"
    assert failing_stage(0.5, 0.5, [], "") == "routing/retrieval"
    assert failing_stage(1.0, 1.0, ["0.71"], "no neighbour here") == "contamination (generation)"


def test_context_text_uses_the_answer_step_labels():
    state = {"records": [{"text": "[records: T] price $1.00"}],
             "chunks": [{"label": "T, page 2", "text": "body"}]}
    assert context_text(state) == "[records: T] price $1.00\n\n[T, page 2]\nbody"


def test_summary_rates():
    rows = [{"correct": 1.0, "contaminated": [], "retried": False, "paused": True, "seconds": 2.0},
            {"correct": 0.5, "contaminated": ["0.71"], "retried": True, "paused": False, "seconds": 4.0}]
    s = summarise(rows, None, 1)
    assert s["fully_correct"] == 0.5 and s["contaminated"] == 0.5 and s["seconds_per_question"] == 3.0
    assert "faithfulness" not in s


def test_judge_cases_built_from_sample_chunks(tmp_path: Path):
    chunks = [
        {"doc_id": "Fore_Jud_598_12", "title": "J", "page": 3, "text": "12.04-3-7 Harlan Voss $412.37"},
        {"doc_id": "Deed_3307_52", "title": "D", "page": 1, "text": "Consideration: $236,000.00"},
        {"doc_id": "Deed_3307_52", "title": "D", "page": 2, "text": "Containing 0.62 acres of land"},
    ]
    f = tmp_path / "chunks.jsonl"
    f.write_text("".join(json.dumps(c) + "\n" for c in chunks))
    cases = judge_cases(f)
    assert [(c[0], c[3]) for c in cases] == [("J1", "supported"), ("J2", "unsupported"), ("J3", "partly")]
    f.write_text(json.dumps(chunks[0]) + "\n")
    assert judge_cases(f) == []  # other data: judge can't be checked


def test_sample_answer_file_items_have_the_required_keys():
    items = json.loads((Path(__file__).resolve().parents[1] / "eval" / "answers_sample.json").read_text())
    for item in items:
        assert {"id", "parcel", "question", "must_include", "must_not_include"} <= set(item)
