"""Shared test fixtures and fakes. Unit tests never need an API key, network or model downloads."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from deedlens.config import Settings
from deedlens.store import records


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with all state under a temporary DATA_DIR and no Gemini key."""
    return Settings.from_env({"DATA_DIR": str(tmp_path / "data")})


class FakeLLM(RunnableLambda):
    """Stands in for ChatGoogleGenerativeAI: fixed route, fixed answer, scripted grading.

    `complete=False` makes the FIRST grade report a missing part (to exercise the retry loop).
    `last_context` keeps the last prompt the answer step saw.
    """

    def __init__(self, route: str = "records", answer: str = "", complete: bool = True):
        self.route_choice, self.answer_text, self.complete_first = route, answer, complete
        self.last_context = ""
        self.router_calls = self.grade_calls = 0
        super().__init__(self._answer)

    def _answer(self, prompt_value) -> AIMessage:
        self.last_context = prompt_value.to_string()
        return AIMessage(content=self.answer_text)

    def with_structured_output(self, model):
        from deedlens.agent.graph import Grade, Part, Route

        def route(_):
            self.router_calls += 1
            return Route(route=self.route_choice, reason="fake router")

        def grade(_):
            self.grade_calls += 1
            ok = self.complete_first or self.grade_calls > 1
            return Grade(parts=[Part(part="the answer", kind="fact", quote="it" if ok else "")])

        return RunnableLambda(route if model is Route else grade)


def _record(volume, page, date, grantors, grantees, in_deed, reported, parcel, prior, evidence):
    return {"instrument_type": "warranty deed",
            "recorded_at": {"volume": volume, "page": page, "kind": "book_page"},
            "date_signed": None, "date_recorded": date, "grantors": grantors, "grantees": grantees,
            "consideration_in_deed": in_deed, "consideration_reported": reported, "transfer_tax": None,
            "prior_deed": {"volume": prior[0], "page": prior[1], "kind": "book_page"} if prior else None,
            "parcel_id": parcel, "property_address": "14 Juniper Hollow Road", "town": "Alder Falls",
            "acreage": 0.62, "evidence": evidence}


def make_records_db(path: Path) -> Path:
    """Two linked deeds of parcel 12.04-3-7; the 2019 grantees have only 'words on page' evidence."""
    p = "12.04-3-7"
    d2004 = _record("2218", "401", "2004-08-27", ["MARTA QUILL"], ["DESMOND OKAFOR", "PRIYA OKAFOR"], 10.0,
                    142500.0, p, None, [{"field": "grantees", "page": 2, "quote": "DESMOND OKAFOR",
                                         "verified": "exact"}])
    d2019 = _record("3307", "52", "2019-05-22", ["DESMOND OKAFOR"], ["LENA MARSH", "TOBIAS RENN"], 1.0,
                    236000.0, p, ("2218", "402"),
                    [{"field": "grantors", "page": 2, "quote": "DESMOND OKAFOR", "verified": "exact"},
                     {"field": "grantees", "page": 1, "quote": "GRANTEE(S) MARSH, LENA", "verified": "words on page"}])
    db = records.connect(path)
    records.rebuild(db, [
        records.DocInput("Deed_2218_401", "Deed 2218 401.pdf", 3, "a", "2004 warranty deed (Book 2218, Page 401)",
                         d2004, [p]),
        records.DocInput("Deed_3307_52", "Deed 3307 52.pdf", 3, "b", "2019 warranty deed (Book 3307, Page 52)",
                         d2019, [p]),
    ])
    db.close()
    return path
