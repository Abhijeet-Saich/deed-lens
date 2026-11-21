"""The LangGraph agent (Lesson 11) with the Lesson 12 guardrails.

    START -> route -> records and/or search -> answer -> grade -> check -> END
                              ^                              |       |
                              +---- part missing? -----------+       pauses (interrupt) for a person
                                    try the unused source, once      when the draft uses a weakly
                                                                     evidenced value or fails a check

- The grader splits the question into parts and must quote the draft for each; CODE decides
  completeness. "Reason" parts need a stated cause, not just the name of a process.
- MAX_ATTEMPTS = 2: the first answer plus at most one retry, so the loop always ends.
- On resume LangGraph re-runs `check` from the top, so everything before `interrupt()` only reads.
- Everything is scoped to the selected parcel: records SQL and search alike.
"""

from __future__ import annotations

import operator
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from deedlens.agent import guardrails as g
from deedlens.agent import prompts
from deedlens.store import records

MAX_ATTEMPTS = 2
ACTIONS = ("approve", "edit", "reject")


class AgentState(TypedDict, total=False):
    question: str
    parcel: str
    property_summary: str
    route: str
    route_reason: str
    records: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    dropped_chunks: list[dict[str, Any]]
    draft: str
    complete: bool
    missing: str
    attempts: int
    flagged: list[dict[str, Any]]
    answer: str
    review: str
    citations: list[dict[str, Any]]
    issues: list[str]
    has_records: bool  # False in search-only mode (nothing extracted for this parcel)
    path: Annotated[list[str], operator.add]  # nodes visited, in order (survives pauses)


class Route(BaseModel):
    route: Literal["records", "search", "both"] = Field(description="Where to look")
    reason: str = Field(description="One short sentence explaining the choice")


class Part(BaseModel):
    part: str = Field(description="One thing the question asks for, in a few words")
    kind: Literal["fact", "reason"] = Field(description="reason for why/how parts, fact for everything else")
    quote: str = Field(description="Exact words from the draft that answer this part; empty string if "
                                   "the draft doesn't answer it")


class Grade(BaseModel):
    parts: list[Part]


@dataclass
class Deps:
    llm: Any  # a LangChain chat model
    search: Callable[[str, str], list[Document]]  # (parcel, question) -> ranked chunks
    db_path: Path
    review_db_path: Path | None = None  # writable, for review_log; defaults to db_path


def _text(reply: Any) -> str:
    """Chat replies can carry content as a string or a list of parts."""
    content = getattr(reply, "content", reply)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _summary_text(s: dict[str, Any]) -> str:
    return (f"tax parcel {s['parcel']}; address {s['address'] or 'unknown'}; "
            f"current owner {s['current_owner'] or 'unknown'}; {s['documents']} recorded documents")


def sources_of(state: AgentState) -> tuple[list[dict[str, Any]], str]:
    """(citable sources, all their text) given to the answer step."""
    srcs = [{"kind": "records", "title": r["title"], "doc_id": r["doc_id"]} for r in state.get("records") or []]
    srcs += [{"kind": "chunk", "title": c["title"], "doc_id": c["doc_id"], "page": c["page"]}
             for c in state.get("chunks") or []]
    text = "\n".join([r["text"] for r in state.get("records") or []] + [c["text"] for c in state.get("chunks") or []])
    return srcs, text


def build_graph(deps: Deps, checkpointer: Any = None) -> Any:
    router = deps.llm.with_structured_output(Route)
    grader = deps.llm.with_structured_output(Grade)
    answer_prompt = ChatPromptTemplate.from_messages([("system", prompts.ANSWER_SYSTEM),
                                                      ("human", prompts.ANSWER_HUMAN)])

    def ro() -> sqlite3.Connection:
        return records.connect(deps.db_path, read_only=True)

    # ----- route -----
    def route(state: AgentState) -> dict[str, Any]:
        db = ro()
        try:
            summary = _summary_text(records.property_summary(db, state["parcel"]))
            available = records.has_records(db, state["parcel"])
        finally:
            db.close()
        if not available:
            return {"route": "search", "route_reason": "No extracted records for this parcel (search-only mode).",
                    "property_summary": summary, "has_records": False, "path": ["route"]}
        decision = router.invoke(prompts.ROUTER_PROMPT.format(property_summary=summary, question=state["question"]))
        return {"route": decision.route, "route_reason": decision.reason, "property_summary": summary,
                "has_records": True, "path": ["route"]}

    def pick_branches(state: AgentState) -> list[str]:
        return ["records", "search"] if state["route"] == "both" else [state["route"]]

    # ----- records: read-only SQL for the selected parcel only -----
    def records_node(state: AgentState) -> dict[str, Any]:
        db = ro()
        try:
            lines = records.records_lines(db, state["parcel"])
        finally:
            db.close()
        return {"records": lines, "path": ["records"]}

    # ----- search: two-stage retrieval filtered to the parcel, then the input scanner -----
    def search_node(state: AgentState) -> dict[str, Any]:
        docs = deps.search(state["parcel"], state["question"])
        chunks = [{"id": d.metadata["id"], "doc_id": d.metadata["doc_id"], "page": d.metadata["page"],
                   "title": d.metadata["title"], "label": f"{d.metadata['title']}, page {d.metadata['page']}",
                   "text": d.page_content, "score": d.metadata.get("rerank_score")} for d in docs]
        kept, dropped = g.input_filter(chunks)
        if dropped:  # logged for a person to look at, never silently discarded
            db = records.connect(deps.review_db_path or deps.db_path)
            try:
                for c in dropped:
                    records.log_review(db, "dropped_chunk", c["doc_id"],
                                       f"{c['id']}: matched {c['matched']} (question: {state['question']!r})")
            finally:
                db.close()
        return {"chunks": kept, "dropped_chunks": (state.get("dropped_chunks") or []) + dropped,
                "path": ["search"]}

    # ----- answer -----
    def answer(state: AgentState) -> dict[str, Any]:
        parts = []
        if state.get("records"):
            parts.append("RECORDS TABLE:\n" + "\n".join(r["text"] for r in state["records"]))
        if state.get("chunks"):
            parts.append("DOCUMENT TEXT:\n" + "\n\n".join(f"[{c['label']}]\n{c['text']}" for c in state["chunks"]))
        context = "\n\n".join(parts) or "(no sources were found for this property)"
        reply = (answer_prompt | deps.llm).invoke({"context": context, "question": state["question"],
                                                   "property_summary": state.get("property_summary", "")})
        return {"draft": _text(reply), "path": ["answer"]}

    # ----- grade: the grader quotes; code decides -----
    def grade(state: AgentState) -> dict[str, Any]:
        result = grader.invoke(prompts.GRADE_PROMPT.format(question=state["question"], draft=state["draft"]))
        missing = [p.part for p in result.parts if not p.quote.strip()]
        return {"complete": not missing, "missing": "; ".join(missing),
                "attempts": state.get("attempts", 0) + 1, "path": ["grade"]}

    def after_grade(state: AgentState) -> str:
        if state["complete"] or state["attempts"] >= MAX_ATTEMPTS:
            return "check"
        if "chunks" not in state:  # retry with the source not used yet
            return "search"
        if "records" not in state and state.get("has_records"):
            return "records"
        return "check"

    # ----- check: output guardrails + weak evidence -> maybe pause for a person -----
    def check(state: AgentState) -> dict[str, Any]:
        # Everything up to interrupt() re-runs on resume, so it must only read.
        draft = state["draft"]
        sources, source_text = sources_of(state)
        db = ro()
        try:
            known = records.known_party_words(db, state["parcel"]) if records.has_records(db, state["parcel"]) else None
            weak = records.weak_values(db, state["parcel"])
        finally:
            db.close()
        citations, _ = g.resolve_citations(draft, sources)
        issues = g.output_issues(draft, sources, source_text, known)
        flagged = g.weak_values_used(draft, weak)
        if not flagged and not issues:
            return {"answer": draft, "citations": citations, "flagged": [], "issues": [],
                    "review": "not needed", "path": ["check"]}

        decision = interrupt({"draft": draft, "flagged": flagged, "issues": issues, "citations": citations})

        action = (decision or {}).get("action")
        base = {"flagged": flagged, "issues": issues, "path": ["check"]}
        if action == "approve":
            return {**base, "answer": draft, "citations": citations, "review": "approved by reviewer"}
        if action == "edit" and (decision.get("text") or "").strip():
            text = decision["text"].strip()
            return {**base, "answer": text, "citations": g.resolve_citations(text, sources)[0],
                    "review": "edited by reviewer"}
        return {**base, "answer": prompts.REJECTED_ANSWER, "citations": [], "review": "rejected by reviewer"}

    graph = StateGraph(AgentState)
    for name, fn in [("route", route), ("records", records_node), ("search", search_node),
                     ("answer", answer), ("grade", grade), ("check", check)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", pick_branches, ["records", "search"])
    graph.add_edge("records", "answer")
    graph.add_edge("search", "answer")
    graph.add_edge("answer", "grade")
    graph.add_conditional_edges("grade", after_grade, ["check", "search", "records"])  # the loop
    graph.add_edge("check", END)
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------- running it

class ReviewNotPending(LookupError):
    """The thread doesn't exist or isn't waiting for a review."""


class Agent:
    """Runs the compiled graph: one new thread per question; resumes paused threads."""

    def __init__(self, graph: Any):
        self.graph = graph

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _result(self, thread_id: str) -> dict[str, Any]:
        snap = self.graph.get_state(self._config(thread_id))
        v = snap.values
        common = {"thread_id": thread_id, "route": v.get("route"), "route_reason": v.get("route_reason"),
                  "path": v.get("path", [])}
        if snap.interrupts:
            payload = snap.interrupts[0].value
            return {"status": "needs_review", **common, "draft": payload["draft"],
                    "flagged": payload["flagged"], "issues": payload["issues"],
                    "citations": payload.get("citations", [])}
        return {"status": "answered", **common, "answer": v.get("answer", ""), "citations": v.get("citations", []),
                "review": v.get("review"), "flagged": v.get("flagged", []), "issues": v.get("issues", [])}

    def ask(self, parcel: str, question: str, thread_id: str | None = None) -> dict[str, Any]:
        thread_id = thread_id or str(uuid.uuid4())
        self.graph.invoke({"question": question, "parcel": parcel}, self._config(thread_id))
        return self._result(thread_id)

    def values(self, thread_id: str) -> dict[str, Any]:
        """The full graph state of a thread (records, chunks, draft, ...), e.g. for evaluation."""
        return dict(self.graph.get_state(self._config(thread_id)).values)

    def pending(self, thread_id: str) -> bool:
        return bool(self.graph.get_state(self._config(thread_id)).interrupts)

    def review(self, thread_id: str, action: str, text: str | None = None) -> dict[str, Any]:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        if not self.pending(thread_id):
            raise ReviewNotPending(thread_id)
        self.graph.invoke(Command(resume={"action": action, "text": text}), self._config(thread_id))
        return self._result(thread_id)
