"""Loads everything the agent needs once (models, Qdrant, checkpoints) for the CLI and the API."""

from __future__ import annotations

import sqlite3
from typing import Any

from deedlens.agent.graph import Agent, Deps, build_graph
from deedlens.config import Settings
from deedlens.store.search import SearchIndex, open_qdrant


class GeminiUnavailable(RuntimeError):
    pass


def make_llm(settings: Settings) -> Any:
    """One shared chat model. temperature 0, but answers still vary a little between runs (Lesson 13)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0,
                                  google_api_key=settings.gemini_api_key)


def make_checkpointer(settings: Settings) -> Any:
    """LangGraph checkpoints in their own SQLite file, so paused reviews survive a restart."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.checkpoints_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoints_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


class AgentRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = open_qdrant(settings.qdrant_dir)
        self.index = SearchIndex.from_settings(settings, self.client)
        self.checkpointer = make_checkpointer(settings)
        self.agent: Agent | None = None
        if settings.has_gemini:
            deps = Deps(llm=make_llm(settings), search=self.index.search, db_path=settings.db_path)
            self.agent = Agent(build_graph(deps, self.checkpointer))

    @property
    def gemini(self) -> bool:
        return self.agent is not None

    def _agent(self) -> Agent:
        if self.agent is None:
            raise GeminiUnavailable("GEMINI_API_KEY is not set, so questions can't be answered. "
                                    "Add it to .env and restart.")
        return self.agent

    def ask(self, parcel: str, question: str) -> dict[str, Any]:
        return self._agent().ask(parcel, question)

    def pending(self, thread_id: str) -> bool:
        return self._agent().pending(thread_id)

    def review(self, thread_id: str, action: str, text: str | None = None) -> dict[str, Any]:
        return self._agent().review(thread_id, action, text)

    def close(self) -> None:
        self.client.close()
        self.checkpointer.conn.close()
