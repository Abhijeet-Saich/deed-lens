"""The HTTP API with a fake agent runtime (no models, no key, no network)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deedlens.agent.graph import ReviewNotPending
from deedlens.agent.runtime import GeminiUnavailable
from deedlens.api.main import create_app
from tests.conftest import make_records_db

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
PARCEL = "12.04-3-7"
CITE = {"label": "records: 2019 warranty deed (Book 3307, Page 52)", "doc_id": "Deed_3307_52", "page": 1,
        "kind": "records"}


class FakeRuntime:
    gemini = True

    def __init__(self):
        self.waiting: dict[str, str] = {}
        self.closed = False

    def ask(self, parcel, question):
        if "owner" in question:
            tid = f"t{len(self.waiting) + 1}"
            self.waiting[tid] = "Lena Marsh owns it."
            return {"status": "needs_review", "thread_id": tid, "route": "records", "path": ["route"],
                    "draft": self.waiting[tid], "flagged": [{"value": "LENA MARSH"}], "issues": []}
        return {"status": "answered", "thread_id": "t0", "route": "records", "path": ["route", "check"],
                "answer": "It was $236,000.00 [records: 2019 warranty deed (Book 3307, Page 52)].",
                "citations": [CITE]}

    def review(self, thread_id, action, text=None):
        if thread_id not in self.waiting:
            raise ReviewNotPending(thread_id)
        draft = self.waiting.pop(thread_id)
        answer = text if action == "edit" else draft
        return {"status": "answered", "thread_id": thread_id, "answer": answer, "citations": [],
                "route": "records", "path": ["route", "check"], "review": f"{action}d"}

    def close(self):
        self.closed = True


@pytest.fixture
def setup(settings):
    make_records_db(settings.db_path)
    settings.raw_dir.mkdir(parents=True)
    shutil.copy(SAMPLES / "Deed 3307 52.pdf", settings.raw_dir / "Deed 3307 52.pdf")
    settings.chunks_file.write_text('{"id": "a"}\n{"id": "b"}\n')
    runtime = FakeRuntime()
    app = create_app(settings, runtime_factory=lambda _s: runtime)
    with TestClient(app) as client:
        yield client, runtime, settings
    assert runtime.closed  # lifespan shut it down


def test_health(setup):
    client, *_ = setup
    assert client.get("/api/health").json() == {"status": "ok", "documents": 2, "chunks": 2, "gemini": True}


def test_properties_and_chain(setup):
    client, *_ = setup
    props = client.get("/api/properties").json()
    assert props == [{"parcel": PARCEL, "address": "14 Juniper Hollow Road", "town": "Alder Falls",
                      "current_owner": "LENA MARSH; TOBIAS RENN", "documents": 2}]
    chain = client.get(f"/api/properties/{PARCEL}/chain").json()
    assert [c["doc_id"] for c in chain] == ["Deed_3307_52", "Deed_2218_401"]  # newest first
    assert chain[0]["nominal"] is True and chain[0]["price_reported"] == 236000.0
    assert chain[0]["sellers"] == ["DESMOND OKAFOR"] and chain[0]["previous"].startswith("2004 warranty deed")


def test_unknown_parcel_is_404_with_error(setup):
    client, *_ = setup
    r = client.get("/api/properties/99.9-9-9/chain")
    assert r.status_code == 404 and r.json() == {"error": "Unknown parcel: 99.9-9-9"}
    r = client.post("/api/ask", json={"parcel": "99.9-9-9", "question": "Who?"})
    assert r.status_code == 404


@pytest.mark.parametrize("body, fragment", [
    ({"parcel": PARCEL, "question": "x" * 501}, "question"),
    ({"parcel": PARCEL, "question": "   "}, "blank"),
    ({"parcel": PARCEL}, "question"),
])
def test_ask_validation(setup, body, fragment):
    client, *_ = setup
    r = client.post("/api/ask", json=body)
    assert r.status_code == 422 and fragment in r.json()["error"]


def test_ask_answered(setup):
    client, *_ = setup
    r = client.post("/api/ask", json={"parcel": PARCEL, "question": "What was the price?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["citations"] == [CITE]


def test_ask_needs_review_then_approve_and_edit(setup):
    client, runtime, _ = setup
    paused = client.post("/api/ask", json={"parcel": PARCEL, "question": "Who is the owner?"}).json()
    assert paused["status"] == "needs_review" and paused["flagged"] == [{"value": "LENA MARSH"}]
    done = client.post(f"/api/review/{paused['thread_id']}", json={"action": "approve"}).json()
    assert done["status"] == "answered" and done["answer"] == "Lena Marsh owns it."

    again = client.post(f"/api/review/{paused['thread_id']}", json={"action": "approve"})
    assert again.status_code == 404 and "No review is waiting" in again.json()["error"]

    tid = client.post("/api/ask", json={"parcel": PARCEL, "question": "owner?"}).json()["thread_id"]
    assert client.post(f"/api/review/{tid}", json={"action": "edit"}).status_code == 422
    assert client.post(f"/api/review/{tid}", json={"action": "bogus"}).status_code == 422
    edited = client.post(f"/api/review/{tid}", json={"action": "edit", "text": " Fixed. "}).json()
    assert edited["answer"] == "Fixed."


def test_page_image_rendered_and_cached(setup):
    client, _, settings = setup
    r = client.get("/api/documents/Deed_3307_52/pages/2.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert (settings.page_cache_dir / "Deed_3307_52" / "2.png").exists()
    assert client.get("/api/documents/Deed_3307_52/pages/4.png").status_code == 404
    assert client.get("/api/documents/Nope/pages/1.png").json() == {"error": "Unknown document: Nope"}
    assert client.get("/api/documents/Deed_2218_401/pages/1.png").status_code == 404  # PDF not in raw/


def test_gemini_unavailable_is_503(settings):
    make_records_db(settings.db_path)

    class NoKey(FakeRuntime):
        gemini = False

        def ask(self, *_a):
            raise GeminiUnavailable("GEMINI_API_KEY is not set")

    with TestClient(create_app(settings, runtime_factory=lambda _s: NoKey())) as client:
        assert client.get("/api/health").json()["gemini"] is False
        r = client.post("/api/ask", json={"parcel": PARCEL, "question": "Who?"})
        assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["error"]


def test_empty_data_dir_still_serves(tmp_path):
    from deedlens.config import Settings

    settings = Settings.from_env({"DATA_DIR": str(tmp_path / "empty")})
    with TestClient(create_app(settings, runtime_factory=lambda _s: FakeRuntime())) as client:
        assert client.get("/api/health").json()["documents"] == 0
        assert client.get("/api/properties").json() == []


def test_frontend_is_served(setup):
    client, *_ = setup
    page = client.get("/")
    assert page.status_code == 200 and "<title>DeedLens</title>" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").headers["content-type"].startswith("text/css")
