"""FastAPI app: JSON API under /api, the static frontend at /.

Run:  uvicorn deedlens.api.main:app            (models load once at startup)

Every error is returned as {"error": "..."} with a proper status code. The agent call blocks
for several seconds, so it runs in a worker thread and the server stays responsive.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from deedlens.api.schemas import AskRequest, ChainEntry, Health, Property, ReviewRequest
from deedlens.config import Settings, get_settings
from deedlens.store import records

log = logging.getLogger("deedlens.api")
PAGE_DPI = 120


def web_dir() -> Path:
    """web/ next to the package in a checkout, else ./web (Docker), or WEB_DIR if set."""
    if os.environ.get("WEB_DIR"):
        return Path(os.environ["WEB_DIR"])
    repo = Path(__file__).resolve().parents[2] / "web"
    return repo if repo.is_dir() else Path.cwd() / "web"


def _default_runtime(settings: Settings) -> Any:
    from deedlens.agent.runtime import AgentRuntime

    return AgentRuntime(settings)


def create_app(settings: Settings | None = None,
               runtime_factory: Callable[[Settings], Any] = _default_runtime) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = await run_in_threadpool(runtime_factory, settings)  # load models once
        try:
            yield
        finally:
            close = getattr(app.state.runtime, "close", None)
            if close:
                close()

    app = FastAPI(title="DeedLens", version="0.1.0", lifespan=lifespan,
                  description="Questions about land records, answered with citations to the scanned page.")
    app.state.settings = settings

    # ---------------------------------------------------------------- errors as {"error": ...}

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_req: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_req: Request, exc: RequestValidationError) -> JSONResponse:
        parts = [f"{'.'.join(str(x) for x in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors()]
        return JSONResponse({"error": "; ".join(parts)}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(_req: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse({"error": f"Internal error: {type(exc).__name__}"}, status_code=500)

    # ---------------------------------------------------------------- helpers

    @contextmanager
    def db() -> Iterator[sqlite3.Connection | None]:
        """Read-only records connection, or None before the first ingestion."""
        if not settings.db_path.exists():
            yield None
            return
        conn = records.connect(settings.db_path, read_only=True)
        try:
            yield conn
        finally:
            conn.close()

    def known_parcel(conn: sqlite3.Connection | None, parcel: str) -> None:
        if conn is None or parcel not in records.all_parcels(conn):
            raise HTTPException(404, f"Unknown parcel: {parcel}")

    def runtime() -> Any:
        return app.state.runtime

    async def call_agent(fn: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
        from deedlens.agent.graph import ReviewNotPending
        from deedlens.agent.runtime import GeminiUnavailable

        try:
            return await run_in_threadpool(fn, *args)
        except GeminiUnavailable as e:
            raise HTTPException(503, str(e)) from e
        except ReviewNotPending as e:
            raise HTTPException(404, f"No review is waiting for thread {e}") from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except HTTPException:
            raise
        except Exception as e:  # Gemini/network failures: tell the user, keep the server up
            log.exception("agent call failed")
            raise HTTPException(502, f"The answer service failed: {type(e).__name__}: {str(e)[:200]}") from e

    # ---------------------------------------------------------------- API routes

    @app.get("/api/health", response_model=Health)
    def health() -> dict[str, Any]:
        with db() as conn:
            documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] if conn else 0
        chunks = 0
        if settings.chunks_file.exists():
            with settings.chunks_file.open() as f:
                chunks = sum(1 for line in f if line.strip())
        return {"status": "ok", "documents": documents, "chunks": chunks,
                "gemini": bool(getattr(runtime(), "gemini", False))}

    @app.get("/api/properties", response_model=list[Property])
    def properties() -> list[dict[str, Any]]:
        with db() as conn:
            if conn is None:
                return []
            out = [records.property_summary(conn, p) for p in records.all_parcels(conn)]
        return sorted(out, key=lambda p: (-p["documents"], p["parcel"]))

    @app.get("/api/properties/{parcel}/chain", response_model=list[ChainEntry])
    def chain(parcel: str) -> list[dict[str, Any]]:
        with db() as conn:
            known_parcel(conn, parcel)
            return records.chain_entries(conn, parcel)

    @app.post("/api/ask")
    async def ask(body: AskRequest) -> dict[str, Any]:
        with db() as conn:
            known_parcel(conn, body.parcel)
        return await call_agent(runtime().ask, body.parcel, body.question)

    @app.post("/api/review/{thread_id}")
    async def review(thread_id: str, body: ReviewRequest) -> dict[str, Any]:
        if body.action == "edit" and not body.text:
            raise HTTPException(422, "text: an edited answer is required for action 'edit'")
        return await call_agent(runtime().review, thread_id, body.action, body.text)

    @app.get("/api/documents/{doc_id}/pages/{page}.png")
    async def page_image(doc_id: str, page: int) -> FileResponse:
        with db() as conn:
            row = conn.execute("SELECT file_name, num_pages FROM documents WHERE doc_id = ?",
                               (doc_id,)).fetchone() if conn else None
        if row is None:  # doc_id comes from the database, never straight from the URL into a path
            raise HTTPException(404, f"Unknown document: {doc_id}")
        if not 1 <= page <= row["num_pages"]:
            raise HTTPException(404, f"{doc_id} has pages 1-{row['num_pages']}, not {page}")
        cached = settings.page_cache_dir / doc_id / f"{page}.png"
        if not cached.exists():
            pdf = settings.raw_dir / row["file_name"]
            if not pdf.exists():
                raise HTTPException(404, f"The PDF for {doc_id} is missing from {settings.raw_dir}")
            await run_in_threadpool(_render_to_cache, pdf, page, cached)
        return FileResponse(cached, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    # ---------------------------------------------------------------- frontend

    static = web_dir()
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(static / "index.html")

    return app


def _render_to_cache(pdf: Path, page: int, out: Path) -> None:
    from deedlens.ingest.render import render_page

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.png")
    render_page(pdf, page, dpi=PAGE_DPI).save(tmp, format="PNG", optimize=True)
    os.replace(tmp, out)


app = create_app()
