from __future__ import annotations

import json
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.compare_agent import compare_terms
from app.config import EMBEDDING_MODEL, MILVUS_COLLECTION, MILVUS_DB, MODEL
from app.logging_conf import get_logger, preview
from app.milvus_store import connect
from app.qa import answer_question, answer_question_stream

log = get_logger("api")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# CORS: allow the separated Vue frontend (dev server + built static) to call the API.
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("service starting | model=%s | embedding=%s", MODEL, EMBEDDING_MODEL)
    connect()
    log.info("service ready | listening on http://127.0.0.1:8000")
    yield
    log.info("service shutting down")


app = FastAPI(title="AskAI Agentic RAG", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QuestionIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class CompareIn(BaseModel):
    term_a: str = Field(..., min_length=1, max_length=200)
    term_b: str = Field(..., min_length=1, max_length=200)


@app.get("/")
def index() -> FileResponse:
    # The built-in page changes often during development; never cache it.
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "milvus_db": MILVUS_DB,
        "collection": MILVUS_COLLECTION,
    }


@app.post("/api/ask")
def ask(body: QuestionIn) -> dict:
    log.info("POST /api/ask | question=%s", preview(body.question, 200))
    try:
        result = answer_question(body.question)
    except ValueError as exc:
        log.warning("POST /api/ask rejected | %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("POST /api/ask failed")
        raise HTTPException(status_code=502, detail=f"问答失败: {exc}") from exc
    log.info(
        "POST /api/ask ok | sources=%d | answer=%d chars",
        len(result.get("sources") or []),
        len(str(result.get("answer") or "")),
    )
    return result


@app.post("/api/ask/stream")
def ask_stream(body: QuestionIn) -> StreamingResponse:
    """Same answer as /api/ask, but sources arrive first and the text streams."""
    log.info("POST /api/ask/stream | question=%s", preview(body.question, 200))

    def gen():
        try:
            for event in answer_question_stream(body.question):
                data = dict(event)
                name = str(data.pop("type", "message"))
                yield _sse(name, data)
        except ValueError as exc:
            log.warning("POST /api/ask/stream rejected | %s", exc)
            yield _sse("error", {"detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            log.exception("POST /api/ask/stream failed")
            yield _sse("error", {"detail": f"问答失败: {exc}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/compare")
def compare(body: CompareIn) -> dict:
    log.info("POST /api/compare | A=%s | B=%s", preview(body.term_a, 80), preview(body.term_b, 80))
    try:
        result = compare_terms(body.term_a, body.term_b)
    except ValueError as exc:
        log.warning("POST /api/compare rejected | %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("POST /api/compare failed")
        raise HTTPException(status_code=502, detail=f"对比失败: {exc}") from exc
    log.info(
        "POST /api/compare ok | milvus=%d | web=%d | enough=%s",
        len(result.get("milvus_sources") or []),
        len(result.get("web_sources") or []),
        result.get("enough"),
    )
    return result


@app.post("/api/compare/stream")
def compare_stream(body: CompareIn) -> StreamingResponse:
    log.info(
        "POST /api/compare/stream | A=%s | B=%s",
        preview(body.term_a, 80),
        preview(body.term_b, 80),
    )

    def gen():
        q: queue.Queue = queue.Queue()

        def on_event(event: dict) -> None:
            q.put(("trace", event))

        def on_token(section: str, text: str) -> None:
            q.put(("token", {"section": section, "text": text}))

        def worker() -> None:
            try:
                result = compare_terms(
                    body.term_a, body.term_b, on_event=on_event, on_token=on_token
                )
                log.info("compare stream finished | milvus=%d | web=%d",
                         len(result.get("milvus_sources") or []),
                         len(result.get("web_sources") or []))
                q.put(("done", result))
            except Exception as exc:  # noqa: BLE001
                log.exception("compare stream failed")
                q.put(("error", {"detail": str(exc)}))
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            event, data = item
            yield _sse(event, data)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
