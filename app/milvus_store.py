from __future__ import annotations

import time
from dataclasses import dataclass

from pymilvus import MilvusClient

from app.config import (
    AGENT_TOP_K,
    CHUNK_CHARS,
    MILVUS_COLLECTION,
    MILVUS_DB,
    MILVUS_METRIC,
    MILVUS_TOKEN,
    MILVUS_URI,
    QA_TOP_K,
)
from app.embeddings import embed_query
from app.logging_conf import get_logger, preview

log = get_logger("milvus")

_client: MilvusClient | None = None


@dataclass
class Hit:
    id: int
    title: str
    doc_id: str
    content: str
    chunk_index: int
    distance: float
    source: str = "milvus"

    def snippet(self, limit: int | None = None) -> str:
        limit = limit or CHUNK_CHARS
        text = (self.content or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "…"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "doc_id": self.doc_id,
            "content": self.snippet(),
            "chunk_index": self.chunk_index,
            "distance": round(float(self.distance), 4),
            "source": self.source,
        }


def connect() -> MilvusClient:
    global _client
    if _client is None:
        log.info(
            "connecting Milvus | uri=%s db=%s collection=%s",
            MILVUS_URI,
            MILVUS_DB,
            MILVUS_COLLECTION,
        )
        started = time.perf_counter()
        try:
            _client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN, db_name=MILVUS_DB)
            if not _client.has_collection(MILVUS_COLLECTION):
                raise RuntimeError(f"collection {MILVUS_COLLECTION!r} not found")
            _client.load_collection(MILVUS_COLLECTION)
        except Exception:
            log.exception("Milvus connect failed after %.2fs", time.perf_counter() - started)
            _client = None
            raise
        log.info(
            "Milvus connected and collection loaded | %.2fs",
            time.perf_counter() - started,
        )
    return _client


def search(query: str, top_k: int | None = None) -> list[Hit]:
    client = connect()
    top_k = top_k or QA_TOP_K
    log.info("milvus search | top_k=%d metric=%s query=%s", top_k, MILVUS_METRIC, preview(query, 120))

    vector = embed_query(query)
    started = time.perf_counter()
    try:
        raw = client.search(
            collection_name=MILVUS_COLLECTION,
            data=[vector],
            anns_field="vector",
            limit=top_k,
            output_fields=["title", "doc_id", "content", "chunk_index"],
            search_params={"metric_type": MILVUS_METRIC, "params": {"nprobe": 16}},
        )
    except Exception:
        log.exception("milvus search failed after %.2fs", time.perf_counter() - started)
        raise

    hits: list[Hit] = []
    for item in raw[0]:
        entity = item.get("entity") or {}
        hits.append(
            Hit(
                id=int(item.get("id") or 0),
                title=str(entity.get("title") or ""),
                doc_id=str(entity.get("doc_id") or ""),
                content=str(entity.get("content") or ""),
                chunk_index=int(entity.get("chunk_index") or 0),
                distance=float(item.get("distance") or 0.0),
            )
        )
    log.info(
        "milvus search done | %.3fs | %d hits | %s",
        time.perf_counter() - started,
        len(hits),
        "; ".join(f"{h.title}#{h.chunk_index} d={h.distance:.3f}" for h in hits) or "none",
    )
    # Retrieved paragraphs are logged here instead of being rendered in the UI.
    for i, hit in enumerate(hits, start=1):
        log.info(
            "  hit #%d | %s | doc=%s | 段落 %d | L2 %.4f | 原文: %s",
            i,
            hit.title,
            hit.doc_id,
            hit.chunk_index,
            hit.distance,
            hit.content,
        )
    return hits


def search_many(queries: list[str], top_k: int | None = None) -> list[Hit]:
    seen: set[int] = set()
    merged: list[Hit] = []
    k = top_k or AGENT_TOP_K
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        for hit in search(q, top_k=k):
            if hit.id in seen:
                continue
            seen.add(hit.id)
            merged.append(hit)
    merged.sort(key=lambda h: h.distance)
    log.info("milvus search_many | queries=%d | unique hits=%d", len(queries), len(merged))
    return merged
