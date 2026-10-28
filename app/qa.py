"""Feature 1: retrieve top-3 notes, let the LLM pick, polish, and summarize."""

from __future__ import annotations

import re
import time
from typing import Iterator

from app.config import QA_TOP_K
from app.llm import chat_json, stream_chat
from app.logging_conf import get_logger, preview
from app.milvus_store import Hit, search

log = get_logger("qa")

# The streamed answer starts with the rank of the note the model considers best,
# so the UI can still highlight "最优" while text arrives token by token.
_SOURCE_TAG = re.compile(r"^\s*<source>\s*(\d+)\s*</source>\s*", re.IGNORECASE)
_SOURCE_TAG_OPEN = "<source>"

NO_HIT_NOTICE = "知识库中没有检索到相关内容，请先把相关知识写入 Milvus。"


def answer_question(question: str) -> dict:
    question = (question or "").strip()
    log.info("QA question | %s", preview(question, 200))
    if not question:
        raise ValueError("问题不能为空")

    hits = search(question, top_k=QA_TOP_K)
    if not hits:
        log.warning("QA found no hits, returning empty-answer notice")
        return {
            "question": question,
            "answer": NO_HIT_NOTICE,
            "best_rank": None,
            "sources": [],
        }

    payload = chat_json(
        [
            {
                "role": "system",
                "content": (
                    "你是 Linux 中间件与 AI 学习笔记助手。"
                    "根据检索到的资料回答用户，只能使用资料中的事实。"
                    "先选出最相关的一段（rank 从 1 开始）。"
                    "回答时把资料里的步骤、路径、命令原样整理成条目，允许改写通顺，但禁止补充资料没有的菜单、按钮或参数。"
                    "可吸收其他段落中不矛盾的补充。"
                    "资料不足时明确说明缺什么。"
                    "只输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": _build_user_prompt(question, hits),
            },
        ],
        fallback={
            "best_rank": 1,
            "answer": hits[0].snippet(),
        },
    )

    best_rank = _as_rank(payload.get("best_rank"), len(hits))
    answer = str(payload.get("answer") or "").strip() or hits[best_rank - 1].snippet()
    log.info(
        "QA answer | best_rank=%s | %d chars | %s",
        best_rank,
        len(answer),
        preview(answer, 200),
    )
    return {
        "question": question,
        "answer": answer,
        "best_rank": best_rank,
        "sources": [h.to_dict() for h in hits],
    }


def answer_question_stream(question: str) -> Iterator[dict]:
    """Stream the answer: sources first, then text deltas, then the final payload."""
    question = (question or "").strip()
    log.info("QA stream question | %s", preview(question, 200))
    if not question:
        raise ValueError("问题不能为空")

    started = time.perf_counter()
    hits = search(question, top_k=QA_TOP_K)
    log.info("QA stream retrieved | %d hits | %.2fs", len(hits), time.perf_counter() - started)
    yield {"type": "sources", "question": question, "sources": [h.to_dict() for h in hits]}

    if not hits:
        log.warning("QA stream found no hits")
        yield {"type": "token", "text": NO_HIT_NOTICE}
        yield {
            "type": "done",
            "question": question,
            "answer": NO_HIT_NOTICE,
            "best_rank": None,
            "sources": [],
        }
        return

    messages = [
        {
            "role": "system",
            "content": (
                "你是 Linux 中间件与 AI 学习笔记助手。"
                "严格根据检索到的资料回答，只能使用资料中的事实。"
                "资料里的步骤、路径、命令原样保留，禁止补充资料没有的菜单、按钮或参数。"
                "资料不足时直接说明缺什么。"
            ),
        },
        {"role": "user", "content": _build_stream_prompt(question, hits)},
    ]

    best_rank: int | None = None
    decided = False
    buffered = ""
    emitted = ""

    for delta in stream_chat(messages):
        if decided:
            emitted += delta
            yield {"type": "token", "text": delta}
            continue

        buffered += delta
        match = _SOURCE_TAG.match(buffered)
        if match:
            best_rank = _as_rank(match.group(1), len(hits))
            rest = buffered[match.end():]
            buffered = ""
            decided = True
            log.info("QA stream picked best_rank=%s", best_rank)
            if rest:
                emitted += rest
                yield {"type": "token", "text": rest}
            continue

        if not _could_be_source_tag(buffered):
            decided = True
            emitted += buffered
            yield {"type": "token", "text": buffered}
            buffered = ""

    if buffered:
        emitted += buffered
        yield {"type": "token", "text": buffered}

    best_rank = best_rank or 1
    answer = emitted.strip()
    if not answer:
        answer = hits[best_rank - 1].snippet()
        log.warning("QA stream produced no text, falling back to note #%d", best_rank)
        yield {"type": "token", "text": answer}

    log.info(
        "QA stream answer | best_rank=%s | %d chars | %.2fs | %s",
        best_rank,
        len(answer),
        time.perf_counter() - started,
        preview(answer, 200),
    )
    yield {
        "type": "done",
        "question": question,
        "answer": answer,
        "best_rank": best_rank,
        "sources": [h.to_dict() for h in hits],
    }


def _format_hits(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[资料{i}]\n标题: {hit.title}\n文档编号: {hit.doc_id}\n"
            f"段落: {hit.chunk_index}\n距离: {hit.distance:.4f}\n内容:\n{hit.snippet()}"
        )
    return "\n\n".join(blocks)


def _build_user_prompt(question: str, hits: list[Hit]) -> str:
    return (
        f"用户问题：{question}\n\n"
        + _format_hits(hits)
        + "\n\n请输出 JSON：{\"best_rank\": 1, \"answer\": \"面向用户的中文回答\"}"
    )


def _build_stream_prompt(question: str, hits: list[Hit]) -> str:
    return (
        f"用户问题：{question}\n\n"
        + _format_hits(hits)
        + "\n\n输出要求：\n"
        f"1. 第一行先输出 <source>资料序号</source>，序号是能回答该问题的最佳资料编号（1 到 {len(hits)}）；\n"
        "2. 紧接着用中文输出答案正文：先给结论，再给命令、路径或步骤；\n"
        "3. 不要输出 JSON，不要解释编号选择过程。"
    )


def _could_be_source_tag(text: str) -> bool:
    """True while ``text`` may still grow into a leading ``<source>N</source>`` tag."""
    head = text.lstrip().lower()
    if not head:
        return True
    if len(head) <= len(_SOURCE_TAG_OPEN):
        return _SOURCE_TAG_OPEN.startswith(head)
    if head.startswith(_SOURCE_TAG_OPEN):
        return "</source>" not in head and len(head) < 48
    return False


def _as_rank(value: object, n: int) -> int:
    try:
        rank = int(value)
    except (TypeError, ValueError):
        rank = 1
    return min(max(rank, 1), n)
