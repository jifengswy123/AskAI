"""Thin OpenAI-compatible client for the local Ollama chat model."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator

from openai import OpenAI

from app.config import API_KEY, BASE_URL, LLM_TEMPERATURE, MODEL
from app.logging_conf import get_logger, preview

log = get_logger("llm")

_client: OpenAI | None = None


#创建client用于请求ollama中的模型
def client() -> OpenAI:
    global _client
    if _client is None:
        log.info("正在创建OpenAI client: model=%s base_url=%s", MODEL, BASE_URL)
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client

#非流式输出，用于Agent内部
def chat(messages: list[dict[str, str]], *, temperature: float | None = None) -> str:
    temp = LLM_TEMPERATURE if temperature is None else temperature
    log.debug(
        "大模型非流式输出请求 | model=%s temp=%s messages=%d | last_user=%s",
        MODEL,
        temp,
        len(messages),
        preview(_last_user(messages), 200),
    )
    started = time.perf_counter()
    #调用大模型
    try:
        resp = client().chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temp,
        )
    except Exception:
        log.exception("LLM request failed after %.2fs", time.perf_counter() - started)
        raise

    content = (resp.choices[0].message.content or "").strip()
    log.info(
        "LLM reply | %.2fs | %d chars | %s",
        time.perf_counter() - started,
        len(content),
        preview(content, 200),
    )
    return content

#流式输出，用户前端页面展示
def stream_chat(
    messages: list[dict[str, str]], *, temperature: float | None = None
) -> Iterator[str]:
    temp = LLM_TEMPERATURE if temperature is None else temperature
    log.info(
        "大模型流式输出请求 | model=%s temp=%s messages=%d | last_user=%s",
        MODEL,
        temp,
        len(messages),
        preview(_last_user(messages), 200),
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    total = 0
    try:
        stream = client().chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temp,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta is not None else None
            if not text:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter() - started
                log.info("LLM first token | %.2fs", first_token_at)
            total += len(text)
            yield text
    except Exception:
        log.exception("LLM stream failed after %.2fs", time.perf_counter() - started)
        raise
    log.info(
        "LLM stream done | %.2fs | ttft=%.2fs | %d chars",
        time.perf_counter() - started,
        first_token_at or 0.0,
        total,
    )


def chat_json(
    messages: list[dict[str, str]],
    *,
    fallback: dict[str, Any] | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Ask the small local model for JSON and recover if it wraps it in markdown."""
    temp = LLM_TEMPERATURE if temperature is None else temperature
    log.debug(
        "LLM json request | model=%s temp=%s messages=%d | last_user=%s",
        MODEL,
        temp,
        len(messages),
        preview(_last_user(messages), 200),
    )
    started = time.perf_counter()
    try:
        resp = client().chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temp,
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        log.warning(
            "LLM json mode failed (%s: %s), retrying as plain chat",
            type(exc).__name__,
            preview(exc, 200),
        )
        text = chat(messages, temperature=temperature)

    elapsed = time.perf_counter() - started
    parsed = extract_json(text)
    if parsed is None:
        log.warning(
            "LLM json unparsable | %.2fs | raw=%s | using fallback",
            elapsed,
            preview(text, 300),
        )
        return fallback or {}
    log.info(
        "LLM json reply | %.2fs | keys=%s | %s",
        elapsed,
        list(parsed.keys()),
        preview(text, 200),
    )
    return parsed


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _last_user(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""
