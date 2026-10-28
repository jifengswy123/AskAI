from collections.abc import Callable, Iterator
from typing import Any

from app.config import AGENT_TOP_K, MAX_WEB_ROUNDS
from app.llm import chat, chat_json, stream_chat
from app.logging_conf import get_logger, preview
from app.milvus_store import Hit, search_many
from app.web_search import WebHit, search_web

log = get_logger("compare")

TraceCallback = Callable[[dict[str, Any]], None]
TokenCallback = Callable[[str, str], None]


def compare_terms(
    term_a: str,
    term_b: str,
    on_event: TraceCallback | None = None,
    on_token: TokenCallback | None = None,
) -> dict:
    a = (term_a or "").strip()
    b = (term_b or "").strip()
    log.info("开始比较 | 关键字A=%s | 关键字B=%s", preview(a), preview(b))
    if not a or not b:
        raise ValueError("两个专有名词都不能为空")

    events: list[dict[str, Any]] = []

    def emit(event: dict[str, Any]) -> None:
        events.append(event)
        if on_event:
            on_event(event)

    plan = _plan(a, b)
    log.info(
        "计划已生成 | goal=%s | steps=%s",
        preview(plan.get("goal")),
        [(s.get("id"), s.get("action"), preview(s.get("query"))) for s in plan.get("steps") or []],
    )
    emit({"type": "plan", "title": "Planner 生成执行计划", "plan": plan})

    milvus_hits: list[Hit] = []
    web_hits: list[WebHit] = []

    for step in plan.get("steps") or []:
        action = str(step.get("action") or "")
        query = str(step.get("query") or "")
        log.info("步骤 %s | 动作=%s | query=%s", step.get("id"), action, preview(query))
        thought = _think(a, b, step, milvus_hits, web_hits)
        emit({"type": "thought", "title": "ReAct Thought", "content": thought})

        if action == "search_milvus":
            emit({"type": "action", "title": "Action: search_milvus", "query": query})
            found = search_many([query], top_k=AGENT_TOP_K)
            milvus_hits = _merge_hits(milvus_hits, found)
            obs = _milvus_observation(found)
            emit({"type": "observation", "title": "Observation", "content": obs, "sources": [h.to_dict() for h in found]})
        elif action == "evaluate":
            emit({"type": "action", "title": "Action: evaluate_sufficiency", "query": ""})
            judge = _judge(a, b, milvus_hits, web_hits)
            emit({"type": "observation", "title": "Observation", "content": judge.get("reason", ""), "enough": judge.get("enough")})
        elif action == "compare":
            continue
        else:
            emit({"type": "observation", "title": "Observation", "content": f"跳过未知步骤 {action}"})

    judge = _judge(a, b, milvus_hits, web_hits)
    log.info(
        "judge | enough=%s | reason=%s",
        bool(judge.get("enough")),
        preview(judge.get("reason")),
    )
    emit(
        {
            "type": "judge",
            "title": "知识是否足够",
            "enough": bool(judge.get("enough")),
            "content": str(judge.get("reason") or ""),
        }
    )

    round_i = 0
    web_failed = False
    while not judge.get("enough") and round_i < MAX_WEB_ROUNDS and not web_failed:
        round_i += 1
        query = _safe_web_query(judge.get("next_query"), a, b, round_i)
        log.info("web round %d/%d | query=%s", round_i, MAX_WEB_ROUNDS, preview(query))
        thought = (
            f"Milvus 中的笔记还不够。第 {round_i} 轮调用网络搜索 Agent，查询：{query}"
        )
        emit({"type": "thought", "title": "ReAct Thought", "content": thought})
        emit({"type": "action", "title": "Action: web_search", "query": query})
        try:
            found_web = search_web(query)
        except Exception as exc:  # network / engine timeout must not abort compare
            web_failed = True
            log.warning(
                "web search failed, continuing with notes only | %s: %s",
                type(exc).__name__,
                preview(exc),
            )
            emit(
                {
                    "type": "observation",
                    "title": "Observation",
                    "content": f"网页搜索失败，改为仅用现有笔记继续对比。原因：{exc}",
                    "sources": [],
                }
            )
            break
        if not found_web:
            web_failed = True
            log.warning("web search returned no results, continuing with notes only")
            emit(
                {
                    "type": "observation",
                    "title": "Observation",
                    "content": "网页搜索没有结果，改为仅用现有笔记继续对比。",
                    "sources": [],
                }
            )
            break
        log.info("web round %d | %d hits", round_i, len(found_web))
        web_hits.extend(found_web)
        obs = _web_observation(found_web)
        emit(
            {
                "type": "observation",
                "title": "Observation",
                "content": obs,
                "sources": [h.to_dict() for h in found_web],
            }
        )
        judge = _judge(a, b, milvus_hits, web_hits)
        log.info(
            "judge after web round %d | enough=%s | reason=%s",
            round_i,
            bool(judge.get("enough")),
            preview(judge.get("reason")),
        )
        emit(
            {
                "type": "judge",
                "title": "知识是否足够",
                "enough": bool(judge.get("enough")),
                "content": str(judge.get("reason") or ""),
            }
        )

    emit({"type": "thought", "title": "ReAct Thought", "content": "知识已收集完毕，生成两个专有名词的对比结论。"})
    emit({"type": "action", "title": "Action: compare", "query": f"{a} vs {b}"})

    parts: list[str] = []
    for _section, text in _synthesize_stream(a, b, milvus_hits, web_hits):
        parts.append(text)
        if on_token:
            on_token("comparison", text)
    comparison = "".join(parts).strip() or _joined(milvus_hits, web_hits)[:1200]
    if not comparison:
        comparison = "资料不足，无法完成对比。"
    result = {"comparison": comparison, **_summarize(a, b, milvus_hits, web_hits)}

    log.info(
        "compare done | milvus=%d web=%d enough=%s | comparison=%s",
        len(milvus_hits),
        len(web_hits),
        bool(judge.get("enough")),
        preview(result.get("comparison")),
    )
    emit({"type": "answer", "title": "最终对比", "content": result.get("comparison", "")})

    return {
        "term_a": a,
        "term_b": b,
        "plan": plan,
        "comparison": result.get("comparison", ""),
        "summary_a": result.get("summary_a", ""),
        "summary_b": result.get("summary_b", ""),
        "enough": bool(judge.get("enough")),
        "milvus_sources": [h.to_dict() for h in milvus_hits],
        "web_sources": [h.to_dict() for h in web_hits],
        "trace": events,
    }


def _plan(a: str, b: str) -> dict[str, Any]:
    data = chat_json(
        [
            {
                "role": "system",
                "content": (
                    "你是检索规划器 Planner。为比较两个专有名词制定简短计划。"
                    "只输出 JSON，步骤要可执行。action 只能是 search_milvus、evaluate、compare。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"比较「{a}」和「{b}」。\n"
                    "输出 JSON：{\"goal\":\"...\",\"steps\":["
                    "{\"id\":1,\"action\":\"search_milvus\",\"query\":\"...\",\"why\":\"...\"},"
                    "{\"id\":2,\"action\":\"search_milvus\",\"query\":\"...\",\"why\":\"...\"},"
                    "{\"id\":3,\"action\":\"evaluate\",\"query\":\"\",\"why\":\"判断笔记是否足够\"},"
                    "{\"id\":4,\"action\":\"compare\",\"query\":\"\",\"why\":\"生成区别\"}"
                    "]}"
                ),
            },
        ],
        fallback=_default_plan(a, b),
    )
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return _default_plan(a, b)
    cleaned = []
    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "search_milvus")
        if action not in {"search_milvus", "evaluate", "compare", "web_search"}:
            action = "search_milvus"
        query = str(step.get("query") or "")
        if action == "search_milvus" and not query:
            query = a if i % 2 else b
        cleaned.append(
            {
                "id": int(step.get("id") or i),
                "action": action,
                "query": query,
                "why": str(step.get("why") or ""),
            }
        )
    if not any(s["action"] == "search_milvus" for s in cleaned):
        return _default_plan(a, b)
    data["steps"] = cleaned
    data.setdefault("goal", f"比较 {a} 与 {b}")
    return data


def _default_plan(a: str, b: str) -> dict[str, Any]:
    return {
        "goal": f"弄清 {a} 与 {b} 的含义并对比区别",
        "steps": [
            {"id": 1, "action": "search_milvus", "query": f"{a} 是什么 定义 作用", "why": f"从笔记检索 {a}"},
            {"id": 2, "action": "search_milvus", "query": f"{b} 是什么 定义 作用", "why": f"从笔记检索 {b}"},
            {"id": 3, "action": "search_milvus", "query": f"{a} {b} 区别 对比", "why": "检索直接对比材料"},
            {"id": 4, "action": "evaluate", "query": "", "why": "判断知识是否足够"},
            {"id": 5, "action": "compare", "query": "", "why": "生成区别说明"},
        ],
    }


def _think(a: str, b: str, step: dict[str, Any], milvus_hits: list[Hit], web_hits: list[WebHit]) -> str:
    prompt = (
        f"当前要比较「{a}」和「{b}」。\n"
        f"下一步: action={step.get('action')} query={step.get('query')} why={step.get('why')}\n"
        f"已有笔记 {len(milvus_hits)} 条，网页 {len(web_hits)} 条。\n"
        "用一两句中文写出 ReAct 的 Thought，不要输出 JSON。"
    )
    thought = chat(
        [
            {"role": "system", "content": "你是检索 Agent，只写简短思考。"},
            {"role": "user", "content": prompt},
        ]
    )
    return thought or str(step.get("why") or "执行下一步检索。")


def _judge(a: str, b: str, milvus_hits: list[Hit], web_hits: list[WebHit]) -> dict[str, Any]:
    context = _context_block(a, b, milvus_hits, web_hits)
    data = chat_json(
        [
            {
                "role": "system",
                "content": (
                    "你负责判断现有资料是否足够比较两个专有名词。"
                    "两者都有较清晰定义、并能说出至少两点区别才算 enough=true。"
                    "只输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    "输出 JSON：{\"enough\": false, \"reason\": \"...\", "
                    "\"next_query\": \"若不够，下一次中文或英文网页检索词\"}"
                ),
            },
        ],
        fallback={
            "enough": bool(milvus_hits) and len(_joined(milvus_hits, web_hits)) > 200,
            "reason": "模型未能判断，按资料长度启发式决定。",
            "next_query": f"{a} vs {b}",
        },
    )
    data["enough"] = bool(data.get("enough"))
    if not data.get("next_query"):
        data["next_query"] = f"{a} versus {b} difference"
    return data


def _synthesize_stream(
    a: str, b: str, milvus_hits: list[Hit], web_hits: list[WebHit]
) -> Iterator[tuple[str, str]]:
    """Stream the comparison conclusion as plain text.

    No tag parsing here on purpose: the local 1.5b model often ignores output
    formats, and any parser could only emit text after the whole answer arrived.
    """
    context = _context_block(a, b, milvus_hits, web_hits)
    messages = [
        {
            "role": "system",
            "content": (
                "根据资料比较两个专有名词。优先使用笔记，网页仅作补充。"
                "不要编造资料中没有的细节。用中文回答。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"{context}\n\n"
                f"请直接输出对比结论：先用两句话分别说明 {a} 和 {b} 各是什么，"
                "再用 1. 2. 3. 分点说明两者的区别和各自适用场景。"
                "不要输出 JSON，不要用 markdown 符号（如 **、#），不要加标签或标题。"
            ),
        },
    ]
    for delta in stream_chat(messages):
        if delta:
            yield "compare", delta


def _summarize(a: str, b: str, milvus_hits: list[Hit], web_hits: list[WebHit]) -> dict[str, str]:
    context = _context_block(a, b, milvus_hits, web_hits)
    data = chat_json(
        [
            {
                "role": "system",
                "content": "根据资料用两三句话解释专有名词，不要编造，只输出 JSON。",
            },
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    f"输出 JSON：{{\"summary_a\": \"{a} 的简明解释\", \"summary_b\": \"{b} 的简明解释\"}}"
                ),
            },
        ],
        fallback={"summary_a": "", "summary_b": ""},
    )
    return {
        "summary_a": str(data.get("summary_a") or "").strip(),
        "summary_b": str(data.get("summary_b") or "").strip(),
    }


def _context_block(a: str, b: str, milvus_hits: list[Hit], web_hits: list[WebHit]) -> str:
    parts = [f"专有名词 A：{a}", f"专有名词 B：{b}", "", "【Milvus 笔记】"]
    if milvus_hits:
        for i, hit in enumerate(milvus_hits[:8], start=1):
            parts.append(f"{i}. {hit.title} / {hit.doc_id} #{hit.chunk_index}\n{hit.snippet()}")
    else:
        parts.append("（无）")
    parts.append("\n【网页检索】")
    if web_hits:
        for i, hit in enumerate(web_hits[:8], start=1):
            parts.append(f"{i}. {hit.title}\n{hit.url}\n{hit.snippet}")
    else:
        parts.append("（无）")
    return "\n".join(parts)


def _joined(milvus_hits: list[Hit], web_hits: list[WebHit]) -> str:
    texts = [h.snippet() for h in milvus_hits] + [h.snippet for h in web_hits]
    return "\n".join(texts)


def _merge_hits(old: list[Hit], new: list[Hit]) -> list[Hit]:
    seen = {h.id for h in old}
    merged = list(old)
    for hit in new:
        if hit.id in seen:
            continue
        seen.add(hit.id)
        merged.append(hit)
    merged.sort(key=lambda h: h.distance)
    return merged


def _milvus_observation(hits: list[Hit]) -> str:
    if not hits:
        return "Milvus 没有返回相关段落。"
    lines = [f"检索到 {len(hits)} 条笔记："]
    for hit in hits:
        lines.append(f"- {hit.title} (doc={hit.doc_id}, chunk={hit.chunk_index}, L2={hit.distance:.4f})")
        lines.append(f"  {hit.snippet(180)}")
    return "\n".join(lines)


def _safe_web_query(raw: object, a: str, b: str, round_i: int) -> str:
    query = str(raw or "").strip()
    poisoned = ("next_query", "若不够", "输出 JSON", "enough", "网页检索词", "只输出")
    if (not query) or len(query) > 80 or any(token in query for token in poisoned):
        fallbacks = [f"{a} vs {b}", f"what is {a}", f"what is {b}"]
        return fallbacks[min(max(round_i, 1) - 1, len(fallbacks) - 1)]
    return query


def _web_observation(hits: list[WebHit]) -> str:
    if not hits:
        return "网页搜索没有返回结果。"
    lines = [f"检索到 {len(hits)} 条网页："]
    for hit in hits:
        lines.append(f"- {hit.title} | {hit.url}")
        lines.append(f"  {hit.snippet[:180]}")
    return "\n".join(lines)
