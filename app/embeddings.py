from __future__ import annotations

import time

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL
from app.llm import client
from app.logging_conf import get_logger, preview

log = get_logger("embeddings")

#使用embedding模型对输入的text进行向量化
def embed_query(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        raise ValueError("embedding请求为空，请重新输入")

    log.info("开始进行向量化 | model=%s | %s", EMBEDDING_MODEL, preview(text, 120))
    started = time.perf_counter()
    try:
        resp = client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    except Exception:
        log.exception("embedding请求失败%.2fs", time.perf_counter() - started)
        raise

    #取出embedding返回中的向量结果
    vector = list(resp.data[0].embedding)
    log.info(
        "embedding已完成 | embedding花费 %.2fs | 向量维度dim=%d",
        time.perf_counter() - started,
        len(vector),
    )
    if len(vector) != EMBEDDING_DIM:
        log.error(
            "embedding维度不匹配: embedding结果为 %d, 集合期望 %d",
            len(vector),
            EMBEDDING_DIM,
        )
        raise RuntimeError(
            f"embedding 维度不匹配: 实际为 {len(vector)}, 集合期望 {EMBEDDING_DIM}; "
            f"请检查embedding模型是否与录入系统使用的模型"
        )
    return vector
