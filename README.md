# AskAI · Agentic RAG 学习笔记问答系统

1. **知识问答** — 用户提问后，系统从 Milvus 向量库检索知识，由大模型润色后返回到前端。
2. **概念对比** — 输入两个专有名词，Planner 生成计划，结合 ReAct + Plan-and-Execute 在 Milvus 中检索；笔记不足时由联网搜索 Agent 补齐，最终生成两者区别。

## 环境安装

| 组件 | 版本 / 说明                                                     |
|------|-------------------------------------------------------------|
| Python | 3.12                                                        |
| Ollama | 本地运行，使用ollama pull下载 `qwen2.5:1.5b` 和 `qwen3-embedding:0.6b` |
| Milvus | 本地运行                                 | 

---

## Milvus字段

- name="id", dtype=DataType.INT64, is_primary=True, auto_id=True。
- name="title", dtype=DataType.VARCHAR, max_length=200。
- name="doc_id", dtype=DataType.VARCHAR, max_length=100。
- name="content", dtype=DataType.VARCHAR, max_length=65535。
- name="chunk_index", dtype=DataType.INT32。
- name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024。

## 运行
运行run.py启动前端，
访问http://127.0.0.1:8000/

## 工作流程

### 功能一：知识问答

```
用户问题 → embedding模型向量化 → Milvus 向量检索知识库 Top3 → LLM 选最优并润色 → 前端流式输出展示
```

### 功能二：概念对比

```
用户输入两个名词
    ↓
Planner 生成执行计划（search_milvus / evaluate / compare）
    ↓
Plan-and-Execute：按计划逐步执行，每步前生成 ReAct Thought
    ↓
search_milvus → 从笔记库检索相关段落
    ↓
evaluate → Judge 判断知识是否足够
    ↓（不足时）
Web Search Agent → DuckDuckGo 联网搜索补齐（最多 MAX_WEB_ROUNDS 轮）
    ↓
Synthesize → 生成 summary_a / summary_b / 区别说明
    ↓
前端展示对比结论
```

---
