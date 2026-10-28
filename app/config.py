from __future__ import annotations

import os

MODEL = os.getenv("ASKAI_MODEL", "qwen2.5:1.5b")
BASE_URL = os.getenv("ASKAI_BASE_URL", "http://127.0.0.1:11434/v1/")
API_KEY = os.getenv("ASKAI_API_KEY", "ollama")

# Local Ollama already has this 1024-dim model, matching the Milvus collection.
EMBEDDING_MODEL = os.getenv("ASKAI_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_DIM = int(os.getenv("ASKAI_EMBEDDING_DIM", "1024"))

MILVUS_URI = os.getenv("ASKAI_MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_TOKEN = os.getenv("ASKAI_MILVUS_TOKEN", "swy:Shiweiye1.")
MILVUS_DB = os.getenv("ASKAI_MILVUS_DB", "MyAILearning")
MILVUS_COLLECTION = os.getenv("ASKAI_MILVUS_COLLECTION", "demo")
MILVUS_METRIC = os.getenv("ASKAI_MILVUS_METRIC", "L2")

QA_TOP_K = int(os.getenv("ASKAI_QA_TOP_K", "3"))
AGENT_TOP_K = int(os.getenv("ASKAI_AGENT_TOP_K", "3"))
MAX_WEB_ROUNDS = int(os.getenv("ASKAI_MAX_WEB_ROUNDS", "3"))
CHUNK_CHARS = int(os.getenv("ASKAI_CHUNK_CHARS", "900"))
LLM_TEMPERATURE = float(os.getenv("ASKAI_LLM_TEMPERATURE", "0.2"))
