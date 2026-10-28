"""Central logging setup for the AskAI service.

Every module logs through ``get_logger`` so the whole request path (user input,
Milvus retrieval, LLM calls, web search) shows up in one consistent format.
Set ``ASKAI_LOG_LEVEL=DEBUG`` to see full prompts and raw model output.
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def setup_logging() -> None:
    """Configure the ``askai`` logger once per process."""
    global _configured
    if _configured:
        return

    # Windows consoles may default to GBK; force UTF-8 so Chinese logs stay readable.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - non-reconfigurable stream
        pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    logger = logging.getLogger("askai")
    logger.setLevel(os.getenv("ASKAI_LOG_LEVEL", "INFO").upper())
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"askai.{name}")


def preview(text: object, limit: int = 160) -> str:
    """Single-line, length-limited rendering of text for log lines."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "..."
