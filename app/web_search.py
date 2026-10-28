"""Web search tool used when Milvus notes are not enough.

Uses direct HTTP requests to DuckDuckGo's lite endpoint (more reliable than
the ddgs library backends from networks where Yahoo routing is blocked).
Falls back to the ddgs library if the direct request fails.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

from app.logging_conf import get_logger, preview

log = get_logger("web_search")


@dataclass
class WebHit:
    title: str
    url: str
    snippet: str
    source: str = "web"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.snippet,
            "source": self.source,
        }


_LITE_URL = "https://lite.duckduckgo.com/lite/"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def search_web(query: str, max_results: int = 5) -> list[WebHit]:
    """Search the web via DuckDuckGo lite. Returns up to ``max_results`` hits."""
    query = (query or "").strip()
    if not query:
        return []
    log.info("web search | query=%s | max_results=%d", preview(query, 120), max_results)

    try:
        hits = _search_lite(query, max_results)
        if hits:
            log.info("web search done (lite) | %d hits | %s", len(hits), preview(hits[0].url, 100))
            return hits
        log.warning("web search (lite) returned no hits, falling back to ddgs")
    except Exception as exc:
        log.warning(
            "web search (lite) failed | %s: %s",
            type(exc).__name__,
            preview(exc, 200),
        )

    # Fallback: try the ddgs library with multiple backends.
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore

    last_error: Exception | None = None
    for backend in ("lite", "duckduckgo", "html"):
        try:
            started = time.perf_counter()
            with DDGS(timeout=15) as ddgs:
                rows = list(
                    ddgs.text(query, region="wt-wt", max_results=max_results, backend=backend)
                )
            log.info(
                "web search (ddgs/%s) | %.2fs | %d rows",
                backend,
                time.perf_counter() - started,
                len(rows or []),
            )
            if rows:
                return [
                    WebHit(
                        title=str(row.get("title") or ""),
                        url=str(row.get("href") or row.get("url") or ""),
                        snippet=str(row.get("body") or row.get("snippet") or ""),
                    )
                    for row in rows
                ]
        except Exception as exc:
            last_error = exc
            log.warning(
                "web search backend %s failed | %s: %s",
                backend,
                type(exc).__name__,
                preview(exc, 150),
            )
            continue
    if last_error is not None:
        raise last_error
    return []


def _search_lite(query: str, max_results: int) -> list[WebHit]:
    """POST to DuckDuckGo lite and parse the resulting HTML."""
    import requests

    resp = requests.post(
        _LITE_URL,
        data={"q": query, "kl": "wt-wt"},
        headers=_HEADERS,
        timeout=20,
    )
    resp.raise_for_status()
    return _parse_lite_html(resp.text, max_results)


def _parse_lite_html(html: str, max_results: int) -> list[WebHit]:
    """Extract search results from DuckDuckGo lite HTML.

    The lite page embeds results inside ``<a class="result-link">`` tags whose
    ``href`` is a DDG redirect URL; the real URL lives in the ``uddg`` query
    param. Snippets sit in the sibling ``<td class="result-snippet">``.
    """
    try:
        from lxml import html as lhtml
    except ImportError:
        lhtml = None  # type: ignore[assignment]

    hits: list[WebHit] = []
    if lhtml is not None:
        tree = lhtml.fromstring(html)
        links = tree.xpath('//a[contains(@class, "result-link")]')
        for a in links:
            if len(hits) >= max_results:
                break
            title = (a.text_content() or "").strip()
            href = a.get("href") or ""
            url = _extract_real_url(href)
            snippet = ""
            # The snippet is in a sibling <td class="result-snippet">
            td = a.getparent()
            if td is not None:
                parent_tr = td.getparent()
                if parent_tr is not None:
                    snippets = parent_tr.xpath('.//td[contains(@class, "result-snippet")]')
                    if snippets:
                        snippet = (snippets[0].text_content() or "").strip()
            if title or url:
                hits.append(WebHit(title=title, url=url, snippet=snippet))
        if hits:
            return hits

    # Fallback: regex-based extraction for environments without lxml.
    return _parse_lite_regex(html, max_results)


def _parse_lite_regex(html: str, max_results: int) -> list[WebHit]:
    """Regex fallback parser for DuckDuckGo lite HTML."""
    hits: list[WebHit] = []
    # Match result links: <a ... class="result-link" href="REDIRECT">TITLE</a>
    pattern = re.compile(
        r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    for m in pattern.finditer(html):
        if len(hits) >= max_results:
            break
        href = m.group(1)
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        url = _extract_real_url(href)
        hits.append(WebHit(title=title, url=url, snippet=""))
    return hits


def _extract_real_url(href: str) -> str:
    """Resolve a DuckDuckGo redirect URL to the real target URL."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    # DDG redirect: //duckduckgo.com/l/?uddg=ENCODED_URL&rut=...
    if "duckduckgo.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg") or qs.get("url")
        if uddg:
            return unquote(uddg[0])
    return href
