const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value == null ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function plain(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(plain).filter(Boolean).join("\n");
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([key, val]) => `${key}: ${plain(val)}`)
      .join("\n");
  }
  return String(value);
}

function renderText(el, value) {
  el.innerHTML = esc(value).replaceAll("\n", "<br>");
}

function setStatus(id, value) {
  const el = $(id);
  el.textContent = value || "";
  el.hidden = !value;
}

function webHtml(s) {
  const link = s.url
    ? `<p><a href="${esc(s.url)}" target="_blank" rel="noreferrer">${esc(s.url)}</a></p>`
    : "";
  return `<p><strong>${esc(s.title || "")}</strong></p><p>${esc(s.content || "")}</p>${link}`;
}

async function readSse(stream, handlers) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const block of parts) {
      let eventName = "message";
      const dataLines = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;
      const data = JSON.parse(dataLines.join("\n"));
      if (handlers[eventName]) handlers[eventName](data);
    }
  }
}

$("qa-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;

  const btn = e.target.querySelector("button");
  btn.disabled = true;
  $("qa-answer").hidden = true;
  $("qa-text").innerHTML = "";
  setStatus("qa-status", "正在检索，请稍候…");

  const startedAt = performance.now();
  let answer = "";

  try {
    const res = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "问答失败");
    }
    await readSse(res.body, {
      sources: () => {
        // 检索到的段落只写服务端日志（askai.milvus），前端不展示原文。
        $("qa-answer").hidden = false;
        setStatus("qa-status", "");
      },
      token: (data) => {
        answer += data.text || "";
        renderText($("qa-text"), answer);
      },
      done: (data) => {
        if (data.answer) answer = data.answer;
        renderText($("qa-text"), answer);
        $("qa-answer").hidden = false;
        setStatus("qa-status", `完成，用时 ${((performance.now() - startedAt) / 1000).toFixed(2)}s`);
      },
      error: (data) => {
        throw new Error(data.detail || "问答失败");
      },
    });
  } catch (err) {
    setStatus("qa-status", err.message || "问答失败");
  } finally {
    btn.disabled = false;
  }
});

$("cmp-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const term_a = $("term-a").value.trim();
  const term_b = $("term-b").value.trim();
  if (!term_a || !term_b) return;

  const btn = e.target.querySelector("button");
  btn.disabled = true;
  $("cmp-answer").hidden = true;
  $("sum-a-title").textContent = term_a;
  $("sum-b-title").textContent = term_b;
  ["sum-a", "sum-b", "cmp-text", "cmp-web"].forEach((id) => {
    $(id).innerHTML = "";
  });
  setStatus("cmp-status", "正在检索并生成对比，请稍候…");

  const acc = { summary_a: "", summary_b: "", comparison: "" };
  const target = { summary_a: "sum-a", summary_b: "sum-b", comparison: "cmp-text" };

  try {
    const res = await fetch("/api/compare/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ term_a, term_b }),
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "对比失败");
    }
    await readSse(res.body, {
      token: (data) => {
        const key = acc[data.section] === undefined ? "comparison" : data.section;
        acc[key] += data.text || "";
        renderText($(target[key]), acc[key]);
        $("cmp-answer").hidden = false;
        setStatus("cmp-status", "");
      },
      done: (data) => {
        $("sum-a-title").textContent = data.term_a;
        $("sum-b-title").textContent = data.term_b;
        renderText($("sum-a"), plain(data.summary_a) || "（无）");
        renderText($("sum-b"), plain(data.summary_b) || "（无）");
        renderText($("cmp-text"), plain(data.comparison));
        $("cmp-web").innerHTML =
          (data.web_sources || []).map(webHtml).join("") || "<p>未使用网页搜索，或没有结果</p>";
        $("cmp-answer").hidden = false;
        setStatus("cmp-status", "");
      },
      error: (data) => {
        throw new Error(data.detail || "对比失败");
      },
    });
  } catch (err) {
    setStatus("cmp-status", err.message || "对比失败");
  } finally {
    btn.disabled = false;
  }
});
