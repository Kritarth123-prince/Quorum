/* Quorum — front-end (no build step, no external dependencies). */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  sessionId: null,
  busy: false,
  controller: null,
  turnCounter: 0,
};

// ── Utilities ────────────────────────────────────────────────────────────────

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function safeUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : null;
  } catch { return null; }
}

function time24(date = new Date()) {
  return date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function dateTime24(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return "";
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay ? time24(d) : `${d.toLocaleDateString("en-GB", { day: "numeric", month: "short" })} ${time24(d)}`;
}

function seconds(ms) { return ms ? `${(ms / 1000).toFixed(1)} s` : ""; }

// ── Markdown (safe subset) ───────────────────────────────────────────────────

function renderInline(text, turnId) {
  // `text` is already HTML-escaped.
  const codes = [];
  text = text.replace(/`([^`]+)`/g, (_, code) => { codes.push(code); return `\u0000${codes.length - 1}\u0000`; });
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) =>
    `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`);
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  text = text.replace(/(^|[\s(])_([^_\s][^_]*)_(?=[\s.,;:)!?]|$)/g, "$1<em>$2</em>");
  text = text.replace(/\[(\d{1,3})\](?!\()/g, (_, n) =>
    `<a class="cite" href="#src-${turnId}-${n}" data-n="${n}" title="Source ${n}">${n}</a>`);
  return text.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
}

function renderMarkdown(md, turnId) {
  const lines = escapeHtml(md).replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0;
  const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
  const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());

  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      const code = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
      i++;
      out.push(`<pre><code>${code.join("\n")}</code></pre>`);
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      const level = Math.min(4, heading[1].length + 1);
      out.push(`<h${level}>${renderInline(heading[2], turnId)}</h${level}>`);
      i++; continue;
    }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { out.push("<hr>"); i++; continue; }
    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
      out.push("<table><thead><tr>" + head.map((c) => `<th>${renderInline(c, turnId)}</th>`).join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${renderInline(c, turnId)}</td>`).join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      const items = [];
      while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*([-*+]|\d+[.)])\s+/, ""));
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
          items[items.length - 1] += " " + lines[i].trim(); i++;
        }
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>` + items.map((it) => `<li>${renderInline(it, turnId)}</li>`).join("") + `</${tag}>`);
      continue;
    }
    if (/^&gt;\s?/.test(line)) {
      const quote = [];
      while (i < lines.length && /^&gt;\s?/.test(lines[i])) quote.push(lines[i++].replace(/^&gt;\s?/, ""));
      out.push(`<blockquote>${renderInline(quote.join(" "), turnId)}</blockquote>`);
      continue;
    }
    if (!line.trim()) { i++; continue; }
    const para = [line];
    i++;
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*```|\s*([-*+]|\d+[.)])\s+|&gt;)/.test(lines[i]) &&
           !(lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]))) {
      para.push(lines[i++]);
    }
    out.push(`<p>${renderInline(para.join(" "), turnId)}</p>`);
  }
  return out.join("\n");
}

// ── SSE over fetch (POST) ────────────────────────────────────────────────────

async function streamResearch(body, onEvent, signal) {
  const resp = await fetch("/research/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try { const j = await resp.json(); detail = j.detail ? JSON.stringify(j.detail) : (j.message || detail); } catch { /* ignore */ }
    throw new Error(detail);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message";
      const data = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith(":")) continue;
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (!data.length) continue;
      let parsed;
      try { parsed = JSON.parse(data.join("\n")); } catch { parsed = data.join("\n"); }
      onEvent(event, parsed);
    }
  }
}

// ── Turn rendering ───────────────────────────────────────────────────────────

function createTurn(question) {
  const turnId = ++state.turnCounter;
  const node = $("#turnTemplate").content.firstElementChild.cloneNode(true);
  node.dataset.turn = String(turnId);
  $(".question", node).textContent = question;
  $(".time", node).textContent = time24();
  $("#thread").append(node);
  node.scrollIntoView({ behavior: "smooth", block: "start" });

  const view = {
    id: turnId, node, question, answerText: "", sources: [], report: null, raf: null,
    answerEl: $(".answer", node),
    setStep(name, status) {
      const li = $(`.steps li[data-node="${name}"]`, node);
      if (!li) return;
      li.classList.remove("running", "done", "failed");
      if (status) li.classList.add(status);
    },
    renderAnswer(streaming) {
      if (this.raf) return;
      this.raf = requestAnimationFrame(() => {
        this.raf = null;
        this.answerEl.innerHTML = renderMarkdown(this.answerText, turnId) || '<p class="placeholder">Working on it…</p>';
        this.answerEl.classList.toggle("streaming", !!streaming);
        decorateCitations(this);
      });
    },
  };

  $$(".tab", node).forEach((tab) => tab.addEventListener("click", () => {
    $$(".tab", node).forEach((t) => t.classList.toggle("active", t === tab));
    $$(".tab-panel", node).forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== tab.dataset.tab));
  }));
  $(".copy-btn", node).addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(view.answerText); flash($(".copy-btn", node), "Copied"); } catch { flash($(".copy-btn", node), "Copy failed"); }
  });
  $(".export-btn", node).addEventListener("click", () => exportMarkdown(view));
  $(".stop-btn", node).addEventListener("click", () => state.controller?.abort());
  return view;
}

function flash(button, text) {
  const old = button.textContent;
  button.textContent = text;
  setTimeout(() => { button.textContent = old; }, 1400);
}

function decorateCitations(view) {
  const byId = Object.fromEntries(view.sources.map((s) => [String(s.id), s]));
  $$(".cite", view.answerEl).forEach((a) => {
    const s = byId[a.dataset.n];
    if (s) a.title = `${s.title} — ${s.domain}`;
  });
}

function addNotice(view, level, message) {
  const box = $(".notices", view.node);
  if ($$(".notice", box).some((n) => n.textContent === message)) return;
  box.append(el("div", { class: `notice ${level || "warning"}`, role: level === "error" ? "alert" : "status", text: message }));
}

function renderPlan(view, plan) {
  const box = $(".plan", view.node);
  box.innerHTML = "";
  if (plan.standalone_query && plan.standalone_query !== view.question) {
    box.append(el("div", {}, "Interpreted as: ", el("strong", { text: plan.standalone_query })));
  }
  const queries = plan.search_queries || [];
  if (queries.length) {
    box.append(el("div", {}, "Searching: ", ...queries.map((q) => el("span", { class: "q", text: q }))));
  }
  const extras = [];
  if (plan.needs_academic) extras.push("academic papers");
  if (plan.needs_crawl) extras.push("deep read");
  if (plan.fallback_used) extras.push("rule-based plan");
  if (extras.length) box.append(el("div", { class: "muted", text: `Includes: ${extras.join(", ")}` }));
  box.classList.remove("hidden");
}

function methodLabel(m) {
  return { web_search: "web", web_crawl: "page", academic: "paper", vector_rag: "memory", hybrid: "memory" }[m] || m;
}

function renderSourcesStrip(view, sources) {
  view.sources = sources;
  const strip = $(".sources-strip", view.node);
  strip.innerHTML = "";
  for (const s of sources) {
    const url = safeUrl(s.url);
    const card = el(url ? "a" : "div", { class: "src-card", href: url, target: url ? "_blank" : null, rel: url ? "noopener noreferrer" : null, id: `src-${view.id}-${s.id}` },
      el("div", { class: "src-top" }, el("span", { class: "src-num", text: s.id }), el("span", { text: s.domain }), el("span", { class: "badge", text: methodLabel(s.method) })),
      el("div", { class: "src-title", text: s.title }));
    strip.append(card);
  }
  strip.classList.toggle("hidden", !sources.length);
}

function renderReport(view, report) {
  view.report = report;
  const node = view.node;
  const sources = report.sources || report.all_citations || [];
  if (sources.length && !view.sources.length) renderSourcesStrip(view, sources);
  view.sources = sources.length ? sources : view.sources;

  view.answerText = report.answer || view.answerText;
  view.answerEl.innerHTML = renderMarkdown(view.answerText, view.id);
  view.answerEl.classList.remove("streaming");
  decorateCitations(view);
  $(".copy-btn", node).disabled = !view.answerText;
  $(".export-btn", node).disabled = false;
  $(".stop-btn", node).classList.add("hidden");

  const trace = report.agent_trace || {};
  const badge = $(".provider-badge", node);
  badge.textContent = trace.answer_provider && trace.answer_provider !== "extractive" ? `by ${trace.answer_provider}` : "extractive";
  badge.classList.remove("hidden");
  if (report.total_latency_ms) $(".latency", node).textContent = seconds(report.total_latency_ms);

  for (const w of report.warnings || []) addNotice(view, /exhausted|rejected|failed/i.test(w) ? "error" : "warning", w);
  if (report.error) addNotice(view, "error", report.error);

  // Confidence summary
  const c = report.confidence_summary || {};
  const conf = $(".confidence", node);
  conf.innerHTML = "";
  if (c.total_claims) {
    const meter = el("div", { class: "meter", role: "img", "aria-label": `Confidence: ${c.high_confidence_pct}% high, ${c.medium_confidence_pct}% medium, ${c.low_confidence_pct}% low` },
      el("span", { class: "m-high", style: `width:${c.high_confidence_pct}%` }),
      el("span", { class: "m-medium", style: `width:${c.medium_confidence_pct}%` }),
      el("span", { class: "m-low", style: `width:${c.low_confidence_pct}%` }));
    conf.append(
      el("strong", { text: `Confidence ${Math.round((c.overall_score || 0) * 100)}%` }),
      meter,
      el("span", { text: `${c.total_claims} claims checked · ${c.contradictions_found || 0} contested · ${c.hallucination_flags || 0} flagged` }));
  }

  // Findings
  const findings = $('[data-panel="findings"]', node);
  findings.innerHTML = "";
  if (report.executive_summary) findings.append(el("p", { class: "summary" }, el("strong", { text: "Summary. " }), report.executive_summary));
  for (const f of report.key_findings || []) {
    const cites = (f.citations || []).map((ct) => {
      const a = el("a", { class: "cite", href: `#src-${view.id}-${ct.id}`, title: `${ct.title} — ${ct.domain}`, text: ct.id });
      return a;
    });
    const detail = el("p");
    detail.innerHTML = renderInline(escapeHtml(f.detail || ""), view.id);
    findings.append(el("div", { class: "finding" },
      el("h4", {}, el("span", { class: `conf ${f.confidence}`, text: f.confidence }), el("span", { text: f.headline })),
      detail,
      el("div", { class: "tags" }, ...(f.tags || []).map((t) => el("span", { class: "badge", text: t })), ...cites)));
  }
  if (report.tradeoffs) {
    const t = el("div", { class: "tradeoffs markdown" });
    t.innerHTML = "<h4>Trade-offs</h4>" + renderMarkdown(report.tradeoffs, view.id);
    findings.append(t);
  }
  if (!findings.children.length) findings.append(el("p", { class: "muted", text: "No findings were produced for this question." }));

  // Sources
  const srcPanel = $('[data-panel="sources"]', node);
  srcPanel.innerHTML = "";
  for (const s of sources) {
    const url = safeUrl(s.url);
    const title = url ? el("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: s.title }) : el("span", { text: s.title });
    const meta = [s.domain, methodLabel(s.method), s.published, s.cited_in_answer ? "cited" : null].filter(Boolean).join(" · ");
    srcPanel.append(el("div", { class: "src-row" },
      el("span", { class: "src-num", text: s.id }),
      el("div", {}, title, el("div", { class: "src-meta", text: meta }), el("div", { class: "snippet", text: s.snippet || "" })),
      el("span", { class: "badge", title: "Relevance", text: (s.relevance ?? 0).toFixed(2) })));
  }
  if (!sources.length) srcPanel.append(el("p", { class: "muted", text: "No sources." }));

  // Claims
  const claimsPanel = $('[data-panel="claims"]', node);
  claimsPanel.innerHTML = "";
  for (const cl of report.verified_claims || []) {
    const refs = (cl.supporting_sources || []).map((n) => `[${n}]`).join(" ");
    const extra = [cl.assessment?.replace("_", " "), cl.contradicting_sources?.length ? `contradicted by ${cl.contradicting_sources.map((n) => `[${n}]`).join(" ")}` : null, cl.note].filter(Boolean).join(" · ");
    const body = el("div");
    body.innerHTML = renderInline(escapeHtml(`${cl.claim} ${refs}`), view.id);
    claimsPanel.append(el("div", { class: "claim" }, el("span", { class: `conf ${cl.confidence}`, text: cl.confidence }), " ", body,
      el("div", { class: "claim-meta", text: `score ${Number(cl.confidence_score || 0).toFixed(2)} · ${extra}` })));
  }
  if (!claimsPanel.children.length) claimsPanel.append(el("p", { class: "muted", text: "No claims were extracted." }));

  // Trace
  const tracePanel = $('[data-panel="trace"]', node);
  tracePanel.innerHTML = "";
  const traceInfo = { agent_trace: trace, retrieval: report.retrieval, notices: report.notices, report_id: report.report_id, session_id: report.session_id };
  tracePanel.append(el("pre", { class: "trace", text: JSON.stringify(traceInfo, null, 2) }));

  // Related questions
  const related = $(".related", node);
  related.innerHTML = "";
  const qs = report.related_questions || [];
  if (qs.length) {
    related.append(el("h4", { text: "Related" }));
    for (const q of qs) related.append(el("button", { type: "button", class: "chip", text: q, onclick: () => ask(q, { academic: false, crawl: false }) }));
  }
  $(".report", node).classList.remove("hidden");
  ["planner", "retriever", "verifier", "writer", "reporter"].forEach((n) => {
    const li = $(`.steps li[data-node="${n}"]`, node);
    if (li && !li.classList.contains("done") && !li.classList.contains("failed")) li.classList.add(report.error ? "failed" : "done");
  });
}

function exportMarkdown(view) {
  const r = view.report || {};
  const sources = r.sources || view.sources || [];
  const lines = [`# ${view.question}`, "", `_Generated ${new Date().toLocaleString("en-GB", { hour12: false })}_`, "", "## Answer", "", view.answerText || "", ""];
  if (r.executive_summary) lines.push("## Summary", "", r.executive_summary, "");
  if ((r.key_findings || []).length) {
    lines.push("## Key findings", "");
    for (const f of r.key_findings) lines.push(`- **${f.headline}** (${f.confidence}) — ${f.detail} ${(f.citations || []).map((c) => `[${c.id}]`).join("")}`);
    lines.push("");
  }
  if (r.tradeoffs) lines.push("## Trade-offs", "", r.tradeoffs, "");
  lines.push("## Sources", "");
  for (const s of sources) lines.push(`${s.id}. [${s.title}](${s.url}) — ${s.domain}`);
  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
  const a = el("a", { href: URL.createObjectURL(blob), download: `research-${(r.report_id || "report")}.md` });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

// ── Asking ───────────────────────────────────────────────────────────────────

const NODE_TO_STEP = { planner: "planner", refine: "retriever", retriever: "retriever", verifier: "verifier", writer: "writer", reporter: "reporter" };

async function ask(question, opts = {}) {
  question = (question || "").trim();
  if (!question || state.busy) return;
  setBusy(true);
  $("#hero").classList.add("hidden");
  $("#followForm").classList.remove("hidden");
  const view = createTurn(question);
  state.controller = new AbortController();

  const body = { query: question, use_academic: !!opts.academic, use_crawl: !!opts.crawl };
  if (state.sessionId) body.session_id = state.sessionId;

  try {
    await streamResearch(body, (event, data) => handleEvent(view, event, data), state.controller.signal);
  } catch (err) {
    if (err.name === "AbortError") {
      addNotice(view, "warning", "Stopped. Partial results are shown above.");
      view.answerEl.classList.remove("streaming");
    } else {
      addNotice(view, "error", `Request failed: ${err.message}`);
      if (!view.answerText) view.answerEl.innerHTML = '<p class="placeholder">No answer — see the message above.</p>';
    }
    $(".stop-btn", view.node).classList.add("hidden");
  } finally {
    setBusy(false);
    state.controller = null;
    loadHistory();
    loadProviders();
  }
}

function handleEvent(view, event, data) {
  switch (event) {
    case "start":
      state.sessionId = data.session_id;
      highlightHistory();
      break;
    case "queued":
      addNotice(view, "info", data.message);
      break;
    case "status": {
      const step = NODE_TO_STEP[data.node];
      if (step) view.setStep(step, data.state === "running" ? "running" : "done");
      break;
    }
    case "plan":
      renderPlan(view, data);
      break;
    case "plan_update":
      addNotice(view, "info", `Few sources found (${data.reason}) — searching again with: ${(data.search_queries || []).join(" · ")}`);
      break;
    case "sources":
      renderSourcesStrip(view, data.sources || []);
      break;
    case "token":
      view.answerText += data.text;
      view.renderAnswer(true);
      break;
    case "answer_reset":
      view.answerText = "";
      view.renderAnswer(true);
      break;
    case "answer":
      view.answerText = data.text;
      view.renderAnswer(false);
      break;
    case "notice":
      addNotice(view, data.level, data.message);
      break;
    case "warning":
      addNotice(view, "warning", data.message);
      break;
    case "error":
      addNotice(view, "error", data.message);
      break;
    case "report":
      renderReport(view, data);
      break;
    default:
      break;
  }
}

function setBusy(busy) {
  state.busy = busy;
  $("#askBtn").disabled = busy;
  $("#followBtn").disabled = busy;
  $("#followBtn").textContent = busy ? "Working…" : "Ask";
}

// ── History ──────────────────────────────────────────────────────────────────

async function loadHistory() {
  const list = $("#historyList");
  try {
    const resp = await fetch("/sessions?limit=30");
    const data = await resp.json();
    list.innerHTML = "";
    if (!data.sessions?.length) { list.append(el("li", { class: "muted", text: "No threads yet" })); return; }
    for (const s of data.sessions) {
      const btn = el("button", { type: "button", "data-session": s.session_id, onclick: () => openSession(s.session_id) },
        el("span", { class: "h-title", text: s.title || s.session_id }),
        el("span", { class: "h-meta", text: `${dateTime24(s.updated_at)} · ${s.turn_count || 0} question${s.turn_count === 1 ? "" : "s"}` }));
      list.append(el("li", {}, btn));
    }
    highlightHistory();
  } catch {
    list.innerHTML = "";
    list.append(el("li", { class: "muted", text: "History unavailable" }));
  }
}

function highlightHistory() {
  $$("#historyList button").forEach((b) => b.classList.toggle("active", b.dataset.session === state.sessionId));
}

async function openSession(sessionId) {
  if (state.busy) return;
  try {
    const resp = await fetch(`/sessions/${encodeURIComponent(sessionId)}`);
    if (!resp.ok) throw new Error(`${resp.status}`);
    const data = await resp.json();
    state.sessionId = sessionId;
    $("#thread").innerHTML = "";
    $("#hero").classList.add("hidden");
    $("#followForm").classList.remove("hidden");
    const reports = data.reports || [];
    const turns = data.turns || [];
    const count = Math.max(reports.length, turns.length);
    for (let i = 0; i < count; i++) {
      const report = reports[i] || {};
      const turn = turns[i] || {};
      const view = createTurn(turn.query || report.query || "Question");
      if (turn.timestamp || report.created_at) $(".time", view.node).textContent = dateTime24(turn.timestamp || report.created_at);
      $(".steps", view.node).classList.add("hidden");
      if (report.answer) renderReport(view, report);
      else { view.answerText = turn.answer || ""; view.renderAnswer(false); $(".stop-btn", view.node).classList.add("hidden"); }
    }
    highlightHistory();
    closeSidebar();
  } catch (err) {
    alert(`Could not open the thread (${err.message}).`);
  }
}

function newThread() {
  if (state.busy) state.controller?.abort();
  state.sessionId = null;
  $("#thread").innerHTML = "";
  $("#hero").classList.remove("hidden");
  $("#followForm").classList.add("hidden");
  $("#askInput").value = "";
  $("#askInput").focus();
  highlightHistory();
  closeSidebar();
}

// ── Providers ────────────────────────────────────────────────────────────────

const STATUS_TEXT = {
  ok: ["ok", "Ready"], degraded: ["warn", "Errors recently"], rate_limited: ["warn", "Rate limited"],
  cooling_down: ["warn", "Paused"], quota_exhausted: ["bad", "Credits exhausted"], auth_failed: ["bad", "Key rejected"],
  not_configured: ["", "No key"], not_running: ["", "Not running"],
};

async function loadProviders(refresh = false) {
  const list = $("#providerList");
  try {
    const resp = await fetch(`/providers${refresh ? "?refresh=true" : ""}`);
    const data = await resp.json();
    list.innerHTML = "";
    const order = { search: 0, llm: 1, academic: 2, infra: 3 };
    const shown = (data.providers || [])
      .filter((p) => p.category !== "infra" && !(p.category === "llm" && p.status === "not_configured" && p.provider !== "anthropic"))
      .sort((a, b) => (order[a.category] ?? 9) - (order[b.category] ?? 9));
    for (const p of shown) {
      const [dot, label] = STATUS_TEXT[p.status] || ["", p.status];
      const row = el("li", {}, el("span", { class: `dot ${dot}` }), el("span", { text: p.display_name }), el("span", { class: "p-status", text: label }));
      const details = [];
      if (p.quota && typeof p.quota.remaining === "number") details.push(`${p.quota.remaining} / ${p.quota.limit ?? "?"} credits left${p.quota.low ? " (low)" : ""}`);
      if (p.model) details.push(p.model);
      if (p.disabled_until) details.push(`until ${p.disabled_until.slice(-5)}`);
      if (details.length) row.append(el("span", { class: "p-detail", text: details.join(" · ") }));
      if (["quota_exhausted", "auth_failed", "rate_limited", "cooling_down"].includes(p.status)) {
        row.append(el("span", { class: "p-detail" }, el("button", { type: "button", class: "link-btn", text: "Retry now", onclick: () => resetProvider(p.provider) })));
      }
      if (p.last_error && p.status !== "ok") row.title = p.last_error;
      list.append(row);
    }
    const hasLlm = (data.llm_order || []).length && (data.providers || []).some((p) => p.category === "llm" && p.status === "ok" && p.configured && p.provider !== "ollama");
    $("#modeLine").textContent = hasLlm ? "Answers: AI-written with citations" : "Answers: extractive (add an LLM key in .env for AI-written answers)";
  } catch {
    list.innerHTML = "";
    list.append(el("li", { class: "muted", text: "Status unavailable" }));
  }
}

async function resetProvider(name) {
  await fetch("/providers/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: name }) });
  loadProviders(true);
}

// ── Wiring ───────────────────────────────────────────────────────────────────

function closeSidebar() { $("#sidebar").classList.remove("open"); }

function submitOnEnter(textarea, form) {
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  $("#askForm").addEventListener("submit", (e) => {
    e.preventDefault();
    ask($("#askInput").value, { academic: $("#optAcademic").checked, crawl: $("#optCrawl").checked });
  });
  $("#followForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("#followInput").value;
    $("#followInput").value = "";
    ask(q, { academic: $("#followAcademic").checked, crawl: $("#followCrawl").checked });
  });
  submitOnEnter($("#askInput"), $("#askForm"));
  submitOnEnter($("#followInput"), $("#followForm"));
  $$("#examples .chip").forEach((chip) => chip.addEventListener("click", () => { $("#askInput").value = chip.textContent; $("#askInput").focus(); }));
  $("#newThread").addEventListener("click", newThread);
  $("#refreshProviders").addEventListener("click", () => loadProviders(true));
  $("#menuBtn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  loadHistory();
  loadProviders();
  setInterval(() => { if (!state.busy) loadProviders(); }, 60000);
  $("#askInput").focus();
});
