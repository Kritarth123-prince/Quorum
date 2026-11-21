# Quorum

**Research your sources agree on.**

A Perplexity-style research assistant that goes further: it **plans** the research, searches
the **web, academic papers and its own knowledge base**, **cross-checks claims** across sources
with evidence-based confidence, and writes a **cited answer plus a structured research report**.
Every page it reads is added to a local knowledge base, so it improves with use.

## Quick start (Windows, `agent` virtual environment)

```bat
run_server.bat                 :: web UI + API → http://127.0.0.1:8000
research.bat "your question"   :: command line   (add --academic, --crawl, --json)
research.bat --status          :: provider health and remaining search credits
```

Manual equivalent:

```bat
agent\Scripts\activate.bat
python -m uvicorn retrieval_pipeline.api:app --host 127.0.0.1 --port 8000
```

> Don't add `--reload` on Windows: it switches to an event loop that cannot start Playwright
> (static page fetching still works, but JavaScript-rendered pages won't).

**AI-written answers need one LLM key** in `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GROQ_API_KEY`, `GEMINI_API_KEY`, …). Without one, everything still runs, and answers are
extractive summaries built from verified source sentences.

Fresh install: `pip install -r retrieval_pipeline/requirements.txt`, then optionally
`python -m playwright install chromium`, and copy `.env.example` to `.env`.

## Architecture

```text
question ─► preflight ─► planner ─► retriever ──(too few sources)──► refine ─┐
                                        ▲                                    │
                                        └────────────────────────────────────┘
                                        │
                              verifier ─► writer (streamed, cited) ─► reporter ─► memory
```

| Stage | What it does |
|---|---|
| Planner | Rewrites follow-ups into standalone questions, sets research goal, 2–4 distinct search queries, decides on academic search / deep reading |
| Retriever | Sub-queries run in parallel: Tavily → SerpAPI → DuckDuckGo, Semantic Scholar + arXiv, local knowledge base; optional full-page reading. Pooled, de-duplicated, **hybrid-ranked** (BM25 + embeddings, RRF) and **reranked** once against the question |
| Refine | Autonomous second pass with broader queries if fewer than 3 sources were found |
| Verifier | Extracts key claims, maps each to supporting/contradicting sources; confidence is computed from independent domains and credibility, not taken from the model |
| Writer | Streams a Markdown answer with `[n]` citations; invalid citations are removed |
| Reporter | Executive summary, findings (confidence capped by evidence), trade-offs, related questions |
| Memory | Threads in SQLite (or Redis); verified findings in Qdrant for later recall |

Orchestrated with **LangGraph**. The vector store is an **embedded Qdrant** in `data/qdrant`
(no Docker needed), or a Qdrant server if one is running.

## Error handling: credits, keys and outages

Every provider error is classified, and the platform reacts accordingly:

| Situation | Examples | Behaviour |
|---|---|---|
| Credits exhausted | Tavily 432/433, SerpAPI "run out of searches", Anthropic "credit balance too low", OpenAI `insufficient_quota`, HTTP 402 | Provider paused (default 6 h), request moves to the next provider, user sees a notice |
| Invalid key | 401/403 | Paused 24 h; fix `.env` and restart |
| Rate limit | 429 | Retries honouring `Retry-After`, then falls back |
| Timeouts / 5xx | | Retries with backoff; after 3 failures, paused 2 min |
| All LLMs down | | Extractive answer and report; the run still completes |
| All search down | | Academic search + knowledge base only, with a clear warning |
| Run too slow | | Hard timeout (7 min) still returns a partial report |

The UI sidebar (and `GET /providers`) shows each provider's status and **remaining Tavily/SerpAPI
credits**. A provider found at 0 credits is paused before it fails. **Retry now** (or
`POST /providers/reset`) re-enables a provider after you top up. API keys are redacted from all
logs and error messages.

## API

| Endpoint | Purpose |
|---|---|
| `POST /research` | Full pipeline → JSON report |
| `POST /research/stream` | Same, as Server-Sent Events (progress, tokens, report) |
| `POST /retrieve`, `POST /retrieve/batch` | Retrieval only (multi-method / multi-query) |
| `POST /index`, `POST /index/urls` | Add documents or URLs to the knowledge base |
| `GET /sessions`, `GET/DELETE /sessions/{id}` | Thread history |
| `GET /providers`, `POST /providers/reset` | Provider health, credits, re-enable |
| `GET /health`, `GET /metrics` | Status and Prometheus metrics |

Interactive docs: <http://127.0.0.1:8000/docs>

## Configuration notes

* Models default to `bge-small-en-v1.5` + `ms-marco-MiniLM-L-6-v2` (fast on a 2-core CPU).
  The large BGE models took minutes per query on this PC; with an NVIDIA GPU set
  `EMBEDDING_DEVICE=cuda` and switch back if you like.
* `TAVILY_SEARCH_DEPTH=basic` (1 credit) and `PLANNER_MAX_QUERIES=4` keep credit use at
  roughly 3–4 searches per question; identical searches within 30 minutes are cached.
* Claude calls use `claude-opus-5-5` with structured outputs and Anthropic's server-side
  refusal fallbacks (`ANTHROPIC_FALLBACKS=false` disables them).

## Tests

```bat
agent\Scripts\python -m pytest -q tests
```

Offline tests use mocked HTTP and cost no credits. They cover error classification,
credit-exhaustion fallback, LLM fallback, chunking, citations and hybrid ranking.

## Docker (optional)

`docker compose up --build` from the project root (API + Qdrant + Redis; add
`--profile local-llm` for Ollama and `--profile monitoring` for Prometheus/Grafana).
