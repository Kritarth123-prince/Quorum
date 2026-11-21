"""
cli.py — command-line research.

  python -m retrieval_pipeline.cli "your question" [--academic] [--crawl] [--json] [--session ID]
  python -m retrieval_pipeline.cli --status        # provider health + remaining search credits
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.logging_setup import setup_logging


async def _status() -> int:
    from retrieval_pipeline.core.provider_health import registry
    from retrieval_pipeline.llm.client import get_llm
    from retrieval_pipeline.retrieval.web_search import search_quota
    llm = get_llm()
    quota = await search_quota(force=True)
    print("\nLLM providers:", ", ".join(f"{p.name} ({p.model})" for p in llm.providers) or "none")
    print("Active LLM   :", await llm.active_provider() or "none — extractive answers")
    print("\nProvider status:")
    for s in registry.snapshot():
        extra = quota.get(s["provider"], {})
        credits = f"  {extra.get('remaining')}/{extra.get('limit')} credits left" if "remaining" in extra else ""
        print(f"  {s['display_name']:<20} {s['status']:<16}{credits}")
    return 0


async def _research(args) -> int:
    from retrieval_pipeline.orchestration.graph import drain_background, stream_research
    report = None
    streamed = False
    async for ev in stream_research(args.query, args.max_results, args.academic, args.crawl, args.session):
        kind, data = ev["event"], ev["data"]
        if args.json:
            if kind == "report":
                report = data
            continue
        if kind == "status" and data.get("state") == "running":
            print(f"… {data['message']}", file=sys.stderr)
        elif kind == "plan":
            print(f"  queries: {' | '.join(data.get('search_queries', []))}", file=sys.stderr)
        elif kind in ("notice", "warning", "error"):
            print(f"  ! {data.get('message')}", file=sys.stderr)
        elif kind == "token":
            if not streamed:
                print("\n" + "=" * 70 + "\nANSWER\n" + "=" * 70)
                streamed = True
            print(data["text"], end="", flush=True)
        elif kind == "answer_reset":
            print("\n[provider switched — restarting answer]\n")
        elif kind == "report":
            report = data
    await drain_background(60)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0
    if not report:
        return 1
    print("\n\n" + "-" * 70 + "\nSOURCES")
    for s in report.get("sources", []):
        print(f"  [{s['id']}] {s['title'][:70]} — {s['domain']}\n      {s['url']}")
    findings = report.get("key_findings", [])
    if findings:
        print("-" * 70 + "\nKEY FINDINGS")
        for f in findings:
            refs = "".join(f"[{c['id']}]" for c in f.get("citations", []))
            print(f"  ({f['confidence'].upper()}) {f['headline']} {refs}")
    conf = report.get("confidence_summary", {})
    print("-" * 70)
    print(f"Confidence {conf.get('overall_score', 0):.0%} | {conf.get('total_claims', 0)} claims | "
          f"{report.get('total_latency_ms', 0) / 1000:.1f} s | session {report.get('session_id')}")
    for w in report.get("warnings", []):
        print(f"! {w}")
    if report.get("related_questions"):
        print("Related: " + " | ".join(report["related_questions"]))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Quorum — research your sources agree on (command line)")
    parser.add_argument("query", nargs="?", help="research question")
    parser.add_argument("--academic", action="store_true", help="include Semantic Scholar + arXiv")
    parser.add_argument("--crawl", action="store_true", help="deep-read the top result pages")
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument("--session", help="continue an existing thread (session id)")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--status", action="store_true", help="show provider status and credits")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging("INFO" if args.verbose else "WARNING")
    get_settings()
    if args.status:
        return asyncio.run(_status())
    if not args.query:
        parser.print_help()
        return 2
    return asyncio.run(_research(args))


if __name__ == "__main__":
    sys.exit(main())
