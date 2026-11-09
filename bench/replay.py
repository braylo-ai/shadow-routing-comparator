"""Replay the synthetic dataset through ShadowRouter and write results/shadow_report.json.

SIMULATED: mock models, synthetic traffic, modelled latencies (not slept). The routing,
sampling, diffing and gating are the real code.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shadow.models import baseline, candidate  # noqa: E402
from shadow.report import build_report  # noqa: E402
from shadow.router import ShadowRouter  # noqa: E402

PERCENT = 25.0


async def main():
    reqs = [json.loads(l) for l in (ROOT / "data" / "synthetic_traffic.jsonl").read_text().splitlines()]
    sr = ShadowRouter(baseline(), candidate(), percent=PERCENT)
    for r in reqs:
        await sr.handle(r)
    await sr.drain()
    rep = build_report(sr.records)
    out = {"kind": "SIMULATED (mock models, synthetic traffic, modelled latency)", "date": time.strftime("%Y-%m-%d"),
           "dataset": "data/synthetic_traffic.jsonl", "requests": len(reqs), "shadow_percent": PERCENT,
           "primary": sr.primary.name, "candidate": sr.shadow.name, "router_stats": sr.stats, **rep}
    (ROOT / "results" / "shadow_report.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("router_stats", "verdict", "gate_failures", "summary", "by_category")}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
