# shadow-routing-comparator

Shadow routing for LLM changes: serve every request from the production model+prompt, silently mirror a configurable percentage to a candidate, and diff the two on text, tool-call trajectory, latency and cost. A gate turns the diff into a **PROMOTE / HOLD** verdict. The report here comes from replaying a seeded synthetic dataset through mock models.

## What it is
- `shadow/router.py` — `ShadowRouter.handle(req)` returns the primary response, then mirrors sampled requests to the candidate in a background task (deterministic hash sampling per request id, in-flight cap, timeout, errors swallowed).
- `shadow/diff.py` — text similarity (difflib ratio, token Jaccard) and trajectory diff (exact match, same tool set, same order, missing / extra tools, duplicate calls, arg mismatches, edit distance over tool names).
- `shadow/report.py` — aggregates per category and applies gates (exact-match rate, per-category exact match and text similarity, critical-tool skips, shadow error rate).
- `shadow/models.py` — mock baseline (`prod-large@prompt-v7`) and candidate (`cand-small@prompt-v8`). The candidate is built to misbehave on some categories so the comparator has real differences to find.

## Why
A new model or prompt can pass offline evals and still regress on live traffic: it skips a policy check before a refund, calls tools in the wrong order, or refuses questions it used to answer. Shadow traffic finds these on real request shapes with zero user impact, and the per-category gate stops a cheaper candidate whose average looks fine.

## Architecture
```mermaid
flowchart LR
  U[User request] --> SR[ShadowRouter]
  SR --> P[Primary model] -->|response| U
  SR -->|hash(id) < pct?| BG[[background task<br/>semaphore + timeout]]
  BG --> C[Candidate model]
  P -.primary response.-> D
  C --> D[diff: text · trajectory · latency · cost]
  D --> REC[(records)]
  REC --> RPT[report + gates] --> V{PROMOTE / HOLD}
```

## Results
### Replay report — SIMULATED
Run: `python3 bench/make_dataset.py && python3 bench/replay.py`. Raw output: [`results/shadow_report.json`](results/shadow_report.json). Dataset: [`data/synthetic_traffic.jsonl`](data/synthetic_traffic.jsonl) (2,000 seeded synthetic requests over 5 categories).

Mock models, synthetic traffic, and **modelled** latency and illustrative prices. The sampling, diffing and gating are the real code. A test re-runs the replay and checks it reproduces the committed numbers exactly.

Shadow rate 25% → **532 of 2,000 requests mirrored**, 0 shadow errors, 0 timeouts.

| Metric | Primary | Candidate |
|---|---|---|
| Latency p50 (modelled) | 1,323 ms | 595 ms |
| Latency p95 (modelled) | 1,559 ms | 725 ms |
| Cost on mirrored traffic (illustrative prices) | $1.7103 | $0.1407 (8.2%) |
| Trajectory exact match | — | 89.3% |
| Mean text similarity | — | 0.919 |

| Category | n | Trajectory exact match | Mean text sim | What differed |
|---|---|---|---|---|
| order_status | 167 | 90.4% | 0.954 | 16 redundant `get_shipping` calls |
| refund | 116 | **72.4%** | 0.955 | **32 skipped `check_refund_policy`** |
| faq | 120 | 100% | **0.808** | 24 answers with similarity < 0.5 (refusals) |
| account | 74 | **87.8%** | 0.951 | 9 reordered tool calls |
| smalltalk | 55 | 100% | 0.934 | — |

**Verdict: HOLD.** Gate failures: overall exact match 0.893 < 0.95; refund 0.724 and account 0.878 < 0.90; faq text similarity 0.808 < 0.85; critical tool `check_refund_policy` skipped 32 times. The candidate is 2.2× faster and ~12× cheaper on paper, and would still have issued refunds without a policy check.

## Tests
18 tests (`pytest`): deterministic sampling and sampling rate at 0/10/50/100%, edit distance, missing / reordered / duplicate / arg-mismatch trajectories, text metrics, identical models → PROMOTE, critical-tool skip → HOLD, user latency not blocked by a 400 ms shadow (real sleeps), shadow exceptions and timeouts never reach the user, in-flight cap drops shadow work, cost/latency deltas, and committed report reproduced by a fresh replay.

```
$ python3 -m pytest -q
..................                                                       [100%]
18 passed in 1.17s
```

## Run it
```bash
pip install pytest
python3 -m pytest -q
python3 bench/make_dataset.py
python3 bench/replay.py
```

## Limits
- Mock models and synthetic traffic. The differences are designed in; the report shows that the comparator detects them, not how any real model behaves.
- Latency is modelled from token counts, not measured from a provider. Prices are illustrative.
- Text similarity is lexical (difflib, Jaccard). It flags refusals and rewrites but cannot judge semantic correctness; an LLM judge or task-specific checks would be the next step.
- Shadow calls must be side-effect-free. Tool calls are compared, never executed; a real deployment needs dry-run tool stubs for the candidate.
- Records are in memory. No persistence, dashboards, or streaming significance tests.

## Stack
Python 3.12 · asyncio · standard library · pytest

## License
MIT
