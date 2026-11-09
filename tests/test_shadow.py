import asyncio
import json
import time
from pathlib import Path

import pytest

from shadow.diff import compare, edit_distance, text_similarity, token_jaccard, trajectory_diff
from shadow.models import MockModel, ModelResponse, baseline, candidate
from shadow.report import build_report
from shadow.router import ShadowRouter, sampled

ROOT = Path(__file__).resolve().parents[1]


def run(c): return asyncio.run(c)


def req(i, cat="refund"):
    return {"id": f"r{i}", "category": cat, "prompt": "refund for ORD-1", "order_id": "ORD-1", "user_id": "u1", "q": cat}


def calls(*names):
    return [{"name": n, "args": {"x": 1}} for n in names]


def test_sampling_is_deterministic():
    assert all(sampled(f"id{i}", 30) == sampled(f"id{i}", 30) for i in range(100))


@pytest.mark.parametrize("pct", [0, 10, 50, 100])
def test_sampling_rate_close_to_percent(pct):
    hits = sum(sampled(f"id{i}", pct) for i in range(10_000))
    assert abs(hits / 10_000 - pct / 100) < 0.02


def test_edit_distance():
    assert edit_distance(list("abc"), list("abc")) == 0
    assert edit_distance(list("abc"), list("ac")) == 1
    assert edit_distance(list("ab"), list("ba")) == 2


def test_trajectory_missing_tool():
    d = trajectory_diff(calls("a", "b", "c"), calls("a", "c"))
    assert d["missing_tools"] == ["b"] and not d["exact_match"] and d["edit_distance"] == 1


def test_trajectory_reordered_same_set():
    d = trajectory_diff(calls("a", "b"), calls("b", "a"))
    assert d["same_tool_set"] and not d["same_order"] and d["missing_tools"] == []


def test_trajectory_duplicate_and_arg_mismatch():
    d = trajectory_diff(calls("a", "b"), calls("a", "b", "b"))
    assert d["duplicate_calls"] == 1 and d["same_tool_set"]
    d2 = trajectory_diff([{"name": "a", "args": {"id": 1}}], [{"name": "a", "args": {"id": 2}}])
    assert d2["arg_mismatches"] == ["a"] and not d2["exact_match"]


def test_text_metrics():
    assert text_similarity("same", "same") == 1.0
    assert token_jaccard("Hello world", "hello, WORLD") == 1.0
    assert token_jaccard("a b", "c d") == 0.0


def test_identical_models_produce_exact_match_and_promote():
    async def go():
        sr = ShadowRouter(baseline(), baseline(), percent=100)
        for i in range(50):
            await sr.handle(req(i, ["refund", "faq", "account"][i % 3]))
        await sr.drain()
        return build_report(sr.records)
    rep = run(go())
    assert rep["verdict"] == "PROMOTE" and rep["summary"]["trajectory_exact_match"] == 1.0


def test_candidate_skipping_critical_tool_is_held():
    async def go():
        sr = ShadowRouter(baseline(), candidate(), percent=100)
        for i in range(200):
            await sr.handle(req(i, "refund"))
        await sr.drain()
        return build_report(sr.records)
    rep = run(go())
    assert rep["verdict"] == "HOLD"
    assert any("critical tool" in f for f in rep["gate_failures"])


def test_user_gets_primary_response_without_waiting_for_shadow():
    async def go():
        p = MockModel("p", 1, 1, 20, 0, real_sleep=True)
        s = MockModel("s", 1, 1, 400, 0, real_sleep=True)
        sr = ShadowRouter(p, s, percent=100)
        t0 = time.perf_counter()
        resp = await sr.handle(req(1, "faq"))
        dt = time.perf_counter() - t0
        assert sr.records == []  # shadow not finished yet
        await sr.drain()
        return dt, resp, sr
    dt, resp, sr = run(go())
    assert dt < 0.2 and resp.text and len(sr.records) == 1


def test_shadow_exception_never_reaches_user():
    async def boom(req):
        raise RuntimeError("candidate exploded")

    async def go():
        sr = ShadowRouter(baseline(), boom, percent=100)
        resp = await sr.handle(req(1))
        await sr.drain()
        return resp, sr
    resp, sr = run(go())
    assert resp.error is None and sr.stats["shadow_errors"] == 1
    assert sr.records[0]["shadow_error"] == "RuntimeError"


def test_shadow_timeout_recorded():
    async def go():
        sr = ShadowRouter(baseline(), MockModel("slow", 1, 1, 500, 0, real_sleep=True), percent=100, shadow_timeout_s=0.05)
        await sr.handle(req(1)); await sr.drain()
        return sr
    sr = run(go())
    assert sr.stats["shadow_timeouts"] == 1


def test_shadow_dropped_when_inflight_limit_reached():
    async def go():
        sr = ShadowRouter(baseline(), MockModel("slow", 1, 1, 200, 0, real_sleep=True), percent=100, max_inflight=1)
        await sr.handle(req(1)); await asyncio.sleep(0)  # first mirror holds the semaphore
        await sr.handle(req(2))
        await sr.drain()
        return sr
    sr = run(go())
    assert sr.stats["shadow_dropped"] == 1 and sr.stats["mirrored"] == 1


def test_compare_cost_and_latency_deltas():
    a = ModelResponse("x", [], latency_ms=100, cost_usd=0.01)
    b = ModelResponse("x", [], latency_ms=40, cost_usd=0.002)
    c = compare(a, b)
    assert c["latency_delta_ms"] == -60 and c["cost_delta_usd"] == pytest.approx(-0.008)


def test_committed_report_matches_rerun():
    """Re-running the replay on the committed dataset reproduces the committed numbers."""
    reqs = [json.loads(l) for l in (ROOT / "data" / "synthetic_traffic.jsonl").read_text().splitlines()]

    async def go():
        sr = ShadowRouter(baseline(), candidate(), percent=25.0)
        for r in reqs:
            await sr.handle(r)
        await sr.drain()
        return build_report(sr.records)
    rep = run(go())
    committed = json.loads((ROOT / "results" / "shadow_report.json").read_text())
    assert committed["kind"].startswith("SIMULATED")
    assert committed["verdict"] == rep["verdict"] == "HOLD"
    assert committed["summary"] == rep["summary"] and committed["by_category"] == rep["by_category"]
