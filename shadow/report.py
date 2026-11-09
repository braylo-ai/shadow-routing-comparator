"""Aggregate shadow records into a report and a promote / hold verdict."""
from __future__ import annotations

import statistics
from collections import defaultdict

DEFAULT_GATES = {
    "min_trajectory_exact_match": 0.95,
    "min_category_exact_match": 0.90,
    "max_missing_critical_tool_rate": 0.0,
    "min_mean_text_similarity": 0.70,
    "min_category_text_similarity": 0.85,
    "max_shadow_error_rate": 0.01,
}
CRITICAL_TOOLS = {"check_refund_policy"}


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * (len(xs) - 1)))] if xs else None


def build_report(records: list[dict], gates: dict | None = None) -> dict:
    g = {**DEFAULT_GATES, **(gates or {})}
    ok = [r for r in records if not r["shadow_error"]]
    n = len(records)
    by_cat = defaultdict(list)
    for r in ok:
        by_cat[r["category"]].append(r)
    cats = {}
    for c, rs in sorted(by_cat.items()):
        cats[c] = {
            "n": len(rs),
            "trajectory_exact_match": round(sum(r["trajectory"]["exact_match"] for r in rs) / len(rs), 4),
            "mean_text_similarity": round(statistics.mean(r["text_similarity"] for r in rs), 4),
            "missing_tool_cases": sum(bool(r["trajectory"]["missing_tools"]) for r in rs),
            "order_mismatch_cases": sum(r["trajectory"]["same_tool_set"] and not r["trajectory"]["same_order"]
                                       and not r["trajectory"]["duplicate_calls"] for r in rs),
            "low_text_similarity_cases": sum(r["text_similarity"] < 0.5 for r in rs),
            "extra_call_cases": sum(bool(r["trajectory"]["duplicate_calls"] or r["trajectory"]["extra_tools"]) for r in rs),
        }
    crit_missing = sum(any(t in CRITICAL_TOOLS for t in r["trajectory"]["missing_tools"]) for r in ok)
    pc, sc = sum(r["primary_cost_usd"] for r in ok), sum(r["shadow_cost_usd"] for r in ok)
    summary = {
        "mirrored": n,
        "shadow_error_rate": round((n - len(ok)) / n, 4) if n else 0.0,
        "trajectory_exact_match": round(sum(r["trajectory"]["exact_match"] for r in ok) / len(ok), 4) if ok else None,
        "mean_text_similarity": round(statistics.mean(r["text_similarity"] for r in ok), 4) if ok else None,
        "p10_text_similarity": _pct([r["text_similarity"] for r in ok], 10),
        "missing_critical_tool_rate": round(crit_missing / len(ok), 4) if ok else None,
        "latency_ms_p50_primary": _pct([r["primary_latency_ms"] for r in ok], 50),
        "latency_ms_p50_shadow": _pct([r["shadow_latency_ms"] for r in ok], 50),
        "latency_ms_p95_primary": _pct([r["primary_latency_ms"] for r in ok], 95),
        "latency_ms_p95_shadow": _pct([r["shadow_latency_ms"] for r in ok], 95),
        "cost_usd_primary": round(pc, 6),
        "cost_usd_shadow": round(sc, 6),
        "cost_ratio_shadow_vs_primary": round(sc / pc, 4) if pc else None,
    }
    failures = []
    if summary["trajectory_exact_match"] is not None and summary["trajectory_exact_match"] < g["min_trajectory_exact_match"]:
        failures.append(f"trajectory exact match {summary['trajectory_exact_match']} < {g['min_trajectory_exact_match']}")
    for c, s in cats.items():
        if s["trajectory_exact_match"] < g["min_category_exact_match"]:
            failures.append(f"category {c}: exact match {s['trajectory_exact_match']} < {g['min_category_exact_match']}")
        if s["mean_text_similarity"] < g["min_category_text_similarity"]:
            failures.append(f"category {c}: mean text similarity {s['mean_text_similarity']} < {g['min_category_text_similarity']}")
    if summary["missing_critical_tool_rate"] and summary["missing_critical_tool_rate"] > g["max_missing_critical_tool_rate"]:
        failures.append(f"critical tool skipped in {crit_missing} cases ({sorted(CRITICAL_TOOLS)})")
    if summary["mean_text_similarity"] is not None and summary["mean_text_similarity"] < g["min_mean_text_similarity"]:
        failures.append(f"mean text similarity {summary['mean_text_similarity']} < {g['min_mean_text_similarity']}")
    if summary["shadow_error_rate"] > g["max_shadow_error_rate"]:
        failures.append(f"shadow error rate {summary['shadow_error_rate']} > {g['max_shadow_error_rate']}")
    worst = sorted(ok, key=lambda r: (r["trajectory"]["edit_distance"], -r["text_similarity"]), reverse=True)[:5]
    return {
        "verdict": "PROMOTE" if not failures else "HOLD",
        "gate_failures": failures,
        "gates": g,
        "summary": summary,
        "by_category": cats,
        "worst_examples": [{"id": r["id"], "category": r["category"], "trajectory": r["trajectory"],
                            "text_similarity": r["text_similarity"]} for r in worst],
    }
