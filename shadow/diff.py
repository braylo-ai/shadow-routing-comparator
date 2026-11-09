"""Diff two responses: text, tool-call trajectory, latency, cost."""
from __future__ import annotations

import difflib
import re

from .models import ModelResponse

_TOK = re.compile(r"\w+")


def text_similarity(a: str, b: str) -> float:
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 4)


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(_TOK.findall(a.lower())), set(_TOK.findall(b.lower()))
    if not sa and not sb:
        return 1.0
    return round(len(sa & sb) / len(sa | sb), 4)


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def trajectory_diff(a: list[dict], b: list[dict]) -> dict:
    na, nb = [c["name"] for c in a], [c["name"] for c in b]
    missing = [n for n in dict.fromkeys(na) if n not in nb]
    extra = [n for n in dict.fromkeys(nb) if n not in na]
    arg_mismatch = [x["name"] for x, y in zip(a, b) if x["name"] == y["name"] and x["args"] != y["args"]]
    return {
        "exact_match": a == b,
        "same_tool_set": set(na) == set(nb),
        "same_order": na == nb,
        "missing_tools": missing,
        "extra_tools": extra,
        "duplicate_calls": len(nb) - len(set(nb)),
        "arg_mismatches": arg_mismatch,
        "edit_distance": edit_distance(na, nb),
    }


def compare(primary: ModelResponse, shadow: ModelResponse) -> dict:
    return {
        "text_similarity": text_similarity(primary.text, shadow.text),
        "token_jaccard": token_jaccard(primary.text, shadow.text),
        "trajectory": trajectory_diff(primary.tool_calls, shadow.tool_calls),
        "latency_delta_ms": round(shadow.latency_ms - primary.latency_ms, 2),
        "cost_delta_usd": shadow.cost_usd - primary.cost_usd,
        "primary_cost_usd": primary.cost_usd,
        "shadow_cost_usd": shadow.cost_usd,
        "primary_latency_ms": primary.latency_ms,
        "shadow_latency_ms": shadow.latency_ms,
        "shadow_error": shadow.error,
    }
