"""Mock models with tool-calling trajectories. SIMULATED: no model is called.

`baseline` is the production model+prompt. `candidate` is a cheaper/faster model with a
new prompt that behaves differently on some categories (by design, so the comparator has
real differences to find).
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field

PLANS = {
    "order_status": [("lookup_order", "order_id"), ("get_shipping", "order_id")],
    "refund": [("lookup_order", "order_id"), ("check_refund_policy", "order_id"), ("issue_refund", "order_id")],
    "faq": [("search_kb", "q")],
    "account": [("lookup_user", "user_id"), ("update_email", "user_id")],
    "smalltalk": [],
}


def _u(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class MockModel:
    name: str
    price_in: float          # USD per 1M tokens (illustrative)
    price_out: float
    base_latency_ms: float
    ms_per_out_token: float
    behaviour: str = "baseline"
    real_sleep: bool = False  # True: actually await latency (used in live-path tests)

    async def __call__(self, req: dict) -> ModelResponse:
        rid, cat, prompt = req["id"], req["category"], req["prompt"]
        plan = list(PLANS[cat])
        text = f"Sure. Regarding your {cat.replace('_', ' ')} request: {prompt.strip().rstrip('?')}."
        if self.behaviour == "candidate":
            r = _u(self.name, rid)
            if cat == "refund" and r < 0.30:
                plan = [p for p in plan if p[0] != "check_refund_policy"]   # skips policy check
            if cat == "account" and r < 0.15:
                plan = list(reversed(plan))                                  # wrong order
            if cat == "order_status" and r < 0.10:
                plan = plan + [("get_shipping", "order_id")]                 # redundant call
            text = f"Regarding your {cat.replace('_', ' ')} request: {prompt.strip().rstrip('?')}."
            if cat == "faq" and r < 0.20:
                text = "I can't help with that, please contact support."    # unhelpful refusal
        calls = [{"name": n, "args": {k: req.get(k, "") }} for n, k in plan]
        tin = 300 + 4 * len(prompt) + 120 * len(calls)
        tout = 40 + len(text) // 4 + 20 * len(calls)
        lat = self.base_latency_ms + self.ms_per_out_token * tout + 30 * len(calls) + 40 * _u("lat", self.name, rid)
        if self.real_sleep:
            await asyncio.sleep(lat / 1000)
        cost = (tin * self.price_in + tout * self.price_out) / 1e6
        return ModelResponse(text, calls, round(lat, 2), tin, tout, cost)


def baseline() -> MockModel:
    return MockModel("prod-large@prompt-v7", 3.0, 15.0, 380, 9.0, "baseline")


def candidate() -> MockModel:
    return MockModel("cand-small@prompt-v8", 0.25, 1.25, 140, 4.0, "candidate")
