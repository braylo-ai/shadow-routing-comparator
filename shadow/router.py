"""Shadow router: serve primary, mirror a sampled % to the candidate off the hot path."""
from __future__ import annotations

import asyncio
import hashlib
import time

from .diff import compare
from .models import ModelResponse


def sampled(request_id: str, percent: float, salt: str = "shadow") -> bool:
    """Deterministic per-request sampling: the same id always gets the same decision."""
    h = int.from_bytes(hashlib.sha256(f"{salt}:{request_id}".encode()).digest()[:8], "big")
    return (h % 10_000) < percent * 100


class ShadowRouter:
    def __init__(self, primary, shadow, percent: float = 10.0, shadow_timeout_s: float = 5.0, max_inflight: int = 100):
        self.primary, self.shadow = primary, shadow
        self.percent, self.shadow_timeout_s = percent, shadow_timeout_s
        self.sem = asyncio.Semaphore(max_inflight)
        self.records: list[dict] = []
        self.stats = {"requests": 0, "mirrored": 0, "shadow_errors": 0, "shadow_timeouts": 0, "shadow_dropped": 0}
        self._tasks: set[asyncio.Task] = set()

    async def handle(self, req: dict) -> ModelResponse:
        self.stats["requests"] += 1
        resp = await self.primary(req)
        if sampled(req["id"], self.percent):
            if self.sem.locked():
                self.stats["shadow_dropped"] += 1  # never queue shadow work behind live traffic
            else:
                t = asyncio.create_task(self._mirror(req, resp))
                self._tasks.add(t)
                t.add_done_callback(self._tasks.discard)
        return resp  # returned before the shadow finishes

    async def _mirror(self, req: dict, primary: ModelResponse) -> None:
        async with self.sem:
            self.stats["mirrored"] += 1
            try:
                sh = await asyncio.wait_for(self.shadow(req), self.shadow_timeout_s)
            except asyncio.TimeoutError:
                self.stats["shadow_timeouts"] += 1
                sh = ModelResponse("", error="timeout")
            except Exception as e:  # shadow failures never reach the user
                self.stats["shadow_errors"] += 1
                sh = ModelResponse("", error=type(e).__name__)
            self.records.append({"id": req["id"], "category": req["category"], "ts": time.time(), **compare(primary, sh)})

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
