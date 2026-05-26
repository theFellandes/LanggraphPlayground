"""Concurrency primitives shared across the pro research-assistant graph.

Kept in a separate module so:
  1. The graph file stays focused on graph topology.
  2. Tests can swap these primitives (e.g. for a Redis-backed lock).
  3. Module-level state is easy to spot.

Patterns implemented:
  - per-key lock map (lesson 27 pattern 3) — `_topic_locks`
  - bounded concurrency semaphore (lesson 27 pattern 2) — `LLM_SEM`, `SEARCH_SEM`
  - circuit breaker (lesson 30 pattern 6) — `tavily_breaker`
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict


# --- semaphores (per-process rate limits) -----------------------------------
LLM_SEM = asyncio.Semaphore(5)        # max 5 in-flight LLM calls
SEARCH_SEM = asyncio.Semaphore(3)     # Tavily free tier ≈ 100/min — be polite


# --- per-topic lock map (lesson 27 pattern 3) -------------------------------
_topic_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def lock_for(topic: str) -> asyncio.Lock:
    """Get-or-create a per-topic lock.

    Keep the topic string short and canonical (lowercase, stripped) so
    near-duplicates share a lock instead of racing.
    """
    return _topic_locks[topic.strip().lower()]


# --- circuit breaker (lesson 30 pattern 6) ----------------------------------
class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: float = 30.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self.fails = 0
        self.opened_at: float | None = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at > self.cooldown:
            self.opened_at = None
            self.fails = 0
            return False
        return True

    def record(self, ok: bool) -> None:
        if ok:
            self.fails = 0
        else:
            self.fails += 1
            if self.fails >= self.threshold:
                self.opened_at = time.time()


tavily_breaker = CircuitBreaker(threshold=3, cooldown=30.0)
