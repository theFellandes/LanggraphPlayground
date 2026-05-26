"""Lesson 27 · Locks & concurrency — four runnable scenarios.

No API key required. Each scenario uses a fake LLM with a small random
sleep so the race conditions are observable.

Run:
    uv run python -m lessons.27_locks_and_concurrency.example          # all four
    uv run python -m lessons.27_locks_and_concurrency.example --race
    uv run python -m lessons.27_locks_and_concurrency.example --mutex
    uv run python -m lessons.27_locks_and_concurrency.example --semaphore
    uv run python -m lessons.27_locks_and_concurrency.example --per-user
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from collections import defaultdict

from shared.pretty import console, section


# --- Fake "LLM call" so the lesson runs offline -----------------------------
async def fake_llm_call(prompt: str, sleep_s: float = 0.05) -> str:
    await asyncio.sleep(sleep_s + random.uniform(0, 0.02))
    return f"echo({len(prompt)} chars)"


# --- Scenario 1 · the race --------------------------------------------------
async def scenario_race() -> None:
    """Two coroutines increment a shared counter — without a lock."""
    section("1 · RACE (no lock) — expect lost updates")

    shared = {"n": 0}

    async def increment(label: str) -> None:
        for _ in range(100):
            val = shared["n"]
            await asyncio.sleep(0)        # yield → the bug surfaces
            shared["n"] = val + 1

    await asyncio.gather(increment("a"), increment("b"))
    console.print(f"Expected 200, got [bold red]{shared['n']}[/]  (lost updates)")


# --- Scenario 2 · fix with a mutex ------------------------------------------
async def scenario_mutex() -> None:
    """Same code, fixed with asyncio.Lock."""
    section("2 · MUTEX (asyncio.Lock) — no lost updates")

    shared = {"n": 0}
    lock = asyncio.Lock()

    async def increment(label: str) -> None:
        for _ in range(100):
            async with lock:                 # critical section
                val = shared["n"]
                await asyncio.sleep(0)
                shared["n"] = val + 1

    await asyncio.gather(increment("a"), increment("b"))
    console.print(f"Expected 200, got [bold green]{shared['n']}[/]  (correct)")


# --- Scenario 3 · semaphore-bounded fan-out ---------------------------------
async def scenario_semaphore() -> None:
    """30 'LLM calls' but cap concurrency at 5 so we don't blow our rate limit."""
    section("3 · SEMAPHORE — bound 30 calls to 5 in-flight at once")

    sem = asyncio.Semaphore(5)
    in_flight = {"current": 0, "peak": 0}

    async def call(i: int) -> str:
        async with sem:
            in_flight["current"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
            try:
                return await fake_llm_call(f"query {i}", sleep_s=0.1)
            finally:
                in_flight["current"] -= 1

    t0 = time.perf_counter()
    results = await asyncio.gather(*(call(i) for i in range(30)))
    elapsed = time.perf_counter() - t0
    console.print(
        f"30 calls done in [bold]{elapsed:.2f}s[/], "
        f"peak concurrency = [bold green]{in_flight['peak']}[/]  (cap was 5)"
    )
    console.print(f"sample result: {results[0]}")


# --- Scenario 4 · per-user lock map -----------------------------------------
async def scenario_per_user() -> None:
    """Two users, two messages each, sent in parallel.

    Within a user, messages must be processed in order. Across users,
    they should run concurrently.
    """
    section("4 · PER-USER LOCK MAP — serial within user, parallel across users")

    user_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    transcript: list[tuple[float, str]] = []
    start = time.perf_counter()

    async def handle(user_id: str, msg: str) -> None:
        async with user_locks[user_id]:
            t_in = time.perf_counter() - start
            transcript.append((t_in, f"{user_id}:start  {msg!r}"))
            await fake_llm_call(msg, sleep_s=0.2)
            t_out = time.perf_counter() - start
            transcript.append((t_out, f"{user_id}:done   {msg!r}"))

    # Two interleaved batches: alice gets two messages, bob gets two.
    await asyncio.gather(
        handle("alice", "hi"),
        handle("alice", "what's the refund policy?"),
        handle("bob",   "hello"),
        handle("bob",   "is my order shipped?"),
    )

    for t, line in transcript:
        console.print(f"  [dim]{t:5.2f}s[/]  {line}")
    console.print(
        "[yellow]→ Notice: alice's two messages never overlap, "
        "but alice and bob run in parallel.[/]"
    )


# --- entry point ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race", action="store_true")
    parser.add_argument("--mutex", action="store_true")
    parser.add_argument("--semaphore", action="store_true")
    parser.add_argument("--per-user", action="store_true")
    args = parser.parse_args()

    selected: list = []
    if args.race:      selected.append(scenario_race)
    if args.mutex:     selected.append(scenario_mutex)
    if args.semaphore: selected.append(scenario_semaphore)
    if args.per_user:  selected.append(scenario_per_user)
    if not selected:
        selected = [scenario_race, scenario_mutex, scenario_semaphore, scenario_per_user]

    async def run_all() -> None:
        for fn in selected:
            await fn()

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
