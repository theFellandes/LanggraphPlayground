"""Lesson 32 · Prompt engineering lab — runnable, offline.

Three demos:
    1. Registry mechanics — list/get/render across versions
    2. A/B routing — sticky-per-user variant assignment
    3. Eval-driven promotion — only promote if the new version beats current

Run:
    uv run python -m lessons.32_prompt_engineering_lab.example
    uv run python -m lessons.32_prompt_engineering_lab.example --registry
    uv run python -m lessons.32_prompt_engineering_lab.example --ab
    uv run python -m lessons.32_prompt_engineering_lab.example --promote
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from shared.pretty import console, section


@dataclass(frozen=True)
class PromptArtefact:
    name: str
    version: str
    template_source: str
    sha256: str
    metadata: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class PromptRegistry:
    def __init__(self, prompts_dir: Path):
        self.dir = prompts_dir
        self._build_env()
        self._cache: dict[tuple[str, str], PromptArtefact] = {}
        self._reload_lock = asyncio.Lock()
        self._current: dict[str, str] = {}              # name → version

    def _build_env(self):
        self._env = Environment(
            loader=FileSystemLoader(self.dir),
            autoescape=select_autoescape(disabled_extensions=("j2",)),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

    def list_versions(self, name: str) -> list[str]:
        return sorted(
            p.stem.split(".")[-1]
            for p in self.dir.glob(f"{name}.*.j2")
        )

    def set_current(self, name: str, version: str) -> None:
        assert version in self.list_versions(name)
        self._current[name] = version

    def _resolve(self, name: str, version: str) -> str:
        if version == "current":
            if name in self._current:
                return self._current[name]
            versions = self.list_versions(name)
            return versions[-1]   # latest by sort
        return version

    def get(self, name: str, version: str = "current") -> PromptArtefact:
        version = self._resolve(name, version)
        key = (name, version)
        if key not in self._cache:
            path = self.dir / f"{name}.{version}.j2"
            src = path.read_text(encoding="utf-8")
            sha = hashlib.sha256(src.encode()).hexdigest()[:16]
            self._cache[key] = PromptArtefact(name, version, src, sha)
        return self._cache[key]

    def render(self, name: str, *, version: str = "current", **vars) -> str:
        artefact = self.get(name, version)
        return self._env.from_string(artefact.template_source).render(**vars)

    async def reload(self) -> None:
        async with self._reload_lock:
            self._cache.clear()
            self._build_env()


# --- helper · seed a tmp dir with sample prompts ----------------------------
def _seed_prompts(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "researcher.v1.j2").write_text(
        "You research things. Answer briefly.\n",
        encoding="utf-8",
    )
    (target / "researcher.v2.j2").write_text(
        "You are a careful research analyst. "
        "Cite sources inline. Be concise but specific.\n"
        "User asked about: {{ topic }}\n",
        encoding="utf-8",
    )
    (target / "researcher.v3.j2").write_text(
        "You are a careful research analyst with a focus on factual grounding.\n"
        "Workflow:\n"
        "  1. Identify the user's intent.\n"
        "  2. Search for 3-5 sources.\n"
        "  3. For each, call cite(url, claim).\n"
        "  4. Return a tidy bullet list with inline citations.\n\n"
        "User asked about: {{ topic }}\n",
        encoding="utf-8",
    )


# --- Demo 1 · registry mechanics --------------------------------------------
def demo_registry(reg: PromptRegistry) -> None:
    section("1 · Registry mechanics")
    versions = reg.list_versions("researcher")
    console.print(f"available versions: {versions}")
    for v in versions:
        a = reg.get("researcher", version=v)
        console.print(f"  {a.version}  sha={a.sha256}  len={len(a.template_source)}")
    console.rule("[bold]render v2[/]")
    console.print(reg.render("researcher", version="v2", topic="quantum cooling"))


# --- Demo 2 · A/B routing ---------------------------------------------------
def assign_variant(user_id: str, variants: list[str]) -> str:
    h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    return variants[h % len(variants)]


def weighted_variant(user_id: str, weights: dict[str, int]) -> str:
    h = int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 100
    cum = 0
    for variant, w in weights.items():
        cum += w
        if h < cum:
            return variant
    return list(weights)[-1]


def demo_ab(reg: PromptRegistry) -> None:
    section("2 · A/B routing — sticky per user")

    users = [f"user_{i}@acme.com" for i in range(10)]
    console.rule("[bold]50/50 split (sticky by user)[/]")
    for u in users:
        v = assign_variant(u, ["v2", "v3"])
        console.print(f"  {u}: {v}")

    console.rule("[bold]Weighted: 80% v2, 20% v3 canary[/]")
    counts = {"v2": 0, "v3": 0}
    for i in range(1000):
        v = weighted_variant(f"u-{i}", {"v2": 80, "v3": 20})
        counts[v] += 1
    console.print(f"  observed: {counts}")


# --- Demo 3 · eval-driven promotion -----------------------------------------
def stub_eval_score(artefact: PromptArtefact) -> float:
    """Pretend longer + has 'Workflow' scores higher.

    A real eval would run the prompt against your eval set (lesson 26 Topic 4),
    score each output, and return a pass-rate. This stub is deterministic so the
    demo output is repeatable.
    """
    base = min(len(artefact.template_source) / 400, 1.0)
    if "Workflow" in artefact.template_source:
        base += 0.05
    return round(base, 3)


def promote_if_better(
    reg: PromptRegistry,
    name: str,
    candidate: str,
    current: str | None = None,
    epsilon: float = 0.02,
) -> str:
    current = current or reg._resolve(name, "current")
    cur_score = stub_eval_score(reg.get(name, version=current))
    new_score = stub_eval_score(reg.get(name, version=candidate))
    if new_score >= cur_score + epsilon:
        reg.set_current(name, candidate)
        return (
            f"PROMOTED {name}@{candidate}: new {new_score:.3f} >= current "
            f"{cur_score:.3f} + ε({epsilon})"
        )
    return (
        f"HOLDING {name}@{current}: candidate {candidate} scored {new_score:.3f} "
        f"< {cur_score:.3f} + ε({epsilon})"
    )


def demo_promote(reg: PromptRegistry) -> None:
    section("3 · Eval-driven promotion")
    reg.set_current("researcher", "v2")
    console.print(f"current → {reg._current['researcher']}")
    console.print(promote_if_better(reg, "researcher", candidate="v3"))
    console.print(f"current → {reg._current['researcher']}")
    # Try downgrade — v1 is shorter; should be rejected.
    console.print(promote_if_better(reg, "researcher", candidate="v1"))
    console.print(f"current → {reg._current['researcher']}  (unchanged)")


# --- entry point ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", action="store_true")
    parser.add_argument("--ab", action="store_true")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    selected = []
    if args.registry: selected.append(demo_registry)
    if args.ab:       selected.append(demo_ab)
    if args.promote:  selected.append(demo_promote)
    if not selected:
        selected = [demo_registry, demo_ab, demo_promote]

    tmp = Path(tempfile.mkdtemp(prefix="lesson32_"))
    try:
        _seed_prompts(tmp)
        reg = PromptRegistry(tmp)
        for fn in selected:
            fn(reg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
