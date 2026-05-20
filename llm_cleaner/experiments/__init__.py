"""Experiment registry.

Each experiment module exposes a top-level `run(ctx: RunContext) -> None`.
The registry below maps the --experiment <name> CLI flag to its module
so the harness can dispatch without code changes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable

from llm_cleaner.core import RunContext


EXPERIMENT_NAMES = (
    "exp01_core_matrix",
    "exp02_prompt_ablation",
    "exp03_batch_vs_single",
    "exp04_variance",
    "exp05_adversarial",
    "exp06_scope_scaling",
    "exp07_baselines",
    "exp08_calibration",
    "exp09_smoke",
)


def get_experiment(name: str) -> Callable[[RunContext], None]:
    if name not in EXPERIMENT_NAMES:
        raise KeyError(
            f"Unknown experiment {name!r}. Available: {', '.join(EXPERIMENT_NAMES)}"
        )
    module = import_module(f"llm_cleaner.experiments.{name}")
    fn = getattr(module, "run", None)
    if not callable(fn):
        raise RuntimeError(f"Experiment {name!r} has no callable run()")
    return fn
