"""Shared builders for the integrated-agent tests (P12).

Canned observation streams reuse the P8 synthetic market (balanced 20-lot
book around mid 1000 with Poisson trade prints) so that the agent's belief
machinery sees exactly the pattern the proposer/workspace suites were
validated on. Feeding ``agent.cycle(obs)`` directly with a canned stream
isolates the loop from engine feedback: same observations in, so any
behavioral divergence is the agent's own.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from tests.proposer.conftest import active_obs
from topos.agent import AblationFlags, AgentConfig, ToposAgent
from topos.contracts.market import Observation

ROOT_SEED = 20260710


def canned_stream(
    n_steps: int, *, seed: int = 11, start_step: int = 0
) -> list[Observation]:
    """A deterministic synthetic active-market observation stream."""
    rng = np.random.default_rng(seed)
    return [active_obs(step, rng) for step in range(start_step, start_step + n_steps)]


def make_agent(
    flags: AblationFlags | None = None,
    config: AgentConfig | None = None,
    root_seed: int = ROOT_SEED,
) -> ToposAgent:
    return ToposAgent(config, root_seed=root_seed, flags=flags)


def run_canned(
    agent: ToposAgent, stream: list[Observation]
) -> None:
    """Drive the agent's cycle over a canned stream (no engine feedback)."""
    for obs in stream:
        agent.cycle(obs)


def spy(
    calls: list[str], obj: object, name: str, label: str
) -> Callable[..., Any]:
    """Shadow ``obj.name`` with a recording wrapper (instance attribute)."""
    orig = getattr(obj, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        calls.append(label)
        return orig(*args, **kwargs)

    setattr(obj, name, wrapped)
    return wrapped
