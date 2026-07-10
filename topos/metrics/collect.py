"""Run collection: one ablation-condition episode reduced to a RunData.

This module is the ONLY place agent internals are read for measurement
(experiment ledger, impact posterior, message log); everything downstream
consumes the frozen ``RunData``. The harness-only channels (ground-truth
regimes, queue positions, engine accounts, the counterfactual twin)
terminate here and in the metric modules (INV-11).

Conditions
----------
``CONDITION_FLAGS`` is the canonical name -> AblationFlags map for the P13
ablation. All behavioral differences between conditions are injected ONLY
through the P12 flags; configs, root seeds, and regime schedules are
identical across conditions, so every comparison in this package is
seed-paired with an identical background event stream (INV-9 makes the
pairing exact until the agent's own footprint causes divergence).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator, Mapping

from topos.agent import AblationFlags, AgentConfig, ToposAgent
from topos.agent.ledger import ResolvedExperiment
from topos.contracts.market import ExchangeMessage
from topos.contracts.workspace import WorkspaceRecord
from topos.env.background import BackgroundConfig
from topos.env.harness import RunConfig, RunLog, StepRecord, null_agent, run

FULL = "FULL"
SURPRISE_CURIOSITY = "SURPRISE_CURIOSITY"
NO_SELF_MODEL = "NO_SELF_MODEL"
NO_REFLEXIVE = "NO_REFLEXIVE"
NO_HOMEOSTAT = "NO_HOMEOSTAT"

CONDITION_FLAGS: Mapping[str, AblationFlags] = MappingProxyType(
    {
        FULL: AblationFlags(),
        SURPRISE_CURIOSITY: AblationFlags(surprise_curiosity=True),
        NO_SELF_MODEL: AblationFlags(no_self_model=True),
        NO_REFLEXIVE: AblationFlags(no_reflexive=True),
        NO_HOMEOSTAT: AblationFlags(no_homeostat=True),
    }
)

CONDITIONS: tuple[str, ...] = tuple(CONDITION_FLAGS)


@dataclass(frozen=True)
class ImpactPosterior:
    """End-of-run snapshot of the agent's impact-model posterior.

    Enough to reproduce ``predictive_own_effect``: mean and scale-free
    covariance of w = [intercept, aggression, resting, *context], plus the
    posterior-mean noise scale E[sigma^2] and the model's fixed horizon.
    """

    coef_mean: tuple[float, ...]
    coef_scale_free_cov: tuple[tuple[float, ...], ...]
    noise_scale_mean: float
    horizon_steps: int

    def own_effect(
        self, aggression_lots: float, resting_touch_lots: float
    ) -> tuple[float, float]:
        """(mean, variance) of the own-action contribution to the h-step
        mid move — the same quantity ``ImpactModel.predictive_own_effect``
        answers, reproduced from the snapshot."""
        d = len(self.coef_mean)
        dx = [0.0] * d
        dx[1] = aggression_lots
        dx[2] = resting_touch_lots
        mean = sum(m * x for m, x in zip(self.coef_mean, dx))
        quad = 0.0
        for i in range(d):
            if dx[i] == 0.0:
                continue
            for j in range(d):
                if dx[j] == 0.0:
                    continue
                quad += dx[i] * self.coef_scale_free_cov[i][j] * dx[j]
        return mean, quad * self.noise_scale_mean


@dataclass(frozen=True)
class RunData:
    """One episode of one ablation condition, frozen for measurement.

    ``records[i + 1]`` is the cycle record computed from
    ``run_log.steps[i].observation`` (``records[0]`` belongs to the reset
    observation, which shares stamp 0 with the first step — DESIGN item
    33); ``paired_steps`` yields that alignment. ``twin_log`` is the
    null-agent counterfactual replay of the identical (config, root_seed)
    — the ground-truth channel for impact validation and for empirical
    background-flow rates.
    """

    condition: str
    root_seed: int
    run_log: RunLog
    records: tuple[WorkspaceRecord, ...]
    experiments: tuple[ResolvedExperiment, ...]
    message_log: tuple[tuple[int, ExchangeMessage], ...]
    impact_posterior: ImpactPosterior | None
    agent_config: AgentConfig
    twin_log: RunLog | None = None

    def paired_steps(self) -> Iterator[tuple[StepRecord, WorkspaceRecord]]:
        """(harness step record, agent cycle record) pairs, step-aligned."""
        for i, step in enumerate(self.run_log.steps):
            if i + 1 < len(self.records):
                yield step, self.records[i + 1]


def snapshot_impact_posterior(agent: ToposAgent) -> ImpactPosterior:
    cov = agent.impact.coef_scale_free_cov
    return ImpactPosterior(
        coef_mean=tuple(float(v) for v in agent.impact.coef_mean),
        coef_scale_free_cov=tuple(
            tuple(float(v) for v in row) for row in cov
        ),
        noise_scale_mean=float(agent.impact.noise_scale_posterior.mean()),
        horizon_steps=agent.impact.impact_horizon_steps,
    )


def collect_run(
    condition: str,
    root_seed: int,
    *,
    n_steps: int,
    background: BackgroundConfig | None = None,
    agent_config: AgentConfig | None = None,
    with_twin: bool = True,
) -> RunData:
    """Run one (condition, seed) episode and reduce it to a RunData.

    The agent and the environment share ``root_seed``: their RNG streams
    are disjoint by actor id (INV-9), and one seed per pair is what makes
    the paired-seed comparisons exact. The twin replay re-runs the same
    (config, root_seed) with the null agent — same regime chain, same
    background draws — so run-vs-twin divergence is the agent's causal
    footprint and nothing else.
    """
    if condition not in CONDITION_FLAGS:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {CONDITIONS}"
        )
    config = RunConfig(
        n_steps=n_steps,
        background=background if background is not None else BackgroundConfig(),
    )
    cfg = agent_config if agent_config is not None else AgentConfig()
    agent = ToposAgent(cfg, root_seed=root_seed, flags=CONDITION_FLAGS[condition])
    run_log = run(config, agent, root_seed)
    twin_log = run(config, null_agent, root_seed) if with_twin else None
    return RunData(
        condition=condition,
        root_seed=root_seed,
        run_log=run_log,
        records=agent.records,
        experiments=agent.ledger.log,
        message_log=agent.message_log,
        impact_posterior=snapshot_impact_posterior(agent),
        agent_config=cfg,
        twin_log=twin_log,
    )
