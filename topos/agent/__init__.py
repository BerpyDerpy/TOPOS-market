"""The integrated cognitive loop (P12).

perceive -> appraise -> compete -> broadcast -> propose -> ignite intent ->
act -> observe -> update posteriors. Realized information gain is computed
from entropy snapshots taken immediately before and after each
outcome-driven update (INV-10).
"""

from topos.agent.ablations import (
    AblationFlags,
    FrozenFillModel,
    FrozenImpactModel,
    NoReflexiveSelection,
    NullDistanceProjector,
    SurpriseAsCuriosity,
    VetoOnlyHomeostat,
)
from topos.agent.config import AgentConfig
from topos.agent.core import StepHandle, ToposAgent
from topos.agent.ledger import (
    ExperimentLedger,
    OpenExperiment,
    ResolvedExperiment,
)
from topos.agent.summary import SlowStats, WorldSummaryTracker
from topos.agent.wiring import (
    BandDistanceProjector,
    assert_registry_covers_known_ids,
    default_registry,
)

__all__ = [
    "AblationFlags",
    "AgentConfig",
    "BandDistanceProjector",
    "ExperimentLedger",
    "FrozenFillModel",
    "FrozenImpactModel",
    "NoReflexiveSelection",
    "NullDistanceProjector",
    "OpenExperiment",
    "ResolvedExperiment",
    "SlowStats",
    "StepHandle",
    "SurpriseAsCuriosity",
    "ToposAgent",
    "VetoOnlyHomeostat",
    "WorldSummaryTracker",
    "assert_registry_covers_known_ids",
    "default_registry",
]
