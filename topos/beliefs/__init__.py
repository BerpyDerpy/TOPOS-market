"""World predictors (P4, P5, P11).

Kalman fair-value and Poisson-Gamma flow-intensity models, the shared
`BeliefModule` protocol and EIG machinery (P4), the queue-position filter
(P5), and the regime tracker with regime-gated forgetting (P11).

All adaptation is closed-form posterior updates plus forgetting (INV-2).
Curiosity quantities are mutual information on parameter posteriors,
never predictive variance or predictive entropy alone (INV-3).
"""

from topos.beliefs.core import (
    EIGTerms,
    GammaPosterior,
    InverseGammaPosterior,
    SurpriseTracker,
    forget_stats,
    gaussian_entropy_nats,
    information_gain_terms,
    negative_binomial_log_pmf,
    poisson_entropy_nats,
    quantile_quadrature,
)
from topos.beliefs.fair_value import FairValueKF, microprice_from_observation
from topos.beliefs.flow_intensity import BANDS, KINDS, FlowIntensity, band_of

__all__ = [
    "BANDS",
    "EIGTerms",
    "FairValueKF",
    "FlowIntensity",
    "GammaPosterior",
    "InverseGammaPosterior",
    "KINDS",
    "SurpriseTracker",
    "band_of",
    "forget_stats",
    "gaussian_entropy_nats",
    "information_gain_terms",
    "microprice_from_observation",
    "negative_binomial_log_pmf",
    "poisson_entropy_nats",
    "quantile_quadrature",
]
