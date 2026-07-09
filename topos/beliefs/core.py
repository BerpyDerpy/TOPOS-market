"""Shared conjugate machinery for belief modules (P4; reused by P5, P6).

Everything a hypothesis-owning module needs to be a well-behaved Bayesian
citizen of the architecture lives here, implemented exactly once:

* the EIG identity (`information_gain_terms`) — mutual information between
  a PARAMETER posterior and a prospective observation (INV-3),
* fixed quantile quadrature over one-dimensional posteriors
  (`quantile_quadrature`) — no fitting loops, no adaptive integration,
* sufficient-statistic forgetting (`forget_stats`) — INV-2-compatible
  adaptation, driven externally (P11),
* EWMA-z-scored surprise (`SurpriseTracker`) — a SALIENCE input only,
* conjugate posterior cells: `GammaPosterior` (Poisson rates) and
  `InverseGammaPosterior` (Gaussian variance scales), each carrying its
  own entropy, credible intervals, quadrature, and EIG.

No torch/jax/tensorflow/sklearn (INV-2): numpy + scipy only, and all
posterior updates are closed-form conjugate increments.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

FloatArray = NDArray[np.float64]

LOG_2PIE = math.log(2.0 * math.pi * math.e)

# --------------------------------------------------------------------------
# The EIG identity (INV-3)
# --------------------------------------------------------------------------


class EIGTerms(NamedTuple):
    """The mutual-information decomposition behind every EIG number.

    ``eig_nats`` is the epistemic part (what the observation can teach us
    about the parameters); ``expected_conditional_entropy_nats`` is the
    aleatoric part (noise that persists even with parameters known).
    """

    eig_nats: float
    predictive_entropy_nats: float
    expected_conditional_entropy_nats: float


def information_gain_terms(
    predictive_entropy_nats: float,
    conditional_entropies_nats: FloatArray,
    weights: FloatArray,
) -> EIGTerms:
    """THE EIG identity, implemented once and reused everywhere:

        EIG(probe) = H[Y | probe] - E_{theta ~ posterior} H[Y | theta, probe]

    This mutual-information form IS the epistemic/aleatoric split (INV-3).
    Predictive entropy alone is FORBIDDEN as a curiosity quantity anywhere
    in this codebase — it enters only as the first term of this difference.

    ``conditional_entropies_nats`` are H[Y | theta_i, probe] evaluated at
    posterior quadrature nodes with the given (normalized) ``weights``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    conditional = np.asarray(conditional_entropies_nats, dtype=np.float64)
    if weights.shape != conditional.shape:
        raise ValueError(
            f"weights {weights.shape} and conditional entropies "
            f"{conditional.shape} must align"
        )
    total_weight = float(weights.sum())
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"quadrature weights must sum to 1, got {total_weight}")
    aleatoric = float(np.dot(weights, conditional))
    eig = predictive_entropy_nats - aleatoric
    return EIGTerms(
        eig_nats=eig,
        predictive_entropy_nats=predictive_entropy_nats,
        expected_conditional_entropy_nats=aleatoric,
    )


# --------------------------------------------------------------------------
# Fixed quadrature over one-dimensional posteriors
# --------------------------------------------------------------------------

_PANEL_EDGES: tuple[float, ...] = (
    0.0,
    1e-7,
    1e-5,
    1e-3,
    1e-2,
    0.05,
    0.2,
    0.5,
    0.8,
    0.95,
    0.99,
    1.0 - 1e-3,
    1.0 - 1e-5,
    1.0 - 1e-7,
    1.0,
)
_NODES_PER_PANEL = 16


def quantile_quadrature(
    ppf: Callable[[FloatArray], FloatArray],
    nodes_per_panel: int = _NODES_PER_PANEL,
) -> tuple[FloatArray, FloatArray]:
    """Fixed composite Gauss-Legendre quadrature in quantile space.

    For a scalar posterior with quantile function ``ppf``,
    ``E[f(theta)] = ∫_0^1 f(ppf(u)) du ≈ Σ_i w_i f(theta_i)`` with
    ``theta_i = ppf(u_i)``. Panels refine geometrically toward both
    endpoints so mild log-type tail singularities (e.g. E[ln theta]) are
    integrated accurately for any shape parameter; interior Gauss nodes
    never evaluate ``ppf`` at exactly 0 or 1. Weights sum to 1.
    """
    base_x, base_w = special.roots_legendre(nodes_per_panel)
    us: list[FloatArray] = []
    ws: list[FloatArray] = []
    for lo, hi in zip(_PANEL_EDGES[:-1], _PANEL_EDGES[1:]):
        half = 0.5 * (hi - lo)
        us.append(lo + half * (base_x + 1.0))
        ws.append(base_w * half)
    u = np.concatenate(us)
    w = np.concatenate(ws)
    nodes = np.asarray(ppf(u), dtype=np.float64)
    return nodes, w


# --------------------------------------------------------------------------
# Forgetting (INV-2-compatible adaptation; P11 drives rho)
# --------------------------------------------------------------------------


def forget_stats(
    stats_now: FloatArray, prior_stats: FloatArray, rho: float
) -> FloatArray:
    """Discount sufficient statistics toward the prior: S <- rho*S + (1-rho)*S0.

    ``rho`` in (0, 1]; ``rho == 1`` is a no-op. Equivalent to scaling every
    accumulated data increment by ``rho``, so in the converged regime
    posterior entropy is non-decreasing under forgetting (the property
    tests/beliefs/test_forgetting_reinflates.py pins down).
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    stats_now = np.asarray(stats_now, dtype=np.float64)
    prior_stats = np.asarray(prior_stats, dtype=np.float64)
    result: FloatArray = rho * stats_now + (1.0 - rho) * prior_stats
    return result


# --------------------------------------------------------------------------
# Surprise (salience only)
# --------------------------------------------------------------------------


class SurpriseTracker:
    """z-scores negative log predictive probability against its own EWMA.

    Surprise is a SALIENCE input only: it feeds the workspace headline and
    salience competition, and it must not appear in any EIG or
    action-scoring code path. It is retrospective prediction error — the
    exact thing INV-3 forbids as a curiosity quantity — which is why it is
    quarantined in its own tracker with no coupling to the posteriors.
    """

    def __init__(self, ewma_decay: float = 0.05, warmup: int = 2) -> None:
        if not 0.0 < ewma_decay < 1.0:
            raise ValueError(f"ewma_decay must be in (0, 1), got {ewma_decay}")
        if warmup < 2:
            raise ValueError(f"warmup must be >= 2, got {warmup}")
        self._decay = ewma_decay
        self._warmup = warmup
        self._count = 0
        self._mean = 0.0
        self._var = 0.0
        self._last_z = 0.0

    @property
    def last_z(self) -> float:
        return self._last_z

    def score(self, nll_nats: float) -> float:
        """z-score ``nll_nats`` against the EWMA history, then fold it in.

        The first ``warmup`` observations score 0 (no history to compare
        against yet).
        """
        if self._count >= self._warmup:
            z = (nll_nats - self._mean) / math.sqrt(self._var + 1e-12)
        else:
            z = 0.0
        if self._count == 0:
            self._mean = nll_nats
            self._var = 0.0
        else:
            beta = self._decay
            delta = nll_nats - self._mean
            self._mean += beta * delta
            self._var = (1.0 - beta) * (self._var + beta * delta * delta)
        self._count += 1
        self._last_z = z
        return z


# --------------------------------------------------------------------------
# Entropy helpers
# --------------------------------------------------------------------------


def gaussian_entropy_nats(variance: float | FloatArray) -> float | FloatArray:
    """H[N(mu, variance)] = 0.5 ln(2*pi*e*variance)."""
    return 0.5 * (LOG_2PIE + np.log(variance))


def bernoulli_entropy_nats(p: float) -> float:
    """H[Bernoulli(p)] = -p ln p - (1-p) ln(1-p), with 0 ln 0 = 0."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    h = 0.0
    if p > 0.0:
        h -= p * math.log(p)
    if p < 1.0:
        h -= (1.0 - p) * math.log(1.0 - p)
    return h


def poisson_entropy_nats(mus: FloatArray) -> FloatArray:
    """Entropy of Poisson(mu) for an array of means, by truncated summation.

    The support is truncated where the pmf is negligible (tail mass below
    ~1e-12 of the entropy scale), which is exact to floating precision for
    the moderate means seen in per-step count models.
    """
    mus = np.asarray(mus, dtype=np.float64)
    out = np.zeros_like(mus)
    positive = mus > 0.0
    if not np.any(positive):
        return out
    mu_pos = mus[positive]
    k_max = int(np.ceil(np.max(mu_pos) + 12.0 * np.sqrt(np.max(mu_pos)) + 25.0))
    k = np.arange(k_max + 1, dtype=np.float64)
    log_pmf = stats.poisson.logpmf(k[np.newaxis, :], mu_pos[:, np.newaxis])
    pmf = np.exp(log_pmf)
    out[positive] = -np.sum(np.where(pmf > 0.0, pmf * log_pmf, 0.0), axis=1)
    return out


def negative_binomial_log_pmf(
    y: FloatArray, shape_a: float, success_p: float
) -> FloatArray:
    """log pmf of NB(a, p): p(y) = C(y+a-1, y) p^a (1-p)^y, real a > 0.

    This is the Gamma-Poisson predictive: rate ~ Gamma(a, b), count over
    exposure dt => NB with success_p = b / (b + dt).
    """
    y = np.asarray(y, dtype=np.float64)
    result: FloatArray = (
        special.gammaln(y + shape_a)
        - special.gammaln(shape_a)
        - special.gammaln(y + 1.0)
        + shape_a * math.log(success_p)
        + y * math.log1p(-success_p)
    )
    return result


# --------------------------------------------------------------------------
# Conjugate cells
# --------------------------------------------------------------------------


class GammaPosterior:
    """Gamma(a, b) posterior over a Poisson rate lambda (b is the RATE).

    Counts y ~ Poisson(lambda * dt) update (a, b) -> (a + y, b + dt); the
    predictive over exposure dt is NegativeBinomial(a, b / (b + dt)).
    (a, b) are the accumulated sufficient statistics (prior pseudo-counts
    included), so `forget` discounts them straight toward (a0, b0).
    """

    def __init__(self, prior_a: float, prior_b: float) -> None:
        if prior_a <= 0.0 or prior_b <= 0.0:
            raise ValueError(f"prior must be positive, got a={prior_a}, b={prior_b}")
        self._a0 = float(prior_a)
        self._b0 = float(prior_b)
        self.a = float(prior_a)
        self.b = float(prior_b)

    @property
    def prior_a(self) -> float:
        return self._a0

    @property
    def prior_b(self) -> float:
        return self._b0

    def observe(self, count: int, exposure: float) -> None:
        """Conjugate update from ``count`` events over ``exposure`` time."""
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if exposure <= 0.0:
            raise ValueError(f"exposure must be > 0, got {exposure}")
        self.a += float(count)
        self.b += float(exposure)

    def forget(self, rho: float) -> None:
        forgotten = forget_stats(
            np.array([self.a, self.b]), np.array([self._a0, self._b0]), rho
        )
        self.a, self.b = float(forgotten[0]), float(forgotten[1])

    def mean(self) -> float:
        return self.a / self.b

    def variance(self) -> float:
        return self.a / (self.b * self.b)

    def entropy_nats(self) -> float:
        return float(stats.gamma.entropy(self.a, scale=1.0 / self.b))

    def interval(self, level: float) -> tuple[float, float]:
        """Central credible interval at the given level (e.g. 0.9)."""
        tail = 0.5 * (1.0 - level)
        lo = float(stats.gamma.ppf(tail, self.a, scale=1.0 / self.b))
        hi = float(stats.gamma.ppf(1.0 - tail, self.a, scale=1.0 / self.b))
        return lo, hi

    def quadrature(self) -> tuple[FloatArray, FloatArray]:
        """Fixed quadrature nodes/weights over the rate posterior."""

        def ppf(u: FloatArray) -> FloatArray:
            values: FloatArray = stats.gamma.ppf(u, self.a, scale=1.0 / self.b)
            return values

        return quantile_quadrature(ppf)

    def predictive_log_pmf(self, y: FloatArray, exposure: float) -> FloatArray:
        """log NB predictive pmf of counts over ``exposure``."""
        return negative_binomial_log_pmf(y, self.a, self.b / (self.b + exposure))

    def predictive_entropy_nats(self, exposure: float) -> float:
        """Entropy of the exact NB predictive, by truncated summation.

        The support is grown (doubled, a bounded number of times) until the
        captured mass is within 1e-11 of 1, so heavy-tailed low-shape
        predictives are summed to floating precision too.
        """
        mean = self.a * exposure / self.b
        var = mean * (1.0 + exposure / self.b)
        y_max = int(math.ceil(mean + 12.0 * math.sqrt(var) + 30.0))
        for _ in range(8):
            y = np.arange(y_max + 1, dtype=np.float64)
            log_pmf = self.predictive_log_pmf(y, exposure)
            pmf = np.exp(log_pmf)
            if float(pmf.sum()) >= 1.0 - 1e-11:
                return float(-np.sum(np.where(pmf > 0.0, pmf * log_pmf, 0.0)))
            y_max *= 2
        raise RuntimeError("negative-binomial support truncation too tight")

    def eig_terms(self, exposure: float) -> EIGTerms:
        """I(lambda; Y) for counts over ``exposure``, via the shared identity.

        H[Y] comes from the exact NB predictive; E_lambda H[Poisson(lambda *
        exposure)] by fixed quadrature over the Gamma posterior.
        """
        if exposure <= 0.0:
            raise ValueError(f"exposure must be > 0, got {exposure}")
        nodes, weights = self.quadrature()
        conditional = poisson_entropy_nats(nodes * exposure)
        return information_gain_terms(
            self.predictive_entropy_nats(exposure), conditional, weights
        )


class InverseGammaPosterior:
    """InverseGamma(a, b) posterior over a Gaussian variance scale c.

    Each standardized squared innovation z = nu^2 / S* (innovation nu with
    scale-free variance S*, so nu | c ~ N(0, c*S*)) updates
    (a, b) -> (a + 1/2, b + z/2) — the exact conjugate update of the
    unknown-scale linear-Gaussian model. `forget` discounts (a, b) toward
    the prior.
    """

    def __init__(self, prior_a: float, prior_b: float) -> None:
        if prior_a <= 1.0 or prior_b <= 0.0:
            raise ValueError(
                f"prior must have a > 1 (finite mean) and b > 0, "
                f"got a={prior_a}, b={prior_b}"
            )
        self._a0 = float(prior_a)
        self._b0 = float(prior_b)
        self.a = float(prior_a)
        self.b = float(prior_b)

    @property
    def prior_a(self) -> float:
        return self._a0

    @property
    def prior_b(self) -> float:
        return self._b0

    def observe_standardized_square(self, z: float) -> None:
        """Conjugate update from one standardized squared innovation."""
        if z < 0.0:
            raise ValueError(f"squared innovation must be >= 0, got {z}")
        self.a += 0.5
        self.b += 0.5 * z

    def forget(self, rho: float) -> None:
        forgotten = forget_stats(
            np.array([self.a, self.b]), np.array([self._a0, self._b0]), rho
        )
        self.a, self.b = float(forgotten[0]), float(forgotten[1])

    def mean(self) -> float:
        return self.b / (self.a - 1.0)

    def entropy_nats(self) -> float:
        return float(stats.invgamma.entropy(self.a, scale=self.b))

    def interval(self, level: float) -> tuple[float, float]:
        """Central credible interval at the given level (e.g. 0.9)."""
        tail = 0.5 * (1.0 - level)
        lo = float(stats.invgamma.ppf(tail, self.a, scale=self.b))
        hi = float(stats.invgamma.ppf(1.0 - tail, self.a, scale=self.b))
        return lo, hi

    def quadrature(self) -> tuple[FloatArray, FloatArray]:
        """Fixed quadrature nodes/weights over the scale posterior."""

        def ppf(u: FloatArray) -> FloatArray:
            values: FloatArray = stats.invgamma.ppf(u, self.a, scale=self.b)
            return values

        return quantile_quadrature(ppf)

    def student_t_predictive(self, scale_free_var: float) -> tuple[float, float]:
        """(df, scale) of the exact predictive of y | c ~ N(mu, c * s0).

        Marginalizing c ~ IG(a, b) gives Student-t with df = 2a and
        scale = sqrt(b * s0 / a) around the same location.
        """
        if scale_free_var <= 0.0:
            raise ValueError(f"scale_free_var must be > 0, got {scale_free_var}")
        return 2.0 * self.a, math.sqrt(self.b * scale_free_var / self.a)

    def eig_terms_for_gaussian(self, scale_free_var: float) -> EIGTerms:
        """I(c; Y) for a single observation Y | c ~ N(mu, c * scale_free_var).

        H[Y] is the exact Student-t predictive entropy; E_c H[Y | c] by
        fixed quadrature over the inverse-gamma posterior.
        """
        df, t_scale = self.student_t_predictive(scale_free_var)
        predictive_entropy = float(stats.t.entropy(df, scale=t_scale))
        nodes, weights = self.quadrature()
        conditional = np.asarray(
            gaussian_entropy_nats(nodes * scale_free_var), dtype=np.float64
        )
        return information_gain_terms(predictive_entropy, conditional, weights)


class BetaPosterior:
    """Beta(a, b) posterior over a Bernoulli success probability p.

    Outcomes may be FRACTIONAL (e.g. an order partially filled by its
    horizon): observing fraction f in [0, 1] updates
    (a, b) -> (a + f, b + (1 - f)) — one pseudo-trial per outcome, split
    between the success and failure counts. (a, b) are the accumulated
    sufficient statistics (prior pseudo-counts included), so `forget`
    discounts them straight toward (a0, b0).

    The Bernoulli parameter EIG I(p; Y) has a fully closed form via
    digamma (see ``eig_terms_bernoulli``); no quadrature is needed.
    """

    def __init__(self, prior_a: float, prior_b: float) -> None:
        if prior_a <= 0.0 or prior_b <= 0.0:
            raise ValueError(f"prior must be positive, got a={prior_a}, b={prior_b}")
        self._a0 = float(prior_a)
        self._b0 = float(prior_b)
        self.a = float(prior_a)
        self.b = float(prior_b)

    @property
    def prior_a(self) -> float:
        return self._a0

    @property
    def prior_b(self) -> float:
        return self._b0

    def observe(self, success_fraction: float) -> None:
        """Conjugate update from one (possibly fractional) trial outcome."""
        if not 0.0 <= success_fraction <= 1.0:
            raise ValueError(
                f"success_fraction must be in [0, 1], got {success_fraction}"
            )
        self.a += success_fraction
        self.b += 1.0 - success_fraction

    def forget(self, rho: float) -> None:
        forgotten = forget_stats(
            np.array([self.a, self.b]), np.array([self._a0, self._b0]), rho
        )
        self.a, self.b = float(forgotten[0]), float(forgotten[1])

    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def variance(self) -> float:
        n = self.a + self.b
        return self.a * self.b / (n * n * (n + 1.0))

    def entropy_nats(self) -> float:
        return float(stats.beta.entropy(self.a, self.b))

    def interval(self, level: float) -> tuple[float, float]:
        """Central credible interval at the given level (e.g. 0.9)."""
        tail = 0.5 * (1.0 - level)
        lo = float(stats.beta.ppf(tail, self.a, self.b))
        hi = float(stats.beta.ppf(1.0 - tail, self.a, self.b))
        return lo, hi

    def quadrature(self) -> tuple[FloatArray, FloatArray]:
        """Fixed quadrature nodes/weights over the probability posterior."""

        def ppf(u: FloatArray) -> FloatArray:
            values: FloatArray = stats.beta.ppf(u, self.a, self.b)
            return values

        return quantile_quadrature(ppf)

    def eig_terms_bernoulli(self) -> EIGTerms:
        """I(p; Y) for a single Bernoulli trial Y | p, fully closed form.

        H[Y] is the binary entropy of the predictive mean a/(a+b);
        E_p H[Y | p] uses E[p ln p] = E[p] (psi(a+1) - psi(a+b+1)) and
        E[(1-p) ln(1-p)] = E[1-p] (psi(b+1) - psi(a+b+1)) — the standard
        Beta log-moment identities via digamma.
        """
        p_bar = self.mean()
        predictive_entropy = bernoulli_entropy_nats(p_bar)
        psi_total = float(special.digamma(self.a + self.b + 1.0))
        e_p_ln_p = p_bar * (float(special.digamma(self.a + 1.0)) - psi_total)
        e_q_ln_q = (1.0 - p_bar) * (float(special.digamma(self.b + 1.0)) - psi_total)
        conditional = -(e_p_ln_p + e_q_ln_q)
        return EIGTerms(
            eig_nats=predictive_entropy - conditional,
            predictive_entropy_nats=predictive_entropy,
            expected_conditional_entropy_nats=conditional,
        )
