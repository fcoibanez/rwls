"""Settings for the regime model and the RWLS estimator.

Every constant used by the estimators is a field of one of the two classes below,
with the default documented in ``docs/methodology.md``. Settings are immutable;
use :meth:`RegimeSettings.replace` or :meth:`RWLSSettings.replace` to derive a
modified copy.
"""

from __future__ import annotations

import dataclasses
import numbers
from dataclasses import dataclass
from typing import Literal

__all__ = ["RegimeSettings", "RWLSSettings"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(value) -> bool:
    """True for Python and numpy integers, False for booleans."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


@dataclass(frozen=True)
class RegimeSettings:
    """Settings of the hidden Markov regime model (methodology, Section 1).

    Parameters
    ----------
    n_regimes : int
        Number of regimes ``M``.
    covariance : {"diag", "full"}
        Form of the within-regime factor covariance matrix.
    variance_prior : float
        Prior ``c_0`` added to the numerator of each variance update. Zero gives
        the maximum-likelihood estimates.
    tol : float
        EM stops when the log-likelihood improves by less than this amount.
    max_iter : int
        Maximum number of EM iterations.
    init_variance_floor : float
        Added to the full-sample variances that initialize every regime on a cold
        start. It affects the starting values only and is not a floor on the
        estimates.
    kmeans_restarts : int
        Number of k-means restarts used to initialize the means on a cold start.
    n_starts : int
        Number of cold starts. With more than one, each start uses a different
        seed and the fit with the highest log-likelihood is kept.
    seed : int
        Seed of the first cold start. Start ``j`` uses ``seed + j``.
    selection : {"best", "modal"}
        How a model is chosen among several cold starts. ``"best"`` keeps the
        fit with the highest likelihood. ``"modal"`` groups the fits that
        reached the same optimum and keeps the best fit of the group reached
        by the most starts.
    modal_order_factor : int or str
        Factor, by position or name, whose mean orders the regimes when fits
        are compared for ``"modal"`` selection. The default is the first
        column, conventionally the market.
    modal_mean_tol, modal_transition_tol : float
        Two fits reached the same optimum if every regime mean agrees within
        ``modal_mean_tol`` and every transition probability within
        ``modal_transition_tol``, with regimes ordered by
        ``modal_order_factor``.
    transition_floor : float
        Lower bound applied to transition probabilities when a fit is used as the
        starting point of the next one, before the rows are renormalized.
    forecast_floor : float
        Lower bound applied to forecast regime probabilities before they are
        renormalized.
    """

    n_regimes: int = 3
    covariance: Literal["diag", "full"] = "diag"
    variance_prior: float = 0.01
    tol: float = 1e-4
    max_iter: int = 1000
    init_variance_floor: float = 1e-3
    kmeans_restarts: int = 10
    n_starts: int = 10
    seed: int = 13
    selection: Literal["best", "modal"] = "best"
    modal_order_factor: int | str = 0
    modal_mean_tol: float = 0.0025
    modal_transition_tol: float = 0.05
    transition_floor: float = 1e-8
    forecast_floor: float = 1e-12

    def __post_init__(self) -> None:
        _require(
            _is_int(self.n_regimes) and self.n_regimes >= 1,
            "n_regimes must be a positive integer",
        )
        _require(
            self.covariance in ("diag", "full"), "covariance must be 'diag' or 'full'"
        )
        _require(self.variance_prior >= 0, "variance_prior must be non-negative")
        _require(self.tol > 0, "tol must be positive")
        _require(
            _is_int(self.max_iter) and self.max_iter >= 1,
            "max_iter must be a positive integer",
        )
        _require(
            self.init_variance_floor >= 0, "init_variance_floor must be non-negative"
        )
        _require(
            _is_int(self.kmeans_restarts) and self.kmeans_restarts >= 1,
            "kmeans_restarts must be a positive integer",
        )
        _require(
            _is_int(self.n_starts) and self.n_starts >= 1,
            "n_starts must be a positive integer",
        )
        _require(_is_int(self.seed), "seed must be an integer")
        _require(
            self.selection in ("best", "modal"), "selection must be 'best' or 'modal'"
        )
        _require(
            isinstance(self.modal_order_factor, str)
            or (_is_int(self.modal_order_factor) and self.modal_order_factor >= 0),
            "modal_order_factor must be a factor name or a non-negative position",
        )
        _require(self.modal_mean_tol > 0, "modal_mean_tol must be positive")
        _require(self.modal_transition_tol > 0, "modal_transition_tol must be positive")
        _require(0 <= self.transition_floor < 1, "transition_floor must be in [0, 1)")
        _require(0 <= self.forecast_floor < 1, "forecast_floor must be in [0, 1)")

    def replace(self, **changes) -> RegimeSettings:
        """Return a copy with the given fields changed."""
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class RWLSSettings:
    """Settings of the RWLS estimator (methodology, Section 2).

    Parameters
    ----------
    intercept : bool
        Whether each regression includes an intercept.
    weighting : {"posterior", "hard"}
        Observation weights: posterior regime probabilities (RWLS), or hard
        classification keeping each regime's most recent ``hard_window``
        classified observations.
    hard_window : int
        Window length ``L`` for hard classification, and for the sparse-regime
        fallback under hard classification.
    min_mass : float or None
        A regime whose total weight is at or below this value is treated as
        sparse. ``None`` uses ``k + 1``, where ``k`` is the number of coefficients
        per regression.
    fallback : {"auto", "pooled", "recent"}
        Weights given to a sparse regime. ``"pooled"`` uses every observation
        equally, so the regime takes the estimates of a model without regimes;
        ``"recent"`` uses the most recent ``hard_window`` observations. ``"auto"``
        is ``"pooled"`` under posterior weighting and ``"recent"`` under hard
        classification.
    use_t : bool
        Whether p-values and confidence intervals use Student's t distribution
        with ``n_s - k`` degrees of freedom. If False, the standard normal is used.
    """

    intercept: bool = True
    weighting: Literal["posterior", "hard"] = "posterior"
    hard_window: int = 24
    min_mass: float | None = None
    fallback: Literal["auto", "pooled", "recent"] = "auto"
    use_t: bool = True

    def __post_init__(self) -> None:
        _require(isinstance(self.intercept, bool), "intercept must be True or False")
        _require(
            self.weighting in ("posterior", "hard"),
            "weighting must be 'posterior' or 'hard'",
        )
        _require(
            _is_int(self.hard_window) and self.hard_window >= 1,
            "hard_window must be a positive integer",
        )
        _require(
            self.min_mass is None or self.min_mass >= 0,
            "min_mass must be None or non-negative",
        )
        _require(
            self.fallback in ("auto", "pooled", "recent"),
            "fallback must be 'auto', 'pooled' or 'recent'",
        )
        _require(isinstance(self.use_t, bool), "use_t must be True or False")

    def replace(self, **changes) -> RWLSSettings:
        """Return a copy with the given fields changed."""
        return dataclasses.replace(self, **changes)

    def resolved_fallback(self) -> Literal["pooled", "recent"]:
        """The fallback rule in effect, with ``"auto"`` resolved."""
        if self.fallback != "auto":
            return self.fallback
        return "pooled" if self.weighting == "posterior" else "recent"

    def resolved_min_mass(self, n_coefficients: int) -> float:
        """The sparse-regime threshold for regressions with ``n_coefficients``."""
        return float(n_coefficients + 1) if self.min_mass is None else self.min_mass
