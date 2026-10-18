"""Regime identification with a Gaussian hidden Markov model (methodology, Section 1).

:class:`RegimeModel` fits the model to factor returns and returns a
:class:`RegimeFit` holding the parameters, the posterior regime probabilities
and the one-step-ahead regime forecast. A fit can start from an earlier fit,
which is how the model is re-estimated date by date (Section 1.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans

from rwls.settings import RegimeSettings

__all__ = ["RegimeModel", "RegimeFit"]


class RegimeModel:
    """Gaussian hidden Markov model of factor returns.

    Parameters
    ----------
    settings : RegimeSettings, optional
        Model settings. Defaults to ``RegimeSettings()``.
    **overrides
        Individual settings to change, for example ``n_regimes=2``. They are
        applied on top of ``settings``.

    Examples
    --------
    >>> fit = RegimeModel(n_regimes=3).fit(factors)          # doctest: +SKIP
    >>> fit.posteriors.tail()                                 # doctest: +SKIP
    >>> fit.forecast()                                        # doctest: +SKIP
    """

    def __init__(self, settings: RegimeSettings | None = None, **overrides):
        settings = settings if settings is not None else RegimeSettings()
        self.settings = settings.replace(**overrides) if overrides else settings

    def fit(self, factors, start_from: RegimeFit | None = None) -> RegimeFit:
        """Estimate the model on ``factors``.

        Parameters
        ----------
        factors : pandas.DataFrame or array-like
            Factor returns, one row per period and one column per factor.
        start_from : RegimeFit, optional
            An earlier fit, typically on a shorter sample, to start from instead
            of a cold start. Its regime labels carry over. If the resulting fit
            fails the health check, the model is refitted from a cold start.

        Returns
        -------
        RegimeFit
        """
        values, index, names = _as_matrix(factors)
        s = self.settings
        if start_from is not None:
            if start_from.n_regimes != s.n_regimes or start_from.factor_names != names:
                raise ValueError("start_from must have the same regimes and factors")
            mdl = self._fit_warm(values, start_from._model)
            if not _healthy(mdl):
                mdl = self._fit_cold(values)
        else:
            mdl = self._fit_cold(values)
        return RegimeFit._from_model(mdl, values, index, names, s)

    def _build(self, n_features: int) -> GaussianHMM:
        s = self.settings
        if s.covariance == "diag":
            prior = s.variance_prior
        else:
            # For full matrices the library adds a scalar prior to every element,
            # off-diagonal ones included. The methodology puts it on the variances.
            prior = s.variance_prior * np.eye(n_features)
        return GaussianHMM(
            n_components=s.n_regimes,
            covariance_type=s.covariance,
            min_covar=s.init_variance_floor,
            covars_prior=prior,
            n_iter=s.max_iter,
            tol=s.tol,
            implementation="scaling",
        )

    def _fit_cold(self, values: np.ndarray) -> GaussianHMM:
        s = self.settings
        best, best_ll = None, -np.inf
        for j in range(s.n_starts):
            seed = s.seed + j
            mdl = self._build(values.shape[1])
            mdl.random_state = seed
            mdl.init_params = "tc"
            mdl.params = "tmc"
            mdl.startprob_ = np.full(s.n_regimes, 1.0 / s.n_regimes)
            kmeans = KMeans(
                n_clusters=s.n_regimes, random_state=seed, n_init=s.kmeans_restarts
            ).fit(values)
            mdl.means_ = kmeans.cluster_centers_
            mdl.fit(values)
            if not _healthy(mdl):
                continue
            ll = mdl.score(values)
            if ll > best_ll:
                best, best_ll = mdl, ll
        if best is None:
            raise RuntimeError("the regime model failed to converge from a cold start")
        return best

    def _fit_warm(self, values: np.ndarray, previous: GaussianHMM) -> GaussianHMM:
        s = self.settings
        mdl = self._build(values.shape[1])
        mdl.init_params = ""
        mdl.params = "tmc"
        mdl.startprob_ = np.full(s.n_regimes, 1.0 / s.n_regimes)
        trans = np.clip(previous.transmat_, s.transition_floor, None)
        mdl.transmat_ = trans / trans.sum(axis=1, keepdims=True)
        mdl.means_ = previous.means_.copy()
        if s.covariance == "diag":
            mdl.covars_ = np.diagonal(previous.covars_, axis1=1, axis2=2).copy()
        else:
            mdl.covars_ = previous.covars_.copy()
        mdl.fit(values)
        return mdl


@dataclass(frozen=True, eq=False)
class RegimeFit:
    """A fitted regime model.

    Attributes
    ----------
    settings : RegimeSettings
        Settings used for the fit.
    factor_names : list of str
        Names of the factors, in column order.
    index : pandas.Index
        Periods of the estimation sample.
    startprob : pandas.Series
        Initial regime distribution (fixed, uniform).
    transmat : pandas.DataFrame
        Transition matrix, rows indexed by the current regime.
    means : pandas.DataFrame
        Factor means, one row per regime.
    covariances : numpy.ndarray
        Factor covariance matrices, shape ``(n_regimes, K, K)``.
    posteriors : pandas.DataFrame
        Posterior regime probabilities given the whole estimation sample, one row
        per period (Section 1.4). The last row is the filtered probability.
    loglike : float
        Log-likelihood of the estimation sample.
    n_iter : int
        Number of EM iterations of the final fit.
    converged : bool
        Whether EM met the tolerance before reaching ``max_iter``.
    """

    settings: RegimeSettings
    factor_names: list[str]
    index: pd.Index
    startprob: pd.Series
    transmat: pd.DataFrame
    means: pd.DataFrame
    covariances: np.ndarray
    posteriors: pd.DataFrame
    loglike: float
    n_iter: int
    converged: bool
    _model: GaussianHMM = field(repr=False)

    @classmethod
    def _from_model(cls, mdl, values, index, names, settings) -> RegimeFit:
        regimes = pd.RangeIndex(mdl.n_components, name="regime")
        return cls(
            settings=settings,
            factor_names=list(names),
            index=index,
            startprob=pd.Series(mdl.startprob_, index=regimes),
            transmat=pd.DataFrame(mdl.transmat_, index=regimes, columns=regimes),
            means=pd.DataFrame(mdl.means_, index=regimes, columns=names),
            covariances=np.array(mdl.covars_),
            posteriors=pd.DataFrame(
                mdl.predict_proba(values), index=index, columns=regimes
            ),
            loglike=float(mdl.score(values)),
            n_iter=int(mdl.monitor_.iter),
            converged=bool(mdl.monitor_.converged),
            _model=mdl,
        )

    @property
    def n_regimes(self) -> int:
        return self.settings.n_regimes

    @property
    def nobs(self) -> int:
        return len(self.index)

    @property
    def variances(self) -> pd.DataFrame:
        """Factor variances, one row per regime."""
        diag = np.diagonal(self.covariances, axis1=1, axis2=2)
        return pd.DataFrame(diag, index=self.means.index, columns=self.factor_names)

    @property
    def filtered(self) -> pd.Series:
        """Filtered regime probabilities at the last period of the sample."""
        return self.posteriors.iloc[-1]

    def forecast(self, steps: int = 1) -> pd.Series:
        """Regime probabilities ``steps`` periods after the end of the sample.

        The filtered probabilities are propagated through the transition matrix,
        floored at ``forecast_floor`` and renormalized (Section 1.5).
        """
        if steps < 1:
            raise ValueError("steps must be at least 1")
        p = self.filtered.to_numpy() @ np.linalg.matrix_power(
            self.transmat.to_numpy(), steps
        )
        p = np.clip(p, self.settings.forecast_floor, None)
        return pd.Series(p / p.sum(), index=self.means.index, name="forecast")


def _healthy(mdl: GaussianHMM) -> bool:
    """Every parameter finite and every row of the transition matrix summing to one."""
    params = [mdl.startprob_, mdl.transmat_, mdl.means_, mdl.covars_]
    return all(np.all(np.isfinite(p)) for p in params) and np.allclose(
        mdl.transmat_.sum(axis=1), 1.0
    )


def _as_matrix(factors) -> tuple[np.ndarray, pd.Index, list[str]]:
    if isinstance(factors, pd.Series):
        factors = factors.to_frame()
    if isinstance(factors, pd.DataFrame):
        values = factors.to_numpy(dtype=float)
        index, names = factors.index, [str(c) for c in factors.columns]
    else:
        values = np.asarray(factors, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        index = pd.RangeIndex(len(values))
        names = [f"f{j}" for j in range(values.shape[1])]
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("factors must be a two-dimensional array with several rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("factors contain missing or infinite values")
    return values, index, names
