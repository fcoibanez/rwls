"""Regime-weighted least squares (methodology, Section 2).

:class:`RWLS` estimates a factor model for every asset in every regime by
weighted least squares, weighting each period by its posterior regime
probability, and returns an :class:`RWLSResults`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rwls.regimes import RegimeFit
from rwls.settings import RWLSSettings

__all__ = ["RWLS", "RWLSResults", "SparseRegimeWarning"]


class SparseRegimeWarning(UserWarning):
    """A regime had too little weight to identify its regression (Section 2.4)."""


class RWLS:
    """Regime-weighted least squares.

    Parameters
    ----------
    returns : pandas.DataFrame or pandas.Series
        Excess returns, one column per asset. Missing values are allowed: each
        asset is estimated on the periods it is observed.
    factors : pandas.DataFrame
        Factor returns, one column per factor.
    regimes : RegimeFit or pandas.DataFrame
        A fitted regime model, or regime probabilities with one column per regime
        and one row per period.
    settings : RWLSSettings, optional
        Estimator settings. Defaults to ``RWLSSettings()``.
    **overrides
        Individual settings to change, for example ``intercept=False``.

    Notes
    -----
    The three inputs are aligned on the periods they have in common. When
    ``regimes`` is a :class:`RegimeFit`, its smoothed probabilities are used as
    weights, so a model fitted on a longer sample than ``returns`` covers can be
    used directly.

    Examples
    --------
    >>> regimes = RegimeModel(n_regimes=3).fit(factors)           # doctest: +SKIP
    >>> res = RWLS(excess_returns, factors, regimes).fit()        # doctest: +SKIP
    >>> res.params["AAPL"]                                        # doctest: +SKIP
    """

    def __init__(
        self,
        returns,
        factors,
        regimes: RegimeFit | pd.DataFrame,
        settings: RWLSSettings | None = None,
        **overrides,
    ):
        settings = settings if settings is not None else RWLSSettings()
        self.settings = settings.replace(**overrides) if overrides else settings
        self.regimes = regimes if isinstance(regimes, RegimeFit) else None
        self.returns, self.factors, self.probabilities = _align(
            returns, factors, regimes
        )

    def fit(self) -> RWLSResults:
        """Estimate the regime-specific factor models of every asset."""
        s = self.settings
        F = self.factors.to_numpy(dtype=float)
        P = self.probabilities.to_numpy(dtype=float)
        R = self.returns.to_numpy(dtype=float)
        T, M = P.shape
        X = np.column_stack([np.ones(T), F]) if s.intercept else F
        k = X.shape[1]
        min_mass = s.resolved_min_mass(k)
        fallback_rule = s.resolved_fallback()

        # Factor moments use every period of the common sample (Section 2.5).
        full = np.ones(T, dtype=bool)
        fw, factor_fallback = _weights(P, full, s, min_mass, fallback_rule)
        cov_type = self.regimes.settings.covariance if self.regimes else "diag"
        means, covs = zip(*(_factor_moments(F, w, cov_type) for w in fw), strict=True)

        N = R.shape[1]
        coefs = np.full((M, k, N), np.nan)
        resid_var = np.full((M, N), np.nan)
        mass = np.full((M, N), np.nan)
        fallback = np.zeros((M, N), dtype=bool)
        groups = []
        observed = np.isfinite(R)
        for rows, cols in _patterns(observed):
            if rows.sum() <= k:
                continue
            w_list, used_fallback = _weights(P, rows, s, min_mass, fallback_rule)
            Xg, Yg = X[rows], R[np.ix_(rows, cols)]
            for j, w in enumerate(w_list):
                wg = w[rows]
                theta, var = _wls(Xg, Yg, wg)
                coefs[j][:, cols] = theta
                resid_var[j, cols] = var
                mass[j, cols] = wg.sum()
                fallback[j, cols] = used_fallback[j]
            groups.append((rows, cols, [w[rows] for w in w_list]))

        n_fallback = int(fallback.sum()) + int(factor_fallback.sum())
        if n_fallback:
            warnings.warn(
                f"{n_fallback} regime estimate(s) had a posterior mass at or below "
                f"{min_mass:g} and used the '{fallback_rule}' fallback; see "
                "RWLSResults.fallback",
                SparseRegimeWarning,
                stacklevel=2,
            )

        return RWLSResults._build(
            self,
            X,
            coefs,
            resid_var,
            mass,
            fallback,
            means,
            covs,
            factor_fallback,
            groups,
        )


@dataclass(frozen=True, eq=False)
class RWLSResults:
    """Estimates of a regime-weighted least-squares fit.

    Attributes
    ----------
    settings : RWLSSettings
        Settings used for the fit.
    params : pandas.DataFrame
        Coefficients, one column per asset, rows indexed by regime and term
        (``const`` and the factor names). ``params["AAPL"]`` gives one asset.
    resid_var : pandas.DataFrame
        Residual variances, one row per regime and one column per asset.
    mass : pandas.DataFrame
        Total weight of each regime in each asset's regression: the posterior
        mass under posterior weighting, the number of periods used under hard
        classification or after a fallback.
    fallback : pandas.DataFrame
        True where a regime was too sparse and its fallback weights were used.
    nobs : pandas.Series
        Number of periods each asset is observed.
    factor_means : pandas.DataFrame
        Factor means, one row per regime (Section 2.5).
    factor_covariances : numpy.ndarray
        Factor covariance matrices, shape ``(n_regimes, K, K)``, in regime order.
    factor_fallback : pandas.Series
        True for regimes whose factor moments used fallback weights.
    """

    settings: RWLSSettings
    index: pd.Index
    regime_labels: list
    factor_names: list[str]
    asset_names: list
    params: pd.DataFrame
    resid_var: pd.DataFrame
    mass: pd.DataFrame
    fallback: pd.DataFrame
    nobs: pd.Series
    factor_means: pd.DataFrame
    factor_covariances: np.ndarray
    factor_fallback: pd.Series
    _model: RWLS = field(repr=False)
    _design: np.ndarray = field(repr=False)
    _groups: list = field(repr=False)

    @classmethod
    def _build(
        cls,
        model,
        X,
        coefs,
        resid_var,
        mass,
        fallback,
        means,
        covs,
        factor_fallback,
        groups,
    ) -> RWLSResults:
        s = model.settings
        labels = list(model.probabilities.columns)
        fnames = [str(c) for c in model.factors.columns]
        assets = list(model.returns.columns)
        terms = (["const"] if s.intercept else []) + fnames
        rows = pd.MultiIndex.from_product([labels, terms], names=["regime", "term"])
        params = pd.DataFrame(
            coefs.reshape(len(labels) * len(terms), -1), index=rows, columns=assets
        )
        by_regime = pd.Index(labels, name="regime")

        def per_regime(values):
            return pd.DataFrame(values, index=by_regime, columns=assets)

        return cls(
            settings=s,
            index=model.returns.index,
            regime_labels=labels,
            factor_names=fnames,
            asset_names=assets,
            params=params,
            resid_var=per_regime(resid_var),
            mass=per_regime(mass),
            fallback=per_regime(fallback),
            nobs=model.returns.notna().sum().rename("nobs"),
            factor_means=pd.DataFrame(np.array(means), index=by_regime, columns=fnames),
            factor_covariances=np.array(covs),
            factor_fallback=pd.Series(
                factor_fallback, index=by_regime, name="fallback"
            ),
            _model=model,
            _design=X,
            _groups=groups,
        )

    @property
    def n_regimes(self) -> int:
        return len(self.regime_labels)

    @property
    def intercepts(self) -> pd.DataFrame:
        """Intercepts, one row per regime and one column per asset."""
        if not self.settings.intercept:
            raise AttributeError("the model was fitted without an intercept")
        return self.params.xs("const", level="term")

    @property
    def loadings(self) -> pd.DataFrame:
        """Factor loadings, rows indexed by regime and factor."""
        if not self.settings.intercept:
            return self.params
        return self.params.drop(index="const", level="term")


# ----------------------------------------------------------------------- weights


def _weights(P, rows, s: RWLSSettings, min_mass, fallback_rule):
    """Observation weights of every regime on ``rows``, with the sparse guard.

    Returns full-length weight vectors (zero outside ``rows``) and, per regime,
    whether the fallback was used.
    """
    T, M = P.shape
    idx = np.flatnonzero(rows)
    if s.weighting == "posterior":
        base = [np.where(rows, P[:, j], 0.0) for j in range(M)]
    else:
        labels = P[idx].argmax(axis=1)
        base = []
        for j in range(M):
            w = np.zeros(T)
            w[idx[np.flatnonzero(labels == j)[-s.hard_window :]]] = 1.0
            base.append(w)

    out, used = [], []
    for w in base:
        if w.sum() > min_mass:
            out.append(w)
            used.append(False)
            continue
        fb = np.zeros(T)
        if fallback_rule == "pooled":
            fb[idx] = 1.0
        else:
            fb[idx[-s.hard_window :]] = 1.0
        out.append(fb)
        used.append(True)
    return out, np.array(used)


# ------------------------------------------------------------------- regressions


def _wls(X, Y, w):
    """Weighted least squares of every column of ``Y`` on ``X``, solved jointly.

    Returns the ``(k, N)`` coefficients and the ``(N,)`` residual variances,
    measured on the total weight (Section 2.3).
    """
    Xw = X * w[:, None]
    theta = np.linalg.solve(X.T @ Xw, Xw.T @ Y)
    resid = Y - X @ theta
    dof = max(w.sum() - X.shape[1], 1.0)
    var = np.maximum((w[:, None] * resid**2).sum(axis=0) / dof, 0.0)
    return theta, var


def _factor_moments(F, w, covariance="diag"):
    """Weighted factor mean and covariance (Section 2.5)."""
    total = w.sum()
    mean = (w[:, None] * F).sum(axis=0) / total
    dev = F - mean
    cov = (w[:, None] * dev).T @ dev / total
    if covariance == "diag":
        cov = np.diag(np.diag(cov))
    return mean, (cov + cov.T) / 2


def _patterns(observed: np.ndarray):
    """Groups of assets observed in exactly the same periods."""
    groups: dict[bytes, list[int]] = {}
    for j in range(observed.shape[1]):
        groups.setdefault(observed[:, j].tobytes(), []).append(j)
    for cols in groups.values():
        yield observed[:, cols[0]].copy(), np.array(cols)


# ---------------------------------------------------------------------- inputs


def _align(returns, factors, regimes):
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(returns.name if returns.name is not None else "y")
    if not isinstance(returns, pd.DataFrame):
        returns = pd.DataFrame(np.asarray(returns, dtype=float))
    if isinstance(factors, pd.Series):
        factors = factors.to_frame()
    if not isinstance(factors, pd.DataFrame):
        factors = pd.DataFrame(np.asarray(factors, dtype=float)).add_prefix("f")

    if isinstance(regimes, RegimeFit):
        probs = regimes.posteriors
    elif isinstance(regimes, pd.DataFrame):
        probs = regimes
        values = probs.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or (values < 0).any():
            raise ValueError("regime probabilities must be finite and non-negative")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("regime probabilities must sum to one in every period")
    else:
        raise TypeError("regimes must be a RegimeFit or a DataFrame of probabilities")

    common = probs.index.intersection(factors.index).intersection(returns.index)
    if len(common) == 0:
        raise ValueError("returns, factors and regimes have no periods in common")
    common = common.sort_values()
    returns = returns.loc[common].astype(float)
    factors = factors.loc[common].astype(float)
    probs = probs.loc[common].astype(float)
    if not np.all(np.isfinite(factors.to_numpy())):
        raise ValueError("factors contain missing values in the common sample")
    return returns, factors, probs
