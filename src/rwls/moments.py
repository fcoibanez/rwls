"""Asset return moments implied by an RWLS fit (methodology, Section 3).

:class:`RegimeMoments` holds each regime's factor model of asset returns in
factor form: intercepts, loadings, factor means and covariances, and residual
variances. Means, covariances and portfolio variances are computed from that
representation; dense ``N x N`` matrices are built only when asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["RegimeMoments"]


@dataclass(frozen=True, eq=False)
class RegimeMoments:
    """Regime-conditional moments of asset returns, in factor form.

    Attributes
    ----------
    regime_labels : list
        Regime labels, in the order of the arrays.
    asset_names : list
        Assets, in the order of the arrays.
    factor_names : list of str
        Factors, in the order of the arrays.
    alphas : numpy.ndarray
        Intercepts, shape ``(M, N)``; zeros for a model without intercepts.
    loadings : numpy.ndarray
        Factor loadings, shape ``(M, N, K)``.
    factor_means : numpy.ndarray
        Factor means, shape ``(M, K)``.
    factor_covariances : numpy.ndarray
        Factor covariance matrices, shape ``(M, K, K)``.
    resid_var : numpy.ndarray
        Residual variances, shape ``(M, N)``.
    """

    regime_labels: list
    asset_names: list
    factor_names: list
    alphas: np.ndarray
    loadings: np.ndarray
    factor_means: np.ndarray
    factor_covariances: np.ndarray
    resid_var: np.ndarray

    def __post_init__(self) -> None:
        M, N, K = len(self.regime_labels), len(self.asset_names), len(self.factor_names)
        shapes = {
            "alphas": (M, N),
            "loadings": (M, N, K),
            "factor_means": (M, K),
            "factor_covariances": (M, K, K),
            "resid_var": (M, N),
        }
        for name, shape in shapes.items():
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name} must have shape {shape}")

    # ---------------------------------------------------------- regime moments

    def _pos(self, regime) -> int:
        try:
            return self.regime_labels.index(regime)
        except ValueError:
            raise KeyError(f"no regime labelled {regime!r}") from None

    def mean(self, regime) -> pd.Series:
        """Expected returns in ``regime``: ``alpha_s + B_s m_s``."""
        s = self._pos(regime)
        mu = self.alphas[s] + self.loadings[s] @ self.factor_means[s]
        return pd.Series(mu, index=self.asset_names, name=regime)

    def cov(self, regime) -> pd.DataFrame:
        """Dense covariance matrix in ``regime``: ``B_s V_s B_s' + D_s``."""
        s = self._pos(regime)
        B = self.loadings[s]
        V = B @ self.factor_covariances[s] @ B.T + np.diag(self.resid_var[s])
        V = (V + V.T) / 2
        return pd.DataFrame(V, index=self.asset_names, columns=self.asset_names)

    def variance(self, weights, regime) -> float:
        """Variance of a portfolio in ``regime``, computed in factor form.

        ``(B_s' w)' V_s (B_s' w) + sum_i w_i^2 sigma^2_is``, without forming the
        ``N x N`` matrix.
        """
        s = self._pos(regime)
        w = self._weights(weights)
        exposure = self.loadings[s].T @ w
        return float(
            exposure @ self.factor_covariances[s] @ exposure + w**2 @ self.resid_var[s]
        )

    # --------------------------------------------------------------- mixture

    def mixture_mean(self, probabilities) -> pd.Series:
        """Mean of the mixture with regime ``probabilities``."""
        p = self._probabilities(probabilities)
        means = np.array([self.mean(r).to_numpy() for r in self.regime_labels])
        return pd.Series(p @ means, index=self.asset_names, name="mixture")

    def mixture(self, probabilities) -> tuple[pd.Series, pd.DataFrame]:
        """Mean and dense covariance matrix of the mixture (Section 3.2).

        ``sum_s p_s (Sigma_s + mu_s mu_s') - mu mu'``: the average within-regime
        covariance plus the dispersion of the regime means.
        """
        p = self._probabilities(probabilities)
        mu = self.mixture_mean(p).to_numpy()
        cov = -np.outer(mu, mu)
        for ps, r in zip(p, self.regime_labels, strict=True):
            m = self.mean(r).to_numpy()
            cov += ps * (self.cov(r).to_numpy() + np.outer(m, m))
        cov = (cov + cov.T) / 2
        return (
            pd.Series(mu, index=self.asset_names, name="mixture"),
            pd.DataFrame(cov, index=self.asset_names, columns=self.asset_names),
        )

    def mixture_variance(self, weights, probabilities) -> float:
        """Variance of a portfolio under the mixture, computed in factor form.

        ``sum_s p_s (w' Sigma_s w + (w' mu_s)^2) - (w' mu)^2``.
        """
        p = self._probabilities(probabilities)
        w = self._weights(weights)
        total, mean = 0.0, 0.0
        for ps, r in zip(p, self.regime_labels, strict=True):
            m = float(w @ self.mean(r).to_numpy())
            total += ps * (self.variance(w, r) + m**2)
            mean += ps * m
        return total - mean**2

    # ---------------------------------------------------------------- inputs

    def _weights(self, weights) -> np.ndarray:
        if isinstance(weights, pd.Series):
            missing = set(self.asset_names) - set(weights.index)
            if missing:
                raise ValueError(f"weights missing for {sorted(map(str, missing))[:5]}")
            weights = weights.reindex(self.asset_names)
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(self.asset_names),):
            raise ValueError("weights must have one entry per asset")
        return w

    def _probabilities(self, probabilities) -> np.ndarray:
        if isinstance(probabilities, pd.Series):
            probabilities = probabilities.reindex(self.regime_labels)
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(self.regime_labels),) or not np.all(np.isfinite(p)):
            raise ValueError("probabilities must have one finite entry per regime")
        if (p < 0).any() or not np.isclose(p.sum(), 1.0):
            raise ValueError("probabilities must be non-negative and sum to one")
        return p
