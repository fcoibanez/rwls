"""Regime-weighted least squares (methodology, Section 2).

:class:`RWLS` estimates a factor model for every asset in every regime by
weighted least squares, weighting each period by its posterior regime
probability, and returns an :class:`RWLSResults`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import stats

from rwls.moments import RegimeMoments
from rwls.regimes import RegimeFit, _period
from rwls.settings import RWLSSettings
from rwls.summary import Summary

__all__ = ["RWLS", "RWLSResults", "AssetResults", "SparseRegimeWarning"]


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

    # -------------------------------------------------------------------- inference

    @property
    def terms(self) -> list[str]:
        return list(self.params.index.get_level_values("term").unique())

    @cached_property
    def _cov(self) -> np.ndarray:
        """Sandwich covariance of every asset's coefficients, all regimes stacked.

        Shape ``(N, M k, M k)``, ordered like the rows of ``params`` (Section 2.6).
        """
        M, k, N = self.n_regimes, len(self.terms), len(self.asset_names)
        theta = self.params.to_numpy().reshape(M, k, N)
        R = self._model.returns.to_numpy(dtype=float)
        out = np.full((N, M * k, M * k), np.nan)
        for rows, cols, weights in self._groups:
            X = self._design[rows]
            Y = R[np.ix_(rows, cols)]
            A = [np.linalg.inv(X.T @ (X * w[:, None])) for w in weights]
            E = [Y - X @ theta[s][:, cols] for s in range(M)]
            for s in range(M):
                for r in range(s, M):
                    u = (weights[s] * weights[r])[:, None] * E[s] * E[r]
                    middle = np.einsum("tk,tl,tn->nkl", X, X, u)
                    block = A[s] @ middle @ A[r].T
                    out[cols, s * k : (s + 1) * k, r * k : (r + 1) * k] = block
                    if r != s:
                        out[cols, r * k : (r + 1) * k, s * k : (s + 1) * k] = (
                            block.transpose(0, 2, 1)
                        )
        return out

    @property
    def bse(self) -> pd.DataFrame:
        """Sandwich standard errors, laid out like ``params``."""
        se = np.sqrt(np.diagonal(self._cov, axis1=1, axis2=2)).T
        return pd.DataFrame(se, index=self.params.index, columns=self.params.columns)

    @property
    def tvalues(self) -> pd.DataFrame:
        return self.params / self.bse

    @property
    def df_resid(self) -> pd.DataFrame:
        """Residual degrees of freedom, ``n_s - k`` floored at one.

        One row per regime and one column per asset.
        """
        return np.maximum(self.mass - len(self.terms), 1.0)

    def _df_by_param(self) -> np.ndarray:
        k = len(self.terms)
        return np.repeat(self.df_resid.to_numpy(), k, axis=0)

    @property
    def pvalues(self) -> pd.DataFrame:
        """Two-sided p-values: Student's t with ``df_resid`` degrees of freedom, or
        the standard normal if ``use_t`` is False."""
        z = np.abs(self.tvalues.to_numpy())
        if self.settings.use_t:
            p = 2 * stats.t.sf(z, self._df_by_param())
        else:
            p = 2 * stats.norm.sf(z)
        return pd.DataFrame(p, index=self.params.index, columns=self.params.columns)

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Confidence intervals at level ``1 - alpha``.

        Columns are ``(asset, "lower")`` and ``(asset, "upper")``.
        """
        if self.settings.use_t:
            q = stats.t.ppf(1 - alpha / 2, self._df_by_param())
        else:
            q = stats.norm.ppf(1 - alpha / 2)
        half = q * self.bse.to_numpy()
        lower = pd.DataFrame(
            self.params.to_numpy() - half,
            index=self.params.index,
            columns=self.params.columns,
        )
        upper = pd.DataFrame(
            self.params.to_numpy() + half,
            index=self.params.index,
            columns=self.params.columns,
        )
        out = pd.concat({"lower": lower, "upper": upper}, axis=1)
        return out.swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)[
            pd.MultiIndex.from_product([self.asset_names, ["lower", "upper"]])
        ]

    def cov_params(self, asset) -> pd.DataFrame:
        """Sandwich covariance of one asset's coefficients across all regimes."""
        j = self.asset_names.index(asset)
        return pd.DataFrame(
            self._cov[j], index=self.params.index, columns=self.params.index
        )

    @cached_property
    def rsquared(self) -> pd.DataFrame:
        """Weighted R-squared of each regime's regression.

        One minus the weighted residual sum of squares over the weighted total sum
        of squares, centred on the weighted mean when there is an intercept.
        """
        M, k, N = self.n_regimes, len(self.terms), len(self.asset_names)
        theta = self.params.to_numpy().reshape(M, k, N)
        R = self._model.returns.to_numpy(dtype=float)
        out = np.full((M, N), np.nan)
        for rows, cols, weights in self._groups:
            X = self._design[rows]
            Y = R[np.ix_(rows, cols)]
            for s, w in enumerate(weights):
                e = Y - X @ theta[s][:, cols]
                ssr = (w[:, None] * e**2).sum(axis=0)
                if self.settings.intercept:
                    mean = (w[:, None] * Y).sum(axis=0) / w.sum()
                    sst = (w[:, None] * (Y - mean) ** 2).sum(axis=0)
                else:
                    sst = (w[:, None] * Y**2).sum(axis=0)
                out[s, cols] = 1 - ssr / sst
        return pd.DataFrame(out, index=self.resid_var.index, columns=self.asset_names)

    def wald_equal_loadings(self) -> pd.DataFrame:
        """Test that each asset's factor loadings are equal in every regime.

        The intercepts are not restricted. Under Student's t (``use_t``) the test is
        an F test with ``(M - 1) K`` and ``T - M k`` degrees of freedom; otherwise
        a chi-square test with ``(M - 1) K``.

        Returns
        -------
        pandas.DataFrame
            One row per asset: statistic, numerator and denominator degrees of
            freedom, and p-value.
        """
        M, k = self.n_regimes, len(self.terms)
        K = len(self.factor_names)
        if M < 2:
            raise ValueError("the test needs at least two regimes")
        first = k - K  # position of the first loading within a regime's block
        C = np.zeros(((M - 1) * K, M * k))
        for s in range(1, M):
            for f in range(K):
                row = (s - 1) * K + f
                C[row, first + f] = -1.0
                C[row, s * k + first + f] = 1.0
        theta = self.params.to_numpy().T  # (N, M k)
        q = C.shape[0]
        rows = {}
        for j, asset in enumerate(self.asset_names):
            d = C @ theta[j]
            V = C @ self._cov[j] @ C.T
            if not np.all(np.isfinite(d)) or not np.all(np.isfinite(V)):
                rows[asset] = (np.nan, q, np.nan, np.nan)
                continue
            # Regimes estimated from identical weights, as after a pooled
            # fallback in every regime, cannot be compared: the covariance of
            # their differences is singular, up to rounding error.
            eig = np.linalg.eigvalsh((V + V.T) / 2)
            if eig.max() <= 0 or eig.min() <= 1e-10 * eig.max():
                rows[asset] = (np.nan, q, np.nan, np.nan)
                continue
            w = float(d @ np.linalg.solve(V, d))
            if self.settings.use_t:
                df2 = max(float(self.nobs[asset]) - M * k, 1.0)
                rows[asset] = (w / q, q, df2, float(stats.f.sf(w / q, q, df2)))
            else:
                rows[asset] = (w, q, np.nan, float(stats.chi2.sf(w, q)))
        return pd.DataFrame.from_dict(
            rows, orient="index", columns=["statistic", "df_num", "df_denom", "pvalue"]
        )

    # ---------------------------------------------------------------------- moments

    def moments(
        self,
        factor_moments: str = "weighted",
        factor_means=None,
        factor_covariances=None,
    ) -> RegimeMoments:
        """Regime-conditional moments of asset returns, in factor form (Section 3).

        Parameters
        ----------
        factor_moments : {"weighted", "regime_model"}
            Source of the factor means and covariances. ``"weighted"`` uses the
            moments computed with the RWLS weights (Section 2.5); ``"regime_model"``
            uses those of the fitted regime model, which requires the fit to have
            been given a :class:`RegimeFit`.
        factor_means, factor_covariances : array-like, optional
            Factor moments to use instead, for example those of a known generating
            model (Section 3.3): means of shape ``(M, K)`` and covariances of shape
            ``(M, K, K)``, in regime order. Both must be given together.

        Returns
        -------
        RegimeMoments
        """
        if (factor_means is None) != (factor_covariances is None):
            raise ValueError(
                "give both factor_means and factor_covariances, or neither"
            )
        if factor_means is not None:
            fm = np.asarray(factor_means, dtype=float)
            fc = np.asarray(factor_covariances, dtype=float)
        elif factor_moments == "weighted":
            fm, fc = self.factor_means.to_numpy(), self.factor_covariances
        elif factor_moments == "regime_model":
            regimes = self._model.regimes
            if regimes is None:
                raise ValueError(
                    "'regime_model' needs the fit to have been given a RegimeFit"
                )
            fm = regimes.means[self.factor_names].to_numpy()
            fc = regimes.covariances
        else:
            raise ValueError("factor_moments must be 'weighted' or 'regime_model'")

        M, N, K = self.n_regimes, len(self.asset_names), len(self.factor_names)
        if self.settings.intercept:
            alphas = self.intercepts.to_numpy()
        else:
            alphas = np.zeros((M, N))
        loadings = self.loadings.to_numpy().reshape(M, K, N).transpose(0, 2, 1)
        return RegimeMoments(
            regime_labels=list(self.regime_labels),
            asset_names=list(self.asset_names),
            factor_names=list(self.factor_names),
            alphas=alphas,
            loadings=loadings,
            factor_means=fm,
            factor_covariances=fc,
            resid_var=self.resid_var.to_numpy(),
        )

    # ---------------------------------------------------------------------- summary

    def summary(self, periods_per_year: int | None = None) -> Summary:
        """Summary across assets: the distribution of loadings by regime.

        Parameters
        ----------
        periods_per_year : int, optional
            If given, residual volatilities are annualized.

        Returns
        -------
        Summary
            Printing it gives the text layout; in a notebook it renders as HTML.
        """
        s = self.settings
        wald = self.wald_equal_loadings()["pvalue"]
        out = Summary("Regime-Weighted Least Squares Results")
        out.add_header(
            _common_header(self, "No. assets:", len(self.asset_names)),
            [
                ("Intercept:", "yes" if s.intercept else "no"),
                ("Std. errors:", "sandwich"),
                ("Inference:", "t" if s.use_t else "normal"),
                ("Assets with a fallback:", int(self.fallback.any().sum())),
                ("Loadings differ (5%):", f"{(wald < 0.05).mean():.1%}"),
            ],
        )

        loadings = self.loadings
        signif = self.pvalues.loc[loadings.index] < 0.05
        values = loadings.to_numpy()
        table = pd.DataFrame(
            {
                "Mean": np.nanmean(values, axis=1),
                "Std": np.nanstd(values, axis=1, ddof=1)
                if values.shape[1] > 1
                else np.nan,
                "10%": np.nanpercentile(values, 10, axis=1),
                "Median": np.nanmedian(values, axis=1),
                "90%": np.nanpercentile(values, 90, axis=1),
                "Signif. 5%": signif.mean(axis=1).to_numpy(),
            },
            index=pd.Index([f"Regime {r}  {f}" for r, f in loadings.index], name=""),
        )
        out.add_table(
            table, title="Loadings across assets", formats={"Signif. 5%": "{:.1%}"}
        )

        vol = np.sqrt(self.resid_var * (periods_per_year or 1))
        fit = pd.DataFrame(
            {
                "Mass": self.mass.median(axis=1).to_numpy(),
                "R-squared": self.rsquared.median(axis=1).to_numpy(),
                "Resid. vol": vol.median(axis=1).to_numpy(),
                "Fallback": self.fallback.mean(axis=1).to_numpy(),
            },
            index=pd.Index([f"Regime {r}" for r in self.regime_labels], name=""),
        )
        out.add_table(
            fit,
            title="Regime fit across assets (medians)",
            formats={"Mass": "{:.1f}", "Fallback": "{:.1%}"},
        )
        out.add_notes(_notes(self, periods_per_year, single=False))
        return out

    def __getitem__(self, asset) -> AssetResults:
        if asset not in self.asset_names:
            raise KeyError(asset)
        return AssetResults(self, asset)


@dataclass(frozen=True, eq=False)
class AssetResults:
    """The results of one asset, in the shape of a single regression.

    Obtained with ``results[asset]``. Every attribute is a view on the parent
    :class:`RWLSResults`.
    """

    parent: RWLSResults
    asset: object

    @property
    def params(self) -> pd.Series:
        return self.parent.params[self.asset]

    @property
    def bse(self) -> pd.Series:
        return self.parent.bse[self.asset]

    @property
    def tvalues(self) -> pd.Series:
        return self.parent.tvalues[self.asset]

    @property
    def pvalues(self) -> pd.Series:
        return self.parent.pvalues[self.asset]

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        return self.parent.conf_int(alpha)[self.asset]

    def cov_params(self) -> pd.DataFrame:
        return self.parent.cov_params(self.asset)

    @property
    def resid_var(self) -> pd.Series:
        return self.parent.resid_var[self.asset]

    @property
    def mass(self) -> pd.Series:
        return self.parent.mass[self.asset]

    @property
    def rsquared(self) -> pd.Series:
        return self.parent.rsquared[self.asset]

    @property
    def nobs(self) -> int:
        return int(self.parent.nobs[self.asset])

    def wald_equal_loadings(self) -> pd.Series:
        return self.parent.wald_equal_loadings().loc[self.asset]

    def summary(self, periods_per_year: int | None = None, alpha: float = 0.05):
        """Regression table of the asset, one coefficient block per regime.

        Parameters
        ----------
        periods_per_year : int, optional
            If given, residual volatilities are annualized.
        alpha : float
            Confidence intervals are at level ``1 - alpha``.
        """
        res, a = self.parent, self.asset
        s = res.settings
        wald = self.wald_equal_loadings()
        out = Summary("Regime-Weighted Least Squares Results")
        wald_label = "F (equal loadings):" if s.use_t else "Chi2 (equal loadings):"
        out.add_header(
            _common_header(res, "Dep. variable:", str(a), nobs=self.nobs),
            [
                ("Intercept:", "yes" if s.intercept else "no"),
                ("Std. errors:", "sandwich"),
                ("Inference:", "t" if s.use_t else "normal"),
                (wald_label, wald["statistic"]),
                ("Prob (equal loadings):", wald["pvalue"]),
            ],
        )

        labels = [f"Regime {r}" for r in res.regime_labels]
        vol = np.sqrt(self.resid_var * (periods_per_year or 1))
        stats_table = pd.DataFrame(
            {
                "Mass": self.mass.to_numpy(),
                "df resid": res.df_resid[a].to_numpy(),
                "R-squared": self.rsquared.to_numpy(),
                "Resid. vol": vol.to_numpy(),
                "Fallback": ["yes" if f else "no" for f in res.fallback[a]],
            },
            index=pd.Index(labels, name=""),
        )
        out.add_table(
            stats_table,
            title="Regime statistics",
            formats={"Mass": "{:.1f}", "df resid": "{:.1f}"},
        )

        stat = "t" if s.use_t else "z"
        ci = self.conf_int(alpha)
        lo, hi = f"[{alpha / 2:g}", f"{1 - alpha / 2:g}]"
        for r, label in zip(res.regime_labels, labels, strict=True):
            block = pd.DataFrame(
                {
                    "coef": self.params.loc[r],
                    "std err": self.bse.loc[r],
                    stat: self.tvalues.loc[r],
                    f"P>|{stat}|": self.pvalues.loc[r],
                    lo: ci["lower"].loc[r],
                    hi: ci["upper"].loc[r],
                }
            )
            block.index.name = label
            out.add_table(block, formats={stat: "{:.3f}", f"P>|{stat}|": "{:.3f}"})
        out.add_notes(_notes(res, periods_per_year, single=True))
        return out


def _common_header(res, first_label, first_value, nobs=None):
    start, end = _period(res.index[0]), _period(res.index[-1])
    rows = [(first_label, first_value)]
    rows.append(("No. observations:", len(res.index) if nobs is None else nobs))
    rows += [
        ("Sample:", start),
        ("", f"- {end}"),
        ("No. regimes:", res.n_regimes),
        ("No. factors:", len(res.factor_names)),
        ("Weighting:", res.settings.weighting),
    ]
    return rows


def _notes(res, periods_per_year, single: bool) -> list[str]:
    s = res.settings
    k = len(res.terms)
    if s.use_t:
        inference = (
            "p-values and intervals use Student's t with n_s - k degrees of freedom, "
            f"where n_s is the regime's posterior mass and k = {k}."
        )
    else:
        inference = "p-values and intervals use the standard normal distribution."
    notes = [
        "Standard errors are sandwich estimates, robust to heteroskedasticity, and "
        "allow for correlation between the estimates of different regimes.",
        inference,
        "The equal-loadings test is a Wald test that every factor loading is the "
        "same in all regimes; intercepts are not restricted.",
        "Residual volatility is "
        + ("annualized." if periods_per_year else "per period."),
    ]
    if res.fallback.to_numpy().any():
        notes.append(
            "Fallback: the regime's posterior mass was too small to identify its "
            f"regression, and '{s.resolved_fallback()}' weights were used instead."
        )
    return notes


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
