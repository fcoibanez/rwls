"""Regime identification with a Gaussian hidden Markov model (methodology, Section 1).

:class:`RegimeModel` fits the model to factor returns and returns a
:class:`RegimeFit` holding the parameters, the posterior regime probabilities
and the one-step-ahead regime forecast. A fit can start from an earlier fit,
which is how the model is re-estimated date by date (Section 1.3).
:func:`select_regimes` compares models with different numbers of regimes.

Regimes are labelled 1 to M. EM numbers them arbitrarily, so the numbering of
a fit depends on its starting values; :meth:`RegimeFit.sort_by`,
:meth:`RegimeFit.align_to` and :meth:`RegimeFit.relabel` fix it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from rwls.settings import RegimeSettings
from rwls.summary import Summary

__all__ = ["RegimeModel", "RegimeFit", "RegimeSelection", "select_regimes"]


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

    def fit(
        self, factors, start_from: RegimeFit | None = None, n_jobs: int = 1
    ) -> RegimeFit:
        """Estimate the model on ``factors``.

        Parameters
        ----------
        factors : pandas.DataFrame or array-like
            Factor returns, one row per period and one column per factor.
        start_from : RegimeFit, optional
            An earlier fit, typically on a shorter sample, to start from instead
            of a cold start. Its regime labels carry over. If the resulting fit
            fails the health check, the model is refitted from a cold start.
        n_jobs : int
            Number of processes used for cold starts when ``n_starts`` exceeds
            one. ``-1`` uses all processors.

        Returns
        -------
        RegimeFit
        """
        values, index, names = _as_matrix(factors)
        s = self.settings
        if start_from is not None:
            if start_from.n_regimes != s.n_regimes or start_from.factor_names != names:
                raise ValueError("start_from must have the same regimes and factors")
            mdl = _fit_warm(values, start_from._model, s)
            if not _healthy(mdl):
                mdl = _fit_cold(values, s, n_jobs)
        else:
            mdl = _fit_cold(values, s, n_jobs)
        return RegimeFit._from_model(mdl, values, index, names, s)

    def _build(self, n_features: int) -> GaussianHMM:
        return _build_hmm(self.settings, n_features)


@dataclass(frozen=True, eq=False)
class RegimeFit:
    """A fitted regime model.

    Regimes are labelled 1 to M in every table: the index of ``means``,
    ``transmat`` and ``forecast()``, and the columns of ``posteriors``.

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
        Factor covariance matrices, shape ``(n_regimes, K, K)``, in regime
        order: ``covariances[0]`` is regime 1. :meth:`covariance` looks a
        matrix up by label.
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
        regimes = _labels(mdl.n_components)
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

    # ------------------------------------------------------------- basic quantities

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

    def covariance(self, regime: int) -> pd.DataFrame:
        """Factor covariance matrix of the regime labelled ``regime``."""
        pos = self._position(regime)
        return pd.DataFrame(
            self.covariances[pos], index=self.factor_names, columns=self.factor_names
        )

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

    # --------------------------------------------------------- information criteria

    @property
    def n_params(self) -> int:
        """Free parameters: transitions, means and covariances.

        The initial distribution is fixed and not counted.
        """
        m, k = self.n_regimes, len(self.factor_names)
        cov = m * k if self.settings.covariance == "diag" else m * k * (k + 1) // 2
        return m * (m - 1) + m * k + cov

    @property
    def entropy(self) -> float:
        """Entropy of the posterior regime assignments, summed over periods."""
        p = self.posteriors.to_numpy()
        return float(-np.sum(p * np.log(np.clip(p, 1e-300, 1.0))))

    @property
    def aic(self) -> float:
        return -2 * self.loglike + 2 * self.n_params

    @property
    def bic(self) -> float:
        return -2 * self.loglike + self.n_params * np.log(self.nobs)

    @property
    def hqic(self) -> float:
        return -2 * self.loglike + 2 * self.n_params * np.log(np.log(self.nobs))

    @property
    def icl(self) -> float:
        """BIC plus twice the entropy of the regime assignments."""
        return self.bic + 2 * self.entropy

    # ---------------------------------------------------------------------- labels

    def relabel(self, order) -> RegimeFit:
        """The same fit with regimes renumbered.

        Parameters
        ----------
        order : sequence of int
            Current labels in their new order: the regime currently labelled
            ``order[0]`` becomes regime 1, ``order[1]`` becomes regime 2, and so
            on. ``relabel([3, 1, 2])`` makes regime 3 the first.
        """
        labels = [int(x) for x in order]
        if sorted(labels) != list(range(1, self.n_regimes + 1)):
            raise ValueError("order must be a permutation of the labels 1 to M")
        order = np.asarray(labels) - 1
        mdl = copy.deepcopy(self._model)
        mdl.startprob_ = self._model.startprob_[order]
        mdl.transmat_ = self._model.transmat_[np.ix_(order, order)]
        mdl.means_ = self._model.means_[order]
        covars = np.array(self._model.covars_)[order]
        if self.settings.covariance == "diag":
            covars = np.diagonal(covars, axis1=1, axis2=2).copy()
        mdl.covars_ = covars

        regimes = _labels(self.n_regimes)
        return RegimeFit(
            settings=self.settings,
            factor_names=list(self.factor_names),
            index=self.index,
            startprob=pd.Series(mdl.startprob_, index=regimes),
            transmat=pd.DataFrame(mdl.transmat_, index=regimes, columns=regimes),
            means=pd.DataFrame(mdl.means_, index=regimes, columns=self.factor_names),
            covariances=np.array(mdl.covars_),
            posteriors=pd.DataFrame(
                self.posteriors.to_numpy()[:, order], index=self.index, columns=regimes
            ),
            loglike=self.loglike,
            n_iter=self.n_iter,
            converged=self.converged,
            _model=mdl,
        )

    def alignment(self, reference: RegimeFit) -> np.ndarray:
        """Match this fit's regimes to those of ``reference`` (Section 1.6).

        Regimes are matched by minimizing the total distance between their mean
        vectors, after scaling each factor by the dispersion of the reference
        means across regimes.

        Returns
        -------
        pandas.Series
            For each regime of this fit, the label of the matching regime of
            ``reference``.
        """
        if reference.n_regimes != self.n_regimes:
            raise ValueError("reference must have the same number of regimes")
        ref = reference.means[self.factor_names].to_numpy()
        scale = ref.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        cost = cdist(self.means.to_numpy() / scale, ref / scale)
        rows, cols = linear_sum_assignment(cost)
        perm = np.empty(len(rows), dtype=int)
        perm[rows] = cols
        return pd.Series(perm + 1, index=self.means.index, name="reference")

    def align_to(self, reference: RegimeFit) -> RegimeFit:
        """The same fit with regimes numbered as in ``reference``."""
        matched = self.alignment(reference)
        return self.relabel(matched.sort_values().index)

    def sort_by(self, factor: str, ascending: bool = False) -> RegimeFit:
        """The same fit renumbered by the mean of one factor.

        With the default, regime 1 has the highest mean of ``factor``. Sorting
        gives a numbering that does not depend on the starting values of EM.
        When two regimes have nearly the same mean on ``factor``, prefer
        :meth:`align_to`.
        """
        if factor not in self.factor_names:
            raise KeyError(f"no factor named {factor!r}")
        col = self.means[factor]
        return self.relabel(col.sort_values(ascending=ascending, kind="stable").index)

    # ---------------------------------------------------------------------- summary

    def summary(self, periods_per_year: int | None = None) -> Summary:
        """Text and HTML summary of the fit.

        Parameters
        ----------
        periods_per_year : int, optional
            If given, factor means are multiplied by it and volatilities by its
            square root, so that both are annual. Otherwise they are per period.

        Returns
        -------
        Summary
            Printing it gives the text layout; in a notebook it renders as HTML.
        """
        s = self.settings
        labels = [f"Regime {k}" for k in self.means.index]
        start, end = _period(self.index[0]), _period(self.index[-1])

        out = Summary("Hidden Markov Regime Model")
        out.add_header(
            [
                ("No. observations:", self.nobs),
                ("Sample:", start),
                ("", f"- {end}"),
                ("No. factors:", len(self.factor_names)),
                ("No. regimes:", self.n_regimes),
                ("Covariance:", s.covariance),
            ],
            [
                ("Log-likelihood:", self.loglike),
                ("No. parameters:", self.n_params),
                ("AIC:", self.aic),
                ("BIC:", self.bic),
                ("HQIC:", self.hqic),
                ("ICL:", self.icl),
            ],
            spec="{:.2f}",
        )

        persistence = np.diag(self.transmat.to_numpy())
        with np.errstate(divide="ignore"):
            duration = 1.0 / (1.0 - persistence)
        post = self.posteriors.to_numpy()
        assigned = np.bincount(post.argmax(axis=1), minlength=self.n_regimes) / len(
            post
        )
        stats = pd.DataFrame(
            {
                "Persistence": persistence,
                "Duration": duration,
                "Occupancy": post.mean(axis=0),
                "Assigned": assigned,
                "Forecast": self.forecast().to_numpy(),
            },
            index=pd.Index(labels, name=""),
        )
        out.add_table(
            stats,
            title="Regime statistics",
            formats={"Duration": "{:.1f}"},
            default="{:.3f}",
        )

        scale_mean = 1.0 if periods_per_year is None else float(periods_per_year)
        scale_vol = np.sqrt(scale_mean)
        unit = "per period" if periods_per_year is None else "annualized"
        means = self.means.T * scale_mean
        vols = np.sqrt(self.variances.T) * scale_vol
        for frame, title in ((means, "Factor means"), (vols, "Factor volatilities")):
            frame = frame.copy()
            frame.columns = labels
            frame.index.name = ""
            out.add_table(frame, title=f"{title} ({unit})")

        trans = self.transmat.copy()
        trans.index = pd.Index(labels, name="From / to")
        trans.columns = labels
        out.add_table(trans, title="Transition probabilities", default="{:.3f}")

        status = (
            f"EM converged after {self.n_iter} iterations."
            if self.converged
            else f"EM stopped after {self.n_iter} iterations without meeting the "
            "tolerance."
        )
        starts = "1 start" if s.n_starts == 1 else f"best of {s.n_starts} starts"
        out.add_notes(
            [
                status,
                f"Variance prior {s.variance_prior:g}, tolerance {s.tol:g}, {starts} "
                f"(seed {s.seed}).",
                "Duration is the expected number of periods in a regime once "
                "entered; Occupancy is the average posterior probability; Assigned "
                "is the share of periods in which the regime is the most probable.",
                "EM numbers regimes arbitrarily; use sort_by or align_to for a fixed "
                "numbering.",
            ]
        )
        return out

    def _position(self, regime: int) -> int:
        if regime not in self.means.index:
            raise KeyError(
                f"no regime labelled {regime!r}; labels are 1 to {self.n_regimes}"
            )
        return int(regime) - 1


@dataclass(frozen=True)
class RegimeSelection:
    """Information criteria for models with different numbers of regimes.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per number of regimes: log-likelihood, number of parameters, AIC,
        BIC, HQIC, ICL, and the number of starts that passed the health check.
    fits : dict
        The best fit for each number of regimes.
    selected : dict
        The number of regimes each criterion selects (its minimum).
    """

    table: pd.DataFrame
    fits: dict[int, RegimeFit] = field(repr=False)
    selected: dict[str, int]


def select_regimes(
    factors,
    n_regimes=range(1, 9),
    settings: RegimeSettings | None = None,
    n_starts: int = 100,
    n_jobs: int = 1,
) -> RegimeSelection:
    """Fit models with each number of regimes and compare them.

    Each model is fitted from ``n_starts`` cold starts and the start with the
    highest likelihood is kept, as in :meth:`RegimeModel.fit`.

    Parameters
    ----------
    factors : pandas.DataFrame or array-like
        Factor returns.
    n_regimes : iterable of int
        Numbers of regimes to compare.
    settings : RegimeSettings, optional
        Settings for every fit; ``n_regimes`` and ``n_starts`` are overridden.
    n_starts : int
        Cold starts per number of regimes.
    n_jobs : int
        Number of processes; ``-1`` uses all processors.

    Returns
    -------
    RegimeSelection
    """
    values, index, names = _as_matrix(factors)
    base = settings if settings is not None else RegimeSettings()
    counts = [int(m) for m in n_regimes]
    jobs = [
        (m, base.replace(n_regimes=m, n_starts=n_starts), base.seed + j)
        for m in counts
        for j in range(n_starts)
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_start)(values, s, seed) for _, s, seed in jobs
    )

    fits, rows = {}, {}
    for m in counts:
        runs = [r for (mm, _, _), r in zip(jobs, results, strict=True) if mm == m]
        good = [r for r in runs if r[1] is not None]
        if not good:
            continue
        _, mdl = max(good, key=lambda r: r[0])
        fit = RegimeFit._from_model(
            mdl, values, index, names, base.replace(n_regimes=m, n_starts=n_starts)
        )
        fits[m] = fit
        rows[m] = {
            "loglike": fit.loglike,
            "n_params": fit.n_params,
            "AIC": fit.aic,
            "BIC": fit.bic,
            "HQIC": fit.hqic,
            "ICL": fit.icl,
            "healthy starts": len(good),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "n_regimes"
    selected = {c: int(table[c].idxmin()) for c in ("AIC", "BIC", "HQIC", "ICL")}
    return RegimeSelection(table=table, fits=fits, selected=selected)


# ------------------------------------------------------------------ fitting helpers


def _build_hmm(s: RegimeSettings, n_features: int) -> GaussianHMM:
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


def _fit_start(values: np.ndarray, s: RegimeSettings, seed: int):
    """One cold start: ``(log-likelihood, model)``, or ``(-inf, None)`` if unhealthy."""
    mdl = _build_hmm(s, values.shape[1])
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
        return -np.inf, None
    return float(mdl.score(values)), mdl


def _fit_cold(values: np.ndarray, s: RegimeSettings, n_jobs: int) -> GaussianHMM:
    seeds = [s.seed + j for j in range(s.n_starts)]
    if s.n_starts == 1 or n_jobs == 1:
        results = [_fit_start(values, s, seed) for seed in seeds]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_start)(values, s, seed) for seed in seeds
        )
    best, best_ll = None, -np.inf
    for ll, mdl in results:
        if mdl is not None and ll > best_ll:
            best, best_ll = mdl, ll
    if best is None:
        raise RuntimeError("the regime model failed to converge from a cold start")
    return best


def _fit_warm(values: np.ndarray, previous: GaussianHMM, s: RegimeSettings):
    mdl = _build_hmm(s, values.shape[1])
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


def _healthy(mdl: GaussianHMM) -> bool:
    """Every parameter finite and every row of the transition matrix summing to one."""
    params = [mdl.startprob_, mdl.transmat_, mdl.means_, mdl.covars_]
    return all(np.all(np.isfinite(p)) for p in params) and np.allclose(
        mdl.transmat_.sum(axis=1), 1.0
    )


# ------------------------------------------------------------------ input handling


def _period(stamp) -> str:
    """A compact label for one period of the index."""
    if isinstance(stamp, pd.Timestamp):
        if stamp == stamp + pd.offsets.MonthEnd(0):
            return f"{stamp:%Y-%m}"
        return f"{stamp:%Y-%m-%d}"
    return str(stamp)


def _labels(n: int) -> pd.RangeIndex:
    return pd.RangeIndex(1, n + 1, name="regime")


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
