"""Re-estimation through time (methodology, Section 1.3).

:func:`expanding` refits the regime model and the RWLS estimator at every
decision date on the data available at that date, starting each regime fit from
the previous one, and yields one :class:`Step` per date.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from rwls.estimator import RWLS, RWLSResults
from rwls.regimes import RegimeFit, RegimeModel
from rwls.settings import RegimeSettings, RWLSSettings

__all__ = ["Step", "expanding"]


@dataclass(frozen=True, eq=False)
class Step:
    """The estimates at one decision date.

    Attributes
    ----------
    date
        The decision date: the last period of data used.
    regimes : RegimeFit
        The regime model fitted on data up to and including ``date``.
    forecast : pandas.Series
        Regime probabilities for the period after ``date``.
    results : RWLSResults or None
        The RWLS fit on the same data, or None if ``estimate`` was False.
    """

    date: object
    regimes: RegimeFit
    forecast: pd.Series
    results: RWLSResults | None


def expanding(
    returns,
    factors: pd.DataFrame,
    start,
    end=None,
    window: int | None = None,
    regime_settings: RegimeSettings | None = None,
    rwls_settings: RWLSSettings | None = None,
    regime_factors: list[str] | None = None,
    initial: RegimeFit | None = None,
    estimate: bool = True,
    n_jobs: int = 1,
) -> Iterator[Step]:
    """Refit the regime model and RWLS at every decision date.

    At each date the models see only the periods up to and including that date.
    The first regime fit is a cold start, with the starts and selection rule of
    ``regime_settings``, unless ``initial`` is given; every later fit starts from
    the previous one (Section 1.3), which keeps the regime labels stable.

    Parameters
    ----------
    returns : pandas.DataFrame or pandas.Series
        Excess returns, one column per asset.
    factors : pandas.DataFrame
        Factor returns. Their index defines the decision dates.
    start : int or date-like
        First decision date, or, as an integer, the number of periods in the first
        estimation sample. A backtest whose first decision follows ``n`` periods
        of in-sample data and also uses the decision period itself, as when
        slicing ``factors.loc[:n]`` on a zero-based index, starts at ``n + 1``.
    end : date-like, optional
        Last decision date. Defaults to the last period of ``factors``.
    window : int, optional
        If given, each estimation uses only the most recent ``window`` periods
        (a rolling sample) instead of all periods so far.
    regime_settings, rwls_settings : optional
        Settings of the two models.
    regime_factors : list of str, optional
        Factors the regime model is fitted on. Defaults to all columns of
        ``factors``. The regressions always use all of them.
    initial : RegimeFit, optional
        Fit to start the first date from, instead of a cold start. It must have the
        same regimes and regime factors.
    estimate : bool
        Whether to run the RWLS regressions at each date. With False only the
        regime model is refitted.
    n_jobs : int
        Processes for the cold start of the first date.

    Yields
    ------
    Step
    """
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a DataFrame indexed by period")
    factors = factors.sort_index()
    regime_cols = (
        list(factors.columns) if regime_factors is None else list(regime_factors)
    )
    missing = [c for c in regime_cols if c not in factors.columns]
    if missing:
        raise ValueError(f"regime_factors not in factors: {missing}")
    if window is not None and window < 2:
        raise ValueError("window must be at least 2")

    dates = factors.index
    first = _first_position(dates, start)
    last = len(dates) - 1 if end is None else dates.get_indexer([end])[0]
    if last < 0:
        raise ValueError("end is not a period of factors")
    if first > last:
        raise ValueError("start is after end")

    model = RegimeModel(regime_settings)
    previous = initial
    for pos in range(first, last + 1):
        lo = 0 if window is None else max(0, pos + 1 - window)
        sample = factors.iloc[lo : pos + 1]
        fit = model.fit(sample[regime_cols], start_from=previous, n_jobs=n_jobs)
        previous = fit
        results = None
        if estimate:
            results = RWLS(returns, sample, fit, rwls_settings).fit()
        yield Step(
            date=dates[pos], regimes=fit, forecast=fit.forecast(), results=results
        )


def _first_position(dates: pd.Index, start) -> int:
    if isinstance(start, int) and not isinstance(start, bool):
        if start < 2 or start > len(dates):
            raise ValueError(
                "start must leave at least two periods in the first sample"
            )
        return start - 1
    pos = dates.get_indexer([start])[0]
    if pos < 0:
        raise ValueError("start is not a period of factors")
    return pos
