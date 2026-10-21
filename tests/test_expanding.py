import numpy as np
import pandas as pd
import pytest

from rwls import RegimeSettings, RWLSSettings
from rwls.expanding import expanding
from rwls.regimes import RegimeModel
from test_estimator import _simulate, ref_fit_wls, ref_guard_weights
from test_regimes import _reference_cold, _reference_warm

QUICK = RegimeSettings(n_regimes=2, tol=0.1, n_starts=1)
RECENT = RWLSSettings(fallback="recent")


@pytest.fixture(scope="module")
def data():
    returns, factors, _, _ = _simulate(n=330, n_assets=6)
    return returns, factors


def test_reproduces_the_backtest_loop(data):
    returns, factors = data
    steps = list(
        expanding(
            returns,
            factors,
            start=301,
            end=factors.index[302],
            regime_settings=QUICK,
            rwls_settings=RECENT,
        )
    )
    assert [s.date for s in steps] == list(factors.index[300:303])
    previous = None
    for t, step in zip(range(300, 303), steps, strict=True):
        X = factors.iloc[: t + 1].to_numpy()
        ref = (
            _reference_cold(X, 2, 0.1)
            if previous is None
            else _reference_warm(X, previous, 0.1)
        )
        previous = ref
        probs = ref.predict_proba(X)
        np.testing.assert_allclose(
            step.regimes.posteriors.to_numpy(), probs, atol=1e-12
        )
        forecast = probs[-1] @ ref.transmat_
        np.testing.assert_allclose(
            step.forecast.to_numpy(), forecast / forecast.sum(), atol=1e-12
        )
        design = np.column_stack([np.ones(len(X)), X])
        Y = returns.iloc[: t + 1].to_numpy()
        for j in range(2):
            w = ref_guard_weights(probs[:, j], 3, 24)
            coefs, var = ref_fit_wls(design, Y, w)
            np.testing.assert_allclose(
                step.results.params.loc[j + 1].to_numpy(), coefs.T, atol=1e-12
            )
            np.testing.assert_allclose(
                step.results.resid_var.loc[j + 1], var, atol=1e-14
            )


def test_start_as_a_count_or_a_date(data):
    returns, factors = data
    by_count = next(expanding(returns, factors, start=301, regime_settings=QUICK))
    by_date = next(
        expanding(returns, factors, start=factors.index[300], regime_settings=QUICK)
    )
    assert by_count.date == by_date.date == factors.index[300]
    assert by_count.regimes.nobs == 301
    pd.testing.assert_frame_equal(by_count.results.params, by_date.results.params)


def test_each_step_sees_only_the_data_available(data):
    returns, factors = data
    for step in expanding(returns, factors, start=320, regime_settings=QUICK):
        assert step.regimes.index[-1] == step.date
        assert step.results.index[-1] == step.date
        assert step.regimes.starts is None or step.date == factors.index[319]


def test_rolling_window(data):
    returns, factors = data
    steps = list(
        expanding(returns, factors, start=320, window=200, regime_settings=QUICK)
    )
    assert all(s.regimes.nobs == 200 for s in steps)
    assert all(len(s.results.index) == 200 for s in steps)


def test_regime_factors_subset(data):
    returns, factors = data
    step = next(
        expanding(
            returns,
            factors,
            start=320,
            regime_factors=["Mkt-RF"],
            regime_settings=QUICK,
        )
    )
    assert step.regimes.factor_names == ["Mkt-RF"]
    assert step.results.factor_names == ["Mkt-RF", "SMB"]
    with pytest.raises(ValueError, match="regime_factors"):
        next(expanding(returns, factors, start=320, regime_factors=["HML"]))


def test_regimes_only(data):
    returns, factors = data
    steps = list(
        expanding(returns, factors, start=328, estimate=False, regime_settings=QUICK)
    )
    assert len(steps) == 3
    assert all(s.results is None for s in steps)


def test_initial_fit_and_stable_labels(data):
    returns, factors = data
    first = RegimeModel(QUICK).fit(factors.iloc[:300])
    swapped = first.relabel([2, 1])
    plain = list(
        expanding(
            returns,
            factors,
            start=301,
            end=factors.index[302],
            regime_settings=QUICK,
            initial=first,
            estimate=False,
        )
    )
    flipped = list(
        expanding(
            returns,
            factors,
            start=301,
            end=factors.index[302],
            regime_settings=QUICK,
            initial=swapped,
            estimate=False,
        )
    )
    assert plain[0].regimes.starts is None
    for a, b in zip(plain, flipped, strict=True):
        np.testing.assert_allclose(
            b.regimes.posteriors.to_numpy(),
            a.regimes.posteriors.to_numpy()[:, [1, 0]],
            atol=1e-8,
        )


def test_invalid_arguments(data):
    returns, factors = data
    with pytest.raises(TypeError):
        next(expanding(returns, factors.to_numpy(), start=300))
    with pytest.raises(ValueError, match="start"):
        next(expanding(returns, factors, start=1))
    with pytest.raises(ValueError, match="start"):
        next(expanding(returns, factors, start=pd.Timestamp("1900-01-31")))
    with pytest.raises(ValueError, match="after end"):
        next(expanding(returns, factors, start=320, end=factors.index[100]))
    with pytest.raises(ValueError, match="window"):
        next(expanding(returns, factors, start=320, window=1))
