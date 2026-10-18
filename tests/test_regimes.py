import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from rwls import RegimeSettings
from rwls.regimes import RegimeModel, _healthy


def _simulate(n=400, seed=0):
    """Two well-separated regimes in two factors."""
    gen = GaussianHMM(n_components=2, covariance_type="diag", random_state=seed)
    gen.startprob_ = np.array([0.5, 0.5])
    gen.transmat_ = np.array([[0.95, 0.05], [0.10, 0.90]])
    gen.means_ = np.array([[0.01, 0.00], [-0.03, 0.02]])
    gen.covars_ = np.array([[0.0010, 0.0005], [0.0040, 0.0020]])
    X, Z = gen.sample(n)
    dates = pd.date_range("1990-01-31", periods=n, freq="ME")
    return pd.DataFrame(X, index=dates, columns=["Mkt-RF", "SMB"]), Z


@pytest.fixture(scope="module")
def data():
    return _simulate()


def _reference_cold(values, n_states, tol, seed=13):
    """The cold start of the research code (regimeaware.core.estimation.build_hmm)."""
    mdl = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        min_covar=1e-3,
        tol=tol,
        implementation="scaling",
        random_state=seed,
        init_params="tmc",
        params="tmc",
    )
    mdl.startprob_ = np.full(n_states, 1 / n_states)
    return mdl.fit(values)


def _reference_warm(values, previous, tol):
    """The warm start of the research code."""
    mdl = GaussianHMM(
        n_components=previous.n_components,
        covariance_type="diag",
        n_iter=1000,
        min_covar=1e-3,
        tol=tol,
        implementation="scaling",
        init_params="",
        params="tmc",
    )
    mdl.startprob_ = np.full(previous.n_components, 1 / previous.n_components)
    trans = np.clip(previous.transmat_, 1e-8, None)
    mdl.transmat_ = trans / trans.sum(axis=1, keepdims=True)
    mdl.means_ = previous.means_
    mdl.covars_ = np.diagonal(previous.covars_, axis1=1, axis2=2).copy()
    return mdl.fit(values)


@pytest.mark.parametrize("tol", [0.1, 1e-4])
def test_cold_start_reproduces_the_research_code(data, tol):
    X, _ = data
    fit = RegimeModel(n_regimes=2, tol=tol, n_starts=1).fit(X)
    ref = _reference_cold(X.to_numpy(), 2, tol)
    np.testing.assert_allclose(fit.means.to_numpy(), ref.means_, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        fit.transmat.to_numpy(), ref.transmat_, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(fit.covariances, ref.covars_, rtol=0, atol=1e-12)
    assert fit.loglike == pytest.approx(ref.score(X.to_numpy()), abs=1e-8)


def test_warm_start_reproduces_the_research_code(data):
    X, _ = data
    model = RegimeModel(n_regimes=2, tol=0.1, n_starts=1)
    first = model.fit(X.iloc[:300])
    second = model.fit(X.iloc[:301], start_from=first)
    ref_first = _reference_cold(X.iloc[:300].to_numpy(), 2, 0.1)
    ref_second = _reference_warm(X.iloc[:301].to_numpy(), ref_first, 0.1)
    np.testing.assert_allclose(second.means.to_numpy(), ref_second.means_, atol=1e-12)
    np.testing.assert_allclose(
        second.transmat.to_numpy(), ref_second.transmat_, atol=1e-12
    )
    np.testing.assert_allclose(
        second.posteriors.to_numpy(),
        ref_second.predict_proba(X.iloc[:301].to_numpy()),
        atol=1e-12,
    )


def test_recovers_well_separated_regimes(data):
    X, Z = data
    fit = RegimeModel(n_regimes=2).fit(X)
    # Order the fitted regimes by market mean to compare with the truth.
    order = np.argsort(-fit.means["Mkt-RF"].to_numpy())
    np.testing.assert_allclose(
        fit.means.to_numpy()[order], [[0.01, 0.00], [-0.03, 0.02]], atol=0.01
    )
    labels = order.argsort()[fit.posteriors.to_numpy().argmax(axis=1)]
    assert (labels == Z).mean() > 0.9


def test_posteriors_and_forecast(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X)
    assert list(fit.posteriors.index) == list(X.index)
    np.testing.assert_allclose(fit.posteriors.sum(axis=1), 1.0)
    pd.testing.assert_series_equal(fit.filtered, fit.posteriors.iloc[-1])
    expected = fit.filtered.to_numpy() @ fit.transmat.to_numpy()
    np.testing.assert_allclose(fit.forecast().to_numpy(), expected / expected.sum())
    two = fit.filtered.to_numpy() @ fit.transmat.to_numpy() @ fit.transmat.to_numpy()
    np.testing.assert_allclose(fit.forecast(2).to_numpy(), two / two.sum())
    with pytest.raises(ValueError):
        fit.forecast(0)


def test_forecast_floor_keeps_every_regime_positive(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2, forecast_floor=0.2).fit(X)
    assert fit.forecast().min() >= 0.2 / (1 + 0.2) - 1e-12


def test_initial_distribution_is_fixed_and_uniform(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X)
    np.testing.assert_allclose(fit.startprob, [0.5, 0.5])


def test_more_starts_never_lower_the_likelihood(data):
    X, _ = data
    one = RegimeModel(n_regimes=2, n_starts=1).fit(X)
    many = RegimeModel(n_regimes=2, n_starts=5).fit(X)
    assert many.loglike >= one.loglike - 1e-9


def test_full_covariance_puts_the_prior_on_the_variances(data):
    X, _ = data
    model = RegimeModel(n_regimes=2, covariance="full", variance_prior=0.5)
    np.testing.assert_allclose(model._build(2).covars_prior, 0.5 * np.eye(2))
    fit = model.fit(X)
    for cov in fit.covariances:
        np.testing.assert_allclose(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_variance_prior_inflates_the_variances(data):
    X, _ = data
    plain = RegimeModel(n_regimes=2, variance_prior=0.0, tol=1e-8).fit(X)
    prior = RegimeModel(n_regimes=2, variance_prior=0.01, tol=1e-8).fit(X)
    assert (prior.variances.to_numpy().min()) > (plain.variances.to_numpy().min())


def test_warm_start_requires_a_matching_fit(data):
    X, _ = data
    first = RegimeModel(n_regimes=2).fit(X)
    with pytest.raises(ValueError):
        RegimeModel(n_regimes=3).fit(X, start_from=first)
    with pytest.raises(ValueError):
        RegimeModel(n_regimes=2).fit(X[["Mkt-RF"]], start_from=first)


def test_health_check(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X)
    assert _healthy(fit._model)
    broken = fit._model
    saved = broken.means_.copy()
    broken.means_ = np.full_like(saved, np.nan)
    assert not _healthy(broken)
    broken.means_ = saved


def test_inputs(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X.to_numpy())
    assert fit.factor_names == ["f0", "f1"]
    assert fit.nobs == len(X)
    one = RegimeModel(n_regimes=2).fit(X["Mkt-RF"])
    assert one.factor_names == ["Mkt-RF"]
    bad = X.copy()
    bad.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        RegimeModel(n_regimes=2).fit(bad)


def test_settings_and_overrides():
    model = RegimeModel(RegimeSettings(tol=0.1), n_regimes=2)
    assert (model.settings.tol, model.settings.n_regimes) == (0.1, 2)


# --------------------------------------------------------------------- labels


def test_regimes_are_labelled_from_one(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X)
    for frame in (fit.means, fit.transmat, fit.variances):
        assert list(frame.index) == [1, 2]
    assert list(fit.transmat.columns) == [1, 2]
    assert list(fit.posteriors.columns) == [1, 2]
    assert list(fit.forecast().index) == [1, 2]
    np.testing.assert_allclose(fit.covariance(2).to_numpy(), fit.covariances[1])
    assert list(fit.covariance(1).index) == ["Mkt-RF", "SMB"]
    with pytest.raises(KeyError):
        fit.covariance(0)


def test_relabel_permutes_everything_consistently(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X)
    swapped = fit.relabel([2, 1])
    np.testing.assert_allclose(swapped.means.to_numpy(), fit.means.to_numpy()[[1, 0]])
    np.testing.assert_allclose(
        swapped.transmat.to_numpy(), fit.transmat.to_numpy()[np.ix_([1, 0], [1, 0])]
    )
    np.testing.assert_allclose(
        swapped.posteriors.to_numpy(), fit.posteriors.to_numpy()[:, [1, 0]]
    )
    np.testing.assert_allclose(
        swapped.forecast().to_numpy(), fit.forecast().to_numpy()[[1, 0]]
    )
    np.testing.assert_allclose(swapped.covariance(1), fit.covariance(2))
    assert swapped.loglike == fit.loglike
    # The relabelled model still describes the data the same way.
    np.testing.assert_allclose(
        swapped._model.predict_proba(X.to_numpy()),
        swapped.posteriors.to_numpy(),
        atol=1e-10,
    )
    with pytest.raises(ValueError):
        fit.relabel([1, 1])
    with pytest.raises(ValueError):
        fit.relabel([0, 1])


def test_warm_start_keeps_the_labels_of_a_relabelled_fit(data):
    X, _ = data
    model = RegimeModel(n_regimes=2)
    first = model.fit(X.iloc[:300])
    plain = model.fit(X.iloc[:301], start_from=first)
    swapped = model.fit(X.iloc[:301], start_from=first.relabel([2, 1]))
    np.testing.assert_allclose(
        swapped.posteriors.to_numpy(),
        plain.posteriors.to_numpy()[:, [1, 0]],
        atol=1e-8,
    )


def test_alignment_undoes_a_relabelling(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X)
    swapped = fit.relabel([2, 1])
    assert swapped.alignment(fit).to_dict() == {1: 2, 2: 1}
    np.testing.assert_allclose(
        swapped.align_to(fit).means.to_numpy(), fit.means.to_numpy()
    )


def test_sort_by_a_factor_mean(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X)
    down = fit.sort_by("Mkt-RF")
    assert down.means["Mkt-RF"].is_monotonic_decreasing
    up = fit.sort_by("Mkt-RF", ascending=True)
    assert up.means["Mkt-RF"].is_monotonic_increasing
    # Sorting is idempotent and does not depend on the starting numbering.
    np.testing.assert_allclose(
        fit.relabel([2, 1]).sort_by("Mkt-RF").means.to_numpy(), down.means.to_numpy()
    )
    with pytest.raises(KeyError):
        fit.sort_by("HML")


# ---------------------------------------------------------- information criteria


def test_parameter_count():
    X, _ = _simulate(n=300)
    diag = RegimeModel(n_regimes=2).fit(X)
    full = RegimeModel(n_regimes=2, covariance="full").fit(X)
    assert diag.n_params == 2 * 1 + 2 * 2 + 2 * 2
    assert full.n_params == 2 * 1 + 2 * 2 + 2 * 3


def test_information_criteria_formulas(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X)
    ll, p, n = fit.loglike, fit.n_params, fit.nobs
    assert fit.aic == pytest.approx(-2 * ll + 2 * p)
    assert fit.bic == pytest.approx(-2 * ll + p * np.log(n))
    assert fit.hqic == pytest.approx(-2 * ll + 2 * p * np.log(np.log(n)))
    post = fit.posteriors.to_numpy()
    entropy = -np.sum(post * np.log(np.clip(post, 1e-300, 1)))
    assert fit.icl == pytest.approx(fit.bic + 2 * entropy)
    assert fit.icl >= fit.bic


def test_select_regimes_finds_two_regimes(data):
    from rwls import select_regimes

    X, _ = data
    sel = select_regimes(X, n_regimes=[1, 2, 3], n_starts=3)
    assert list(sel.table.index) == [1, 2, 3]
    assert {"loglike", "n_params", "AIC", "BIC", "HQIC", "ICL"} <= set(
        sel.table.columns
    )
    assert sel.selected["BIC"] == 2
    assert set(sel.fits) == {1, 2, 3}
    assert sel.fits[2].settings.n_regimes == 2


# --------------------------------------------------------------------- summary


def test_summary_content(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X)
    s = fit.summary()
    text = str(s)
    assert text.splitlines()[0].strip() == "Hidden Markov Regime Model"
    for part in ("Regime statistics", "Factor means (per period)", "Transition"):
        assert part in text
    assert all(len(line) <= 78 for line in text.splitlines())
    stats, means, vols, trans = s.tables
    np.testing.assert_allclose(stats["Persistence"], np.diag(fit.transmat))
    np.testing.assert_allclose(stats["Duration"], 1 / (1 - np.diag(fit.transmat)))
    np.testing.assert_allclose(stats["Occupancy"], fit.posteriors.mean())
    np.testing.assert_allclose(stats["Forecast"], fit.forecast())
    assert stats["Assigned"].sum() == pytest.approx(1.0)
    np.testing.assert_allclose(means.to_numpy(), fit.means.T.to_numpy())
    np.testing.assert_allclose(vols.to_numpy(), np.sqrt(fit.variances.T.to_numpy()))
    np.testing.assert_allclose(trans.to_numpy(), fit.transmat.to_numpy())
    assert list(trans.columns) == ["Regime 1", "Regime 2"]
    assert "<table" in s._repr_html_()


def test_summary_annualizes_on_request(data):
    X, _ = data
    fit = RegimeModel(n_regimes=2).fit(X)
    _, means, vols, _ = fit.summary(periods_per_year=12).tables
    np.testing.assert_allclose(means.to_numpy(), 12 * fit.means.T.to_numpy())
    np.testing.assert_allclose(
        vols.to_numpy(), np.sqrt(12 * fit.variances.T.to_numpy())
    )
    assert "Factor means (annualized)" in str(fit.summary(periods_per_year=12))
