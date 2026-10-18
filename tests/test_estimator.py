import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from rwls import RWLSSettings
from rwls.estimator import RWLS, SparseRegimeWarning
from rwls.regimes import RegimeModel

# --------------------------------------------------------- reference functions
# Copies of the research code (regimeaware.core.moments and core.estimation), so
# that the package can be checked against it without importing that project.


def ref_fit_wls(X, Y, weights):
    Xw = X * weights[:, None]
    coefs = np.linalg.solve(X.T @ Xw, Xw.T @ Y).T
    resid = Y - X @ coefs.T
    dof = max(weights.sum() - X.shape[1], 1.0)
    resid_var = np.maximum((weights[:, None] * resid**2).sum(axis=0) / dof, 0.0)
    return coefs, resid_var


def ref_guard_weights(weights, n_params, window):
    if weights.sum() > n_params + 1:
        return weights
    fallback = np.zeros(len(weights))
    fallback[-window:] = 1.0
    return fallback


def ref_regime_weights(probs, loadings, window=None):
    n_obs, n_states = probs.shape
    if loadings == "rwls":
        return [probs[:, s] for s in range(n_states)]
    labels = probs.argmax(axis=1)
    weights = []
    for s in range(n_states):
        w = np.zeros(n_obs)
        idx = np.flatnonzero(labels == s)[-window:]
        w[idx] = 1.0
        weights.append(w)
    return weights


def ref_factor_moments(X, w):
    total = w.sum()
    mean = (w[:, None] * X).sum(axis=0) / total
    dev = X - mean
    cov = (w[:, None] * dev).T @ dev / total
    return mean, np.diag(np.diag(cov))


# ------------------------------------------------------------------- the data


def _simulate(n=360, n_assets=12, seed=1):
    """Two regimes in two factors, and assets with regime-dependent loadings."""
    gen = GaussianHMM(n_components=2, covariance_type="diag", random_state=seed)
    gen.startprob_ = np.array([0.5, 0.5])
    gen.transmat_ = np.array([[0.95, 0.05], [0.10, 0.90]])
    gen.means_ = np.array([[0.01, 0.00], [-0.03, 0.02]])
    gen.covars_ = np.array([[0.0010, 0.0005], [0.0040, 0.0020]])
    X, Z = gen.sample(n)
    rng = np.random.default_rng(seed)
    beta = np.stack(
        [rng.normal(1.0, 0.3, (n_assets, 2)), rng.normal(1.3, 0.3, (n_assets, 2))]
    )
    R = np.einsum("tk,tnk->tn", X, beta[Z]) + rng.normal(0, 0.03, (n, n_assets))
    dates = pd.date_range("1990-01-31", periods=n, freq="ME")
    factors = pd.DataFrame(X, index=dates, columns=["Mkt-RF", "SMB"])
    returns = pd.DataFrame(R, index=dates, columns=[f"A{i}" for i in range(n_assets)])
    return returns, factors, Z, beta


@pytest.fixture(scope="module")
def setup():
    returns, factors, Z, beta = _simulate()
    regimes = RegimeModel(n_regimes=2).fit(factors)
    return returns, factors, regimes, Z, beta


def _design(factors, intercept=True):
    F = factors.to_numpy()
    return np.column_stack([np.ones(len(F)), F]) if intercept else F


# ---------------------------------------------------------------------- tests


def test_posterior_weighting_reproduces_the_research_code(setup):
    returns, factors, regimes, _, _ = setup
    res = RWLS(returns, factors, regimes).fit()
    X, P = _design(factors), regimes.posteriors.to_numpy()
    for j, label in enumerate(regimes.posteriors.columns):
        w = ref_guard_weights(ref_regime_weights(P, "rwls")[j], X.shape[1], 24)
        coefs, var = ref_fit_wls(X, returns.to_numpy(), w)
        np.testing.assert_allclose(
            res.params.loc[label].to_numpy(), coefs.T, atol=1e-12
        )
        np.testing.assert_allclose(res.resid_var.loc[label], var, atol=1e-14)
        mean, cov = ref_factor_moments(factors.to_numpy(), w)
        np.testing.assert_allclose(res.factor_means.loc[label], mean, atol=1e-14)
        np.testing.assert_allclose(res.factor_covariances[j], cov, atol=1e-14)
        np.testing.assert_allclose(res.mass.loc[label], w.sum())


def test_hard_classification_reproduces_the_research_code(setup):
    returns, factors, regimes, _, _ = setup
    res = RWLS(returns, factors, regimes, weighting="hard", hard_window=24).fit()
    X, P = _design(factors), regimes.posteriors.to_numpy()
    for j, label in enumerate(regimes.posteriors.columns):
        w = ref_guard_weights(ref_regime_weights(P, "hard", 24)[j], X.shape[1], 24)
        coefs, var = ref_fit_wls(X, returns.to_numpy(), w)
        np.testing.assert_allclose(
            res.params.loc[label].to_numpy(), coefs.T, atol=1e-12
        )
        np.testing.assert_allclose(res.resid_var.loc[label], var, atol=1e-14)
        assert res.mass.loc[label].iloc[0] == w.sum() <= 24


def test_joint_solution_equals_asset_by_asset_regressions(setup):
    returns, factors, regimes, _, _ = setup
    res = RWLS(returns, factors, regimes).fit()
    X = _design(factors)
    w = regimes.posteriors.iloc[:, 1].to_numpy()
    sw = np.sqrt(w)
    for asset in returns.columns[:3]:
        beta, *_ = np.linalg.lstsq(
            X * sw[:, None], returns[asset].to_numpy() * sw, rcond=None
        )
        np.testing.assert_allclose(res.params[asset].loc[2], beta, atol=1e-10)


def test_recovers_the_true_loadings(setup):
    returns, factors, regimes, Z, beta = setup
    fit = regimes.sort_by(
        "Mkt-RF"
    )  # regime 1 has the higher market mean, as in the truth
    res = RWLS(returns, factors, fit).fit()
    est = np.stack([res.loadings.loc[r].to_numpy().T for r in (1, 2)])
    assert np.abs(est - beta).max() < 0.25


def test_residual_variance_uses_the_posterior_mass(setup):
    returns, factors, regimes, _, _ = setup
    res = RWLS(returns, factors, regimes).fit()
    X, w = _design(factors), regimes.posteriors.iloc[:, 0].to_numpy()
    e = returns.to_numpy() - X @ res.params.loc[1].to_numpy()
    expected = (w[:, None] * e**2).sum(axis=0) / (w.sum() - 3)
    np.testing.assert_allclose(res.resid_var.loc[1], expected, rtol=1e-12)


def test_factor_moments_match_the_regime_model(setup):
    returns, factors, regimes, _, _ = setup
    res = RWLS(returns, factors, regimes).fit()
    # The same weighted means as the EM update, up to its convergence tolerance.
    np.testing.assert_allclose(res.factor_means, regimes.means, atol=1e-3)


def test_missing_returns_are_estimated_on_the_observed_periods(setup):
    returns, factors, regimes, _, _ = setup
    gappy = returns.copy()
    gappy.iloc[:100, 0] = np.nan
    res = RWLS(gappy, factors, regimes).fit()
    alone = RWLS(gappy.iloc[100:, [0]], factors, regimes).fit()
    np.testing.assert_allclose(res.params["A0"], alone.params["A0"], atol=1e-12)
    np.testing.assert_allclose(
        res.params["A1"], RWLS(returns, factors, regimes).fit().params["A1"]
    )
    assert res.nobs["A0"] == len(returns) - 100


def test_sparse_regime_falls_back_with_a_warning(setup):
    returns, factors, regimes, _, _ = setup
    with pytest.warns(SparseRegimeWarning):
        res = RWLS(returns, factors, regimes, min_mass=10_000).fit()
    assert res.fallback.to_numpy().all() and res.factor_fallback.all()
    # The pooled fallback is the regression without regimes.
    coefs, _ = ref_fit_wls(_design(factors), returns.to_numpy(), np.ones(len(returns)))
    np.testing.assert_allclose(res.params.loc[1].to_numpy(), coefs.T, atol=1e-12)
    with pytest.warns(SparseRegimeWarning):
        recent = RWLS(
            returns, factors, regimes, min_mass=10_000, fallback="recent"
        ).fit()
    coefs, _ = ref_fit_wls(
        _design(factors)[-24:], returns.to_numpy()[-24:], np.ones(24)
    )
    np.testing.assert_allclose(recent.params.loc[1].to_numpy(), coefs.T, atol=1e-12)


def test_without_intercept(setup):
    returns, factors, regimes, _, _ = setup
    res = RWLS(returns, factors, regimes, intercept=False).fit()
    assert list(res.params.index.get_level_values("term").unique()) == ["Mkt-RF", "SMB"]
    coefs, _ = ref_fit_wls(
        _design(factors, False),
        returns.to_numpy(),
        regimes.posteriors.iloc[:, 0].to_numpy(),
    )
    np.testing.assert_allclose(res.params.loc[1].to_numpy(), coefs.T, atol=1e-12)
    with pytest.raises(AttributeError):
        _ = res.intercepts


def test_result_layout(setup):
    returns, factors, regimes, _, _ = setup
    res = RWLS(returns, factors, regimes).fit()
    assert res.params.index.names == ["regime", "term"]
    assert list(res.params.columns) == list(returns.columns)
    assert list(res.intercepts.index) == [1, 2]
    assert res.loadings.shape == (4, returns.shape[1])
    assert list(res.resid_var.index) == [1, 2]
    assert res.n_regimes == 2


def test_inputs_align_on_common_periods(setup):
    returns, factors, regimes, _, _ = setup
    res = RWLS(returns.iloc[50:], factors, regimes).fit()
    assert res.index.equals(returns.index[50:])
    single = RWLS(returns["A3"], factors, regimes).fit()
    np.testing.assert_allclose(
        single.params["A3"], RWLS(returns, factors, regimes).fit().params["A3"]
    )
    probs = RWLS(returns, factors, regimes.posteriors).fit()
    np.testing.assert_allclose(
        probs.params, RWLS(returns, factors, regimes).fit().params
    )


def test_invalid_inputs(setup):
    returns, factors, regimes, _, _ = setup
    bad = regimes.posteriors * 2
    with pytest.raises(ValueError, match="sum to one"):
        RWLS(returns, factors, bad)
    with pytest.raises(TypeError):
        RWLS(returns, factors, regimes.posteriors.to_numpy())
    with pytest.raises(ValueError, match="no periods"):
        RWLS(
            returns.set_index(returns.index + pd.DateOffset(years=100)),
            factors,
            regimes,
        )


def test_settings_and_overrides(setup):
    returns, factors, regimes, _, _ = setup
    model = RWLS(
        returns, factors, regimes, RWLSSettings(hard_window=12), weighting="hard"
    )
    assert (model.settings.hard_window, model.settings.weighting) == (12, "hard")
