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


# ------------------------------------------------------------------ inference


def _ref_sandwich(X, y, weights, thetas):
    """Per-asset sandwich covariance across regimes, written out directly."""
    M, k = len(weights), X.shape[1]
    A = [np.linalg.inv(X.T @ np.diag(w) @ X) for w in weights]
    e = [y - X @ th for th in thetas]
    V = np.zeros((M * k, M * k))
    for s in range(M):
        for r in range(M):
            mid = sum(
                np.outer(X[t], X[t]) * weights[s][t] * weights[r][t] * e[s][t] * e[r][t]
                for t in range(len(y))
            )
            V[s * k : (s + 1) * k, r * k : (r + 1) * k] = A[s] @ mid @ A[r].T
    return V


def test_sandwich_covariance_matches_a_direct_calculation(setup):
    returns, factors, regimes, _, _ = setup
    res = RWLS(returns.iloc[:120], factors, regimes).fit()
    X = _design(factors.iloc[:120])
    P = regimes.posteriors.iloc[:120].to_numpy()
    weights = [P[:, 0], P[:, 1]]
    for asset in ["A0", "A5"]:
        thetas = [res.params[asset].loc[r].to_numpy() for r in (1, 2)]
        ref = _ref_sandwich(X, returns[asset].iloc[:120].to_numpy(), weights, thetas)
        np.testing.assert_allclose(res.cov_params(asset).to_numpy(), ref, rtol=1e-9)
    np.testing.assert_allclose(
        res.bse["A0"].to_numpy(), np.sqrt(np.diag(res.cov_params("A0"))), rtol=1e-12
    )


def test_one_regime_gives_the_robust_ols_covariance(setup):
    returns, factors, _, _, _ = setup
    one = pd.DataFrame(1.0, index=returns.index, columns=[1])
    res = RWLS(returns, factors, one).fit()
    X, y = _design(factors), returns["A2"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    hc0 = XtXi @ (X.T * e**2) @ X @ XtXi
    np.testing.assert_allclose(res.cov_params("A2").to_numpy(), hc0, rtol=1e-9)
    r2 = 1 - (e**2).sum() / ((y - y.mean()) ** 2).sum()
    assert res.rsquared.loc[1, "A2"] == pytest.approx(r2)
    assert res.df_resid.loc[1, "A2"] == len(y) - 3
    with pytest.raises(ValueError):
        res.wald_equal_loadings()


def test_t_and_normal_inference(setup):
    returns, factors, regimes, _, _ = setup
    t_res = RWLS(returns, factors, regimes).fit()
    n_res = RWLS(returns, factors, regimes, use_t=False).fit()
    pd.testing.assert_frame_equal(t_res.tvalues, n_res.tvalues)
    assert (n_res.pvalues <= t_res.pvalues + 1e-15).all().all()
    from scipy import stats

    z = np.abs(n_res.tvalues.to_numpy())
    np.testing.assert_allclose(n_res.pvalues.to_numpy(), 2 * stats.norm.sf(z))
    ci = t_res.conf_int(0.05)
    assert list(ci.columns[:2]) == [("A0", "lower"), ("A0", "upper")]
    mid = (ci.xs("lower", axis=1, level=1) + ci.xs("upper", axis=1, level=1)) / 2
    np.testing.assert_allclose(mid.to_numpy(), t_res.params.to_numpy())
    wide_t = ci.xs("upper", axis=1, level=1) - ci.xs("lower", axis=1, level=1)
    nci = n_res.conf_int(0.05)
    wide_n = nci.xs("upper", axis=1, level=1) - nci.xs("lower", axis=1, level=1)
    assert (wide_n.to_numpy() < wide_t.to_numpy()).all()


def test_wald_statistic_matches_a_direct_calculation(setup):
    returns, factors, regimes, _, _ = setup
    for use_t in (True, False):
        res = RWLS(returns, factors, regimes, use_t=use_t).fit()
        theta = res.params["A1"].to_numpy()
        V = res.cov_params("A1").to_numpy()
        # Loadings of regime 2 minus those of regime 1; intercepts at 0 and 3.
        C = np.zeros((2, 6))
        C[0, [1, 4]] = [-1, 1]
        C[1, [2, 5]] = [-1, 1]
        d = C @ theta
        w = d @ np.linalg.solve(C @ V @ C.T, d)
        row = res.wald_equal_loadings().loc["A1"]
        if use_t:
            assert row["statistic"] == pytest.approx(w / 2)
            assert (row["df_num"], row["df_denom"]) == (2, len(returns) - 6)
        else:
            assert row["statistic"] == pytest.approx(w)
            assert np.isnan(row["df_denom"])


def test_wald_test_separates_equal_and_different_loadings():
    rng = np.random.default_rng(3)
    n, n_assets = 600, 20
    Z = (rng.random(n) < 0.4).astype(int)
    F = rng.normal(0, 0.04, (n, 2))
    probs = pd.DataFrame(np.eye(2)[Z], columns=[1, 2])
    idx = pd.RangeIndex(n)
    factors = pd.DataFrame(F, index=idx, columns=["Mkt-RF", "SMB"])
    noise = rng.normal(0, 0.03, (n, n_assets))
    same = pd.DataFrame(F @ np.full((2, n_assets), 1.0) + noise, index=idx)
    beta = np.where(Z[:, None] == 1, 1.5, 0.5)
    differ = pd.DataFrame((F.sum(axis=1)[:, None] * beta) + noise, index=idx)
    p_same = RWLS(same, factors, probs).fit().wald_equal_loadings()["pvalue"]
    p_differ = RWLS(differ, factors, probs).fit().wald_equal_loadings()["pvalue"]
    assert p_differ.max() < 1e-6
    assert p_same.median() > 0.2


def test_asset_view(setup):
    returns, factors, regimes, _, _ = setup
    res = RWLS(returns, factors, regimes).fit()
    a = res["A4"]
    pd.testing.assert_series_equal(a.params, res.params["A4"])
    pd.testing.assert_series_equal(a.bse, res.bse["A4"])
    pd.testing.assert_series_equal(a.pvalues, res.pvalues["A4"])
    pd.testing.assert_frame_equal(a.conf_int(), res.conf_int()["A4"])
    pd.testing.assert_series_equal(a.rsquared, res.rsquared["A4"])
    assert a.nobs == len(returns)
    assert (
        a.wald_equal_loadings()["pvalue"]
        == res.wald_equal_loadings().loc["A4", "pvalue"]
    )
    with pytest.raises(KeyError):
        res["nope"]


def test_inference_with_missing_returns(setup):
    returns, factors, regimes, _, _ = setup
    gappy = returns.copy()
    gappy.iloc[:100, 0] = np.nan
    res = RWLS(gappy, factors, regimes).fit()
    alone = RWLS(gappy.iloc[100:, [0]], factors, regimes).fit()
    np.testing.assert_allclose(res.cov_params("A0"), alone.cov_params("A0"), rtol=1e-10)
    np.testing.assert_allclose(res.rsquared["A0"], alone.rsquared["A0"])
