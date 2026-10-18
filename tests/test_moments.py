import numpy as np
import pandas as pd
import pytest

from rwls import RWLS, RegimeModel
from rwls.moments import RegimeMoments
from test_estimator import _simulate


@pytest.fixture(scope="module")
def fitted():
    returns, factors, _, _ = _simulate(n_assets=8)
    regimes = RegimeModel(n_regimes=2).fit(factors)
    res = RWLS(returns, factors, regimes).fit()
    return res, regimes


def test_regime_moments_follow_the_factor_model(fitted):
    res, _ = fitted
    m = res.moments()
    for j, r in enumerate(res.regime_labels):
        B = res.loadings.loc[r].to_numpy().T  # (N, K)
        a = res.intercepts.loc[r].to_numpy()
        mu = a + B @ res.factor_means.loc[r].to_numpy()
        cov = B @ res.factor_covariances[j] @ B.T + np.diag(res.resid_var.loc[r])
        np.testing.assert_allclose(m.mean(r).to_numpy(), mu, atol=1e-15)
        np.testing.assert_allclose(m.cov(r).to_numpy(), cov, atol=1e-15)
        assert np.all(np.linalg.eigvalsh(cov) > 0)
    assert list(m.mean(1).index) == res.asset_names


def test_factor_form_variance_equals_the_dense_calculation(fitted):
    res, _ = fitted
    m = res.moments()
    w = np.random.default_rng(0).dirichlet(np.ones(len(res.asset_names)))
    for r in res.regime_labels:
        assert m.variance(w, r) == pytest.approx(w @ m.cov(r).to_numpy() @ w)
    named = pd.Series(w, index=res.asset_names)[::-1]  # order does not matter
    assert m.variance(named, 1) == pytest.approx(m.variance(w, 1))


def test_mixture_moments(fitted):
    res, regimes = fitted
    m = res.moments()
    p = regimes.forecast()
    mean, cov = m.mixture(p)
    means = [m.mean(r).to_numpy() for r in res.regime_labels]
    covs = [m.cov(r).to_numpy() for r in res.regime_labels]
    mu = sum(pi * mi for pi, mi in zip(p, means, strict=True))
    expected = sum(
        pi * (ci + np.outer(mi, mi)) for pi, ci, mi in zip(p, covs, means, strict=True)
    ) - np.outer(mu, mu)
    np.testing.assert_allclose(mean.to_numpy(), mu, atol=1e-15)
    np.testing.assert_allclose(cov.to_numpy(), expected, atol=1e-15)
    np.testing.assert_allclose(m.mixture_mean(p), mean)
    w = np.full(len(res.asset_names), 1 / len(res.asset_names))
    assert m.mixture_variance(w, p) == pytest.approx(w @ cov.to_numpy() @ w)


def test_a_degenerate_mixture_is_one_regime(fitted):
    res, _ = fitted
    m = res.moments()
    mean, cov = m.mixture([0.0, 1.0])
    np.testing.assert_allclose(mean, m.mean(2), atol=1e-15)
    np.testing.assert_allclose(cov, m.cov(2), atol=1e-15)


def test_sources_of_the_factor_moments(fitted):
    res, regimes = fitted
    model = res.moments("regime_model")
    np.testing.assert_allclose(model.factor_means, regimes.means.to_numpy())
    np.testing.assert_allclose(model.factor_covariances, regimes.covariances)
    known_means = np.array([[0.01, 0.0], [-0.03, 0.02]])
    known_covs = np.array([np.diag([0.001, 0.0005]), np.diag([0.004, 0.002])])
    known = res.moments(factor_means=known_means, factor_covariances=known_covs)
    np.testing.assert_allclose(known.factor_means, known_means)
    np.testing.assert_allclose(known.loadings, res.moments().loadings)
    with pytest.raises(ValueError, match="both"):
        res.moments(factor_means=known_means)
    with pytest.raises(ValueError, match="shape"):
        res.moments(factor_means=known_means[:1], factor_covariances=known_covs[:1])
    with pytest.raises(ValueError):
        res.moments("unknown")
    returns, factors = res._model.returns, res._model.factors
    plain = RWLS(returns, factors, regimes.posteriors).fit()
    with pytest.raises(ValueError, match="RegimeFit"):
        plain.moments("regime_model")


def test_without_intercepts_the_alphas_are_zero(fitted):
    res, regimes = fitted
    returns, factors = res._model.returns, res._model.factors
    m = RWLS(returns, factors, regimes, intercept=False).fit().moments()
    assert not m.alphas.any()


def test_input_checks(fitted):
    res, _ = fitted
    m = res.moments()
    with pytest.raises(ValueError, match="sum to one"):
        m.mixture([0.5, 0.6])
    with pytest.raises(ValueError, match="one finite entry"):
        m.mixture([1.0])
    with pytest.raises(ValueError, match="one entry per asset"):
        m.variance(np.ones(3), 1)
    with pytest.raises(ValueError, match="missing"):
        m.variance(pd.Series([1.0], index=["A0"]), 1)
    with pytest.raises(KeyError):
        m.mean(0)
    with pytest.raises(ValueError, match="shape"):
        RegimeMoments(
            [1],
            ["a"],
            ["f"],
            np.zeros((1, 2)),
            np.zeros((1, 1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1, 1)),
            np.zeros((1, 1)),
        )
