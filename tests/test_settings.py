import dataclasses

import pytest

from rwls import RegimeSettings, RWLSSettings


def test_regime_defaults_match_the_methodology():
    s = RegimeSettings()
    assert s.n_regimes == 3
    assert s.covariance == "diag"
    assert s.variance_prior == 0.01
    assert s.tol == 1e-4
    assert s.max_iter == 1000
    assert s.init_variance_floor == 0.01
    assert s.kmeans_restarts == 10
    assert s.n_starts == 10
    assert s.seed == 13
    assert s.transition_floor == 1e-8
    assert s.forecast_floor == 1e-12


def test_rwls_defaults_match_the_methodology():
    s = RWLSSettings()
    assert s.intercept is True
    assert s.weighting == "posterior"
    assert s.hard_window == 24
    assert s.min_mass is None
    assert s.fallback == "auto"
    assert s.use_t is True


def test_settings_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        RegimeSettings().tol = 0.1
    with pytest.raises(dataclasses.FrozenInstanceError):
        RWLSSettings().intercept = False


def test_replace_returns_a_modified_copy():
    base = RegimeSettings()
    changed = base.replace(tol=0.1, variance_prior=0.0)
    assert (changed.tol, changed.variance_prior) == (0.1, 0.0)
    assert (base.tol, base.variance_prior) == (1e-4, 0.01)


@pytest.mark.parametrize(
    "field, value",
    [
        ("n_regimes", 0),
        ("n_regimes", 2.0),
        ("covariance", "spherical"),
        ("variance_prior", -0.1),
        ("tol", 0.0),
        ("max_iter", 0),
        ("init_variance_floor", -1e-3),
        ("kmeans_restarts", 0),
        ("n_starts", 0),
        ("seed", 1.5),
        ("transition_floor", 1.0),
        ("forecast_floor", -1e-12),
    ],
)
def test_regime_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        RegimeSettings(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("intercept", 1),
        ("weighting", "soft"),
        ("hard_window", 0),
        ("min_mass", -1.0),
        ("fallback", "none"),
        ("use_t", "yes"),
    ],
)
def test_rwls_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        RWLSSettings(**{field: value})


def test_fallback_resolves_by_weighting():
    assert RWLSSettings().resolved_fallback() == "pooled"
    assert RWLSSettings(weighting="hard").resolved_fallback() == "recent"
    assert (
        RWLSSettings(weighting="hard", fallback="pooled").resolved_fallback()
        == "pooled"
    )


def test_min_mass_defaults_to_coefficients_plus_one():
    assert RWLSSettings().resolved_min_mass(7) == 8.0
    assert RWLSSettings(min_mass=20.0).resolved_min_mass(7) == 20.0


def test_integer_fields_accept_numpy_integers_and_reject_booleans():
    import numpy as np

    assert RegimeSettings(n_regimes=np.int64(2)).n_regimes == 2
    with pytest.raises(ValueError):
        RegimeSettings(n_regimes=True)


def test_selection_defaults_and_validation():
    s = RegimeSettings()
    assert s.selection == "best"
    assert s.modal_order_factor == 0
    assert (s.modal_mean_tol, s.modal_transition_tol) == (0.05, 0.05)
    assert RegimeSettings(modal_order_factor="Mkt-RF").modal_order_factor == "Mkt-RF"
    for bad in (
        {"selection": "median"},
        {"modal_order_factor": -1},
        {"modal_order_factor": 1.5},
        {"modal_mean_tol": 0.0},
        {"modal_transition_tol": -0.1},
    ):
        with pytest.raises(ValueError):
            RegimeSettings(**bad)
