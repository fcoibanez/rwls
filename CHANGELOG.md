# Changelog

All notable changes to this project are recorded here. Versions follow
[semantic versioning](https://semver.org); before 1.0, minor versions may change
the API.

## 0.1.0 (2026-10-18)

First usable release.

- Regime identification with a multivariate Gaussian hidden Markov model on
  factor returns: cold starts from several seeds with the highest-likelihood or
  most frequently reached model kept, warm-started refits, smoothed and filtered
  regime probabilities, regime forecasts, relabelling, alignment and sorting of
  regimes, information criteria and selection of the number of regimes.
- Regime-weighted least squares for large cross-sections, with posterior or hard
  classification weights, residual variances on each regime's posterior mass, a
  safeguard for regimes with little posterior mass, and assets with missing
  returns estimated on the periods they are observed.
- Inference: sandwich standard errors across regimes, t or normal p-values and
  confidence intervals, weighted R-squared, and a Wald test of equal loadings
  across regimes.
- Summaries in the layout of statsmodels, as text and as HTML in notebooks.
- Regime-conditional and mixture moments of asset returns in factor form.
- Re-estimation through time on the data available at each date.
- A loader for the Kenneth R. French Data Library that records the data vintage.
- Every estimator constant exposed as a documented setting.

## 0.0.1 (2026-10-18)

Placeholder release reserving the package name.
