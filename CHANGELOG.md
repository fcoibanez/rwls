# Changelog

All notable changes to this project are recorded here. Versions follow
[semantic versioning](https://semver.org); before 1.0, minor versions may change
the API.

## 0.2.0 (2026-10-20)

Corrections to the regime model. Fits can differ from those of 0.1.0.

- EM stops on the penalized log-likelihood, the objective its updates maximize
  when the variance prior is positive. Version 0.1.0 stopped on the
  log-likelihood, which those updates can lower, so many fits ended before they
  had converged.
- The variance prior is proportional to each factor's variance in the
  estimation sample: `variance_prior` is now a share of that variance, 0.01 by
  default. In 0.1.0 it was an absolute amount, large relative to the variance of
  monthly returns and dependent on their units.
- Cold starts begin from the full-sample variances raised by a share of
  themselves, `init_variance_floor` (0.01 by default), instead of by an absolute
  amount.
- Fits are grouped for `selection="modal"` with mean differences measured in
  standard deviations of each factor: `modal_mean_tol` is now 0.05 standard
  deviations instead of an absolute 0.0025. With these two changes, fits do not
  depend on the units or frequency of the returns.
- With several starts, `selection="best"` keeps the fit with the highest
  penalized log-likelihood.
- A refit that falls back to a cold start has its regimes aligned to those of
  the fit it started from, so that labels stay stable through time.

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
